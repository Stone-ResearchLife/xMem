import copy
import time
import uuid
import fire
import torch
import json
import tqdm
import re
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional
from ures.string import format_memory
from ures.files import filter_files
from perf_estimator.config import Config
from utils import search_nvml_file, search_profiler_file, search_snapshot_file
from exp.config import (
    ExperimentConfig,
    CNNExperiments,
    TransformerExperiments,
    LargeTransformerExperiments,
)
from exp.trainer.trainer import ModelPreparer
from paper_container.evaluations import Experiments
from ures.docker.container import Container

logger = logging.getLogger(__name__)

if torch.cuda.device_count() > 1:
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============== If Multiprocessing Needed =============== #
# This is a workaround for the issue with CUDA and multiprocessing
# Even in linux, better to set the start method to 'spawn'
# mp = torch.multiprocessing.set_start_method('spawn', force=True)
# ========================================================= #


class SummarySectionName(Enum):
    train = "train info"
    config = "config"
    groundtruth = "groundtruth"
    solution = "solution"
    schedtune = "SchedTune"
    DNNmem = "DNNmem"
    LLmem = "LLmem"


class _ExperimentExecutor:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        optimizer: str,
        gpu_id: int,
        config: Config,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is not available. CUDA is mandatory for this experiment."
            )
        else:
            gpu_info = torch.cuda.get_device_properties(gpu_id)
        self._info = {
            "model": model_name,
            "batch_size": batch_size,
            "target_iteration": 3,
            "optimiser": optimizer,
            "device": gpu_id,
            "input_size": (86, 86, 3),
            "zero_grad_mode": config.trainer.zero_out,
            "total_gpu_memory": gpu_info.total_memory,
            "used_gpu_memory": None,
            "gpu_name": gpu_info.name,
        }
        # supplement run_id with gpu name
        config.run_id = f"{config.run_id}/{str(gpu_info.name).replace(' ', '-')}"
        config.trainer.huggingface_model_name = model_name
        self._config = config

    @property
    def config(self) -> Config:
        return copy.deepcopy(self._config)

    @property
    def model_name(self):
        return self._info["model"]

    @property
    def batch_size(self):
        return self._info["batch_size"]

    @property
    def gpu_id(self):
        return self._info["device"]

    @property
    def gpu_total_memory(self):
        return self._info["total_gpu_memory"]

    @property
    def optimiser(self):
        return self._info["optimiser"]

    @property
    def zero_out(self):
        return self._info["zero_grad_mode"]

    @property
    def summary_json_path(self) -> Path:
        return self.config.base_dir.joinpath("summary.json")

    @property
    def _get_framework_mem(self) -> Optional[int]:
        ground_data = self.load_json()
        return int(
            ground_data.get(SummarySectionName.groundtruth.value, {}).get(
                "framework_mem", 0
            )
        )

    def load_json(self) -> dict:
        """
        Create a JSON file to store the summary of the experiment.

        Returns:
                        dict: The summary data.

        """
        if not self.summary_json_path.parent.is_dir():
            self.summary_json_path.parent.mkdir(parents=True, exist_ok=True)

        if self.summary_json_path.is_file():
            with open(self.summary_json_path) as f:
                summary_data = json.load(f)
        else:
            summary_data = {
                SummarySectionName.train.value: copy.deepcopy(self._info),
                SummarySectionName.config.value: self.config.model_dump(),
            }
            with open(self.summary_json_path, "w") as f:
                json.dump(summary_data, f, indent=4)
        return summary_data

    def write_data2json(self, key_name: str, data: dict):
        summary_data = self.load_json()
        summary_data[key_name] = data
        with open(self.summary_json_path, "w") as f:
            json.dump(summary_data, f, indent=4)

    def run_ddnmem(self, verification: bool = True):
        from exp.baselines.dnnmem import DNNmem

        model_p = self._get_model_p_instance()
        dnn = DNNmem(
            model=model_p.model,
            dataloader=model_p.dl,
            optimizer=model_p.optimiser,
            is_transformer=model_p.is_transformer,
            max_est_memory_in_bytes=self.gpu_total_memory,
        )
        dnn.estimate()
        return self._unified_format(
            name=SummarySectionName.DNNmem,
            memory=dnn.estimate_memory + self._get_framework_mem,
            runtime=dnn.execute_time,
            oom=dnn.estimate_memory > self.gpu_total_memory,
            verify=verification,
        )

    def run_schedtune(self, verification: bool = True):
        from exp.baselines.schedtune import ScheduleTune

        model_p = self._get_model_p_instance()
        schedtune = ScheduleTune(
            model=model_p.model,
            dataloader=model_p.dl,
            optimizer=model_p.optimiser,
            is_transformer=model_p.is_transformer,
            device_id=self.gpu_id,
        )
        schedtune.estimate()
        return self._unified_format(
            name=SummarySectionName.schedtune,
            memory=schedtune.estimate_memory,
            runtime=schedtune.execute_time,
            oom=schedtune.estimate_memory > self.gpu_total_memory,
            verify=verification,
        )

    def run_solution(self):
        from exp.trainer import FastRunner

        print(f"Solution: CPU-based Running...")
        runner = FastRunner(
            model_name=self.model_name,
            batch_size=self.batch_size,
            optimiser=self.optimiser,
            gpu_id=self.gpu_id,
            config=self.config,
        )
        try:
            c_config, oom = runner.train_on_cpu()
        except Exception as e:
            raise RuntimeError(f"Non OOM error occurred: {e}") from e
        time.sleep(5)
        result = self._get_solution_est(c_config)
        return result

    def run_ground_truth(self):
        import GPUtil
        from exp.trainer import FastRunner

        s_time = time.time_ns()
        runner = FastRunner(
            model_name=self.model_name,
            batch_size=self.batch_size,
            optimiser=self.optimiser,
            gpu_id=self.gpu_id,
            config=self.config,
        )
        try:
            g_config, oom = runner.train_on_gpu()
        except Exception as e:
            raise RuntimeError(f"Non OOM error occurred: {e}") from e

        e_time = time.time_ns()
        torch.cuda.empty_cache()
        time.sleep(2)
        frame_mem = GPUtil.getGPUs()[self.gpu_id].memoryUsed * 1024**2
        ground_truth = self._get_ground_truth(g_config)
        snap_truth = self._get_snap_ground_truth(g_config)
        result = self._unified_format(
            name=SummarySectionName.groundtruth,
            memory=ground_truth,
            oom=oom,
            runtime=e_time - s_time,
            verify=False,
            framework_mem=frame_mem,
            snap_truth=snap_truth,
        )
        return result

    def _get_mem_fraction(self, mem_in_bytes: float) -> float:
        return round(mem_in_bytes / self.gpu_total_memory, 2)

    def _verify_est_mem(self, mem_fraction: float) -> tuple[int, bool]:
        from exp.trainer import FastRunner

        mem_fraction = 1.0 if mem_fraction > 1 else mem_fraction
        runner = FastRunner(
            model_name=self.model_name,
            batch_size=self.batch_size,
            optimiser=self.optimiser,
            gpu_id=self.gpu_id,
            config=self.config,
        )
        try:
            g_config, oom = runner.train_on_gpu(fraction_gpu=mem_fraction)
        except Exception as e:
            raise RuntimeError(f"Non OOM error occurred: {e}") from e

        torch.cuda.empty_cache()
        # system take time writing data into fs, so waiting here for a while
        time.sleep(3)
        ground_truth = self._get_ground_truth(g_config)
        logger.warning(
            f"GPU {self.gpu_id}'s memory fraction: restore to {mem_fraction * 100}%"
        )
        torch.cuda.set_per_process_memory_fraction(
            1.0, device=torch.device(f"cuda:{self.gpu_id}")
        )
        return ground_truth, oom

    def _unified_format(
        self,
        name: SummarySectionName,
        memory: int,
        oom: bool,
        runtime: int,
        verify: bool = True,
        **kwargs,
    ):
        if memory > self.gpu_total_memory and oom is False:
            oom = True
        _formatted_data = {
            "tool": name.value,
            "memory": memory,
            "memory_str": format_memory(memory),
            "oom": oom,
            "runtime": runtime,
            "runtime_str": f"{runtime/10**9:.2f} seconds",
            "verification": {},
        }
        _formatted_data.update(kwargs)
        if verify and not oom:
            print(
                f"{name}: Verification of Estimated Memory {format_memory(memory)}..."
            )
            mem_fraction = self._get_mem_fraction(memory)
            ground, oom = self._verify_est_mem(mem_fraction=mem_fraction)
            _formatted_data["verification"] = {
                "ground": ground,
                "ground_str": format_memory(ground),
                "oom": oom,
            }
        self.write_data2json(name.value, _formatted_data)
        return _formatted_data

    def _get_ground_truth(self, config: Config):
        result_dir = config.result_dir
        n_files = search_nvml_file(result_dir)
        with open(n_files[-1], "r") as f:
            _data = json.load(f)
        _gpu_memory_usage = {}
        start_memory = {}
        for index, gpu_metric in enumerate(_data["records"]):
            gpu_data = gpu_metric["HostGPUs"]
            for device_id, data in gpu_data.items():
                if index == 0:
                    start_memory[device_id] = data["memory"]["used"]
                if device_id not in _gpu_memory_usage:
                    _gpu_memory_usage[device_id] = []
                _gpu_memory_usage[device_id].append(
                    data["memory"]["used"] - start_memory[device_id]
                )
        return max(_gpu_memory_usage[str(self.gpu_id)])

    def _get_snap_ground_truth(self, config: Config):
        from exp.snapshot import SnapshotAnalyser

        result_dir = config.result_dir
        s_files = search_snapshot_file(result_dir)
        snap = SnapshotAnalyser(str(s_files[-1]))
        seg, _ = snap.gpu_and_segment_max_memory_changes_data()
        return max(seg)

    def _get_solution_est(self, config: Config) -> dict:
        from exp.baselines.solution import MySolution

        print(f"Solution: Estimating memory usage...")
        model_p = self._get_model_p_instance()

        config.trainer.huggingface_enable = model_p.is_transformer
        config.trainer.huggingface_model_name = self.model_name
        result_dir = config.result_dir
        p_files = search_profiler_file(result_dir)

        if len(p_files) == 0:
            raise FileNotFoundError(f"No profiler file found at {result_dir}")

        solution = MySolution(
            batch_size=self.batch_size,
            max_gpu_memory_in_gb=self.gpu_total_memory / 1024**3,
            config=config,
            profiler_file=p_files[-1],
        )
        solution.estimate()
        est_date = self._unified_format(
            name=SummarySectionName.solution,
            memory=solution.estimate_memory + self._get_framework_mem,
            oom=solution.oom,
            runtime=solution.execute_time,
            fp16=config.trainer.fp16,
        )

        return est_date

    def _get_model_p_instance(self) -> ModelPreparer:
        return ModelPreparer(
            model_name=self.model_name,
            batch_size=self.batch_size,
            optimiser=self.optimiser,
            fp16=self.config.trainer.fp16,
        )


class ExperimentRun:
    def __init__(self, config: ExperimentConfig):
        self._job_list: list[_ExperimentExecutor] = []
        self._config = config

    @property
    def jobs(self) -> list[_ExperimentExecutor]:
        return self._job_list

    @property
    def base_dir(self) -> Path:
        """
        Returns: Path point to the base directory for the experiment.
        """
        _path = Path().home() / self._config.run_id
        if not _path.is_dir():
            _path.mkdir(parents=True, exist_ok=True)
        return _path

    def _model_name_split(self, model_name: str) -> tuple[str, list[str]]:
        char_to_find = "/"
        pattern = re.escape(char_to_find)
        matches = re.finditer(pattern, model_name)
        indices = [str(match.start()) for match in matches]
        if len(indices) == 0:
            return model_name, []
        else:
            model_name = model_name.replace("/", "-")
            return model_name, indices

    def add_task(
        self,
        model_name: str,
        batch_size: int,
        optimizer: str,
        gpu_id: int,
        zero_out: int = 0,
        task_id: Optional[str] = None,
    ):
        project_name = self.base_dir.name
        if task_id is None:
            formatted_model_name, indices = self._model_name_split(model_name)
            task_uuid = str(uuid.uuid4().hex[:4])
            if len(indices) == 0:
                task_id = task_uuid
            else:
                task_id = f"{'-'.join(indices)}-{task_uuid}"
        else:
            # Ensure task_id is a string
            # task_is is treated as an int when task_id consists of digits
            task_id = str(task_id)
            # The code provides a capability to restore the model name
            formatted_model_name = model_name
            task_id_pieces = task_id.split("-")
            if len(task_id_pieces) > 1:
                model_name_list = list(model_name)
                for index in task_id_pieces[:-1]:
                    model_name_list[int(index)] = "/"
                model_name = "".join(model_name_list)

        logger.info(f"Task ID: {task_id} has been created for {model_name}")
        run_id = f"{formatted_model_name}_{optimizer}_{batch_size}_{gpu_id}_{task_id}"
        _exe_config = Config(name=project_name, run_id=run_id, save2tmp=False)
        _exe_config.trainer.zero_out = zero_out
        _exe_config.trainer.fp16 = self._config.fp16
        _exe_config.trainer.bf16 = self._config.bf16
        _exe_config.debug = self._config.debug

        _exe_instance = _ExperimentExecutor(
            model_name=model_name,
            batch_size=batch_size,
            optimizer=optimizer,
            gpu_id=gpu_id,
            config=_exe_config,
        )
        self._job_list.append(_exe_instance)

    def prepare_regular_experiments_data(self):
        """
        Run CNN experiments with the given configuration.
        """
        conf = self._config
        gpu_id = conf.gpu_id
        for model in conf.models:
            for opt in conf.optimisers:
                for batch_number in range(
                    conf.batch_range[0], conf.batch_range[1], conf.batch_range[2]
                ):
                    for i in range(conf.repeats):
                        self.add_task(
                            model_name=model,
                            batch_size=batch_number,
                            optimizer=opt,
                            gpu_id=gpu_id,
                        )

    def prepare_monte_carlo_experiments_data(
        self, number: int, gpus: Optional[list[int]] = None
    ):
        assert isinstance(number, int)
        import random

        conf = self._config
        for index in range(number):
            model_name = random.choice(conf.models)
            batch_size = random.randint(conf.batch_range[0], conf.batch_range[1])
            optimizer = random.choice(conf.optimisers)
            gpu_id = random.choice(gpus or [0, 1])
            zero_out = random.choice([0, 1, 2])
            self.add_task(
                model_name=model_name,
                batch_size=batch_size,
                optimizer=optimizer,
                gpu_id=gpu_id,
                zero_out=zero_out,
            )

    def load_from_exist_data(self):
        all_summary_files = filter_files(
            "summary.json", directory=str(self.base_dir), fuzz=False
        )
        if len(all_summary_files) == 0:
            raise FileNotFoundError(f"No summary.json file found at {self.base_dir}")

        for summary_file in all_summary_files:
            info_dir_name = Path(summary_file).parent.parent.name
            name_pieces = info_dir_name.split("_")
            self.add_task(
                model_name=name_pieces[0],
                batch_size=int(name_pieces[2]),
                optimizer=name_pieces[1],
                gpu_id=int(name_pieces[3]),
                task_id=str(name_pieces[4]),
            )

    def run_group_truth(self, in_docker: bool = False):
        """
        Run the experiment to get the ground truth.
        """
        if len(self._job_list) == 0:
            self.prepare_regular_experiments_data()

        if in_docker:
            exp = Experiments()
            for index, task in enumerate(tqdm.tqdm(self._job_list)):
                self._build_container(
                    container_manager=exp,
                    task=task,
                    ground=True,
                    paper=True,
                    fp16=self._config.fp16,
                    enable_large_model=isinstance(
                        self._config, LargeTransformerExperiments
                    ),
                )
            exp.execute(manual_container=True)
        else:
            for index, task in enumerate(tqdm.tqdm(self._job_list)):
                task.run_ground_truth()
                torch.cuda.empty_cache()
                time.sleep(2)

    def _build_container(
        self,
        container_manager: Experiments,
        task: _ExperimentExecutor,
        dnnmem: bool = False,
        schedtune: bool = False,
        llmem: bool = False,
        paper: bool = False,
        ground: bool = False,
        **kwargs,
    ) -> Optional[Container]:
        run_id = str(task.config.run_id).split("/")[0]
        task_id = str(str(run_id).split("_")[-1])
        formatted_model_name = run_id.split("_")[0]
        args = {
            "model": task.model_name,
            "batch": task.batch_size,
            "optimizer": task.optimiser,
            "gpu_id": task.gpu_id,
            "zero_out": task.zero_out,
            "task_id": task_id,
            "is_transformer": (
                True if isinstance(self._config, TransformerExperiments) else False
            ),
            "paper": paper,
            "dnnmem": dnnmem,
            "schedtune": schedtune,
            "llmem": llmem,
            "ground": ground,
            "debug": self._config.debug,
            "fp16": kwargs.get("fp16", False),
            "enable_large_model": kwargs.get("enable_large_model", False),
            "result_verification": kwargs.get("result_verification", True),
        }

        # Only add the container if at least one estimator is selected
        if any(
            [
                args["paper"],
                args["dnnmem"],
                args["schedtune"],
                args["llmem"],
                args["ground"],
            ]
        ):
            containers = container_manager.add_container(**args)
        else:
            containers = None
        return containers

    def basic_info(self):
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch built with CUDA version: {torch.version.cuda}")
        print(f"CUDA version: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")

    def run_monte_carlo_experiments(self, number: int = 600):
        self.basic_info()
        self._job_list: list[_ExperimentExecutor] = []
        self.prepare_monte_carlo_experiments_data(number)
        print(f"{'='*10} Monte Carlo Experiment #{len(self._job_list)} runs {'='*10}")
        if isinstance(self._config, LargeTransformerExperiments):
            self.run_experiments_solving_compatibility_issue()
        else:
            self.run_experiments()

    def run_anova_experiments(self):
        self.basic_info()
        self._job_list: list[_ExperimentExecutor] = []
        self.prepare_regular_experiments_data()
        print(f"{'='*10} ANOVA Experiment #{len(self._job_list)} runs {'='*10}")
        if isinstance(self._config, LargeTransformerExperiments):
            self.run_experiments_solving_compatibility_issue()
        else:
            self.run_experiments()

    def run_experiments(self):
        est_list = [SummarySectionName.DNNmem, SummarySectionName.schedtune]
        self.run_group_truth(in_docker=True)
        if isinstance(self._config, TransformerExperiments):
            est_list.append(SummarySectionName.LLmem)
        self.run_estimation(estimators=est_list, in_docker=True)
        if isinstance(self._config, TransformerExperiments):
            self.verify_llmem_result()

    def run_experiments_solving_compatibility_issue(self):
        est_list = [
            SummarySectionName.DNNmem,
        ]
        self._config.result_verification = True
        self.run_group_truth(in_docker=True)
        if isinstance(self._config, TransformerExperiments):
            est_list.append(SummarySectionName.LLmem)
        self.run_estimation(estimators=est_list, in_docker=True)
        if isinstance(self._config, TransformerExperiments):
            self.verify_llmem_result()
        print("Dedicatly Run SchedTune to resolve python package confilct issue")
        self._config.result_verification = False
        self.run_estimation(estimators=[SummarySectionName.schedtune], in_docker=True)
        self.verify_schedtune_result()

    def run_estimation(
        self,
        estimators: list[SummarySectionName],
        force: bool = False,
        in_docker: bool = False,
    ) -> Optional[dict]:
        """
        Run the experiment to get the estimated results.
        """
        if len(self._job_list) == 0:
            self._job_list: list[_ExperimentExecutor] = []
            self.load_from_exist_data()

        if in_docker:
            exp = Experiments()
            containers = []
            print("================== Create docker containers ==================")
            for index, task in enumerate(tqdm.tqdm(self._job_list)):
                summary_data = task.load_json()
                args = {
                    "container_manager": exp,
                    "task": task,
                    "paper": False,
                    "dnnmem": False,
                    "schedtune": False,
                    "llmem": False,
                    "ground": False,
                    "fp16": self._config.fp16,
                    "enable_large_model": isinstance(
                        self._config, LargeTransformerExperiments
                    ),
                    "result_verification": self._config.result_verification,
                }
                for est in estimators:
                    if est.value in summary_data.keys() and force is False:
                        # Skip the task if the estimator is already present in the summary
                        continue
                    if est == SummarySectionName.solution:
                        args["paper"] = True
                    elif est == SummarySectionName.DNNmem:
                        args["dnnmem"] = True
                    elif est == SummarySectionName.schedtune:
                        args["schedtune"] = True
                    elif est == SummarySectionName.LLmem:
                        args["llmem"] = True

                container = self._build_container(**args)
                if container is not None:
                    containers.extend(container)
                time.sleep(2)
            print("================== Execute docker containers ==================")
            exp.execute(manual_container=True)
            print("================== Statistics ==================")
            print(
                f"Run(success/total): {len([(int(cont.exit_code) == 0) is True for cont in containers])}/{len(containers)}"
            )
        else:
            summary = {}
            for index, task in enumerate(tqdm.tqdm(self._job_list)):
                summary_data = task.load_json()
                results = []
                for est in estimators:
                    if est.value not in summary_data.keys() or force is True:
                        print(f"{task.model_name} estimated by {est.value}")
                        try:
                            if est == SummarySectionName.solution:
                                result = task.run_solution()
                            elif est == SummarySectionName.DNNmem:
                                result = task.run_ddnmem(
                                    self._config.result_verification
                                )
                            elif est == SummarySectionName.schedtune:
                                result = task.run_schedtune(
                                    self._config.result_verification
                                )
                            elif est == SummarySectionName.LLmem:
                                logger.warning(
                                    "LLmem Estimator could be only run in-docker mode, skipped"
                                )
                            else:
                                raise ValueError(f"Unknown estimator: {est.value}")
                        except Exception as e:
                            logger.error(
                                f"Error occurred while running {est.value}: {e}"
                            )
                            continue
                        else:
                            results.append(result)
                        time.sleep(1)
                if not in_docker:
                    summary[task.config.run_id] = results
            return summary

    def statistics(self):
        from ures.tools.enum import EnumManipulator

        key_section_enum = EnumManipulator(SummarySectionName)
        key_list = key_section_enum.fetch_keys()
        self._job_list: list[_ExperimentExecutor] = []
        self.load_from_exist_data()
        summary = {
            "total": len(self._job_list),
        }
        for index, task in enumerate(tqdm.tqdm(self._job_list)):
            summary_data = task.load_json()
            for key_name in key_list:
                if key_name not in summary:
                    summary[key_name] = 0
                if key_section_enum.fetch_value(key_name) in summary_data.keys():
                    summary[key_name] += 1

        print(
            f"=============== Statistics for {self._config.run_id} =================="
        )
        for key_name in key_list:
            print(f"{key_name}: {summary[key_name]}/{summary['total']}")

    def verify_llmem_result(self):
        if len(self._job_list) == 0:
            self._job_list: list[_ExperimentExecutor] = []
            self.load_from_exist_data()
        for index, task in enumerate(tqdm.tqdm(self._job_list)):
            llmem_file = task.config.base_dir.joinpath("llmem_result.json")
            if llmem_file.is_file():
                with open(llmem_file) as f:
                    llmem_json = json.load(f)
                is_supported = llmem_json["support"]
                task._unified_format(
                    name=SummarySectionName.LLmem,
                    memory=llmem_json["memory"],
                    runtime=llmem_json["time"],
                    oom=llmem_json["oom"],
                    version=is_supported,
                )

    def verify_schedtune_result(self):
        if len(self._job_list) == 0:
            self._job_list: list[_ExperimentExecutor] = []
            self.load_from_exist_data()
        for index, task in enumerate(tqdm.tqdm(self._job_list)):
            summary_file = task.config.base_dir.joinpath("summary.json")
            if summary_file.is_file():
                with open(summary_file) as f:
                    summary_json = json.load(f)
                if SummarySectionName.schedtune.value in summary_json.keys():
                    sche_data = summary_json[SummarySectionName.schedtune.value]
                    task._unified_format(
                        name=SummarySectionName.schedtune,
                        memory=sche_data["memory"],
                        runtime=sche_data["runtime"],
                        oom=sche_data["oom"],
                        verify=True,
                    )

    def to_evaluation_result(self):
        """
        In order to reduce redundant work for ploting diagram, the function is used to
        convert the summary.json file to the old form of the evaluation result.
        """
        self._job_list: list[_ExperimentExecutor] = []
        self.load_from_exist_data()
        pandas_list = []
        for index, task in enumerate(tqdm.tqdm(self._job_list)):
            summary_data = task.load_json()
            old_evaluation_json_path = task.summary_json_path.parent.joinpath(
                "evaluation_result.json"
            )
            old_summary_data = {
                SummarySectionName.train.value: copy.deepcopy(
                    summary_data[SummarySectionName.train.value]
                ),
                SummarySectionName.config.value: copy.deepcopy(
                    summary_data[SummarySectionName.config.value]
                ),
            }
            # get groundtruth
            if SummarySectionName.groundtruth.value not in summary_data.keys():
                logger.warning(f"Ground truth not found for {task.config.run_id}")
                continue

            gt_data = summary_data[SummarySectionName.groundtruth.value]
            gt_mem_1st = gt_data["memory"]
            gt_oom_1st = gt_data["oom"]

            for est in [
                SummarySectionName.schedtune,
                SummarySectionName.solution,
                SummarySectionName.DNNmem,
                SummarySectionName.LLmem,
            ]:
                if est.value in summary_data.keys():
                    est_data = summary_data[est.value]
                    formatted_data = self.format_old_json_data(
                        name=est_data["tool"],
                        memory=est_data["memory"],
                        oom=est_data["oom"],
                        runtime=est_data["runtime"],
                        ground=gt_mem_1st,
                        real_oom=gt_oom_1st,
                        verification_oom=est_data.get("verification", {}).get(
                            "oom", None
                        ),
                        verification_ground=est_data.get("verification", {}).get(
                            "ground", None
                        ),
                    )
                    old_summary_data[est.value] = formatted_data
                    pandas_list.append(formatted_data)

            with open(old_evaluation_json_path, "w") as f:
                json.dump(old_summary_data, f, indent=4)

        return pandas_list

    def format_old_json_data(
        self,
        name: str,
        memory: int,
        oom: bool,
        runtime: int,
        ground: int,
        real_oom: bool,
        verification_ground: Optional[int] = None,
        verification_oom: bool = True,
    ):
        if verification_oom is True:
            verification_ground = None
        _data = {
            "tool": name,
            "memory": memory,
            "oom": oom,
            "runtime": runtime,
            "ground": ground,
            "error": abs(memory - ground) / ground,
            "real_oom": real_oom,
            "correct_estimation": oom == real_oom,
            "2nd verification": {
                "oom": verification_oom,
                "error": (
                    abs(verification_ground - memory) / ground
                    if verification_ground is not None
                    else None
                ),
            },
        }
        return _data


def estimate(
    model: str,
    batch: int,
    optimizer: str,
    gpu_id: int,
    zero_out: int = 0,
    task_id: Optional[str] = None,
    is_transformer: bool = True,
    dnnmem: bool = False,
    llmem: bool = False,
    schedtune: bool = False,
    paper: bool = False,
    ground: bool = False,
    debug: bool = False,
    fp16: bool = False,
    enable_large_model: bool = False,
    result_verification: bool = False,
):
    if not any([llmem, schedtune, paper, dnnmem, ground]):
        raise ValueError(
            f"At least one estimator should be selected.(dnnmem, llmem, schedtune, paper, ground)"
        )
    else:
        estimate_list = []
        if dnnmem:
            estimate_list.append(SummarySectionName.DNNmem)
        if llmem:
            estimate_list.append(SummarySectionName.LLmem)
        if schedtune:
            estimate_list.append(SummarySectionName.schedtune)
        if paper:
            estimate_list.append(SummarySectionName.solution)

    conf = TransformerExperiments() if is_transformer else CNNExperiments()
    if enable_large_model:
        conf = LargeTransformerExperiments()
    conf.fp16 = bool(fp16)
    conf.debug = bool(debug)
    conf.result_verification = bool(result_verification)
    exp = ExperimentRun(config=conf)
    exp.add_task(
        model_name=model,
        batch_size=batch,
        optimizer=optimizer,
        gpu_id=gpu_id,
        task_id=str(task_id),
        zero_out=zero_out,
    )

    if ground:
        exp.run_group_truth()
        time.sleep(2)
        torch.cuda.empty_cache()
        time.sleep(2)

    results = exp.run_estimation(estimators=estimate_list)
    print(results)


if __name__ == "__main__":
    fire.Fire(estimate)
