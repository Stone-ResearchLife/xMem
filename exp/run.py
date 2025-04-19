import copy
import time
import uuid
import torch
import json
import tqdm
import re
import logging
from enum import Enum
from pathlib import Path
from typing import Optional
from ures.string import format_memory
from ures.files import filter_files
from perf_estimator.config import Config
from utils import search_nvml_file, search_profiler_file
# from .config import ExperimentConfig
# from .trainer.trainer import ModelPreparer
from exp.config import ExperimentConfig
from exp.trainer.trainer import ModelPreparer


logger = logging.getLogger(__name__)


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
            raise RuntimeError(f"CUDA is not available. CUDA is mandatory for this experiment.")
        else:
            gpu_info = torch.cuda.get_device_properties(gpu_id)
        self._info = {
            "model": model_name,
            "batch_size": batch_size,
            "target_iteration": 3,
            "optimiser": optimizer,
            "device": gpu_id,
            'input_size': (86, 86, 3),
            "zero_grad_mode": None,
            'total_gpu_memory': gpu_info.total_memory,
            "used_gpu_memory": None,
            'gpu_name': gpu_info.name,
        }
        # supplement run_id with gpu name
        config.run_id = f"{config.run_id}/{str(gpu_info.name).replace(' ', '-')}"
        self.config = config

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
    def summary_json_path(self) -> Path:
        return self.config.base_dir.joinpath("summary.json")

    def load_json(self) -> dict:
        """
        Create a JSON file to store the summary of the experiment.

        Returns:
            dict: The summary data.

        """
        if self.summary_json_path.is_file():
            with open(self.summary_json_path) as f:
                summary_data = json.load(f)
        else:
            summary_data = {
                SummarySectionName.train.value: copy.deepcopy(self._info),
                SummarySectionName.config.value: self.config.model_dump(),
            }
            with open(self.summary_json_path, 'w') as f:
                json.dump(summary_data, f, indent=4)
        return summary_data

    def write_data2json(self, key_name: str, data: dict):
        summary_data = self.load_json()
        summary_data[key_name] = data
        with open(self.summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=4)

    def run_ddnmem(self):
        from exp.baselines.dnnmem import DNNmem
        model_p = self._get_model_p_instance()
        dnn = DNNmem(
            model= model_p.model,
            dataloader= model_p.dl,
            optimizer=model_p.optimiser,
            is_transformer=model_p.is_transformer,
            max_est_memory_in_bytes=self.gpu_total_memory,
        )
        dnn.estimate()
        _result = self._unified_format(
            name=SummarySectionName.DNNmem,
            memory=dnn.estimate_memory,
            runtime=dnn.execute_time,
            oom=dnn.estimate_memory > self.gpu_total_memory
        )
        return _result


    def run_solution(self):
        from exp.trainer import FastRunner
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
        time.sleep(1)

        self._get_solution_est(c_config)

        return c_config

    def run_ground_truth(self):
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
        finally:
            e_time = time.time_ns()
            torch.cuda.empty_cache()

        time.sleep(2)
        ground_truth = self._get_ground_truth(g_config)
        self._unified_format(
            name=SummarySectionName.groundtruth,
            memory=ground_truth,
            oom=oom,
            runtime= e_time - s_time,
        )

        return g_config

    def _unified_format(self, name: SummarySectionName, memory: int, oom: bool, runtime: int):
        memory = copy.deepcopy(memory)
        oom = copy.deepcopy(oom)
        runtime = copy.deepcopy(runtime)
        _formatted_data ={
            "tool": name.value,
            "memory": memory,
            "memory_str": format_memory(memory),
            "oom": oom,
            "runtime": runtime,
            "runtime_str": f"{runtime/10**9:.2f} seconds",
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

    def _get_solution_est(self, config: Config) -> dict:
        from exp.baselines.solution import MySolution
        model_p = self._get_model_p_instance()

        config.trainer.huggingface_enable = model_p.is_transformer
        config.trainer.huggingface_model_name = self.model_name
        result_dir = config.result_dir
        p_files = search_profiler_file(result_dir)

        if len(p_files) == 0:
            raise FileNotFoundError(f"No profiler file found at {result_dir}")

        solution = MySolution(
            batch_size=self.batch_size,
            max_gpu_memory_in_gb=self.gpu_total_memory,
            config=config,
            profiler_file=p_files[-1],
        )
        solution.estimate()
        return self._unified_format(
            name=SummarySectionName.solution,
            memory=solution.estimate_memory,
            oom=solution.oom,
            runtime=solution.execute_time
        )

    def _get_model_p_instance(self) -> ModelPreparer:
        return ModelPreparer(
            model_name=self.model_name,
            batch_size=self.batch_size,
            optimiser=self.optimiser
        )


class ExperimentRun:
    def __init__(self, config: ExperimentConfig):
        self._job_list: list[_ExperimentExecutor] = []
        self._config = config

    @property
    def base_dir(self) -> Path:
        """
        Returns: Path point to the base directory for the experiment.
        """
        _path = Path().home() / self._config.run_id
        return _path

    def _model_name_split(self, model_name: str) -> tuple[str, list[str]]:
        char_to_find = "/"
        pattern = re.escape(char_to_find)
        matches = re.finditer(pattern, model_name)
        indices= [str(match.start()) for match in matches]
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
            task_id: Optional[str] = None,
    ):
        project_name = self.base_dir.name
        if task_id is None:
            formatted_model_name, indices = self._model_name_split(model_name)
            if len(indices) == 0:
                task_id = uuid.uuid4().hex[:3]
            else:
                task_id = f"{'-'.join(indices)}-{uuid.uuid4().hex[:3]}"
        else:
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
        _exe_config = Config(
            name=project_name,
            run_id=run_id,
            save2tmp=False
        )
        _exe_config.debug = True

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
        gpu_id = 0
        for model in conf.models:
            for opt in conf.optimisers:
                for batch_number in range(conf.batch_range[0], conf.batch_range[1], conf.batch_range[2]):
                    self.add_task(
                        model_name=model,
                        batch_size=batch_number,
                        optimizer=opt,
                        gpu_id=gpu_id,
                    )

    def load_from_exist_data(self):
        all_summary_files = filter_files('summary.json', directory=str(self.base_dir), fuzz=False)
        if len(all_summary_files) == 0:
            raise FileNotFoundError(f"No summary.json file found at {self.base_dir}")

        for summary_file in all_summary_files:
            info_dir_name = Path(summary_file).parent.parent.name
            name_pieces = info_dir_name.split('_')
            self.add_task(
                model_name=name_pieces[0],
                batch_size=int(name_pieces[2]),
                optimizer=name_pieces[1],
                gpu_id=int(name_pieces[3]),
                task_id=name_pieces[4],
            )

    def run_group_truth(self):
        """
        Run the experiment.
        """
        if len(self._job_list) == 0:
            self.prepare_regular_experiments_data()
        for index, task in enumerate(tqdm.tqdm(self._job_list)):
            task.run_ground_truth()
            time.sleep(1)

    def run_estimation(self, estimators: list[SummarySectionName]):
        """
        Run the experiment.
        """
        if len(self._job_list) > 0:
            self._job_list: list[_ExperimentExecutor] = []
        self.load_from_exist_data()
            
        for index, task in enumerate(tqdm.tqdm(self._job_list)):
            summary_data = task.load_json()
            for est in estimators:
                if est.value not in summary_data.keys():
                    print(f"{task.model_name} estimated by {est.value}")
                    try:
                        if est == SummarySectionName.solution:
                            task.run_solution()
                        elif est == SummarySectionName.DNNmem:
                            task.run_ddnmem()
                        elif est == SummarySectionName.schedtune:
                            pass
                        elif est == SummarySectionName.LLmem:
                            pass
                        else:
                            raise ValueError(f"Unknown estimator: {est.value}")
                    except Exception as e:
                        logger.error(f"Error occurred while running {est.value}: {e}")
                        continue
                    time.sleep(1)


if __name__ == '__main__':
    from exp.config import TransformerExperiments
    config = TransformerExperiments()
    exp = ExperimentRun(config=config)
    est_list = [SummarySectionName.DNNmem]
    exp.run_estimation(estimators=est_list)




