import time
import torch
import logging
import os
import copy
import json
import GPUtil
from uuid import uuid4
from typing import Optional, Union, Tuple
from perf_estimator.models import AllModels
from perf_estimator.utilis.utilis import filter_files
from perf_estimator.dataset import image_dataset
from perf_estimator.config import Config
from perf_estimator.allocator import AllocatorSim, CachingAllocator
from perf_estimator.estimator import Estimator
from perf_estimator.log import init_logging
from experiments.snapshot import SnapshotAnalyser
from experiments.trainer import (
    ModelTrainer,
    ProfilerPlugin,
    SnapshotPlugin,
    HostMonitorPlugin,
)


logger = logging.getLogger(__name__)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
torch.backends.cuda.cufft_plan_cache.max_size = 1


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: Optional[torch.utils.data.DataLoader] = None,
        batch_size: int = 200,
        input_size: int = 86,
        max_gpu_memory_in_gb: Union[int, float] = 4,
        config: Optional[Config] = None,
        run_id: Optional[str] = None,
    ):
        run_id = run_id or f"{model.__class__.__name__}-{batch_size}-{uuid4().hex[:8]}"
        self._model = model
        if data_loader is None:
            data_loader = image_dataset(
                batch=batch_size, image_size=(input_size, input_size)
            )

        self._data_loader = data_loader
        self._batch_size = batch_size
        self._config = config or Config(run_id=run_id, save2tmp=False)
        self._max_gpu_memory_in_gb = max_gpu_memory_in_gb

    @property
    def conf(self) -> Config:
        return self._config

    @property
    def allocator_sim(self) -> AllocatorSim:
        return self._sim

    def _train(self, **kwargs):
        logger.debug(f"Start Training with {kwargs}")
        trainer = ModelTrainer(**kwargs)
        torch.cuda.empty_cache()
        time.sleep(1)
        trainer.train()
        time.sleep(1)

    def train_on_gpu(
        self,
        gpu_id: int = 0,
        iteration: int = 2,
        limit_in_gp: Optional[int] = None,
        entire_training: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: Optional[torch.nn.Module] = None,
        zero_grad_mode: int = 1,
    ) -> tuple:
        torch.cuda.empty_cache()
        time.sleep(1)  # wait for the memory to be released
        permanent_target_device_memory_in_gb = (
            self._config.permanent_gpu_memory_in_gb.get(gpu_id, 0)
        )
        if limit_in_gp is None:
            limit_in_gp = self._max_gpu_memory_in_gb
        else:
            limit_in_gp = limit_in_gp + permanent_target_device_memory_in_gb

        total_gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        fraction = (limit_in_gp * 1024**3) / total_gpu_memory
        if fraction > 1:
            logger.warning(
                f"The limit is over the total GPU memory, set to 1. input max: {limit_in_gp}GB, total: {total_gpu_memory/1024**3}GB"
            )
            fraction = 1
        logger.info(
            f"Set GPU Memory Fraction: {round(fraction, 2)*100}%, limit: {limit_in_gp}GB"
        )
        torch.cuda.set_per_process_memory_fraction(
            round(fraction, 2), device=torch.device(f"cuda:{gpu_id}")
        )
        iteration = iteration - 1
        if entire_training:
            iteration = None
        trainer_conf = {
            "model": copy.deepcopy(self._model),
            "data_loader": copy.deepcopy(self._data_loader),
            "batch_size": self._batch_size,
            "iterations": iteration,
            "on_cpu": False,
            "config": self._config,
            "gpu_id": gpu_id,
            "plugins": [
                HostMonitorPlugin(
                    interval_ms=1,
                    cpu_enable=False,
                    gpu_enable=True,
                    network_enable=False,
                    config=self._config,
                ),
                SnapshotPlugin(config=self._config),
            ],
            "optimiser": optimizer,
            "loss": loss,
            "zero_grad_mode": zero_grad_mode,
        }
        logger.info(f"================== Evaluate on GPU: {gpu_id} ==================")
        try:
            self._train(**trainer_conf)
        finally:
            host_monitor_files = filter_files(
                "host_metrics", str(self._config.result_dir), fuzz=True
            )
            snapshot_files = filter_files(
                ".pickle", str(self._config.result_dir), fuzz=True
            )
            host_monitor_files.sort(key=lambda x: os.path.getmtime(x))
            snapshot_files.sort(key=lambda x: os.path.getmtime(x))
        return host_monitor_files[-1], snapshot_files[-1]

    def train_on_cpu(
        self,
        iteration: int = 2,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: Optional[torch.nn.Module] = None,
        zero_grad_mode: int = 1,
    ) -> str:
        trainer_conf = {
            "model": copy.deepcopy(self._model),
            "data_loader": copy.deepcopy(self._data_loader),
            "batch_size": self._batch_size,
            "iterations": iteration,
            "on_cpu": True,
            "config": self._config,
            "plugins": [
                ProfilerPlugin(config=self._config),
            ],
            "optimiser": optimizer,
            "loss": loss,
            "zero_grad_mode": zero_grad_mode,
        }
        logger.info(f"================== Evaluate on CPU ==================")
        try:
            self._train(**trainer_conf)
        finally:
            profiler_files = filter_files(
                "pt.trace.json", str(self._config.result_dir), fuzz=True
            )
            profiler_files.sort(key=lambda x: os.path.getmtime(x))
        return profiler_files[-1]

    def get_ground_value_from_nvml(self, host_monitor_file: str) -> dict:
        with open(host_monitor_file, "r") as f:
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
        return _gpu_memory_usage

    def get_ground_value_from_snapshot(self, snapshot_file: str) -> dict:
        _snapshot = SnapshotAnalyser(snapshot_file)
        return _snapshot.gpu_and_segment_in_same_time_length()

    def evaluate_my_solution(
        self, profiler_file: str, iteration: int = 2
    ) -> Tuple[CachingAllocator, dict]:
        estimator = Estimator(
            dataloader=copy.deepcopy(self._data_loader),
            profiler_file=profiler_file,
            max_gpu_memory_in_gb=self._max_gpu_memory_in_gb,
        )
        return estimator.estimate(target_iteration=iteration)

    def evaluate_DNNmem(self) -> CachingAllocator:
        from experiments.baselines import DNNmem

        data_x, data_y = next(iter(copy.deepcopy(self._data_loader)))
        dnnmem = DNNmem(
            model=copy.deepcopy(self._model),
            data_x=data_x,
            data_y=data_y,
            loss_fn=torch.nn.CrossEntropyLoss(),
            max_gpu_memory=self._max_gpu_memory_in_gb,
        )
        return dnnmem.estimate()

    def evaluate_schedtune(self, device_id: int = 0):
        from experiments.baselines import Schedtune
        from experiments.fx import FXAnalyser
        from pathlib import Path

        data_x, data_y = next(iter(copy.deepcopy(self._data_loader)))
        loss = torch.nn.CrossEntropyLoss()
        model_analysis = FXAnalyser(
            model=copy.deepcopy(self._model), data_x=data_x, data_y=data_y, loss=loss
        )
        activation_size = 0
        parameter_size = 0
        for key, layer in model_analysis.layers_info.items():
            if layer.op_type == "call_module":
                if layer.bias is not None:
                    parameter_size += layer.bias.nbytes
                if layer.weight is not None:
                    parameter_size += layer.weight.nbytes
                if isinstance(layer.output, torch.Tensor):
                    activation_size += layer.output.nbytes

        device = "4070ti" if device_id == 0 else "4060"
        conf_dir = Path(__file__).parent.joinpath("experiments/baselines/schedtune")
        schedtune = Schedtune(
            jobname="batchsize",
            option="1",
            activations=activation_size / 1024**2,
            parameters=parameter_size / 1024**2,
            inputsize=data_x.nbytes / 1024**2,
            gpu=device,
            conf_dir=conf_dir,
        )
        schedtune_output = schedtune.estimate()
        return schedtune_output["mem"] * 1024**2

    def evaluate_llmem(self, gpu_id: int):
        import GPUtil
        from experiments.trainer.plugins.monitor import HostGPUs
        from experiments.baselines import LLmemEstimator
        from colossalai.booster import Booster

        dataloader = copy.deepcopy(self._data_loader)
        data_x, data_y = next(iter(dataloader))
        host_gpus = HostGPUs()
        used_nvml = int(host_gpus.get_gpu(gpu_id).get_memory_info()["used"] / 1024**2)
        cuda_context_mem = used_nvml - GPUtil.getGPUs()[gpu_id].memoryUsed
        model = copy.deepcopy(self._model)
        llmem = LLmemEstimator(
            model=copy.deepcopy(self._model),
            batch=data_x,
            real_bs=self._data_loader.batch_size,
            bytes=4,
            bytes_input=data_x.element_size(),
            gpu_n=1,
            tp=0,
            lm_fp32=True,
            m_total=self._max_gpu_memory_in_gb * 1024,
        )
        device = f"cuda:{gpu_id}"
        model.to(device)
        torch.cuda.empty_cache()

        # Set lr scheduler
        lr = 0.001

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        torch.cuda.empty_cache()
        prev_get_output = GPUtil.getGPUs()[gpu_id].memoryUsed
        llmem.get_output_sizes()
        torch.cuda.empty_cache()
        after_get_output = GPUtil.getGPUs()[gpu_id].memoryUsed

        booster = Booster()
        model, optimizer, _, _, _ = booster.boost(model, optimizer)
        torch.cuda.empty_cache()

        # after load
        torch.cuda.empty_cache()
        booster_chunk_mem = GPUtil.getGPUs()[gpu_id].memoryUsed
        m_pbase = (
            booster_chunk_mem + cuda_context_mem - (after_get_output - prev_get_output)
        )
        # Unit of llm_mem is MB
        llm_mem, llm_bs = llmem.estimate_size(m_init=m_pbase)
        return llm_mem * 1024**2

    def verification(
        self,
        device_id: int = 0,
        iteration: int = 2,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: Optional[torch.nn.Module] = None,
        zero_grad_mode: int = 0,
    ):
        logger.info(f"================== Verification Start ==================")
        logger.info(f"================== Training on CPU ==================")
        profiler_file = self.train_on_cpu(
            iteration=iteration,
            optimizer=optimizer,
            loss=loss,
            zero_grad_mode=zero_grad_mode,
        )
        verification_result = {}
        logger.info(f"================== xMem Estimation ==================")
        time.sleep(1)
        try:
            before_run = time.time()
            my_result, _ = self.evaluate_my_solution(profiler_file, iteration=iteration)
        except Exception as e:
            logger.error(f"xMem estimation failed, error: {e}")
        else:
            after_run = time.time()
            verification_result["solution"] = {
                "runtime": after_run - before_run,
                "memory": max(my_result._trace.max_segment_changes),
                "oom": my_result.oom,
            }

        logger.info(f"================== DNNmem Estimation==================")
        time.sleep(1)
        try:
            before_run = time.time()
            dnnmem_result = self.evaluate_DNNmem()
        except Exception as e:
            logger.error(f"DNNmem estimation failed, error: {e}")
        else:
            after_run = time.time()
            verification_result["dnnmem"] = {
                "runtime": after_run - before_run,
                "memory": max(dnnmem_result._trace.max_segment_changes),
                "oom": my_result.oom,
            }

        logger.info(f"================== Schedtune ==================")
        time.sleep(1)
        try:
            before_run = time.time()
            schedtune_result = self.evaluate_schedtune(device_id=device_id)
        except Exception as e:
            logger.error(f"Schedtune estimation failed, error: {e}")
        else:
            after_run = time.time()
            verification_result["schedtune"] = {
                "runtime": after_run - before_run,
                "memory": schedtune_result,
                "oom": bool(schedtune_result > self._max_gpu_memory_in_gb * 1024**3),
            }

        logger.info(f"================== LLmem ==================")
        time.sleep(1)
        try:
            before_run = time.time()
            llmem_result = self.evaluate_llmem(device_id)
        except Exception as e:
            logger.error(f"LLmem estimation failed, error: {e}")
        else:
            after_run = time.time()
            verification_result["llmem"] = {
                "runtime": after_run - before_run,
                "memory": llmem_result,
                "oom": bool(llmem_result > self._max_gpu_memory_in_gb * 1024**3),
            }

        logger.info(f"================== Initial Validation Round ==================")
        try:
            test_iteration = iteration * 2
            if test_iteration < 10:
                test_iteration = 10

            vf_host_monitor_file, vf_snapshot_file = self.train_on_gpu(
                gpu_id=device_id,
                iteration=test_iteration,
                optimizer=optimizer,
                loss=loss,
                zero_grad_mode=zero_grad_mode,
                limit_in_gp=self._max_gpu_memory_in_gb,
            )
        except Exception as e:
            logger.warning(f"OOM occur, error: {e}")
            real_oom = True
            ground = self._max_gpu_memory_in_gb * 1024**3
        else:
            real_oom = False
            ground = self.get_ground_value_from_nvml(vf_host_monitor_file)
            ground = max(ground[str(device_id)])

        logger.info(
            f"================== Subsequent Validation Round =================="
        )
        for name, value in verification_result.items():
            _memory = int(value["memory"])
            value["ground"] = ground
            value["error"] = abs(_memory - ground) / ground
            value["real_oom"] = real_oom
            value["correct_estimation"] = real_oom == value["oom"]
            value["2nd verification"] = {}
            min_runnable_memory = _memory / 1024**3
            if value["real_oom"] is False and value["oom"] is False:
                try:
                    test_iteration = iteration * 2
                    if test_iteration < 10:
                        test_iteration = 10
                    logger.info(
                        f"================== 2nd Verification {name} =================="
                    )
                    logger.info(f"Minimum Runnable Memory: {min_runnable_memory} GB")
                    logger.info(f"Maximum iterations: {test_iteration}")
                    vvf_host_monitor_file, vvf_snapshot_file = self.train_on_gpu(
                        gpu_id=device_id,
                        iteration=test_iteration,
                        limit_in_gp=min_runnable_memory,
                        optimizer=optimizer,
                        loss=loss,
                        zero_grad_mode=zero_grad_mode,
                    )
                except Exception as e:
                    logger.error(f"Subsequnt validation failed for {name}, error: {e}")
                    value["2nd verification"] = {"oom": True, "error": None}
                else:
                    vvf_ground = self.get_ground_value_from_nvml(vvf_host_monitor_file)
                    vvf_ground = max(vvf_ground[str(device_id)])
                    value["2nd verification"] = {
                        "oom": False,
                        "error": abs(_memory - vvf_ground) / vvf_ground,
                    }
        return verification_result


def main(
    model: str = "VGG16",
    device_id: int = 0,
    batch: int = 200,
    target_iteration: int = 2,
    optimiser: str = "Adam",
    zero_grad_mode: int = 0,
    input_size: int = 86,
):
    data_loader = None
    model_name = model
    model = AllModels[model].value
    _conf = Config(save2tmp=False)
    init_logging(level="DEBUG", conf=_conf)

    gpu_info = GPUtil.getGPUs()[device_id]
    max_gpu_in_gb = GPUtil.getGPUs()[device_id].memoryFree / 1024
    logger.warning(
        f"Used Memory: {gpu_info.memoryUsed} MB, Free Memory: {gpu_info.memoryFree} MB"
    )

    eva = Evaluator(
        model=model,
        data_loader=data_loader,
        max_gpu_memory_in_gb=max_gpu_in_gb,
        batch_size=batch,
        config=_conf,
        input_size=input_size,
    )
    all_result = eva.verification(
        device_id=device_id,
        iteration=target_iteration,
        optimizer=getattr(torch.optim, optimiser, torch.optim.SGD),
        zero_grad_mode=zero_grad_mode,
    )

    all_result["train info"] = {
        "model": model_name,
        "batch_size": batch,
        "target_iteration": target_iteration,
        "optimiser": optimiser,
        "device": device_id,
        "input_size": input_size,
        "zero_grad_mode": zero_grad_mode,
        "total_gpu_memory": max_gpu_in_gb,
        "used_gpu_memory": gpu_info.memoryUsed / 1024,
    }
    logger.info(all_result)
    all_result["config"] = _conf.model_dump()
    with open(eva.conf.result_dir.joinpath("evaluation_result.json"), "w") as f:
        json.dump(all_result, f, indent=4)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
