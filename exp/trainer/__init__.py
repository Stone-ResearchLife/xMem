import logging
import torch
import copy
import uuid
from typing import Optional
from ures.string import format_memory
from perf_estimator.config import Config, default_setting
from .plugins import SnapshotPlugin, ProfilerPlugin, HostMonitorPlugin
from .trainer import ModelTrainer, ModelPreparer



logger = logging.getLogger(__name__)


class FastRunner:
    def __init__(
            self,
            model_name: str,
            batch_size: int = 32,
            gpu_id: int = 0,
            optimiser: str = None,
    ):
        self.model_preparer = ModelPreparer(
            model_name=model_name,
            batch_size=batch_size,
            optimiser=optimiser,
        )
        self._info = {
            "model_name": model_name,
            "batch_size": batch_size,
            "gpu_id": gpu_id,
            "optimiser": optimiser,
            "gpu_total_memory": self.get_total_gpu_memory(),
            "gpu_name": self.get_gpu_name(),
        }
        self.config = Config(
            save2tmp=False,
            run_id=f"{model_name}_{optimiser}_{batch_size}_{uuid.uuid4().hex[:4]}",
        )


    @property
    def gpu_id(self) -> int:
        return self._info["gpu_id"]

    def get_total_gpu_memory(self) -> Optional[int]:
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Therefore, GPU memory cannot be retrieved.")
            return None
        else:
            gpu_id = self.gpu_id
            return torch.cuda.get_device_properties(gpu_id).total_memory

    def get_gpu_name(self) -> Optional[str]:
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Therefore, GPU name cannot be retrieved.")
            return None
        else:
            gpu_id = self.gpu_id
            return torch.cuda.get_device_properties(gpu_id).name

    def set_fraction_gpu_memory(self, gpu_memory_in_bytes: Optional[int] = 8*1024**3):
        total_gpu_memory = self.get_total_gpu_memory()
        device_id = self.gpu_id
        if total_gpu_memory is None:
            logger.error("CUDA is not available. Therefore, GPU memory fraction cannot be set.")
            raise RuntimeError("CUDA is not available.")

        if gpu_memory_in_bytes is None:
            gpu_memory = total_gpu_memory
        else:
            gpu_memory = gpu_memory_in_bytes

        fraction = round(gpu_memory / total_gpu_memory, 2)

        if fraction > 1:
            logger.warning(
                f"The limit is over the total GPU memory, set to 1. input max: {format_memory(gpu_memory_in_bytes)}, total: {format_memory(total_gpu_memory)}"
            )
            fraction = 1

        logger.info(
            f"Set GPU Memory Fraction: {round(fraction, 2)*100}%, limit: {format_memory(gpu_memory_in_bytes)}"
        )
        torch.cuda.set_per_process_memory_fraction(
            round(fraction, 2), device=torch.device(f"cuda:{device_id}")
        )

    def train_on_cpu(self):
        cpu_config = self.config.model_copy(deep=True)
        cpu_config.task_id = f"CPU_{uuid.uuid4().hex[:3]}"
        profilers = [
            ProfilerPlugin(config=cpu_config),
        ]
        trainer = ModelTrainer(
            model=copy.deepcopy(self.model),
            data_loader=copy.deepcopy(self.dl),
            on_cpu=True,
            gpu_id=self.gpu_id,
            iterations=3,
            optimiser=self.optimiser,
            config=cpu_config,
            plugins=profilers,
        )
        trainer.train(is_transformer=self.is_transformer)
        return cpu_config

    def train_on_gpu(self, set_gpu_memory_in_byte: Optional[int] = None):
        self.set_fraction_gpu_memory(set_gpu_memory_in_byte)
        gpu_config = self.config.model_copy(deep=True)
        gpu_config.task_id = f"G_{self.get_gpu_name().replace(' ', '-')}_{uuid.uuid4().hex[:3]}"
        profilers = [
            ProfilerPlugin(config=gpu_config),
            HostMonitorPlugin(config=gpu_config),
            SnapshotPlugin(config=gpu_config),
        ]
        trainer = ModelTrainer(
            model=copy.deepcopy(self.model),
            data_loader=copy.deepcopy(self.dl),
            on_cpu=False,
            gpu_id=self.gpu_id,
            iterations=3,
            optimiser=self.optimiser,
            config=gpu_config,
            plugins=profilers,
        )
        trainer.train(is_transformer=self.is_transformer)
        return gpu_config


__all__ = ["FastRunner", "SnapshotPlugin", "ProfilerPlugin", "HostMonitorPlugin"]
