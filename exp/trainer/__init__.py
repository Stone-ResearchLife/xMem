import logging
import torch
import copy
import uuid
from typing import Optional
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
        config: Config = default_setting,
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
            "zero_out_pos": config.trainer.zero_out,
            "gpu_total_memory": None,
            "gpu_name": None,
        }
        self.config = config

    @property
    def gpu_id(self) -> int:
        return self._info["gpu_id"]

    @property
    def gpu_name(self):
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA is not available. Therefore, GPU name cannot be retrieved."
            )
            return None
        else:
            gpu_id = self.gpu_id
            _name = str(torch.cuda.get_device_properties(gpu_id).name).replace(" ", "-")
            self._info["gpu_name"] = _name
            return _name

    def get_total_gpu_memory(self) -> Optional[int]:
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA is not available. Therefore, GPU memory cannot be retrieved."
            )
            return None
        else:
            gpu_id = self.gpu_id
            total_memory_in_bytes = torch.cuda.get_device_properties(
                gpu_id
            ).total_memory
            self._info["gpu_total_memory"] = total_memory_in_bytes
            return total_memory_in_bytes

    def set_fraction_gpu_memory(self, fraction_gpu: Optional[float] = 1):
        fraction = float(fraction_gpu or 1)
        if fraction > 1:
            logger.warning(f"The limit is over the total GPU memory, set to 1.0")
            fraction = 1.0

        logger.info(
            f"Set GPU ({self.gpu_name}) Memory Fraction: {round(fraction, 2)*100}%"
        )
        torch.cuda.set_per_process_memory_fraction(
            round(fraction, 2), device=torch.device(f"cuda:{self.gpu_id}")
        )

    def train_on_cpu(self) -> tuple[Config, bool]:
        """
        Train the model on CPU and return the configuration.

        Returns:
            tuple: A tuple containing the configuration and a boolean indicating if the training occurred OOM.

        """
        cpu_config = self.config.model_copy(deep=True)
        cpu_config.task_id = f"CPU_{uuid.uuid4().hex[:3]}"
        profilers = [
            ProfilerPlugin(config=cpu_config),
        ]
        if self.config.debug:
            profilers.append(
                HostMonitorPlugin(
                    interval_ms=1,
                    config=cpu_config,
                    gpu_enable=True,
                    cpu_enable=False,
                    network_enable=False,
                )
            )
            profilers.append(SnapshotPlugin(config=cpu_config))
        trainer = ModelTrainer(
            model=copy.deepcopy(self.model_preparer.model),
            data_loader=copy.deepcopy(self.model_preparer.dl),
            on_cpu=True,
            gpu_id=self.gpu_id,
            iterations=3,
            optimiser=self.model_preparer.optimiser,
            config=cpu_config,
            plugins=profilers,
            zero_grad_mode=self.config.trainer.zero_out,
        )
        return trainer.train(is_transformer=self.model_preparer.is_transformer)

    def train_on_gpu(self, fraction_gpu: Optional[float] = None):
        """
        Train the model on CPU and return the configuration.

        Returns:
            tuple: A tuple containing the configuration and a boolean indicating if the training occurred OOM.

        """
        self.set_fraction_gpu_memory(fraction_gpu)
        gpu_config = self.config.model_copy(deep=True)
        gpu_config.task_id = f"GPU_{uuid.uuid4().hex[:3]}"
        profilers = [
            HostMonitorPlugin(
                interval_ms=1,
                config=gpu_config,
                gpu_enable=True,
                cpu_enable=False,
                network_enable=False,
            ),
        ]
        if self.config.debug:
            profilers.append(SnapshotPlugin(config=gpu_config))
            profilers.append(ProfilerPlugin(config=gpu_config))
        trainer = ModelTrainer(
            model=copy.deepcopy(self.model_preparer.model),
            data_loader=copy.deepcopy(self.model_preparer.dl),
            on_cpu=False,
            gpu_id=self.gpu_id,
            iterations=3,
            optimiser=self.model_preparer.optimiser,
            config=gpu_config,
            plugins=profilers,
            zero_grad_mode=self.config.trainer.zero_out,
        )
        return trainer.train(is_transformer=self.model_preparer.is_transformer)


__all__ = ["FastRunner", "SnapshotPlugin", "ProfilerPlugin", "HostMonitorPlugin"]
