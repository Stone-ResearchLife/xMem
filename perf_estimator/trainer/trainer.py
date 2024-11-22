from venv import logger

import torch
import platform
from typing import Optional, List, Tuple
from perf_estimator.config import Config, default_setting
from .train_loop import conv_train_loop


class ModelTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            data_loader: Optional[torch.utils.data.DataLoader],
            on_cpu: bool = False,
            gpu_id: int = 0,
            batch_size: int = 32,
            iterations: Optional[int] = 1,
            loss: Optional[torch.nn.Module] = None,
            zero_grad_mode: int = 1,
            plugins: List['InterfacePlugin'] = None,
            optimiser: Optional[torch.optim.Optimizer] = None,
            config: Config = default_setting,
    ):
        if on_cpu:
            self._device = torch.device("cpu")
        elif isinstance(gpu_id, int) and gpu_id <= (torch.cuda.device_count() - 1):
            if platform.system() == "darwin":
                self._device = torch.device("mps")
            else:
                self._device = torch.device(f"cuda:{gpu_id}")
        else:
            self._device = torch.device("cpu")

        self._model = model
        self._data_loader = data_loader
        self._plugins = plugins or []
        self._batch_size = batch_size
        self._iterations = iterations
        self._loss = loss
        self._optimiser = optimiser
        self._config = config
        self._epochs = config.trainer.epochs
        self._lr = config.trainer.lr
        self._zero_grad_mode = zero_grad_mode
        # update configuration data for each plugin
        for plugin in self._plugins:
            if plugin.config != self._config:
                logger.warning(f"Plugin {plugin.tool_name} has different config, will be updated")
                plugin.config = self._config

    def show_summary(self):
        print("=============== Device Information ===============")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {self._device}")
        print(f"Model: {self._model.__class__.__name__}")
        print(f"Batch Size: {self._data_loader.batch_size if self._data_loader is not None else self._batch_size}")
        print(f"Run ID: {self._config.run_id}")
        print(f"Directory: {self._config.base_dir}")
        print("================================================")

    def train(self):
        self.show_summary()
        train_func = conv_train_loop

        train_func(
            model=self._model,
            data_loader=self._data_loader,
            epochs=self._epochs,
            device=self._device,
            batch_size= self._data_loader.batch_size if self._data_loader is not None else self._batch_size,
            iterations=self._iterations,
            loss=self._loss,
            lr=self._lr,
            plugins=self._plugins,
            optimizer=self._optimiser,
            zero_grad_mode=self._zero_grad_mode
        )