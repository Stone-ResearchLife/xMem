from venv import logger

import torch
import platform
from typing import Optional, List, Dict
from perf_estimator.config import Config, default_setting
from .train_loop import conv_train_loop


class HuggingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        image, label = self.dataset[idx]
        return {"pixel_values": image, "labels": label}


class HuggingFaceModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, loss: torch.nn.Module = None):
        super(HuggingFaceModel, self).__init__()
        self.model = model
        self.loss = loss or torch.nn.CrossEntropyLoss()
        self.model_input_names = ["pixel_values", "labels"]

    def forward(self, pixel_values, labels):
        logits = self.model(pixel_values)
        if labels is not None:
            loss = self.loss(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return logits


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
        plugins: List["InterfacePlugin"] = None,
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
                logger.warning(
                    f"Plugin {plugin.tool_name} has different config, will be updated"
                )
                plugin.config = self._config

    def show_summary(self):
        print("=============== Device Information ===============")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {self._device}")
        print(f"Model: {self._model.__class__.__name__}")
        print(
            f"Batch Size: {self._data_loader.batch_size if self._data_loader is not None else self._batch_size}"
        )
        print(f"Run ID: {self._config.run_id}")
        print(f"Directory: {self._config.base_dir}")
        print("================================================")

    def train(self, mode=None):
        if mode is None:
            mode = "torch"
        self.show_summary()
        if mode == "torch":
            train_func = conv_train_loop

            train_func(
                model=self._model,
                data_loader=self._data_loader,
                epochs=self._epochs,
                device=self._device,
                batch_size=(
                    self._data_loader.batch_size
                    if self._data_loader is not None
                    else self._batch_size
                ),
                iterations=self._iterations,
                loss=self._loss,
                lr=self._lr,
                plugins=self._plugins,
                optimizer=self._optimiser,
                zero_grad_mode=self._zero_grad_mode,
            )
        elif mode == "huggingface-conv":
            from transformers import TrainingArguments, Trainer
            from perf_estimator.trainer.plugins import ProfilerCallback, StepBasedStopCallback
            _dataset = HuggingDataset(self._data_loader.dataset)
            _model = HuggingFaceModel(self._model, self._loss)
            # --- Training Arguments ---
            training_args = TrainingArguments(
                output_dir=str(self._config.result_dir.joinpath('huggingface')),
                num_train_epochs=self._epochs,
                per_device_train_batch_size=self._data_loader.batch_size if self._data_loader is not None else self._batch_size,
                logging_dir=str(self._config.log_dir.joinpath('huggingface')),
                logging_steps=50,
                save_strategy="epoch",
                report_to="tensorboard",
                use_cpu=("cpu" in str(self._device)),
            )
            if self._optimiser is None:
                self._optimiser = torch.optim.SGD
            optimiser = self._optimiser(params=self._model.parameters(), lr=self._lr)
            trainer = Trainer(
                model=_model,
                args=training_args,
                train_dataset=_dataset,
                optimizers=(
                    optimiser,
                    torch.optim.lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1),
                ),
                callbacks=[ProfilerCallback(self._config)],
            )
            trainer.train()



