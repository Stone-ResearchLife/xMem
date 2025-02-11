import logging
import torch
import platform

import torchvision.models
from transformers import TrainingArguments, Trainer
from typing import Optional, List, Dict
from perf_estimator.config import Config, default_setting
from perf_estimator.trainer.plugins import ProfilerCallback


logger = logging.getLogger(__name__)


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
        self.model = torchvision.models.resnet101(weights=None)
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
        data_loader: torch.utils.data.DataLoader,
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
        self._model = HuggingFaceModel(model, loss=loss)
        self._dataset = HuggingDataset(data_loader.dataset)
        self._config = config or default_setting
        optimiser = optimiser or torch.optim.SGD
        self._optimiser = optimiser(params=self._model.parameters(), lr=self._config.trainer.lr)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimiser, step_size=7, gamma=0.1)
        self._train_args = TrainingArguments(
            output_dir=str(self._config.result_dir.joinpath('huggingface')),
            num_train_epochs=1,
            per_device_train_batch_size=data_loader.batch_size or batch_size,
            logging_dir=str(self._config.log_dir.joinpath('huggingface')),
            save_strategy="epoch",
            report_to="tensorboard",
            max_steps=iterations,
        )

        if on_cpu:
            self._train_args.use_cpu = True
        elif isinstance(gpu_id, int) and gpu_id <= (torch.cuda.device_count() - 1):
            if platform.system() == "darwin":
                self._train_args.use_mps_device = True
        else:
            self._train_args.use_cpu = True
    
    def train(self):
        trainer = Trainer(
            model=self._model,
            args=self._train_args,
            train_dataset=self._dataset,
            callbacks=[ProfilerCallback(self._config)],
            optimizers=(self._optimiser, self._scheduler)
        )
        trainer.train()
        
