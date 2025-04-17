from venv import logger
import torch
import platform
import uuid
from typing import Optional, List, Tuple
from perf_estimator.config import Config, default_setting
from perf_estimator.models import AllModels
from .train_loop import conv_train_loop, transformer_train_loop


class ModelPreparer:
    def __init__(
            self,
            model_name: str,
            batch_size: int = 32,
            optimiser: str = None,
    ):
        self.is_transformer = False
        self.model = self.get_model(model_name)
        self.dl = self.get_dataloader(batch_size)
        self.optimiser = optimiser or torch.optim.AdamW

    def get_model(self, model_name: str) -> torch.nn.Module:
        if getattr(AllModels, model_name, None) is None:
            from transformers import AutoConfig
            from utils.huggingface import suggest_automodel_class
            model_class = suggest_automodel_class(model_name)
            logger.info(f"Model class: {model_class.__name__}")
            config = AutoConfig.from_pretrained(
                model_name
            )  # load config; do NOT load pretrained weights
            model = model_class.from_config(config)
            self.is_transformer = True
            model.train()
        else:
            model = getattr(AllModels, model_name).value

        return model

    def get_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        if self.is_transformer:
            from transformers import AutoTokenizer, DataCollatorForLanguageModeling
            from datasets import load_dataset
            model_name = self.model.name_or_path
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

            # 3. participle
            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=True, max_length=128)

            tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

            # 4. create DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # 5. create DataLoader
            tokenized_datasets.set_format("torch")
            dataloader = torch.utils.data.DataLoader(
                tokenized_datasets,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=data_collator
            )
        else:
            from perf_estimator.dataset import image_dataset
            dataloader = image_dataset(batch=batch_size)
        return dataloader


class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: Optional[torch.utils.data.DataLoader],
        on_cpu: bool = False,
        gpu_id: int = 0,
        iterations: Optional[int] = 1,
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
        self._iterations = iterations
        self._optimiser = optimiser
        self._config = config
        self._epochs = config.trainer.epochs
        self._lr = config.trainer.lr
        self._zero_grad_mode = zero_grad_mode

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

    def train(self, is_transformer: bool = False):
        self.show_summary()
        if is_transformer:
            func = transformer_train_loop
        else:
            func = conv_train_loop

        func(
            model=self._model,
            data_loader=self._data_loader,
            epochs=self._epochs,
            device=self._device,
            iterations=self._iterations,
            lr=self._lr,
            plugins=self._plugins,
            optimizer=self._optimiser,
            zero_grad_mode=self._zero_grad_mode,
        )
