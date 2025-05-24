from venv import logger
import torch
import platform
from typing import Optional, List, Tuple
from perf_estimator.config import Config, default_setting
from perf_estimator.models import AllModels
from .train_loop import (
    conv_train_loop,
    transformer_train_loop,
    transformer_mixed_precision_train_loop,
    transformer_cpu_f16_train_loop,
)


class ModelPreparer:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        optimiser: str = None,
        fp16: bool = False,
    ):
        print(f"Preparing {model_name} with fp16: {fp16} and optimiser: {optimiser}")
        self._mlm = False
        self._fp16 = fp16
        self.is_transformer = False
        self.model = self.get_model(model_name)
        self.dl = self.get_dataloader(batch_size)
        self.optimiser = getattr(torch.optim, optimiser or "AdamW", None)
        if self.optimiser is None:
            import transformers

            self.optimiser = getattr(transformers, optimiser, None)
            if self.optimiser is None:
                raise ValueError(
                    f"Invalid optimiser: {optimiser}. The optimiser must be available in torch.optim or transformers."
                )

        print(
            f"Loaded {model_name} in data type: {next(self.model.parameters()).dtype}"
        )

    def get_model(self, model_name: str) -> torch.nn.Module:
        if getattr(AllModels, model_name, None) is None:
            from transformers import AutoConfig, AutoModelForMaskedLM
            from utils.huggingface import suggest_automodel_class

            model_class = suggest_automodel_class(model_name)
            if isinstance(model_class, AutoModelForMaskedLM):
                self._mlm = True
            logger.info(f"Model class: {model_class.__name__}")
            config = AutoConfig.from_pretrained(
                model_name
            )  # load config; do NOT load pretrained weights
            if self._fp16:
                config.torch_dtype = torch.float16
            else:
                config.torch_dtype = torch.float32
            model = model_class.from_config(config)
            # if self._fp16:
            #     model.half()
            self.is_transformer = True
            model.train()
        else:
            model = getattr(AllModels, model_name).value

        return model

    def get_dataloader(
        self, batch_size: int, token_padding: bool = False
    ) -> torch.utils.data.DataLoader:
        if self.is_transformer:
            from transformers import AutoTokenizer, DataCollatorForLanguageModeling
            from datasets import load_dataset

            model_name = self.model.name_or_path

            model_list = ["EleutherAI/pythia", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen3"]
            token_padding = any([model in model_name for model in model_list])

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

            # 3. participle
            def tokenize_function(examples):
                if token_padding:
                    return tokenizer(
                        examples["text"],
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                        padding=True,
                    )
                else:
                    return tokenizer(examples["text"], truncation=True, max_length=128)

            tokenized_datasets = dataset.map(
                tokenize_function, batched=True, remove_columns=["text"]
            )

            # 4. create DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=self._mlm
            )

            # 5. create DataLoader
            tokenized_datasets.set_format("torch")
            dataloader = torch.utils.data.DataLoader(
                tokenized_datasets,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=data_collator,
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
        zero_grad_mode: int = 0,
        plugins: List["AbcPlugin"] = None,
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

    def train(
        self, is_transformer: bool = False, display_info: bool = False
    ) -> tuple[Config, bool]:
        """
        Train the model on CPU and return the configuration.

        Returns:
            tuple: A tuple containing the configuration and a boolean indicating if the training occurred OOM.

        """
        if display_info:
            self.show_summary()

        if is_transformer:
            if self._config.trainer.fp16:
                func = transformer_mixed_precision_train_loop
            else:
                print(f"Using Mixed Precision (FP32) Training loop.")
                func = transformer_train_loop
        else:
            print(f"Using Convolutional Training loop.")
            func = conv_train_loop

        _conf = self._config.model_copy(deep=True)
        try:
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
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logger.warning(
                    f"CUDA out of memory, please check the GPU memory usage."
                )
                return _conf, True
            else:
                raise RuntimeError(f"Unexpected error: {e}") from e
        else:
            logger.info("Training completed successfully.")
            return _conf, False
