from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModel,
    AutoConfig
)
from huggingface_hub import model_info
from datasets import load_dataset
from torch.utils.data import DataLoader
from experiments.trainer.plugins import ProfilerPlugin, HostMonitorPlugin, SnapshotPlugin
from perf_estimator.config import Config
from typing import Optional
import torch.optim as optim
import torch
import os
import time
import logging

logger = logging.getLogger(__name__)


class LLMTrainer:
    def __init__(
            self,
            model_name,
            cpu_only: bool = False,
            optimizer: str = "AdamW",
            batch_size: int = 10,
    ):
        self.model_name: str = model_name
        self.cpu_only: bool = cpu_only
        self.optimizer: str = optimizer
        self.batch_size: int = batch_size


    def suggest_automodel_class(self):
        model_name = self.model_name
        try:
            info = model_info(model_name)
            pipeline_tag = info.pipeline_tag

            if pipeline_tag:
                if pipeline_tag == "feature-extraction":
                    return AutoModel
                elif pipeline_tag == "fill-mask":
                    return AutoModelForMaskedLM
                elif pipeline_tag == "sentiment-analysis" or pipeline_tag == "text-classification":
                    return AutoModelForCausalLM
                elif pipeline_tag == "text2text-generation":
                    return AutoModelForSeq2SeqLM
                elif pipeline_tag == "summarization":
                    return AutoModelForSeq2SeqLM
                elif pipeline_tag == "translation":
                    return AutoModelForSeq2SeqLM
                elif pipeline_tag == "text-generation":
                    return AutoModelForCausalLM
                else:
                    return AutoModel  # Default if pipeline_tag is unknown
            else:
                # Fallback based on model name keywords (less reliable)
                model_name_lower = model_name.lower()
                if "gpt" in model_name_lower or "llama" in model_name_lower or "codegen" in model_name_lower:
                    return AutoModelForCausalLM
                elif "bert" in model_name_lower or "roberta" in model_name_lower or "distilbert" in model_name_lower or "albert" in model_name_lower:
                    return AutoModel
                elif "t5" in model_name_lower or "bart" in model_name_lower or "mt5" in model_name_lower:
                    return AutoModelForSeq2SeqLM
                else:
                    return AutoModel  # More generic fallback
        except Exception as e:
            print(f"Could not retrieve model info for {model_name}: {e}")
            return AutoModel  # Fallback to a generic AutoModel

    def prepare_model(self):
        model_class = self.suggest_automodel_class()
        print(f"Model class: {model_class.__name__}")
        config = AutoConfig.from_pretrained(
            self.model_name
        )  # load config; do NOT load pretrained weights
        model = model_class.from_config(config)
        model.train()
        return model

    def prepare_data(self) -> DataLoader:
        model_name = self.model_name
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
        dataloader = DataLoader(tokenized_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        return dataloader

    def get_device(self):
        if self.cpu_only:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return device

    def train(self):
        snap_conf = Config(save2tmp=False)
        profiler = ProfilerPlugin(config=snap_conf)
        snap = SnapshotPlugin(config=snap_conf)
        host_monitor = HostMonitorPlugin(
            interval_ms=1,
            cpu_enable=False,
            gpu_enable=False if self.cpu_only else True,
            network_enable=False,
            config=snap_conf
        )

        profiler.start()
        host_monitor.start()
        snap.start()
        time.sleep(2)

        device = self.get_device()
        model = self.prepare_model()
        dataloader = self.prepare_data()


        optimizer = getattr(optim, self.optimizer, optim.AdamW)(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        try:
            model.to(device)
            epochs = 1
            for epoch in range(epochs):
                for index, batch in enumerate(dataloader):
                    profiler.step()
                    with torch.set_grad_enabled(True):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        print(f"Epoch {epoch}, Loss: {loss.item()}")
                        if index == 3:
                            break
                    scheduler.step()
        except Exception as e:
            print("Exception occurred during training.")
            print(e)
        finally:
            host_monitor.stop()
            profiler.stop()
            snap.stop()
            print("Train Finsihed！")


class LLMEvaluator:
    def __init__(
            self,
            model_name,
            optimizer: str = "AdamW",
            batch_size: int = 10,
            gpu_id: int = 1,
            limited_gpu_in_gb: Optional[float] = None,
    ):
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        self.model_name: str = model_name
        self.optimizer: str = optimizer
        self.batch_size: int = batch_size
        self.g_id = int(gpu_id)

    def train_on_cpu(self):
        self._train(on_cpu=True)
        
    def set_fraction_gpu_memory(self, gpu_memory_in_gb: float = 8.0):
        no_devices = torch.cuda.device_count()
        if no_devices > 1:
            devide_id = self.g_id
        elif no_devices == 1:
            devide_id = 0
        else:
            raise ValueError("No GPU device found")
        
        total_gpu_memory = torch.cuda.get_device_properties(devide_id).total_memory
        fraction = (gpu_memory_in_gb * 1024**3) / total_gpu_memory

        if fraction > 1:
            logger.warning(
                f"The limit is over the total GPU memory, set to 1. input max: {gpu_memory_in_gb}GB, total: {total_gpu_memory/1024**3}GB"
            )
            fraction = 1
        logger.info(
            f"Set GPU Memory Fraction: {round(fraction, 2)*100}%, limit: {gpu_memory_in_gb}GB"
        )
        torch.cuda.set_per_process_memory_fraction(
            round(fraction, 2), device=torch.device(f"cuda:{devide_id}")
        )

    def train_on_gpu(self, gpu_memory_in_gb: Optional[float] = None):
        torch.cuda.empty_cache()
        time.sleep(1)  # wait for the memory to be released
        if gpu_memory_in_gb is not None:
            self.set_fraction_gpu_memory(gpu_memory_in_gb)
        time.sleep(1)  # wait for the memory to be released
        self._train(on_cpu=False)

    def _train(self, on_cpu: bool = False):
        evaluation = LLMTrainer(
            model_name=self.model_name,
            cpu_only=on_cpu,
            optimizer=self.optimizer,
            batch_size=self.batch_size
        )
        evaluation.train()

    def eva(self):
        self.train_on_cpu()
        time.sleep(5)
        self.train_on_gpu()


if __name__ == "__main__":
    model_name = "EleutherAI/gpt-neo-125M"
    optimizer = "AdamW"
    batch_size = 10
    eva = LLMEvaluator(
        model_name=model_name,
        optimizer=optimizer,
        batch_size=batch_size,
        gpu_id=1
    )
    eva.train_on_gpu(gpu_memory_in_gb=4.1)

