import torch.optim as optim
import torch
import os
import time
import logging
import json
import uuid
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
from utility import search_profiler_file, search_nvml_file

logger = logging.getLogger(__name__)


class LLMTrainer:
    def __init__(
            self,
            model_name,
            cpu_only: bool = False,
            optimizer: str = "AdamW",
            batch_size: int = 10,
            config: Config = None,
    ):
        self.model_name: str = model_name
        self.cpu_only: bool = cpu_only
        self.optimizer: str = optimizer
        self.batch_size: int = batch_size
        self.config: Config = config or Config(save2tmp=False)


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
            logger.info(f"Could not retrieve model info for {model_name}: {e}")
            return AutoModel  # Fallback to a generic AutoModel

    def prepare_model(self):
        model_class = self.suggest_automodel_class()
        logger.info(f"Model class: {model_class.__name__}")
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
        _plugin_config = self.config
        profiler = ProfilerPlugin(config=_plugin_config)
        snap = SnapshotPlugin(config=_plugin_config)
        host_monitor = HostMonitorPlugin(
            interval_ms=1,
            cpu_enable=False,
            gpu_enable=False if self.cpu_only else True,
            network_enable=False,
            config=_plugin_config
        )

        host_monitor.start()
        time.sleep(2)

        device = self.get_device()
        model = self.prepare_model()
        dataloader = self.prepare_data()

        optimizer = getattr(optim, self.optimizer, optim.AdamW)(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        snap.start()
        profiler.start()
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
                        logger.info(f"Epoch {epoch}, Loss: {loss.item()}")
                        if index == 3:
                            break
                    scheduler.step()
        except Exception as e:
            logger.error("Exception occurred during training.")
            logger.error(e)
        finally:
            host_monitor.stop()
            profiler.stop()
            snap.stop()
            logger.info("Train Finsihed！")


class LLMEvaluator:
    def __init__(
            self,
            model_name,
            optimizer: str = "AdamW",
            batch_size: int = 10,
            gpu_id: int = 1,
            config: Config = None,
            tool_name: str = "xMem",
            iterations: int = 2,
    ):
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        self.tool_name = tool_name
        self.model_name: str = model_name
        self.optimizer: str = optimizer
        self.batch_size: int = batch_size
        self.g_id = int(gpu_id)
        self.iterations: int = iterations
        run_id = f"{model_name.replace('/', '-')}_{optimizer}_{batch_size}_{str(uuid.uuid4())[:4]}"
        self.config: Config = config or Config(save2tmp=False, run_id=run_id)

        total_gpu_memory, _ = self.get_total_gpu_memory()
        self.all_result = {
            "train info": {
                "model": self.model_name,
                "batch_size": self.batch_size,
                "target_iteration": self.iterations,
                "optimiser": self.optimizer,
                "device": gpu_id,
                "input_size": (0, 0),
                "zero_grad_mode": None,
                "total_gpu_memory": total_gpu_memory/1024**3,
                "used_gpu_memory": None,
            }
        }

    def train_on_cpu(self, config: Config = None) -> Config:
        if config is None:
            _config = self.config.model_copy(deep=True)
            _config.task_id = "CPU"
            config = _config
        return self._train(on_cpu=True, config=config)

    def get_total_gpu_memory(self):
        no_devices = torch.cuda.device_count()
        if no_devices > 1:
            devide_id = self.g_id
        elif no_devices == 1:
            devide_id = 0
        else:
            raise ValueError("No GPU device found")

        return torch.cuda.get_device_properties(devide_id).total_memory, devide_id

    def set_fraction_gpu_memory(self, gpu_memory_in_gb: Optional[float] = 8.0):
        total_gpu_memory, devide_id = self.get_total_gpu_memory()
        if gpu_memory_in_gb is None:
            gpu_memory = total_gpu_memory
        else:
            gpu_memory = gpu_memory_in_gb * 1024**3

        fraction = gpu_memory / total_gpu_memory

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

    def train_on_gpu(self, gpu_memory_in_gb: Optional[float] = None, config: Config = None) -> Config:
        if config is None:
            _config = self.config.model_copy(deep=True)
            if gpu_memory_in_gb is None:
                task_id = "GPU"
            else:
                task_id = f"GPU-{gpu_memory_in_gb}GB"
            _config.task_id = task_id
            config = _config

        torch.cuda.empty_cache()
        time.sleep(1)  # wait for the memory to be released
        self.set_fraction_gpu_memory(gpu_memory_in_gb)
        time.sleep(1)  # wait for the memory to be released
        return self._train(on_cpu=False, config=config)


    def _train(self, on_cpu: bool = False, config: Config = None) -> Config:
        evaluation = LLMTrainer(
            model_name=self.model_name,
            cpu_only=on_cpu,
            optimizer=self.optimizer,
            batch_size=self.batch_size,
            config=config or self.config,
        )
        evaluation.train()
        return config

    def get_estimate_max_gpu(self, config: Config):
        result_dir = config.result_dir
        p_files = search_profiler_file(result_dir)

        from perf_estimator.estimator import TrainerEstimator
        from perf_estimator.dataset import image_dataset
        estimator = TrainerEstimator(
            dataloader=image_dataset(batch=100),
            profiler_file=p_files[-1],
            max_gpu_memory_in_gb=64,
            config=config,
        )
        return estimator.estimate()
    
    def get_truth_ground_max_gpu(self, config: Config):
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
        return _gpu_memory_usage

    def eva(self):
        c_config = self.train_on_cpu()
        c_est, c_result = self.get_estimate_max_gpu(c_config)
        estimate_max_gpu = c_result["memory"]["segment"]
        est_oom = c_result["OOM"]

        time.sleep(5)
        try:
            g_config = self.train_on_gpu()
            g_nvml = self.get_truth_ground_max_gpu(g_config)

        except Exception as e:
            logger.warning(f"OOM occur, error: {e}")
            real_oom = True
            ground = self.get_total_gpu_memory()
        else:
            real_oom = False
            ground = max(g_nvml[str(self.g_id)])


        try:
            g_config_2nd = self.train_on_gpu(gpu_memory_in_gb=estimate_max_gpu/1024**3)
            g_nvml_2nd = self.get_truth_ground_max_gpu(g_config_2nd)
        except Exception as e:
            logger.warning(f"OOM occur, error: {e}")
            real_oom_2nd = True
            ground_2nd = None
        else:
            real_oom_2nd = False
            ground_2nd = max(g_nvml_2nd[str(self.g_id)])


        self.all_result[self.tool_name] = {
            "ground": ground,
            "memory": estimate_max_gpu,
            "oom": est_oom,
            "real_oom": real_oom,
            "error": abs(estimate_max_gpu - ground) / ground,
            "correct_estimation": est_oom == real_oom,
            "2nd verification": {
                "oom": real_oom_2nd,
                "ground": ground_2nd,
                "error": None if real_oom_2nd else abs(estimate_max_gpu - ground_2nd) / ground_2nd,
            }
        }

        self.all_result["config"] = self.config.model_dump()
        report_dir = self.config.base_dir
        with open(os.path.join(report_dir, "evaluation_result.json"), "w") as f:
            json.dump(self.all_result, f, indent=4)


        return self.all_result

def main(
    model: str,
    device_id: int = 0,
    batch: int = 10,
    target_iteration: int = 2,
    optimiser: str = "AdamW",
):
    eva = LLMEvaluator(
        model_name=model,
        optimizer=optimiser,
        batch_size=batch,
        gpu_id=device_id,
        iterations=target_iteration
    )
    eva.eva()
    torch.cuda.empty_cache()
    time.sleep(5)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
