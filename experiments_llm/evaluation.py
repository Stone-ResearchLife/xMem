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
from ures.string import format_memory
from typing import Optional
from utils import search_profiler_file, search_nvml_file

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
            
            # if pipeline_tag == "feature-extraction":
            #     return AutoModel
            # elif pipeline_tag == "fill-mask":
            #     return AutoModelForMaskedLM
            # elif pipeline_tag == "question-answering":
            #     return AutoModelForQuestionAnswering
            # elif pipeline_tag == "sentiment-analysis" or pipeline_tag == "text-classification":
            #     return AutoModelForSequenceClassification
            # elif pipeline_tag == "token-classification":
            #     return AutoModelForTokenClassification
            # elif pipeline_tag == "text-generation":
            #     return AutoModelForCausalLM
            # elif pipeline_tag == "text2text-generation":
            #     return AutoModelForSeq2SeqLM
            # elif pipeline_tag == "zero-shot-classification":
            #     return AutoModelForSequenceClassification  # Or potentially AutoModel
            # elif pipeline_tag == "table-question-answering":
            #     return AutoModelForQuestionAnswering # May need a specific class
            # elif pipeline_tag == "visual-question-answering":
            #     return AutoModelForQuestionAnswering # May need a specific class
            # elif pipeline_tag == "image-classification":
            #     return AutoModelForImageClassification
            # elif pipeline_tag == "object-detection":
            #     return AutoModelForObjectDetection
            # elif pipeline_tag == "semantic-segmentation":
            #     return AutoModelForSemanticSegmentation
            # elif pipeline_tag == "audio-classification":
            #     return AutoModelForAudioClassification
            # elif pipeline_tag == "automatic-speech-recognition":
            #     return AutoModelForSpeechRecognition
            # elif pipeline_tag == "summarization":
            #     return AutoModelForSeq2SeqLM
            # elif pipeline_tag == "translation":
            #     return AutoModelForSeq2SeqLM
            # elif pipeline_tag == "text-to-speech":
            #     # No direct AutoModel yet, might need to use specific model class
            #     return None # Or suggest a base AutoModel
            # else:
            #     return AutoModel  # Default if pipeline_tag is unknown
            
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
                    host_monitor.step()
                    snap.step()
                    with torch.set_grad_enabled(True):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    if index == 3:
                        break
                    scheduler.step()
            logger.info("Train Finsihed！")
        except Exception as e:
            logger.error("Exception occurred during training.")
            logger.error(e)
            raise RuntimeError(f"Training failed: {e}") from e
        finally:
            profiler.stop()
            host_monitor.stop()
            snap.stop()
            logger.info("Post-action finished!")


class LLMEvaluator:
    def __init__(
            self,
            model_name,
            optimizer: str = "AdamW",
            batch_size: int = 10,
            gpu_id: int = 0,
            config: Config = None,
            tool_name: str = "xMem",
            iterations: int = 2,
    ):
        if torch.cuda.device_count() > 1:
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_IB_DISABLE"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
        torch.backends.cuda.cufft_plan_cache.max_size = 1
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
            device_id = self.g_id
        elif no_devices == 1:
            device_id = 0
        else:
            raise ValueError("No GPU device found")
        return torch.cuda.get_device_properties(device_id).total_memory, device_id

    def set_fraction_gpu_memory(self, gpu_memory_in_bytes: Optional[int] = 8*1024**3):
        total_gpu_memory, device_id = self.get_total_gpu_memory()
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
        print(f"Set GPU Memory Fraction: {round(fraction, 2)*100}%, limit: {format_memory(gpu_memory_in_bytes)}")
        torch.cuda.set_per_process_memory_fraction(
            round(fraction, 2), device=torch.device(f"cuda:{device_id}")
        )

    def train_on_gpu(self, gpu_memory_in_bytes: Optional[int] = None, config: Config = None) -> Config:
        if config is None:
            _config = self.config.model_copy(deep=True)
            _id = str(uuid.uuid4())[:4]
            if gpu_memory_in_bytes is None:
                task_id = f"GPU-{_id}"
            else:
                task_id = f"GPU-{format_memory(gpu_memory_in_bytes)}-{_id}"
            _config.task_id = task_id
            config = _config

        torch.cuda.empty_cache()
        time.sleep(1)  # wait for the memory to be released
        self.set_fraction_gpu_memory(gpu_memory_in_bytes)
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
        time.sleep(1)
        return config

    def get_estimate_max_gpu(self, config: Config):
        result_dir = config.result_dir
        print(result_dir)
        p_files = search_profiler_file(result_dir)

        from perf_estimator.estimator import TrainerEstimator
        from perf_estimator.dataset import image_dataset
        max_m_in_bytes, _ = self.get_total_gpu_memory()
        estimator = TrainerEstimator(
            profiler_file=p_files[-1],
            max_gpu_memory_in_gb=round(max_m_in_bytes/1024**3, 2),
            config=config,
        )
        return estimator.estimate()
    
    def get_truth_ground_max_gpu(self, config: Config):
        result_dir = config.result_dir
        n_files = search_nvml_file(result_dir)
        print(n_files[-1])

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
        import GPUtil
        import json
        # Get Estimated GPU Memory
        c_config = self.train_on_cpu()
        c_est, c_result = self.get_estimate_max_gpu(c_config)
        est_seg = max(c_est._trace.max_segment_changes)
        est_oom  = c_est.oom

        # Get Truth Ground GPU Memory
        try:
            g_config = self.train_on_gpu()
            g_nvml = self.get_truth_ground_max_gpu(g_config)
        except Exception as e:
            print(e)
            real_oom = True
            ground = None
        else:
            real_oom = False
            ground = max(g_nvml[str(self.g_id)])

        # Get Framework GPU Memory
        torch.cuda.empty_cache()
        time.sleep(3)
        framework_memory_usage = GPUtil.getGPUs()[self.g_id].memoryUsed * 1024**2
        total_memory_used = est_seg + framework_memory_usage

        if est_oom is False:
            # todo: Some works should be done in the future
            # As estimation program only consider the GPU memory changes during the training, it misses
            # the framework memory usage. Therefore, OOM should be re-evaluated.
            total_gpu_memory, _ = self.get_total_gpu_memory()
            est_oom = True if total_memory_used > total_gpu_memory else False

        # Record Results
        all_info = self.all_result
        all_info[self.tool_name] = {
            "ground": ground,
            "framework": framework_memory_usage,
            "memory": est_seg,
            "oom": est_oom,
            "real_oom": real_oom,
            "error": None if real_oom else abs(est_seg - ground) / ground,
            "correct_estimation": est_oom == real_oom,
            "2nd verification": {}
        }
        
        # 2nd verification

        if real_oom is False and all_info[self.tool_name]["correct_estimation"]:
            try:
                print(f"The max memory set to {format_memory(est_seg)}(Est) + {format_memory(framework_memory_usage)}(Framework)")
                g_config_2nd = self.train_on_gpu(gpu_memory_in_bytes=total_memory_used)
                g_nvml_2nd = self.get_truth_ground_max_gpu(g_config_2nd)
            except Exception as e:
                print(f"2nd verification run error: {e}")
                real_oom_2nd = True
                ground_2nd = None
            else:
                real_oom_2nd = False
                ground_2nd = max(g_nvml_2nd[str(self.g_id)])

            torch.cuda.empty_cache()
            time.sleep(3)
            tool_data = all_info[self.tool_name]
            tool_data["2nd verification"] = {
                "oom": real_oom_2nd,
                "ground": ground_2nd,
                "error": None if real_oom_2nd else abs(est_seg - ground_2nd) / ground_2nd,
            }


        all_info["config"] = self.config.model_dump()
        report_dir = self.config.base_dir
        with open(report_dir.joinpath(f"evaluation_result.json"), "w") as f:
            json.dump(all_info, f, indent=4)


def main(
    model: str = "EleutherAI/gpt-neo-125M",
    device_id: int = 1,
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
    time.sleep(10)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
