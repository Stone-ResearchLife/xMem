import os
import json
from pathlib import Path
from typing import Union, List
from utility import *


class TrainerComparsion(DataProcessorInterface):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def get_paper_dir(self, model: str, batch: int, optimizer: str = "SDG") -> Path:
        pytorch_dir_name_format = "recurrence-{}-{}-{}-1"
        entrypoint_dir = (
            self.data_dir
            / "003-Paper-profiling"
            / pytorch_dir_name_format.format(model, optimizer, batch)
        )
        all_torch_dirs = os.listdir(entrypoint_dir)
        all_torch_dirs = [
            entrypoint_dir.joinpath(d)
            for d in all_torch_dirs
            if str(d).startswith(".") is False
        ]
        dirs_sorted = sorted(all_torch_dirs, key=lambda d: d.stat().st_ctime)
        return dirs_sorted[0]

    def get_trainer_cpu_dir(
        self, model: str, batch: int, optimizer: str = "SDG"
    ) -> Path:
        huggingface_dir_name_format_llm = "{}-{}-{}"
        return (
            self.data_dir
            / "001-CPU-profiling"
            / huggingface_dir_name_format_llm.format(model, batch, optimizer)
        )

    def get_trainer_gpu_dir(
        self, model: str, batch: int, optimizer: str = "SDG"
    ) -> Path:
        tprofiler_dir_name_format = "tprofiler-{}-{}-batch-{}-Nvidia"
        return (
            self.data_dir
            / "004-GPU-profiling-local"
            / tprofiler_dir_name_format.format(model, optimizer, batch)
        )

    def _get_paper_data(
        self, model: str, batch: int, optimizer: str = "SDG"
    ) -> EstimatedMemoryRecord:
        summary_jsons = search_files(
            "evaluation_result.json",
            self.get_paper_dir(model=model, batch=batch, optimizer=optimizer),
            fuzz=False,
        )
        with open(str(summary_jsons[-1]), "r") as f:
            paper_result = json.load(f)

        paper_result = paper_result["solution"]
        paper_est = EstimatedMemoryRecord(
            tool="xMem",
            gt_tool="NVML",
            platform="Paper",
            model=model,
            batch_size=batch,
            optimizer=optimizer,
            gpu_capacity=(8 * 1024**3),
            runtime=paper_result["runtime"],
            est_memory=paper_result["memory"],
            gt_memory=paper_result["ground"],
            est_oom=paper_result["oom"],
            oom=paper_result["real_oom"],
            accuracy_mode=False,
        )
        return paper_est

    def _get_trainer_data(
        self, model: str, batch: int, optimizer: str = "SDG"
    ) -> EstimatedMemoryRecord:
        ground_truth_json = search_nvml_file(
            target_dir=self.get_trainer_gpu_dir(
                model=model, batch=batch, optimizer=optimizer
            )
        )
        if len(ground_truth_json) == 0:
            raise FileNotFoundError(
                f"No ground truth json file for {model}/{batch}/{optimizer}"
            )
        ground_truth_data = get_nvml_result(ground_truth_json[-1])

        # unified_data = self.data_dir.joinpath("001-CPU-profiling", "output.json")
        # if unified_data.exists() is False:
        #     raise FileNotFoundError(
        #         f"No output json file for {model}/{batch}/{optimizer}"
        #     )
        #
        # with open(str(unified_data), "r") as f:
        #     unified_data = json.load(f)
        # estimated_data = unified_data[model][optimizer][str(batch)]["huggingface"]["estimated"]

        profile_files = search_profiler_file(
            target_dir=self.get_trainer_cpu_dir(
                model=model, batch=batch, optimizer=optimizer
            )
        )
        estimated_data = get_profiler_result(
            profile_file=profile_files[-1],
            batch_size=batch,
            gpu_capacity=8,
            huggingface_enabled=True,
        )

        est = EstimatedMemoryRecord(
            tool="xMem",
            gt_tool="NVML",
            platform="HuggingFace/Tainer",
            model=model,
            batch_size=batch,
            optimizer=optimizer,
            gpu_capacity=estimated_data.allowed_memory_maximum,
            runtime=-1,
            est_memory=max(estimated_data._trace.max_segment_changes),
            gt_memory=max(ground_truth_data["1"]),
            est_oom=estimated_data.oom,
            oom=False,
            accuracy_mode=False,
        )
        return est

    def get_data(self, *args, **kwargs) -> List[EstimatedMemoryRecord]:
        models = ["ConvNeXtTiny", "ResNet50", "VGG16"]
        batchs = range(10, 570, 40)
        optimizers = ["SGD"]
        gpu_capacity = 8

        models = kwargs.pop("models", models)
        batchs = kwargs.pop("batchs", batchs)
        optimizers = kwargs.pop("optimizers", optimizers)
        gpu_capacity = kwargs.pop("gpu_capacity", gpu_capacity)
        records = []
        for model in models:
            for batch in batchs:
                for opt in optimizers:
                    try:
                        trainer_record = self._get_trainer_data(
                            model=model, batch=batch, optimizer=opt
                        )
                    except Exception:
                        print(f"No trainer json file for {model}/{batch}/{opt}")
                        continue
                    else:
                        records.append(trainer_record)
                    try:
                        paper_record = self._get_paper_data(
                            model=model, batch=batch, optimizer=opt
                        )
                    except Exception as e:
                        print(f"No paper json file for {model}/{batch}/{opt}")
                        continue
                    else:
                        records.append(paper_record)
        return records
