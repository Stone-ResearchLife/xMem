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
            / "002-GPU-profiling"
            / tprofiler_dir_name_format.format(model, optimizer, batch)
        )

    def _get_paper_data(
        self, model: str, batch: int, optimizer: str = "SDG"
    ) -> EstimatedMemoryRecord:
        summary_jsons = search_files(
            "evaluation_result.json",
            self.get_paper_dir(model=model, batch=batch, optimizer=optimizer),
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
        unified_data = self.data_dir.joinpath("001-CPU-profiling", "output.json")
        with open(str(unified_data), "r") as f:
            unified_data = json.load(f)

        ground_truth_data = get_nvml_result(ground_truth_json[-1])
        estimated_data = unified_data[model][optimizer][batch]["huggingface"][
            "estimated"
        ]

        est = EstimatedMemoryRecord(
            tool="xMem",
            gt_tool="NVML",
            platform="HuggingFace/Tainer",
            model=model,
            batch_size=batch,
            optimizer=optimizer,
            gpu_capacity=estimated_data["Max GPU Memory"],
            runtime=-1,
            est_memory=estimated_data["memory"]["segment"],
            gt_memory=max(ground_truth_data["0"]),
            est_oom=estimated_data["OOM"],
            oom=False,
            accuracy_mode=False,
        )
        return est

    def get_data(self, *args, **kwargs) -> List[EstimatedMemoryRecord]:
        models = ["ConvNeXtTiny", "ResNet50", "VGG16"]
        batchs = range(10, 570, 40)
        optimizers = ["SGD"]
        gpu_capacity = 8
        records = []
        for model in models:
            for batch in batchs:
                for opt in optimizers:
                    records.append(
                        self._get_trainer_data(model=model, batch=batch, optimizer=opt)
                    )
                    records.append(
                        self._get_paper_data(model=model, batch=batch, optimizer=opt)
                    )
        return records
