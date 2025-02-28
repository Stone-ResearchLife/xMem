import os
import json
from pathlib import Path
from typing import Union, List
from ures.files import filter_files
from perf_estimator.estimator import Estimator, TrainerEstimator
from perf_estimator.dataset import image_dataset
from experiments.snapshot import SnapshotAnalyser
from utility.data import DataAggregation, DataProcessorInterface, EstimatedMemoryRecord


class TrainerComparsion(DataProcessorInterface):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def get_pytorch_dir(self, model: str, batch: int, optimizer: str = "SDG") -> Path:
        pytorch_dir_name_format = "recurrence-{}-{}-{}-1"
        entrypoint_dir = (
            self.data_dir
            / "001-PyTorch"
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

    def get_huggingface_dir(
        self, model: str, batch: int, optimizer: str = "SDG"
    ) -> Path:
        huggingface_dir_name_format_llm = "{}-{}-LLM"
        return (
            self.data_dir
            / "002-HuggingFace"
            / huggingface_dir_name_format_llm.format(model, batch)
        )

    def get_tprofiler_dir(self, model: str, batch: int, optimizer: str = "SDG") -> Path:
        tprofiler_dir_name_format = "tprofiler-{}-batch-{}-Nvidia"
        return (
            self.data_dir
            / "003-NVML Ground"
            / tprofiler_dir_name_format.format(model, batch)
        )

    def get_paper_result(self, dest_dir: Union[str, Path]) -> dict:
        summary_json = filter_files("evaluation_result.json", dest_dir, fuzz=False)[-1]
        with open(summary_json, "r") as f:
            paper_result = json.load(f)
        return paper_result["solution"]

    def get_snapshot_result(self, dest_dir: Union[str, Path]) -> dict:
        snapshot_file = filter_files(".pickle", dest_dir, fuzz=True)[-1]
        _snapshot = SnapshotAnalyser(snapshot_file)
        return _snapshot.gpu_and_segment_in_same_time_length()

    def get_nvml_result(self, dest_dir: Union[str, Path]) -> dict:
        dest_file = filter_files("host_metrics", dest_dir, fuzz=True)[-1]
        with open(dest_file, "r") as f:
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

    def get_profiler_result(
        self,
        dest_dir: Union[str, Path],
        batch_size: int,
        gpu_capacity: int,
        huggingface_enabled: bool = True,
    ):
        profile_file = filter_files(".pt.trace.json", dest_dir, fuzz=True)[-1]
        if huggingface_enabled:
            estimator = TrainerEstimator(
                dataloader=image_dataset(batch=int(batch_size)),
                profiler_file=profile_file,
                max_gpu_memory_in_gb=gpu_capacity,
            )
        else:
            estimator = Estimator(
                dataloader=image_dataset(batch=int(batch_size)),
                profiler_file=profile_file,
                max_gpu_memory_in_gb=gpu_capacity,
            )
        my_result, _ = estimator.estimate()
        return my_result

    def get_data(self, *args, **kwargs) -> List[EstimatedMemoryRecord]:
        models = ["ConvNeXtTiny", "ResNet50", "VGG16"]
        batchs = range(10, 570, 40)
        optimizers = ["SGD"]
        gpu_capacity = 8
        records = []
        for model in models:
            for batch in batchs:
                for opt in optimizers:
                    ## Get HuggingFace.Trainer Result
                    nvml_result = self.get_nvml_result(
                        dest_dir=self.get_tprofiler_dir(model, batch, opt)
                    )
                    estimated_result = self.get_profiler_result(
                        dest_dir=self.get_huggingface_dir(model, batch, opt),
                        batch_size=batch,
                        gpu_capacity=gpu_capacity,
                        huggingface_enabled=True,
                    )
                    est = EstimatedMemoryRecord(
                        tool="xMem",
                        gt_tool="NVML",
                        platform="HuggingFace/Tainer",
                        model=model,
                        batch_size=batch,
                        optimizer=opt,
                        gpu_capacity=(gpu_capacity * 1024**3),
                        runtime=-1,
                        est_memory=max(estimated_result._trace.max_segment_changes),
                        gt_memory=max(nvml_result["0"]),
                        est_oom=estimated_result.oom,
                        oom=False,
                        accuracy_mode=False,
                    )
                    records.append(est)

                    ## Get Paper Result
                    paper_result = self.get_paper_result(
                        dest_dir=self.get_pytorch_dir(model, batch, opt)
                    )
                    paper_est = EstimatedMemoryRecord(
                        tool="xMem",
                        gt_tool="NVML",
                        platform="PyTorch",
                        model=model,
                        batch_size=batch,
                        optimizer=opt,
                        gpu_capacity=(gpu_capacity * 1024**3),
                        runtime=paper_result["runtime"],
                        est_memory=paper_result["memory"],
                        gt_memory=paper_result["ground"],
                        est_oom=paper_result["oom"],
                        oom=paper_result["real_oom"],
                        accuracy_mode=False,
                    )
                    records.append(paper_est)
        return records
