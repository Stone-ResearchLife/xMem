import json
from typing import Union, List
from pathlib import Path
from ures.files import filter_files
from perf_estimator.estimator import Estimator, TrainerEstimator
from perf_estimator.dataset import image_dataset
from experiments.snapshot import SnapshotAnalyser
from utility.data import DataAggregation, DataProcessorInterface, EstimatedMemoryRecord


def search_nvml_file(target_dir: Union[Path, str]) -> List[Path]:
    return search_files("host_metrics", target_dir)


def search_snapshot_file(target_dir: Union[Path, str]) -> List[Path]:
    return search_files(".pickle", target_dir)


def search_profiler_file(target_dir: Union[Path, str]) -> List[Path]:
    return search_files(".pt.trace.json", target_dir)


def search_files(pattern: str, target_dir: Union[Path, str]) -> List[Path]:
    return [
        Path(file_path) for file_path in filter_files(pattern, target_dir, fuzz=True)
    ]


def get_nvml_result(nvml_json: Union[str, Path]) -> dict:
    with open(nvml_json, "r") as f:
        _data = json.load(f)
    _gpu_memory_usage = {}
    start_memory = {}
    for index, gpu_metric in enumerate(_data["records"]):
        gpu_data = gpu_metric["hostgpus"]
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
    profile_file: Union[str, Path],
    batch_size: int,
    gpu_capacity: int,
    huggingface_enabled: bool = True,
):
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


def get_snapshot_result(snapshot_file: Union[str, Path]) -> dict:
    _snapshot = SnapshotAnalyser(snapshot_file)
    return _snapshot.gpu_and_segment_in_same_time_length()


__all__ = [
    "search_files",
    "search_nvml_file",
    "search_snapshot_file",
    "search_profiler_file",
    "get_nvml_result",
    "get_profiler_result",
    "get_snapshot_result",
    "get_nvml_result",
    "get_profiler_result",
    "get_snapshot_result",
    "DataAggregation",
    "DataProcessorInterface",
    "EstimatedMemoryRecord",
]
