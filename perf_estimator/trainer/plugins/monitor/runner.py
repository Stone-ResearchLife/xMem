import threading
import logging
import time
import importlib
from pathlib import Path
from enum import Enum
from typing import List, Dict
from .interface import InterfaceHostMetric


logger = logging.getLogger(__name__)


class HostMetricsFeatures(Enum):
    CPU = "CGroupMonitor"
    Network = "EthernetMonitor"
    GPU = "HostGPUs"


class _HostMetrics:
    def __init__(
            self,
            interval_ms: int = 10,
            features: List[Enum] = None
    ):
        self.interval_ms = interval_ms
        self.records = []
        self.count = 0
        self.features: Dict[str, InterfaceHostMetric] = {}
        if features is None:
            features = []
        for feature in features:
            module = importlib.import_module('perf_estimator.trainer.plugins.monitor')
            _class = getattr(module, feature.value)
            _initialized_class = _class()
            _is_pass = _initialized_class.self_check()
            if _is_pass:
                self.features[feature.value] = _initialized_class

    def record(self):
        _metrics = {}
        _p_metrics = {}
        for _name, _feature in self.features.items():
            _p_metrics[_name] = _feature.record()
        time.sleep(self.interval_ms/1000)
        for _name, _feature in self.features.items():
            _result = _feature.summary(_p_metrics[_name], _feature.record(), interval_ms=self.interval_ms)
            _metrics[_name] = _result
        _metrics['timestamp'] = round(time.time_ns()/1e6, 2)
        self.records.append(_metrics)
        self.count += 1

    def to_json(self):
        _records = self.records
        _data = {
            "interval": self.interval_ms,  # unit: ms
            "num": self.count,
            "makespan": round((_records[-1]["timestamp"] - _records[0]["timestamp"]), 2),  # unit: ms
            "records": _records
        }
        return _data

    def save(self, file_path: Path):
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_json(), f, indent=4)


class MonitorThreading:
    def __init__(self, name: str = None):
        self.name = "monitor" if name is None else name
        self.stop_flat = threading.Event()
        self.thread = None

    @property
    def is_alive(self):
        if self.thread is not None:
            return self.thread.is_alive()
        return False

    def run(self, **kwargs):
        self.thread = threading.Thread(target=self._monitor_runner, kwargs=kwargs)
        self.thread.start()
        logger.info(f"Start monitoring thread: {self.name}, thread id: {self.thread.ident}")

    def stop(self):
        if self.thread is not None:
            logger.info(f"Stop monitoring thread: {self.name}, thread id: {self.thread.ident}")
            self.stop_flat.set()

    def _monitor_runner(
            self,
            interval_ms: int,
            cpu_enable: bool = True,
            gpu_enable: bool = True,
            network_enable: bool = True,
            output_dir: Path = None
    ):
        _features = []
        if cpu_enable:
            _features.append(HostMetricsFeatures.CPU)
        if gpu_enable:
            _features.append(HostMetricsFeatures.GPU)
        if network_enable:
            _features.append(HostMetricsFeatures.Network)
        _monitor = _HostMetrics(interval_ms=interval_ms, features=_features)
        while not self.stop_flat.is_set():
            _monitor.record()
        logger.info(f"Stop monitoring thread: {self.name}, thread id: {self.thread.ident}")
        file_name = f"host_metrics-{int(time.time())}.json"
        _monitor.save(output_dir.joinpath(file_name))


