import logging
import torch
import copy
import time
from abc import ABC, abstractmethod
from pathlib import Path
from perf_estimator.config import default_setting, Config
from .monitor import MonitorThreading


logger = logging.getLogger(__name__)


class InterfacePlugin(ABC):
    @abstractmethod
    def start(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass


class AbcPlugin(InterfacePlugin, ABC):
    def __init__(self, tool_name: str, config: Config = default_setting):
        self.tool_name = tool_name
        self.config = config

    @property
    def output_dir(self) -> Path:
        _result_dir = self.config.result_dir.joinpath(self.tool_name)
        _result_dir.mkdir(parents=True, exist_ok=True)
        return _result_dir


class SnapshotPlugin(AbcPlugin):
    TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 5000000


    def __init__(self, config: Config = default_setting):
        super(SnapshotPlugin, self).__init__("snapshot", config)

    def start(self, *args, **kwargs):
        logger.debug(f"Start Snapshot Plugin, max_entires: {self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT}")
        torch.cuda.memory._record_memory_history(
            stacks="all",
            max_entries=self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )

    def step(self, *args, **kwargs):
        pass

    def stop(self, output: str = None):
        file_name = f"{self.tool_name}_result-{int(time.time())}.pickle"
        out_file = self.output_dir.joinpath(file_name)
        logger.debug(f"Stop Snapshot Plugin and save the result to {out_file}")
        torch.cuda.memory._dump_snapshot(out_file)
        torch.cuda.memory._record_memory_history(enabled=None)


class ProfilerPlugin(AbcPlugin):
    DefaultConfig = {
        "activities": [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        "schedule": torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        "on_trace_ready": None,
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": True,
        "with_flops": False,
        "with_modules": True,
        "experimental_config": None,
        "execution_trace_observer": None,
    }

    def __init__(self, config: Config = default_setting):
        super(ProfilerPlugin, self).__init__("profiler", config)
        self._profiler = None

    def start(self, *args, **kwargs):
        _config = copy.deepcopy(self.DefaultConfig)
        _config.update(kwargs)
        _config["on_trace_ready"] = torch.profiler.tensorboard_trace_handler(
            str(self.output_dir)
        )
        logger.debug(f"Start Profiler Plugin, config: {_config}")
        self._profiler = torch.profiler.profile(**_config)
        self._profiler.start()

    def stop(self, *args, **kwargs):
        logger.debug(f"Stop Profiler, the result will be saved to tensorboard {self.output_dir}")
        if self._profiler is not None:
            self._profiler.stop()

    def step(self, *args, **kwargs):
        logger.debug("Step Profiler")
        if self._profiler is not None:
            self._profiler.step()


class HostMonitorPlugin(AbcPlugin):
    def __init__(
            self,
            interval_ms: int = 10,
            cpu_enable: bool = True,
            gpu_enable: bool = True,
            network_enable: bool = True,
            config: Config = default_setting
    ):
        super(HostMonitorPlugin, self).__init__("host_monitor", config)
        self._monitor = MonitorThreading(name=self.tool_name)
        self._params = {
            "interval_ms": interval_ms,
            "cpu_enable": cpu_enable,
            "gpu_enable": gpu_enable,
            "network_enable": network_enable,
            "output_dir": self.output_dir
        }

    def start(self, *args, **kwargs):
        logger.debug("Start Host Monitor Plugin")
        _params = kwargs
        _params.update(self._params)
        self._monitor.run(**_params)

    def stop(self, *args, **kwargs):
        logger.debug(f"Stop Host Monitor Plugin, the result will be saved to {self._params['output_dir']}")
        self._monitor.stop()

    def step(self, *args, **kwargs):
        pass


