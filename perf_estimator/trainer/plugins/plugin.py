import logging
import torch
import copy
from abc import ABC, abstractmethod
from pathlib import Path
from perf_estimator.config import default_setting, Config


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
        _result_dir = self.config.result_dir.joinpath("plugins", self.tool_name)
        _result_dir.mkdir(parents=True, exist_ok=True)
        return _result_dir


class ProfilerPlugin(AbcPlugin):
    DefaultConfig = {
        "activities": [
            torch.profiler.ProfilerActivity.CPU,
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
        logger.debug(
            f"Stop Profiler, the result will be saved to tensorboard {self.output_dir}"
        )
        if self._profiler is not None:
            self._profiler.stop()

    def step(self, *args, **kwargs):
        logger.debug("Step Profiler")
        if self._profiler is not None:
            self._profiler.step()
