import torch
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from typing import Optional
from torch.profiler import profile, ProfilerActivity
from perf_estimator.config import Config, default_setting

logger = logging.getLogger(__name__)


class AbsCallback(TrainerCallback, ABC):
    def __init__(self, name: str, config: Optional[Config] = None):
        if config is None:
            config = default_setting
        self.config = config
        self.name = name

    def output_dir(self) -> Path:
        return self.config.result_dir.joinpath("callback", self.name)


class ProfilerCallback(AbsCallback):
    def __init__(self, config: Optional[Config] = None):
        super(ProfilerCallback, self).__init__(name="Profiler", config=config)
        activities = [ProfilerActivity.CPU]
        self.profiler = profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.output_dir())
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=False,
            with_modules=True,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting profiler...")
        self.profiler.start()

    def on_step_begin(self, args, state, control, **kwargs):
        self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        print("Stopping profiler...")
        self.profiler.stop()


class SnapshotCallback(AbsCallback):
    TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 5000000

    def __init__(self, config: Optional[Config] = None):
        super(SnapshotCallback, self).__init__(name="Snapshot", config=config)

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        torch.cuda.memory._record_memory_history(
            stacks="all", max_entries=self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        file_name = f"{self.name}_result-{int(time.time())}.pickle"
        out_file = self.output_dir().joinpath(file_name)
        if out_file.parent.is_dir() is False:
            out_file.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Stop Snapshot Plugin and save the result to {out_file}")
        torch.cuda.memory._dump_snapshot(out_file)
        torch.cuda.memory._record_memory_history(enabled=None)
