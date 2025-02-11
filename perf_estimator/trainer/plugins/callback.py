import torch
from abc import ABC, abstractmethod
from pathlib import Path
from transformers import TrainerCallback
from typing import Optional
from torch.profiler import profile, ProfilerActivity
from perf_estimator.config import Config, default_setting


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
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=5, repeat=1
            ),
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
    

class StepBasedStopCallback(AbsCallback):
    def __init__(self, stop_after_steps: int, config: Optional[Config] = None):
        super(StepBasedStopCallback, self).__init__(name="StepBasedStopCallback", config=config)
        self.stop_after_steps = stop_after_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Check if the number of steps has reached the limit
        if state.global_step >= self.stop_after_steps:
            control.should_training_stop = True  # Signal to stop training
        return control