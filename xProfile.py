import time
import torch
import logging
import os
import copy
import fire
from uuid import uuid4
from typing import Optional, Union
from perf_estimator.dataset import image_dataset
from perf_estimator.trainer import ModelTrainer, ProfilerPlugin
from perf_estimator.utilis.enum import EnumManipulator
from perf_estimator.utilis.utilis import filter_files
from perf_estimator.config import Config
from perf_estimator.log import init_logging
from perf_estimator.models import AllModels


logger = logging.getLogger(__name__)


class XMemProfiler:
    def __init__(
            self,
            model: torch.nn.Module,
            batch_size: int = 200,
            input_size: int = 86,
            config: Optional[Config] = None,
            run_id: Optional[str] = None
    ):
        run_id = run_id or f"{model.__class__.__name__}-{batch_size}-{uuid4().hex[:8]}"
        self._model = model
        data_loader = image_dataset(batch=batch_size, image_size=(input_size, input_size))
        self._data_loader = data_loader
        self._batch_size = batch_size
        self._config = config or Config(run_id=run_id, save2tmp=False)

    @property
    def conf(self) -> Config:
        return self._config

    def _train(self, **kwargs):
        logger.debug(f"Start Training with {kwargs}")
        trainer = ModelTrainer(**kwargs)
        torch.cuda.empty_cache()
        time.sleep(1)
        trainer.train()
        time.sleep(1)
    
    def train_on_cpu(
            self,
            iteration: int = 2,
            optimizer: Optional[torch.optim.Optimizer] = None,
            loss: Optional[torch.nn.Module] = None,
            zero_grad_mode: int = 0
    ) -> str:
        trainer_conf = {
            "model": copy.deepcopy(self._model),
            "data_loader":copy.deepcopy(self._data_loader),
            "batch_size": self._batch_size,
            "iterations": iteration,
            "on_cpu": True,
            "config": self._config,
            "plugins": [
                ProfilerPlugin(config=self._config),
            ],
            "optimiser": optimizer,
            "loss": loss,
            "zero_grad_mode": zero_grad_mode
        }
        print(f"================== Evaluate on CPU ==================")
        try:
            self._train(**trainer_conf)
        finally:
            profiler_files = filter_files("pt.trace.json", str(self._config.result_dir), fuzz=True)
            profiler_files.sort(key=lambda x: os.path.getmtime(x))
        return profiler_files[-1]


def main(
        model: str,
        optimizer: str = "SGD",
        batch_size: int = 200,
        input_size: int = 86,
    ):
    models_enum = EnumManipulator(AllModels)
    models_list = models_enum.fetch_keys()
    if model not in models_list:
        raise ValueError(f"Model {model} is not supported. Supported models are {','.join(models_list)}")
    _conf = Config(save2tmp=False)
    init_logging(level="WARNING", conf=_conf)
    model = AllModels[model].value
    profiler = XMemProfiler(
        model=model,
        batch_size=batch_size,
        input_size=input_size,
        config=_conf
    )
    profiler_file = profiler.train_on_cpu(
        optimizer=getattr(torch.optim, optimizer, torch.optim.SGD)
    )
    print(f"Profiler file is saved in {profiler_file}")



if __name__ == '__main__':
    fire.Fire(main)