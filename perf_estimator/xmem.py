import logging
import torch
import copy
import fire
from typing import Optional, Union
from uuid import uuid4
from perf_estimator.config import Config
from perf_estimator.dataset import image_dataset
from perf_estimator.estimator import Estimator
from perf_estimator.log import init_logging


logger = logging.getLogger(__name__)


class XMem:
    def __init__(
            self,
            model: torch.nn.Module,
            data_loader: Optional[torch.utils.data.DataLoader] = None,
            batch_size: int = 200,
            input_size: int = 86,
            max_gpu_memory_in_gb: Union[int, float] = 4,
            config: Optional[Config] = None,
            run_id: Optional[str] = None
    ):
        run_id = run_id or f"{model.__class__.__name__}-{batch_size}-{uuid4().hex[:8]}"
        self._model = model
        if data_loader is None:
            data_loader = image_dataset(batch=batch_size, image_size=(input_size, input_size))

        self._data_loader = data_loader
        self._batch_size = batch_size
        self._config = config or Config(run_id=run_id, save2tmp=False)
        self._max_gpu_memory_in_gb = max_gpu_memory_in_gb

    @property
    def conf(self) -> Config:
        return self._config

    def estimate(self, profiler_file: str) -> dict:
        iteration = 2 # default value, better to keep it as default
        estimator = Estimator(
            model=copy.deepcopy(self._model),
            dataloader=copy.deepcopy(self._data_loader),
            profiler_file=profiler_file,
            max_gpu_memory_in_gb=self._max_gpu_memory_in_gb,
            config=self.conf
        )
        _, estimation_result = estimator.estimate(target_iteration=iteration)
        return self.display_output(estimation_result)

    def display_output(self, estimated_result: dict) -> dict:
        print(f"======================== Basic Information ========================")
        print(f"Model: {self._model.__class__.__name__}")
        print(f"Batch Size: {self._batch_size}")
        print(f"Input Size: {list(self._data_loader.dataset[0][0].shape)}")
        print(f"Max GPU Memory: {self._max_gpu_memory_in_gb}GB")
        print(f"======================== Estimated Result ========================")
        is_OOM = estimated_result['OOM']
        if is_OOM:
            print(f"OOM: {is_OOM}")
            print(f"{estimated_result['Max GPU Memory']}GB is not enough to run the model")
        else:
            print(f"OOM: {is_OOM}")
            print(f"Estimated Peak GPU Memory: {round(estimated_result['memory']['segment']/1024**3, 2)}GB")
            print(f"Estimated Peak Tensor Memory: {round(estimated_result['memory']['tensor']/1024**3, 2)}GB")

        return estimated_result
