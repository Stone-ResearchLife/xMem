import logging
import torch
import copy
import time
from typing import Optional, Union
from uuid import uuid4
from perf_estimator.config import Config
from perf_estimator.dataset import image_dataset
from perf_estimator.estimator import Estimator
from perf_estimator.utilis.utilis import format_memory


logger = logging.getLogger(__name__)


class XMem:
    def __init__(
        self,
        data_loader: Optional[torch.utils.data.DataLoader] = None,
        batch_size: int = 200,
        input_size: int = 86,
        max_gpu_memory_in_gb: Union[int, float] = 4,
        config: Optional[Config] = None,
        run_id: Optional[str] = None,
    ):
        run_id = run_id or f"xMem-{batch_size}-{uuid4().hex[:8]}"
        if data_loader is None:
            data_loader = image_dataset(
                batch=batch_size, image_size=(input_size, input_size)
            )

        self._data_loader = data_loader
        self._batch_size = batch_size
        self._config = config or Config(run_id=run_id, save2tmp=False)
        self._max_gpu_memory_in_gb = max_gpu_memory_in_gb

    @property
    def conf(self) -> Config:
        return self._config

    def estimate(self, profiler_file: str, output_only: bool = False) -> dict:
        iteration = 2  # default value, better to keep it as default
        before_run = time.time()
        estimator = Estimator(
            dataloader=copy.deepcopy(self._data_loader),
            profiler_file=profiler_file,
            max_gpu_memory_in_gb=self._max_gpu_memory_in_gb,
            config=self.conf,
        )
        _, estimation_result = estimator.estimate(target_iteration=iteration)
        after_run = time.time()
        estimation_result.update({"runtime": round(after_run - before_run, 2)})
        if not output_only:
            estimation_result = self.display_output(estimation_result)
        return estimation_result

    def display_output(self, estimated_result: dict) -> dict:
        _estimated_result = copy.deepcopy(estimated_result)
        print(f"======================== Basic Information ========================")
        print(f"Batch Size: {self._batch_size}")
        print(f"Input Size: {list(self._data_loader.dataset[0][0].shape)}")
        print(f"Max GPU Memory: {self._max_gpu_memory_in_gb} GB")
        print(f"Runtime: {estimated_result.get('runtime', -1)} s")
        print(f"======================== Estimated Result ========================")
        is_OOM = estimated_result["OOM"]
        if is_OOM:
            print(f"OOM: {is_OOM}")
            print(
                f"{format_memory(estimated_result['Max GPU Memory'])} is not enough to run the model"
            )
        else:
            print(f"OOM: {is_OOM}")
            print(
                f"Estimated Peak GPU Memory: {format_memory(estimated_result['memory']['segment'])}"
            )
            print(
                f"Estimated Peak Tensor Memory: {format_memory(estimated_result['memory']['tensor'])}"
            )

        return _estimated_result
