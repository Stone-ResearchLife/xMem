from ..abc import EstimatorInterface
from typing import Optional, Union
from pathlib import Path
from perf_estimator.xmem import XMem, Config


class MySolution(EstimatorInterface):
    def __init__(
            self,
            batch_size: int,
            max_gpu_memory_in_gb: Union[int, float],
            config: Config,
            profiler_file: Union[str, Path],
    ):
        self.batch_size = batch_size
        self.max_gpu_memory_in_gb = max_gpu_memory_in_gb
        self.config = config
        self.profiler_file = profiler_file
        self._result = None

    @property
    def estimate_memory(self) -> Optional[int]:
        return self._result['memory']['segment'] if self._result is not None else None

    @property
    def execute_time(self) -> Optional[float]:
        return self._result['runtime'] if self._result is not None else None

    @property
    def oom(self) -> Optional[bool]:
        return self._result['OOM'] if self._result is not None else None

    def estimate(self, *args, **kwargs) -> None:
        xmem = XMem(
            batch_size=self.batch_size,
            max_gpu_memory_in_gb=self.max_gpu_memory_in_gb,
            config=self.config,
        )
        result = xmem.estimate(
            profiler_file=self.profiler_file,
            output_only=True,
        )
        self._result = result

