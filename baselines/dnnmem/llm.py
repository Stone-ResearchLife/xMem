import torch
import time
from typing import Optional, Union
from perf_estimator.allocator import AllocatorSim, CachingAllocator
from baselines.utils import GenComputationalGraph
from baselines.abc import EstimatorInterface


class Estimator(EstimatorInterface):
    def __init__(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            max_est_memory_in_bytes: int,
            optimizer: Optional[type(torch.optim.Optimizer)] = None,
    ):
        self.cg = GenComputationalGraph(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
        )
        self.max_est_memory = max_est_memory_in_bytes/1024**3
        self._alloc: Optional[CachingAllocator]= None
        self._execute_time: Optional[Union[float, int]] = None

    @property
    def execute_time(self) -> Optional[Union[float, int]]:
        return self._execute_time

    @property
    def estimate_memory(self) -> Optional[Union[float, int]]:
        return max(self._alloc._trace.max_segment_changes) if self._alloc else None

    def estimate(self):
        s_time = time.time_ns()
        self.cg.prepare_computational_graph_data()
        _sim = AllocatorSim(self.max_est_memory)
        _cache_alloc = _sim.simulate(
            data_analysis=self.cg.gen_memory_blocks(),
            segment_plot=False,
        )
        self._execute_time = time.time_ns() - s_time
        self._alloc = _cache_alloc



