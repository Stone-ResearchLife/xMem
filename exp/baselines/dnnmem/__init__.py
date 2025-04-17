import torch
import copy
from typing import Union, Optional
from .llm import Estimator as LLMEstimator
from .cnn import Estimator as CNNEstimator
from ..abc import EstimatorInterface


class DNNmem(EstimatorInterface):
    def __init__(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            max_est_memory_in_bytes: int,
            optimizer: Optional[type(torch.optim)] = None,
            is_transformer: bool = False,
    ):
        self.estimator = None
        if is_transformer:
            self.estimator = LLMEstimator(
                model=model,
                dataloader=dataloader,
                max_est_memory_in_bytes=max_est_memory_in_bytes,
                optimizer=optimizer,
            )
        else:
            data_x, data_y = next(iter(copy.deepcopy(dataloader)))
            self.estimator = CNNEstimator(
                model=model,
                data_x=data_x,
                data_y=data_y,
                max_est_memory_in_bytes=max_est_memory_in_bytes,
            )

    @property
    def execute_time(self) -> Optional[Union[float, int]]:
        """
        The execute time of the estimation in ns
        """
        return self.estimator.execute_time

    @property
    def estimate_memory(self):
        """
        The estimated memory in bytes
        """
        return self.estimator.estimate_memory

    def estimate(self, *args, **kwargs) -> None:
        self.estimator.estimate()


__all__ = ['DNNmem']
