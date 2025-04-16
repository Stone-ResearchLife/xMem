from abc import ABC, abstractmethod
from typing import Optional, Union


class EstimatorInterface(ABC):
    @property
    @abstractmethod
    def execute_time(self) -> Union[int, float]:
        "The return value is in ns"
        pass

    @property
    @abstractmethod
    def estimate_memory(self) -> Union[int, float]:
        "The return value is in bytes"
        pass

    @abstractmethod
    def estimate(self, *args, **kwargs) -> None:
        """
        execute the estimation
        """
        pass