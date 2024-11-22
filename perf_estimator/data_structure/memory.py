from abc import ABC, abstractmethod
from typing import Union


class MemoryBlock(ABC):
    @property
    @abstractmethod
    def bytes(self) -> int:
        pass

    @property
    @abstractmethod
    def address(self) -> str:
        pass

    @property
    @abstractmethod
    def alloc_time(self) -> Union[int, float]:
        pass

    @property
    @abstractmethod
    def free_time(self) -> Union[int, float]:
        pass

    @property
    @abstractmethod
    def duration(self) -> Union[int, float]:
        pass

    def __repr__(self):
        return f"Memory({self.address}): {self.bytes} bytes, start: {self.alloc_time}, end: {self.free_time}, duration: {self.duration}"
