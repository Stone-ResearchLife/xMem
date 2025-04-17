from abc import ABC, abstractmethod


class InterfaceHostMetric(ABC):
    def record(self, *args, **kwargs) -> dict:
        pass

    def summary(self, p_record: dict, c_record: dict, interval_ms: int) -> dict:
        pass

    def self_check(self) -> bool:
        pass
