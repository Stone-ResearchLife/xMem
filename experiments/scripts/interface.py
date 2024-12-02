from abc import ABC, abstractmethod


class InterfaceApp(ABC):
    @abstractmethod
    def build(self, **kwargs):
        pass

    @abstractmethod
    def execute(self, **kwargs):
        pass