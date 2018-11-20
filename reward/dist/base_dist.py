from abc import ABC, abstractmethod


class BaseDist(ABC):
    @abstractmethod
    def __getitem__(self, key):
        pass
