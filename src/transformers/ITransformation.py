from abc import ABC, abstractmethod


class ITransformation(ABC):

    @abstractmethod
    def transform(self,data):
        ...

    @abstractmethod
    def __str__(self):
        ...