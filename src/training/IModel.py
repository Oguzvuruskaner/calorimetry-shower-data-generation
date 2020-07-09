from abc import ABC,abstractmethod


class IModel(ABC):


    @abstractmethod
    def train(self,data) -> "train_results":
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def load(self):
        ...