from abc import ABC,abstractmethod


class ISplitStrategy(ABC):


    @abstractmethod
    def get_split(self) -> '("x_train","y_train","x_test","y_test")':
        ...

    