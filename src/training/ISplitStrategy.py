from abc import ABC,abstractmethod


class ISplitStrategy(ABC):


    @abstractmethod
    def get_split(self,data) -> '("x_train","y_train","x_test","y_test")':
        ...

    