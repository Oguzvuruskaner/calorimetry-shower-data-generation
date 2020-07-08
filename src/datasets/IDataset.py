from abc import ABC, abstractmethod

from src.transformers.IInvertibleTransformation import IInvertibleTransformation
from src.transformers.ITransformation import ITransformation


class IDataset(ABC):

    """
        Applying transformations are handled by
        double-ended queue.

        A - B - ... B' - A'
    """

    @abstractmethod
    def obtain(self):
        ...

    @abstractmethod
    def store(self):
        ...

    @abstractmethod
    def add_invertible_transformation(self, transformation: IInvertibleTransformation):
        ...

    @abstractmethod
    def add_post_transformation(self, transformation: ITransformation):
        ...

    @abstractmethod
    def add_pre_transformation(self, transformation: ITransformation):
        ...

    @abstractmethod
    def apply_all_transformations(self):
        ...

    @abstractmethod
    def apply_pre_transformations(self):
        ...

    @abstractmethod
    def apply_post_transformations(self):
        ...

    @abstractmethod
    def apply_next_transformation(self):
        ...

    @abstractmethod
    def array(self):
        ...


