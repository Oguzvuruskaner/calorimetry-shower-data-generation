from abc import abstractmethod

from src.transformers.ITransformation import ITransformation


class IInvertibleTransformation(ITransformation):

    """
    # Concrete implementation of InvertibleTransformation
    A = InvertibleTransform()
    A.transform(data).inverse_transform(data) == data
    """

    @abstractmethod
    def inverse_transform(self,data):
        ...