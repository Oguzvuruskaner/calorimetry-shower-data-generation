from src.transformers.IInvertibleTransformation import IInvertibleTransformation

class Flatten(IInvertibleTransformation):

    def __init__(self):

        self._data_shape = None

    def inverse_transform(self, data):

        return data.reshape(self._data_shape)

    def transform(self, data):
        self._data_shape = data.shape

        return data.reshape((data.shape[0],data.shape[1]*data.shape[2]))

    def __str__(self):
        return "flatten"

