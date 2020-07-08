from src.transformers.IInvertibleTransformation import IInvertibleTransformation


class Lambda(IInvertibleTransformation):

    def __init__(self,transformation,inverse_transformation=None,name=""):

        self._transformation = transformation
        self._inverse_transformation = inverse_transformation
        self._name = name

    def transform(self, data):
        return self._transformation(data)

    def inverse_transform(self, data):
        return self._inverse_transformation(data)

    def __str__(self):
        if self._name == "":
            return str(self._transformation)
        else:
            return self._name

