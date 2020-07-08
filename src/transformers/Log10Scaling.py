import numpy as np
from src.transformers.IInvertibleTransformation import IInvertibleTransformation


class Log10Scaling(IInvertibleTransformation):


    def inverse_transform(self, data):

        return np.array([10]) **(data * 2) - 1

    def transform(self, data):

        return np.log10(data + 1)/2

    def __str__(self):

        return "log_10_scaler"

