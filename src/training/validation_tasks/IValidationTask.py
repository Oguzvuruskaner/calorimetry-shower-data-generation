from abc import ABC,abstractmethod
import tensorflow as tf



class IValidationTask(ABC):

    @abstractmethod
    def validate(self,models:[tf.keras.Model],data,results):
        ...

