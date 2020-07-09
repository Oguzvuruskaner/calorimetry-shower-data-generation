from functools import wraps

from src.training.TrainingModel import TrainingModel


def TrainingWrapper(train_func):

    @wraps(train_func)
    def wrapper(*args,**kwargs):
        self : TrainingModel = args[0]
        data = args[1]
        self._data = data
        self._results = train_func(*args,**kwargs)

        self.run_all_validation_tasks()

        return self._results

    return wrapper