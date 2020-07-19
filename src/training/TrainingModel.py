from src.training.IModel import IModel
from src.training.validation_tasks.IValidationTask import IValidationTask


class TrainingModel(IModel):

    def __init__(self):

        self._models = []
        self._results = None
        self._validation_tasks : [IValidationTask] = []
        self._data = None
        self._training_params = {}

    def get_models(self):
        return self._models

    def add_validation_task(self, validation_task:IValidationTask):
        self._validation_tasks.append(validation_task)
        return self

    def run_all_validation_tasks(self):

        for task in self._validation_tasks:
            task.validate(self._models,self._data,self._results)


