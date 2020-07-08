from src.datasets.IDataset import IDataset
from src.transformers.ITransformation import ITransformation
from src.transformers.IInvertibleTransformation import IInvertibleTransformation
from collections import deque
from logging import Logger,INFO
import numpy as np
import os
from src.decorators.Builder import BuilderMethod


class Dataset(IDataset):

    def __init__(self,name=""):

        self._post_stack : [ITransformation] = deque()
        self._pre_stack : [ITransformation] = deque()
        self._data = None
        self._logger = Logger(name=name,level=INFO)

    @BuilderMethod
    def add_invertible_transformation(self, transformation: IInvertibleTransformation):

        self._pre_stack.appendleft(transformation.transform)
        self._post_stack.appendleft(transformation.inverse_transform)
        return self

    @BuilderMethod
    def apply_all_transformations(self):

        while len(self._pre_stack) > 0 or len(self._post_stack) > 0:
            self.apply_next_transformation()

    @BuilderMethod
    def apply_next_transformation(self):
        transformation = None

        if len(self._pre_stack) > 0:
            transformation = self._pre_stack.popleft()

        elif len(self._post_stack) > 0:
            transformation = self._post_stack.popleft()

        self._logger.info("{} | Before {} Data shape:{}".format(self._logger.name, transformation, self._data.shape))
        self._data = transformation(self._data)
        self._logger.info("{} | Applied {} Data shape:{}".format(self._logger.name, transformation, self._data.shape))

    @BuilderMethod
    def apply_pre_transformations(self):
        while len(self._pre_stack) > 0:
            self.apply_next_transformation()

    @BuilderMethod
    def apply_post_transformations(self):
        while len(self._post_stack) > 0:
            self.apply_next_transformation()


    @BuilderMethod
    def add_post_transformation(self, transformation: ITransformation):
        self._post_stack.appendleft(transformation.transform)


    @BuilderMethod
    def add_pre_transformation(self, transformation: ITransformation):
        self._pre_stack.appendleft(transformation.transform)

    def array(self):
        return np.array(self._data)

    def __list__(self):
        return list(self._data)


    @staticmethod
    def get_root_files_in_directory(directory_path=os.path.join("root_files")):
        return [
            os.path.join("root_files", root_file)
            for root_file in os.listdir(directory_path)
            if root_file.endswith(".root")
        ]

    @staticmethod
    def get_root_files_in_multiple_directories(directory_paths):
        root_paths = []

        for root_dir in directory_paths:
            root_paths.extend(Dataset.get_root_files_in_directory(root_dir))

        return root_paths



