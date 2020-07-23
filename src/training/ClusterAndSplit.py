from src.training.ISplitStrategy import ISplitStrategy


class ClusterAndSplit(ISplitStrategy):

    def get_split(self,data) -> '("x_train","y_train","x_test","y_test")':
        pass