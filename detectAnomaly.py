from sklearn.ensemble import IsolationForest
from pickle import dump,load
import numpy as np
from typing import Union
from os.path import join

N_ESTIMATORS = 100

def loadModel(modelPath):

    return load(modelPath)

def getAbnormalValues(data : Union[np.array,np.ndarray],model_path=None) -> np.ndarray:

    # In this context outlier means abnormal.

    if not model_path:

        fp = open(join("models","isolation_forest_{}.pkl".format(N_ESTIMATORS)),"wb")

        isolation_forest = IsolationForest(n_estimators=N_ESTIMATORS,verbose=1,n_jobs=8)
        isolation_forest.fit(data)
        dump(isolation_forest,fp)

    else:
        fp = open(model_path,"rb")
        isolation_forest = load(fp)



    outlierProbability = isolation_forest.predict(data)

    return data[outlierProbability == -1]


def filterAbnormalValues(data,filterValues):

    return data[filterValues]