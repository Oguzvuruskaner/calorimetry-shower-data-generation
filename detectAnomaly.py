from sklearn.ensemble import IsolationForest
import matplotlib
from pickle import dump,load
from tqdm import tqdm
import numpy as np
from typing import Union

N_ESTIMATORS = 100

def loadModel(modelPath):

    return load(modelPath)

def getAbnormalValues(data : Union[np.array,np.ndarray],model_path=None) -> np.ndarray:

    # In this context outlier means abnormal.

    if not model_path:

        fp = open("isolation_forest_{}.pkl".format(N_ESTIMATORS),"wb")

        isolation_forest = IsolationForest(n_estimators=N_ESTIMATORS,verbose=1,n_jobs=8)
        isolation_forest.fit(data)
        dump(isolation_forest,fp)

    else:
        fp = open(model_path,"rb")
        isolation_forest = load(fp)



    outlierProbability = isolation_forest.predict(data)

    asdf = data[outlierProbability == -1]
    print(asdf)


def filterAbnormalValues(data,filterValues):

    return data[filterValues]