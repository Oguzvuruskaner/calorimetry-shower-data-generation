from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler
from pickle import load,dump

def train_preprocessors(data,dataset_name):

    standardScaler = StandardScaler()
    standardScaler.fit(data)
    with open("scalers/{}_standard_scaler.pkl".format(dataset_name),"wb") as fp:
        dump(standardScaler,fp)

    minMaxScaler = MinMaxScaler()
    minMaxScaler.fit(data)
    with open("scalers/{}_min_max_scaler.pkl".format(dataset_name), "wb") as fp:
        dump(minMaxScaler, fp)

    robustScaler = RobustScaler()
    robustScaler.fit(data)
    with open("scalers/{}_robust_scaler.pkl".format(dataset_name), "wb") as fp:
        dump(robustScaler, fp)

    maxAbsScaler = MaxAbsScaler()
    maxAbsScaler.fit(data)
    with open("scalers/{}_max_abs_scaler.pkl".format(dataset_name),"wb") as fp:
        dump(maxAbsScaler,fp)
