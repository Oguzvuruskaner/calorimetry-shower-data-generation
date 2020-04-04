from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from pickle import load,dump

def train_preprocessors(data,dataset_name):

    standardScaler = StandardScaler()
    standardScaler.fit(data)
    with open("scalers/standard_scaler_{}.pkl".format(dataset_name),"rb") as fp:
        dump(standardScaler,fp)

    minMaxScaler = MinMaxScaler()
    minMaxScaler.fit(data)
    with open("scalers/min_max_scaler_{}.pkl".format(dataset_name), "rb") as fp:
        dump(minMaxScaler, fp)

    robustScaler = RobustScaler()
    robustScaler.fit(data)
    with open("scalers/robust_scaler_{}.pkl".format(dataset_name), "rb") as fp:
        dump(robustScaler, fp)


