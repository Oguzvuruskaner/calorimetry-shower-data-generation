from detectAnomaly import getAbnormalValues
import numpy as np
from readRoot import create_all_quadruple_array_file
from scripts import loadAndSplitArray


def main(args):
    pass


if __name__ == "__main__":

    create_all_quadruple_array_file(["pion50GeVshowers.root"])
    loadAndSplitArray("npy/quadruple_all.npy",10)

    # data = np.load("npy/quadruple_all_chunk_10.npy")
    # data = np.array_split(data,90)[0]
    # data.resize((data.shape[0],1))
    # getAbnormalValues(data,model_path="isolation_forest_100.pkl")
