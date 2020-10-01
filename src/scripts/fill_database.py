import uproot4 as uproot
from tqdm import trange

from src.config import MATRIX_DIMENSION, TENSOR_DIMENSION, HIT_R_MAX, HIT_Z_MIN, HIT_Z_MAX, HIT_X_MAX, HIT_Y_MAX
from src.datasets import DATASETS
from pymongo import MongoClient
import bson
import numpy as np
import pickle
import hashlib


def deserialize_array(binary ) :

    return pickle.loads(binary)

def serialize_array(arrr : np.array):

    return bson.Binary(pickle.dumps(arrr,protocol=4))

def get_connection(username="root",password="example") -> MongoClient:

    return MongoClient(
        "mongodb://localhost:27017",
        username=username,
        password=password
    )

entries = lambda root : root["g4SimHits"]["eventTree"]

def fill_database():

    client  = get_connection()
    ml4sim = client["ml4sim"]
    jets_collection = ml4sim["jets"]

    for GeV in DATASETS.keys():

        for dataset in DATASETS[GeV]:

            entry_strat = dataset.get("entries",None) or entries
            dataset_path = dataset["path"]

            with uproot.open(dataset_path) as root:


                entry_directory = entry_strat(root)

                hit_y = entry_directory["hit_y"]
                for y_ind in trange(hit_y.num_entries):

                    x = entry_directory["hit_x"].array(entry_start=y_ind,entry_stop=y_ind+1,library="np")[0]
                    y = hit_y.array(entry_start=y_ind,entry_stop=y_ind+1,library="np")[0]
                    z = entry_directory["hit_z"].array(entry_start=y_ind,entry_stop=y_ind+1,library="np")[0]
                    e = entry_directory["hit_e"].array(entry_start=y_ind,entry_stop=y_ind+1,library="np")[0]


                    m = hashlib.sha512()

                    tmp_x = x / HIT_X_MAX
                    tmp_y = y / HIT_Y_MAX
                    tmp_r = np.sqrt(x*x + y*y) / HIT_R_MAX
                    tmp_z = (z - HIT_Z_MIN) / (HIT_Z_MAX - HIT_Z_MIN)
                    tmp_e = e/ GeV


                    matrix_view = np.hstack([
                        tmp_r.reshape(-1,1),
                        tmp_z.reshape(-1,1),
                        tmp_e.reshape(-1,1)
                    ])

                    tensor_view = np.hstack([
                        tmp_x.reshape(-1, 1),
                        tmp_y.reshape(-1, 1),
                        tmp_z.reshape(-1, 1),
                        tmp_e.reshape(-1, 1)
                    ])

                    hist_2d = np.histogramdd(matrix_view[:, :2],
                                             bins=MATRIX_DIMENSION,
                                             range=np.array([[0, 1], [0, 1]]),
                                             weights=matrix_view[:, 2])[0]

                    hist_3d = np.histogramdd(tensor_view[:, :3],
                                             bins=TENSOR_DIMENSION,
                                             range=np.array([[0, 1], [0, 1], [0, 1]]),
                                             weights=tensor_view[:, 3])[0]


                    jet_obj = {
                        "energy":GeV,
                        "matrix_view":serialize_array(hist_2d),
                        "tensor_view":serialize_array(hist_3d),
                        "particles":serialize_array(tensor_view)
                    }

                    m.update(jet_obj["particles"])
                    jet_obj["_id"] = m.hexdigest()

                    jets_collection.insert_one(jet_obj)