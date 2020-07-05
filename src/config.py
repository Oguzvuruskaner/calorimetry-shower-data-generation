import os

__DATASETS_WITH_OUTLIERS__ = ["triple_all"]

__DATASETS__ = ["triple_all"]

__DATA_MAPPING__ = [
    "hit_r",
    "hit_z",
    "hit_e"
]

__MODEL_VERSION__ = 10

HIT_Z_MAX = 1001
HIT_Z_MIN = -1000

HIT_R_MIN = -1
HIT_R_MAX = 1532

DIMENSION = 100

ENERGY = 50 #GeV

# N_COMPONENTS should be a square number
# to use 2d convolutions and locally connected layers
# in neural network.
N_COMPONENTS = 25

ROOT_FOLDER = os.path.join(".")