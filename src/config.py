EPOCH = 1500
STEPS_PER_EPOCH = 200
BATCH_SIZE = 64
LATENT_SIZE = 1024

PLOT_ERROR = 1e-15
TEST_IMAGES = 16
NUMBER_OF_LABELS = 10

TEST_BATCH = 16
DISCRIMINATOR_STEP = 5
DEPTH_PARAMETER = 3

CHECKPOINT_RATE = 10

MATRIX_DIMENSION = 32
TENSOR_DIMENSION = 16

HIT_Z_MIN = -15000.
HIT_Z_MAX = 15000.
HIT_X_MAX = 30000.
HIT_X_MIN = -30000.
HIT_Y_MAX = 15000.
HIT_Y_MIN = -15000.
HIT_R_MAX = (HIT_X_MAX * HIT_X_MAX + HIT_Y_MAX * HIT_Y_MAX)**.5


LEARNING_RATE = 1e-5
LAMBDA = 10


MODEL_VERSION = 4
STATE_DECAY = 1e-3
WEIGHT_DECAY = 1e-4
STATE_SIZE = 64

