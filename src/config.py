DATA_PATH = ["../data/part-r-00001"]

MODEL_SAVE_PATH = "../model/"
PB_SAVE_PATH = "../pb/"

EPOCH = 10
BATCH_SIZE = 1024
NUM_PARALLEL = 4


FEATURE_LEN = 27
LABEL_LEN = 3
USER_VOCAB_SIZE = 100000
USER_EMB_SIZE = 4
RANDOM_SEED = 1
POI_VOCAB_SIZE = 100000
POI_EMB_SIZE = 4
CONTEXT_VOCAB_SIZE = 100000
CONTEXT_EMB_SIZE = 4
LEARNING_RATE = 1e-3

USE_AUX_RES_LOSS = True
AUX_RES_LOSS_WIEHGT = 1.0
TARGET_RES = 0.8

CHANNEL_CNT = 4
CHANNEL_MASK = [
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
]


WHOLE_ACTION = [
    [0,0,0,0,0],
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [1,1,0,0,0],
    [1,0,1,0,0],
    [1,0,0,1,0],
    [1,0,0,0,1],
    [0,1,1,0,0],
    [0,1,0,1,0],
    [0,1,0,0,1],
    [0,0,1,1,0],
    [0,0,1,0,1],
    [0,0,0,1,1],
    [1,1,1,0,0],
    [1,1,0,1,0],
    [1,1,0,0,1],
    [1,0,1,1,0],
    [1,0,1,0,1],
    [1,0,0,1,1],
    [0,1,1,1,0],
    [0,1,1,0,1],
    [0,1,0,1,1],
    [0,0,1,1,1],
    [1,1,1,1,0],
    [1,1,1,0,1],
    [1,1,0,1,1],
    [1,0,1,1,1],
    [0,1,1,1,1],
    [1,1,1,1,1]
]

WHOLE_ACTION_RES = [
    0.0,0.10,0.11,0.12,0.13,0.14,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.5
]

AD_WEIGHT,FEE_WEIGHT,REX_WEIGHT = [1.0,1.0,1.0]