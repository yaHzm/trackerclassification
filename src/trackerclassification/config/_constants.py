import numpy as np
import math

# tracker sampling space bounds
X_MIN = -100.0
X_MAX = 100.0
Y_MIN = -100.0
Y_MAX = 100.0
Z_MIN = 150.0
Z_MAX = 200.0
MAX_ROT_ANGLE_DEG = 85.0

# tracker geometry
L = 64.0  # side length in mm
RS_VAL_E = np.array([[0.25, 0.50, 0.75],
                        [-0.02, 0.02, -0.02]], dtype=float)   
TRAFO_BE = np.array([[1.0, -1.0 / math.sqrt(3.0)],
                        [0.0,  2.0 / math.sqrt(3.0)]], dtype=float)  

# projection parameters
WIDTH = 1440
HEIGHT = 1080
HFOV_DEG = 90.0



# training arguments
OUTPUT_DIR = "./results"
PER_DEVICE_TRAIN_BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.0
NUM_TRAIN_EPOCHS = 100
LOGGING_STEPS = 100
EVAL_STRATEGY = "steps"
SAVE_STRATEGY = "steps"
SAVE_STEPS = 500
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 1
DATALOADER_NUM_WORKERS = 4
FP16 = False
SEED = 42
REPORT_TO = ["wandb"]  
METRIC_FOR_BEST_MODEL = "loss"
LOAD_BEST_MODEL_AT_END = True