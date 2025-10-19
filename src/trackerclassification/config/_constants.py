import numpy as np
import math

# tracker sampling space bounds
X_MIN = -100.0
X_MAX = 100.0
Y_MIN = -100.0
Y_MAX = 100.0
Z_MIN = 150.0
Z_MAX = 200.0
MAX_ROT_ANGLE_DEG = 45.0

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