from __future__ import annotations
from typing import Annotated
import numpy as np
from numpy.typing import NDArray

from ._typechecking import Shape, DType

VARIABLE = None


# L = number of LEDs per tracker
# T = number of trackers



# -- Numpy -----------------------------------------------------------------
# 1D Vectors
Vector_3_f = Annotated[NDArray[np.float64], Shape(3,), DType(np.float64)]
Vector_2_f = Annotated[NDArray[np.float64], Shape(2,), DType(np.float64)]

# 2D Matrices
Matrix_Lx3_f = Annotated[NDArray[np.float64], Shape(VARIABLE, 3), DType(np.float64)]

Matrix_3x3_f = Annotated[NDArray[np.float64], Shape(3, 3), DType(np.float64)]
Matrix_7x3_f = Annotated[NDArray[np.float64], Shape(7, 3), DType(np.float64)]
Matrix_Nx3_f = Annotated[NDArray[np.float64], Shape(VARIABLE, 3), DType(np.float64)]
Matrix_Nx2_f = Annotated[NDArray[np.float64], Shape(VARIABLE, 2), DType(np.float64)]
Matrix_TxL_b = Annotated[NDArray[np.bool_], Shape(VARIABLE, VARIABLE), DType(np.bool_)]

# ND Tensors
Tensor_TxLx3_f = Annotated[NDArray[np.float64], Shape(VARIABLE, VARIABLE, 3), DType(np.float64)]
Tensor_TxLx2_f = Annotated[NDArray[np.float64], Shape(VARIABLE, VARIABLE, 2), DType(np.float64)]

