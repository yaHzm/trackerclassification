from __future__ import annotations
from typing import Annotated
import numpy as np
from numpy.typing import NDArray

from ._typechecking import Shape, DType


# 1D Vectors
Vector_3_f = Annotated[NDArray[np.float64], Shape(3,), DType(np.float64)]
Vector_2_f = Annotated[NDArray[np.float64], Shape(2,), DType(np.float64)]

# 2D Matrices
Matrix_3x3_f = Annotated[NDArray[np.float64], Shape(3, 3), DType(np.float64)]
Matrix_7x3_f = Annotated[NDArray[np.float64], Shape(7, 3), DType(np.float64)]
Matrix_Tx7_b = Annotated[NDArray[np.bool_],   Shape(None, 7), DType(np.bool_)]
Matrix_Nx3_f = Annotated[NDArray[np.float64], Shape(None, 3), DType(np.float64)]
Matrix_Nx2_f = Annotated[NDArray[np.float64], Shape(None, 2), DType(np.float64)]

# ND Tensors
Tensor_Tx7x3_f = Annotated[NDArray[np.float64], Shape(None, 7, 3), DType(np.float64)]
Tensor_Tx7x2_f = Annotated[NDArray[np.float64], Shape(None, 7, 2), DType(np.float64)]

