from __future__ import annotations
import inspect
from functools import wraps
from typing import Annotated, Union, Optional, Any, Callable, TypeVar, ParamSpec, get_type_hints
import numpy as np
from numpy.typing import NDArray


class Shape:
    """
    Helper for specifying expected shape of numpy arrays in type annotations.

    Parameters:
        *dims (Optional[int]): dimensions of the array shape, use None for variable size

    Examples:
      Shape(3, 3)     -> exactly (3, 3)   
      Shape(None, 7)  -> any rows x 7    
      Shape(3,)       -> 1D of length 3    
    """
    def __init__(self, *dims: Optional[int]):
        self._dims = dims


class DType:
    """
    Helper for specifying expected dtype of numpy arrays in type annotations.

    Parameters:
        dtype (Union[np.dtype, str]): expected numpy dtype
    """
    def __init__(self, dtype: Union[np.dtype, str]):
        self._dtype = np.dtype(dtype)


def _check_one_arg(name: str, val: Any, meta: Union[Shape, DType]) -> None:
    """
    Check a single argument `val` against one metadata `meta` (Shape or DType).

    Raises: 
        TypeError, ValueError: if the check fails
    """
    if isinstance(meta, (Shape, DType)) and not isinstance(val, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray for checks, got {type(val)!r}")
    if isinstance(meta, Shape):
        want = meta._dims
        if val.ndim != len(want):
            raise ValueError(f"{name} must have {len(want)} dims, got {val.ndim}")
        for i, dim in enumerate(want):
            if dim is not None and val.shape[i] != dim:
                raise ValueError(f"{name} dim {i} must be {dim}, got {val.shape[i]}")
    if isinstance(meta, DType):
        if val.dtype != meta._dtype:
            raise TypeError(f"{name} must have dtype {meta._dtype}, got {val.dtype}")


P = ParamSpec("P")
R = TypeVar("R")


def check_dtypes(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Function decorator: enforces numpy array type checks based on Annotated type hints.

    Parameters:
        fn (Callable): function to wrap

    Returns:
        Callable: wrapped function with numpy type checks
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, val in bound.arguments.items():
            anno = hints.get(name)
            # looking for Annotated[NDArray[...], <metadata>...]
            if getattr(anno, "__origin__", None) is Annotated:
                base, *metas = anno.__args__
                # only enforce for NDArray (skip non-numpy params that might be Annotated)
                if getattr(base, "__origin__", None) is NDArray:
                    for meta in metas:
                        _check_one_arg(name, val, meta)
        return fn(*args, **kwargs)
    wrapper.__signature__ = sig
    return wrapper


def check_dtypes_cls(cls: Any):
    """
    Class decorator: enforces numpy array type checks on all methods of the class based on Annotated type hints.

    Parameters:
        cls (Any): class to wrap

    Returns:
        cls (Any): wrapped class with numpy type checks on methods
    """
    for attr_name, attr_val in list(cls.__dict__.items()):
        # Skip dunder and properties/attributes
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue
        if isinstance(attr_val, property):
            continue
        # staticmethod
        if isinstance(attr_val, staticmethod):
            func = attr_val.__func__
            wrapped = check_dtypes(func)
            setattr(cls, attr_name, staticmethod(wrapped))
            continue
        # classmethod
        if isinstance(attr_val, classmethod):
            func = attr_val.__func__
            wrapped = check_dtypes(func)
            setattr(cls, attr_name, classmethod(wrapped))
            continue
        # plain function (instance method)
        if inspect.isfunction(attr_val):
            setattr(cls, attr_name, check_dtypes(attr_val))
    return cls


