"""Shared type aliases and protocols used across the package."""

from collections.abc import Callable
from typing import Protocol, overload

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type PotentialFunction = Callable[[FloatArray], FloatArray]


class _ScalarArrayFunction(Protocol):
    """A function evaluating elementwise on a scalar or a 1D array."""

    @overload
    def __call__(self, R: float) -> float: ...
    @overload
    def __call__(self, R: FloatArray) -> FloatArray: ...
