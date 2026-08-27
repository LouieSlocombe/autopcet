"""Shared type aliases and protocols used across the package."""

from collections.abc import Callable
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int_]
type PotentialFunction = Callable[[FloatArray], FloatArray]

type FitMethod = Literal["poly6", "poly8", "bspline"]
"""Name of a method for smoothing a tabulated potential."""

# A tabulated potential, given as the coordinate grid paired with the energies.
type TabulatedPotential = tuple[FloatArray, FloatArray] | list[FloatArray]


class ScalarOrArrayFunction(Protocol):
    """A function evaluating elementwise on a scalar or a 1D array."""

    def __call__[T: (float, FloatArray)](self, x: T, /) -> T: ...
