"""Small numerical and type-checking helpers."""

import numpy as np

from ._types import FloatArray


def is_number(value: object) -> bool:
    """Return True if ``value`` is a real (non-boolean) scalar number."""
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(
        value, bool
    )


def is_array(value: object) -> bool:
    """Return True if ``value`` is a tuple, list, or NumPy array."""
    return isinstance(value, tuple | list | np.ndarray)


def find_roots(x: FloatArray, y: FloatArray) -> list[float]:
    """Locate the grid points closest to the sign changes of ``y``."""
    crossings = np.flatnonzero(np.sign(y[1:]) != np.sign(y[:-1]))
    return [float(x[i + 1] if abs(y[i + 1]) < abs(y[i]) else x[i]) for i in crossings]


def _find_first_crossing(x: FloatArray, y: FloatArray) -> tuple[int, float]:
    """Return the index and midpoint position of the first sign change of ``y``.

    The index is that of the point just past the crossing, so ``y[i - 1]`` and
    ``y[i]`` bracket it.
    """
    crossings = np.flatnonzero(y[1:] * y[:-1] <= 0)
    if crossings.size == 0:
        raise RuntimeError(
            "The reactant and product proton potentials do not cross within the "
            "proton coordinate grid."
        )
    index = int(crossings[0]) + 1
    return index, float((x[index] + x[index - 1]) / 2)
