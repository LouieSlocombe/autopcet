"""Small numerical and type-checking helpers."""

import numpy as np

from ._types import FloatArray


def is_number(dat: object) -> bool:
    """Return True if ``dat`` is a real (non-boolean) scalar number."""
    return isinstance(dat, (int, float, np.integer, np.floating)) and not isinstance(
        dat, bool
    )


def is_array(dat: object) -> bool:
    """Return True if ``dat`` is a tuple, list, or NumPy array."""
    return isinstance(dat, (tuple, list, np.ndarray))


def find_roots(xdata: FloatArray, ydata: FloatArray) -> list[float]:
    """Locate the grid points closest to the sign changes of ``ydata``."""
    xo = xdata[0]
    yo = ydata[0]
    roots: list[float] = []
    for xi, yi in zip(xdata[1:], ydata[1:], strict=True):
        if np.sign(yi) != np.sign(yo):
            if np.abs(yi) < np.abs(yo):
                roots.append(xi)
            else:
                roots.append(xo)
        xo = xi
        yo = yi

    return roots


def _find_first_crossing(y: FloatArray, x: FloatArray) -> tuple[int, float]:
    """Return the index and midpoint position of the first sign change of ``y``."""
    for i in range(1, len(x)):
        if y[i] * y[i - 1] <= 0:
            return i, (x[i] + x[i - 1]) / 2
    raise RuntimeError(
        "The reactant and product proton potentials do not cross within the rp grid."
    )
