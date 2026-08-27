"""Structure file input shared by the command-line helpers.

xyz files are read with :func:`autopcet.structure.read_xyz`; anything else, and
anything needing an ASE ``Atoms`` object, goes through ASE, which the ``ase``
extra installs.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

from .._types import FloatArray
from ..ase_io import ASE_HINT as _ASE_HINT
from ..structure import read_xyz

if TYPE_CHECKING:
    from ase import Atoms


def read_atoms(path: str | Path, need: str = "Reading this structure") -> Atoms:
    """Read a structure into an ASE ``Atoms`` object."""
    try:
        from ase.io import read
    except ImportError as error:
        raise ImportError(_ASE_HINT.format(need=need)) from error

    # the default index selects the last frame, so this is always a single Atoms
    return cast("Atoms", read(path))


def read_structure(path: str | Path) -> tuple[list[str], FloatArray]:
    """Read a structure into its element symbols and ``(n_atoms, 3)`` positions."""
    path = Path(path)
    if path.suffix.lower() == ".xyz":
        return read_xyz(path)

    atoms = read_atoms(path, need=f"Reading '{path.suffix}' files")
    return list(atoms.symbols), atoms.get_positions()


def symbols_from_atomic_numbers(atomic_numbers: list[int]) -> list[str]:
    """Map atomic numbers onto element symbols, using ASE's periodic table."""
    try:
        from ase.data import chemical_symbols
    except ImportError as error:
        raise ImportError(
            _ASE_HINT.format(need="Writing the optimized geometry")
        ) from error

    return [chemical_symbols[number] for number in atomic_numbers]


def load_template(path: str | Path | None, default: str) -> str:
    """Return the Gaussian header template at ``path``, or ``default`` if unset."""
    return default if path is None else Path(path).read_text()
