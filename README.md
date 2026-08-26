# autopcet

Automated proton-coupled electron transfer (PCET) tooling, packaged as a
modern, typed Python project. It uses setuptools and `pyproject.toml`
packaging, Ruff, mypy, pytest with full branch coverage, pre-commit, and
GitHub Actions.

The package currently ships a tiny placeholder NumPy API and a command-line
entry point, so the quality gates run end to end before the real code lands.

## Requirements

- Python 3.14 or newer
- pip 25.1 or newer (for dependency groups)

## Quick start

Create an isolated environment and install the package with its development
tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --group dev -e .
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead.

Run the command-line entry point:

```bash
autopcet Ada
python -m autopcet Ada
```

Or use the library:

```python
from autopcet import line, print_hello

print_hello("Ada")
samples = line(-1.0, 1.0, num=5)
print(samples)
```

## Development

Run the complete local checks:

```bash
ruff check .
ruff format --check .
mypy
pytest
python -m build
python -m twine check dist/*
```

Ruff can apply safe lint and formatting changes with:

```bash
ruff check --fix .
ruff format .
```

Install the Git hooks once, then pre-commit will run the fast checks before
each commit:

```bash
pre-commit install
pre-commit run --all-files
```

If you prefer Conda, `build_tools/environment.yml` creates the base environment:

```bash
conda env create -f build_tools/environment.yml
conda activate autopcet
python -m pip install --group dev -e .
```

## Project layout

```text
.
├── .github/workflows/ci.yml   # automated quality and packaging checks
├── build_tools/               # optional Conda setup
├── autopcet/                  # installable package
├── tests/                     # behavior-focused tests
└── pyproject.toml             # project metadata and tool configuration
```

## Next steps

1. Replace the placeholder API and tests with the real PCET code, keeping the
   quality gates green.
2. Confirm the Python versions the project supports.
3. Set a real release version and configure trusted publishing once the
   package is ready.

## License

Released under the [MIT License](LICENSE).
