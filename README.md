# UMA Catalysis Tutorial

This repository is refactoring the *AI for Accelerated Materials Science: ML
Potentials for Catalyst Discovery* notebook into a reproducible Python
codebase. The package will implement UMA-based bulk optimization, surface
energies, Wulff construction, adsorption, coverage, and reaction-barrier
workflows for the Ni tutorial system.

The refactor plan is available in
[REFACTORING_PROPOSAL.md](REFACTORING_PROPOSAL.md). The original notebook is
currently retained as the scientific and teaching reference.

## Status

The project scaffold is in place. Scientific workflows, TOML configuration,
and test cases will be added in subsequent refactoring steps.

## Requirements

- Python 3.12 or 3.13 (Python 3.12 is recommended);
- [Poetry](https://python-poetry.org/); and
- access to the gated [UMA model](https://huggingface.co/facebook/UMA) for
  model-backed calculations.

Store a Hugging Face token with UMA access outside the repository:

```bash
export HF_TOKEN="your-token"
```

## Installation

After generating the lock file, install the project dependencies with:

```bash
poetry install
```

The notebook's dependencies include FAIR Chemistry, ASE, PyTorch, pymatgen,
and CodeCarbon. Installing and running the full scientific workflows can
require substantial compute and model-download time.

## Development checks

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
PYTHONPATH=src poetry run python -m unittest discover -s tests -v
```

## License and citation

License and citation information will be added with the complete codebase.
Until then, preserve the citations and responsible-use guidance in the source
tutorial notebook when adapting or redistributing this work.
