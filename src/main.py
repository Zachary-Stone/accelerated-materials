"""Run a configured UMA catalysis experiment from the command line."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from uma_catalysis.runner import load_experiment_config, run_experiment


def main(arguments: Sequence[str] | None = None) -> int:
    """Load one TOML experiment configuration, execute it, and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "experiment.toml",
        help="Path to the experiment TOML file.",
    )
    parsed = parser.parse_args(arguments)
    config = load_experiment_config(parsed.config)
    experiment = run_experiment(config)
    print(f"Executed: {', '.join(experiment.executed_workflows)}")
    print(f"Summary: {experiment.summary_path}")
    if experiment.emissions_path is not None:
        print(f"Emissions: {experiment.emissions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
