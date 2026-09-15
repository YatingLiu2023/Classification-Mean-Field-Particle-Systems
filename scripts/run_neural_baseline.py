#!/usr/bin/env python3
"""Run the leakage-free direct neural-network baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from mkv_classification.config import NeuralBaselineConfig, load_config, neural_baseline_profile
from mkv_classification.experiments_nn import run_neural_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("A", "B"), required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = (
        load_config(args.config, NeuralBaselineConfig)
        if args.config else neural_baseline_profile(args.profile, args.scenario)
    )
    if config.scenario != args.scenario:
        raise ValueError("--scenario must match the scenario stored in --config.")
    output = args.output or Path("results") / f"nn_{args.scenario}_{config.profile}"
    rows = run_neural_baseline(config, output)
    print(f"Completed {len(rows)} neural-baseline rows in {output}.")


if __name__ == "__main__":
    main()

