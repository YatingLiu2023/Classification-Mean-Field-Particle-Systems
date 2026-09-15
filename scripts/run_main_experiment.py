#!/usr/bin/env python3
"""Run the fixed-D main experiment from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from mkv_classification.config import MainExperimentConfig, load_config, main_profile
from mkv_classification.experiments import run_main_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--config", type=Path, help="Optional custom JSON configuration.")
    parser.add_argument("--output", type=Path, help="Output directory; existing checkpoints resume.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = (
        load_config(args.config, MainExperimentConfig)
        if args.config else main_profile(args.profile)
    )
    output = args.output or Path("results") / f"main_{config.profile}"
    rows = run_main_experiment(config, output)
    print(f"Completed {len(rows)} repetition-level rows in {output}.")


if __name__ == "__main__":
    main()

