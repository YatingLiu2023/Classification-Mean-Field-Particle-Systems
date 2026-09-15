#!/usr/bin/env python3
"""Run the paired hyperparameter-D validation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from mkv_classification.config import DSelectionConfig, d_selection_profile, load_config
from mkv_classification.experiments import run_d_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--config", type=Path, help="Optional custom JSON configuration.")
    parser.add_argument("--output", type=Path, help="Output directory; existing checkpoints resume.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = (
        load_config(args.config, DSelectionConfig)
        if args.config else d_selection_profile(args.profile)
    )
    output = args.output or Path("results") / f"D_selection_{config.profile}"
    rows = run_d_selection(config, output)
    print(f"Completed {len(rows)} D-validation rows in {output}.")


if __name__ == "__main__":
    main()

