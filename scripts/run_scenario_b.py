#!/usr/bin/env python3
"""Run the five-class Scenario B plug-in experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from mkv_classification.config import ScenarioBConfig, load_config, scenario_b_profile
from mkv_classification.experiments_b import run_scenario_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, ScenarioBConfig) if args.config else scenario_b_profile(args.profile)
    output = args.output or Path("results") / f"scenario_b_{config.profile}"
    rows = run_scenario_b(config, output)
    print(f"Completed {len(rows)} Scenario-B rows in {output}.")


if __name__ == "__main__":
    main()

