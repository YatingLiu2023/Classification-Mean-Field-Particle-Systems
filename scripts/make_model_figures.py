#!/usr/bin/env python3
"""Generate paper model figures in vector and high-resolution raster formats."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mkv_classification.randomness import make_rng
from mkv_classification.scenario_b import simulate_ips as simulate_b
from mkv_classification.simulation import pair_drift


def save_both(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/model_figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(grid, grid)
    surfaces = [
        ("A(i): class 1", pair_drift(1, X, Y, 1.0)),
        ("A(ii): class 1", pair_drift(1, X, Y, 0.5)),
        ("Scenario A: class 2", pair_drift(2, X, Y, 1.0)),
    ]
    figure = plt.figure(figsize=(11.2, 3.5))
    for index, (title, values) in enumerate(surfaces, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.plot_surface(X, Y, values, cmap="viridis", linewidth=0)
        axis.set_title(title); axis.set_xlabel("x"); axis.set_ylabel("y")
    figure.tight_layout()
    save_both(figure, args.output / "scenario_a_true_drifts")
    plt.close(figure)

    theta_values = (-6.0, -3.0, 0.0, 3.0, 6.0)
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    times = np.linspace(0.0, 1.0, 101)
    for class_index, theta in enumerate(theta_values):
        paths = simulate_b(theta, 50, make_rng(20260908, class_index), n_steps=100)
        axis.plot(times, paths[:10].T, alpha=0.35)
        axis.plot([], [], label=rf"$\theta={theta:g}$")
    axis.set_xlabel("Time"); axis.set_ylabel("Particle position")
    axis.legend(ncol=5, frameon=False, fontsize=8); axis.grid(alpha=0.15)
    figure.tight_layout()
    save_both(figure, args.output / "scenario_b_representative_paths")
    plt.close(figure)


if __name__ == "__main__":
    main()

