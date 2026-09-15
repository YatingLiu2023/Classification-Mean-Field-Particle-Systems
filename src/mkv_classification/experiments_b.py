"""Checkpointed five-class Scenario B experiment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

import numpy as np

from .classifier import classification_accuracy
from .config import ScenarioBConfig, save_config
from .estimator import fit_drift, mean_estimated_drift
from .io import read_rows, sample_summary, write_manifest, write_rows_atomic
from .randomness import make_rng
from .scenario_b import simulate_ips, true_mean_drift_paths
from .simulation import observe_paths


def _observed_system(
    config: ScenarioBConfig,
    *,
    theta: float,
    n_particles: int,
    M: int,
    stream_ids: tuple[int, ...],
) -> np.ndarray:
    fine = simulate_ips(
        theta, n_particles, make_rng(config.base_seed, *stream_ids),
        total_time=config.total_time, n_steps=config.simulation_steps,
    )
    return observe_paths(fine, M)


def _drift_mse(
    paths: np.ndarray,
    measures: np.ndarray,
    model,
    theta: float,
    clip: bool,
) -> float:
    estimated, _ = mean_estimated_drift(
        paths, model, measure_paths=measures, clip=clip
    )
    target = true_mean_drift_paths(paths, measures, theta)
    return float(np.mean((estimated - target) ** 2))


def _summary(rows: list[dict[str, Any]], n_classes: int) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["M"]), int(row["N_train_per_class"]))].append(row)
    output: list[dict[str, Any]] = []
    metrics = ["plugin_accuracy", *[f"drift_mse_class{index}" for index in range(1, n_classes + 1)]]
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = {
            "scenario": "B", "M": key[0], "N_train_per_class": key[1],
            "repetitions": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            stats = sample_summary(values)
            item.update({f"{metric}_{name}": value for name, value in stats.items()})
        item["drift_mse_mean_across_classes"] = float(
            np.mean([float(item[f"drift_mse_class{k}_mean"]) for k in range(1, n_classes + 1)])
        )
        output.append(item)
    return output


def run_scenario_b(config: ScenarioBConfig, output_dir: Path) -> list[dict[str, Any]]:
    """Run or resume Scenario B plug-in classification and drift evaluation."""
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")
    checkpoint = output_dir / "repetition_results.csv"
    rows: list[dict[str, Any]] = list(read_rows(checkpoint))
    done = {
        (int(row["M"]), int(row["N_train_per_class"]), int(row["repetition"]))
        for row in rows
    }
    started = time.perf_counter()
    n_classes = len(config.theta_values)
    priors = tuple([1.0 / n_classes] * n_classes)

    for M in config.observation_steps:
        for n_train in config.train_particles_per_class:
            for repetition in range(config.repetitions):
                key = (M, n_train, repetition)
                if key in done:
                    continue
                banks: dict[str, list[np.ndarray]] = {}
                for name, size, bank_id in (
                    ("train_paths", n_train, 10),
                    ("train_measures", n_train, 20),
                    ("test_paths", config.test_particles_per_class, 30),
                    ("auxiliary_paths", config.auxiliary_particles_per_class, 40),
                    ("auxiliary_measures", config.auxiliary_particles_per_class, 50),
                ):
                    banks[name] = [
                        _observed_system(
                            config, theta=theta, n_particles=size, M=M,
                            stream_ids=(M, n_train, repetition, bank_id, class_index),
                        )
                        for class_index, theta in enumerate(config.theta_values)
                    ]
                fit_started = time.perf_counter()
                models = [
                    fit_drift(
                        banks["train_paths"][index], banks["train_measures"][index],
                        D=config.D, degree=config.degree, A=config.domain_half_width,
                        total_time=config.total_time,
                    )
                    for index in range(n_classes)
                ]
                fit_seconds = time.perf_counter() - fit_started
                accuracy, confusion = classification_accuracy(
                    banks["test_paths"], models, total_time=config.total_time,
                    priors=priors, clip=config.clip_estimator,
                )
                errors = [
                    _drift_mse(
                        banks["auxiliary_paths"][index], banks["auxiliary_measures"][index],
                        models[index], config.theta_values[index], config.clip_estimator,
                    )
                    for index in range(n_classes)
                ]
                row: dict[str, Any] = {
                    "scenario": "B", "M": M, "N_train_per_class": n_train,
                    "N_test_per_class": config.test_particles_per_class,
                    "N_auxiliary_per_class": config.auxiliary_particles_per_class,
                    "repetition": repetition, "D": config.D,
                    "plugin_accuracy": accuracy, "fit_seconds": fit_seconds,
                    "maximum_kkt_residual": max(
                        model.diagnostics.kkt_residual for model in models
                    ),
                    "maximum_train_outside_rate": max(model.train_outside_rate for model in models),
                }
                for index in range(n_classes):
                    row[f"class{index + 1}_accuracy"] = (
                        confusion[index, index] / confusion[index].sum()
                    )
                    row[f"drift_mse_class{index + 1}"] = errors[index]
                rows.append(row)
                write_rows_atomic(checkpoint, rows)
                print(
                    f"Scenario B M={M} N={n_train} rep={repetition}: "
                    f"accuracy={accuracy:.4f}"
                )

    elapsed = time.perf_counter() - started
    write_rows_atomic(output_dir / "summary.csv", _summary(rows, n_classes))
    write_manifest(output_dir / "manifest.json", config=asdict(config), elapsed_seconds=elapsed)
    return rows

