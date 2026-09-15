"""Checkpointed main and D-selection experiment runners."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

import numpy as np

from .classifier import classification_accuracy
from .config import DSelectionConfig, MainExperimentConfig, save_config
from .estimator import DriftModel, fit_drift
from .io import read_rows, sample_summary, write_manifest, write_rows_atomic
from .metrics import drift_mean_squared_error
from .randomness import make_rng
from .simulation import observe_paths, simulate_ips


def _regime_name(theta0: float) -> str:
    return "A(i)" if np.isclose(theta0, 1.0) else "A(ii)"


def _observed_system(
    *,
    label: int,
    n_particles: int,
    theta0: float,
    n_observations: int,
    simulation_steps: int,
    total_time: float,
    pair_chunk_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    fine = simulate_ips(
        label,
        n_particles,
        theta0,
        rng,
        total_time=total_time,
        n_steps=simulation_steps,
        pair_chunk_size=pair_chunk_size,
    )
    return observe_paths(fine, n_observations)


def _fit_models(
    paths_by_class: list[np.ndarray],
    measures_by_class: list[np.ndarray],
    *,
    D: int,
    degree: int,
    A: float,
    total_time: float,
) -> list[DriftModel]:
    return [
        fit_drift(
            paths_by_class[index], measures_by_class[index], D=D,
            degree=degree, A=A, total_time=total_time,
        )
        for index in range(2)
    ]


def _main_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), int(row["M"]), int(row["N_train_per_class"]))].append(row)
    summaries: list[dict[str, Any]] = []
    metrics = ("plugin_accuracy", "drift_mse_class1", "drift_mse_class2")
    for key, group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "regime": key[0], "M": key[1], "N_train_per_class": key[2],
            "repetitions": len(group),
        }
        for metric in metrics:
            summary = sample_summary([float(row[metric]) for row in group])
            result.update({f"{metric}_{name}": value for name, value in summary.items()})
        summaries.append(result)
    return summaries


def run_main_experiment(config: MainExperimentConfig, output_dir: Path) -> list[dict[str, Any]]:
    """Run or resume the main test experiment with fully independent systems."""
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")
    checkpoint = output_dir / "repetition_results.csv"
    rows: list[dict[str, Any]] = list(read_rows(checkpoint))
    completed = {
        (str(row["regime"]), int(row["M"]), int(row["N_train_per_class"]), int(row["repetition"]))
        for row in rows
    }
    started = time.perf_counter()

    for theta_index, theta0 in enumerate(config.theta0_values):
        regime = _regime_name(theta0)
        for M in config.observation_steps:
            for n_train in config.train_particles_per_class:
                for repetition in range(config.repetitions):
                    key = (regime, M, n_train, repetition)
                    if key in completed:
                        continue
                    banks: dict[str, list[np.ndarray]] = {}
                    specifications = (
                        ("train_paths", n_train, 10),
                        ("train_measures", n_train, 20),
                        ("test_paths", config.test_particles_per_class, 30),
                        ("auxiliary_paths", config.auxiliary_particles_per_class, 40),
                        ("auxiliary_measures", config.auxiliary_particles_per_class, 50),
                    )
                    for name, size, bank_id in specifications:
                        banks[name] = [
                            _observed_system(
                                label=label, n_particles=size, theta0=theta0,
                                n_observations=M, simulation_steps=config.simulation_steps,
                                total_time=config.total_time,
                                pair_chunk_size=config.pair_chunk_size,
                                rng=make_rng(
                                    config.base_seed, theta_index, M, n_train,
                                    repetition, bank_id, label,
                                ),
                            )
                            for label in (1, 2)
                        ]
                    fit_started = time.perf_counter()
                    models = _fit_models(
                        banks["train_paths"], banks["train_measures"], D=config.D,
                        degree=config.degree, A=config.domain_half_width,
                        total_time=config.total_time,
                    )
                    fit_seconds = time.perf_counter() - fit_started
                    accuracy, confusion = classification_accuracy(
                        banks["test_paths"], models, total_time=config.total_time,
                        priors=config.priors, clip=config.clip_estimator,
                    )
                    drift_errors = [
                        drift_mean_squared_error(
                            banks["auxiliary_paths"][index],
                            banks["auxiliary_measures"][index], models[index],
                            label=index + 1, theta0=theta0,
                            clip=config.clip_estimator,
                            pair_chunk_size=config.pair_chunk_size,
                        )
                        for index in range(2)
                    ]
                    row: dict[str, Any] = {
                        "regime": regime, "theta0": theta0, "M": M,
                        "N_train_per_class": n_train,
                        "N_test_per_class": config.test_particles_per_class,
                        "N_auxiliary_per_class": config.auxiliary_particles_per_class,
                        "repetition": repetition, "D": config.D,
                        "plugin_accuracy": accuracy,
                        "class1_accuracy": confusion[0, 0] / confusion[0].sum(),
                        "class2_accuracy": confusion[1, 1] / confusion[1].sum(),
                        "drift_mse_class1": drift_errors[0],
                        "drift_mse_class2": drift_errors[1],
                        "fit_seconds": fit_seconds,
                    }
                    for class_index, model in enumerate(models, start=1):
                        diagnostic = model.diagnostics
                        row.update(
                            {
                                f"class{class_index}_lambda": diagnostic.lambda_value,
                                f"class{class_index}_constraint_active": diagnostic.constraint_active,
                                f"class{class_index}_rank": diagnostic.numerical_rank,
                                f"class{class_index}_dimension": diagnostic.dimension,
                                f"class{class_index}_condition_number": diagnostic.condition_number,
                                f"class{class_index}_kkt_residual": diagnostic.kkt_residual,
                                f"class{class_index}_train_outside_rate": model.train_outside_rate,
                            }
                        )
                    rows.append(row)
                    write_rows_atomic(checkpoint, rows)
                    print(
                        f"{regime} M={M} N={n_train} rep={repetition}: "
                        f"accuracy={accuracy:.4f}"
                    )

    elapsed = time.perf_counter() - started
    write_rows_atomic(output_dir / "summary.csv", _main_summary(rows))
    write_manifest(output_dir / "manifest.json", config=asdict(config), elapsed_seconds=elapsed)
    return rows


def _paired_validation_banks(
    config: DSelectionConfig,
    *,
    theta_index: int,
    theta0: float,
    M: int,
    n_total: int,
    repetition: int,
) -> dict[str, Any]:
    n_fit = int(config.train_fraction * n_total)
    n_validation = n_total - n_fit
    if min(n_fit, n_validation) < 2:
        raise ValueError("Both fitting and validation subsets require at least two particles.")
    output: dict[str, Any] = {
        "train_paths": [], "train_measures": [],
        "within_paths": [], "within_measures": [],
        "independent_paths": [], "independent_measures": [],
        "n_fit": n_fit, "n_validation": n_validation,
    }
    for label in (1, 2):
        systems = {}
        for name, bank_id in (
            ("paths", 10), ("measures", 20),
            ("independent_paths", 30), ("independent_measures", 40),
        ):
            systems[name] = _observed_system(
                label=label, n_particles=n_total, theta0=theta0,
                n_observations=M, simulation_steps=config.simulation_steps,
                total_time=config.total_time, pair_chunk_size=config.pair_chunk_size,
                rng=make_rng(
                    config.base_seed, theta_index, M, n_total, repetition, bank_id, label,
                ),
            )
        path_order = make_rng(
            config.base_seed, theta_index, M, n_total, repetition, 50, label
        ).permutation(n_total)
        measure_order = make_rng(
            config.base_seed, theta_index, M, n_total, repetition, 60, label
        ).permutation(n_total)
        independent_path_order = make_rng(
            config.base_seed, theta_index, M, n_total, repetition, 70, label
        ).permutation(n_total)[:n_validation]
        independent_measure_order = make_rng(
            config.base_seed, theta_index, M, n_total, repetition, 80, label
        ).permutation(n_total)[:n_validation]
        output["train_paths"].append(systems["paths"][path_order[:n_fit]])
        output["within_paths"].append(systems["paths"][path_order[n_fit:]])
        output["train_measures"].append(systems["measures"][measure_order[:n_fit]])
        output["within_measures"].append(systems["measures"][measure_order[n_fit:]])
        output["independent_paths"].append(
            systems["independent_paths"][independent_path_order]
        )
        output["independent_measures"].append(
            systems["independent_measures"][independent_measure_order]
        )
    return output


def _d_analysis(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    summary_groups: dict[tuple[str, int, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    repetition_groups: dict[tuple[str, int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        summary_key = (
            str(row["regime"]), int(row["M"]), int(row["N_total"]),
            str(row["strategy"]), int(row["D"]),
        )
        repetition_key = (
            str(row["regime"]), int(row["M"]), int(row["N_total"]),
            int(row["repetition"]), str(row["strategy"]),
        )
        summary_groups[summary_key].append(row)
        repetition_groups[repetition_key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(summary_groups.items()):
        result: dict[str, Any] = {
            "regime": key[0], "M": key[1], "N_total": key[2],
            "strategy": key[3], "D": key[4], "repetitions": len(group),
        }
        for metric in ("validation_accuracy", "drift_mse_mean", "fit_seconds"):
            metric_summary = sample_summary([float(row[metric]) for row in group])
            result.update({f"{metric}_{name}": value for name, value in metric_summary.items()})
        summaries.append(result)

    selections: list[dict[str, Any]] = []
    for key, group in sorted(repetition_groups.items()):
        accuracy_winner = min(
            group, key=lambda row: (-float(row["validation_accuracy"]), int(row["D"]))
        )
        mse_winner = min(group, key=lambda row: (float(row["drift_mse_mean"]), int(row["D"])))
        selections.append(
            {
                "regime": key[0], "M": key[1], "N_total": key[2],
                "repetition": key[3], "strategy": key[4],
                "classification_selected_D": int(accuracy_winner["D"]),
                "drift_selected_D": int(mse_winner["D"]),
            }
        )

    frequency_groups: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        frequency_groups[
            (str(row["regime"]), int(row["M"]), int(row["N_total"]), str(row["strategy"]))
        ].append(row)
    frequencies: list[dict[str, Any]] = []
    for key, group in sorted(frequency_groups.items()):
        for metric in ("classification", "drift"):
            counts = Counter(int(row[f"{metric}_selected_D"]) for row in group)
            for D in sorted({int(row["D"]) for row in rows}):
                frequencies.append(
                    {
                        "regime": key[0], "M": key[1], "N_total": key[2],
                        "strategy": key[3], "metric": metric, "D": D,
                        "count": counts[D], "total": len(group),
                        "frequency": counts[D] / len(group),
                    }
                )

    one_se: list[dict[str, Any]] = []
    config_groups: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        config_groups[
            (str(row["regime"]), int(row["M"]), int(row["N_total"]), str(row["strategy"]))
        ].append(row)
    for key, group in sorted(config_groups.items()):
        best_accuracy = max(group, key=lambda row: float(row["validation_accuracy_mean"]))
        accuracy_se = float(best_accuracy["validation_accuracy_standard_error"])
        if not np.isfinite(accuracy_se):
            accuracy_se = 0.0
        accuracy_cutoff = (
            float(best_accuracy["validation_accuracy_mean"])
            - accuracy_se
        )
        accuracy_choice = min(
            int(row["D"]) for row in group
            if float(row["validation_accuracy_mean"]) >= accuracy_cutoff
        )
        best_mse = min(group, key=lambda row: float(row["drift_mse_mean_mean"]))
        mse_se = float(best_mse["drift_mse_mean_standard_error"])
        if not np.isfinite(mse_se):
            mse_se = 0.0
        mse_cutoff = (
            float(best_mse["drift_mse_mean_mean"])
            + mse_se
        )
        mse_choice = min(
            int(row["D"]) for row in group
            if float(row["drift_mse_mean_mean"]) <= mse_cutoff
        )
        one_se.append(
            {
                "regime": key[0], "M": key[1], "N_total": key[2], "strategy": key[3],
                "classification_best_mean_D": int(best_accuracy["D"]),
                "classification_one_se_D": accuracy_choice,
                "drift_best_mean_D": int(best_mse["D"]),
                "drift_one_se_D": mse_choice,
            }
        )
    return {
        "summary": summaries, "selections": selections,
        "frequencies": frequencies, "one_se": one_se,
    }


def run_d_selection(config: DSelectionConfig, output_dir: Path) -> list[dict[str, Any]]:
    """Run paired within-system and independent-system validation for D."""
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")
    checkpoint = output_dir / "D_validation_raw.csv"
    rows: list[dict[str, Any]] = list(read_rows(checkpoint))
    expected_per_group = 2 * len(config.D_candidates)
    started = time.perf_counter()

    for theta_index, theta0 in enumerate(config.theta0_values):
        regime = _regime_name(theta0)
        for M in config.observation_steps:
            for n_total in config.total_particles_per_class:
                for repetition in range(config.repetitions):
                    matches = [
                        row for row in rows
                        if str(row["regime"]) == regime and int(row["M"]) == M
                        and int(row["N_total"]) == n_total
                        and int(row["repetition"]) == repetition
                    ]
                    if len(matches) == expected_per_group:
                        continue
                    rows = [row for row in rows if row not in matches]
                    banks = _paired_validation_banks(
                        config, theta_index=theta_index, theta0=theta0, M=M,
                        n_total=n_total, repetition=repetition,
                    )
                    for D in config.D_candidates:
                        fit_started = time.perf_counter()
                        models = _fit_models(
                            banks["train_paths"], banks["train_measures"], D=D,
                            degree=config.degree, A=config.domain_half_width,
                            total_time=config.total_time,
                        )
                        fit_seconds = time.perf_counter() - fit_started
                        for strategy in ("within", "independent"):
                            validation_paths = banks[f"{strategy}_paths"]
                            validation_measures = banks[f"{strategy}_measures"]
                            accuracy, _ = classification_accuracy(
                                validation_paths, models, total_time=config.total_time,
                                priors=config.priors, clip=config.clip_estimator,
                            )
                            errors = [
                                drift_mean_squared_error(
                                    validation_paths[index], validation_measures[index],
                                    models[index], label=index + 1, theta0=theta0,
                                    clip=config.clip_estimator,
                                    pair_chunk_size=config.pair_chunk_size,
                                )
                                for index in range(2)
                            ]
                            rows.append(
                                {
                                    "regime": regime, "theta0": theta0, "M": M,
                                    "N_total": n_total, "N_fit": banks["n_fit"],
                                    "N_validation": banks["n_validation"],
                                    "repetition": repetition, "strategy": strategy,
                                    "D": D, "validation_accuracy": accuracy,
                                    "drift_mse_mean": 0.5 * (errors[0] + errors[1]),
                                    "drift_mse_class1": errors[0],
                                    "drift_mse_class2": errors[1],
                                    "fit_seconds": fit_seconds,
                                    "maximum_kkt_residual": max(
                                        model.diagnostics.kkt_residual for model in models
                                    ),
                                    "maximum_train_outside_rate": max(
                                        model.train_outside_rate for model in models
                                    ),
                                }
                            )
                    write_rows_atomic(checkpoint, rows)
                    print(f"{regime} M={M} N={n_total} rep={repetition}: complete")

    analysis = _d_analysis(rows)
    write_rows_atomic(output_dir / "D_validation_summary.csv", analysis["summary"])
    write_rows_atomic(output_dir / "D_selection_by_repetition.csv", analysis["selections"])
    write_rows_atomic(output_dir / "D_selection_frequencies.csv", analysis["frequencies"])
    write_rows_atomic(output_dir / "D_one_se_recommendations.csv", analysis["one_se"])
    elapsed = time.perf_counter() - started
    write_manifest(output_dir / "manifest.json", config=asdict(config), elapsed_seconds=elapsed)
    return rows
