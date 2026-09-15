"""Leakage-free direct neural-network baseline for Scenarios A and B."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import NeuralBaselineConfig, save_config
from .io import read_rows, sample_summary, write_manifest, write_rows_atomic
from .neural_network import tune_and_evaluate
from .randomness import make_rng
from .scenario_b import simulate_ips as simulate_b
from .simulation import observe_paths, simulate_ips as simulate_a


def _simulate(
    config: NeuralBaselineConfig,
    *,
    class_index: int,
    regime_parameter: float | None,
    n_particles: int,
    M: int,
    stream_ids: tuple[int, ...],
) -> np.ndarray:
    rng = make_rng(config.base_seed, *stream_ids)
    if config.scenario == "A":
        assert regime_parameter is not None
        fine = simulate_a(
            class_index + 1, n_particles, regime_parameter, rng,
            total_time=config.total_time, n_steps=config.simulation_steps,
        )
    else:
        fine = simulate_b(
            config.theta_values[class_index], n_particles, rng,
            total_time=config.total_time, n_steps=config.simulation_steps,
        )
    return observe_paths(fine, M)


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["regime"]), int(row["M"]), int(row["N_train_per_class"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        stats = sample_summary([float(row["direct_nn_accuracy"]) for row in group])
        accuracies = np.asarray(
            [float(row["direct_nn_accuracy"]) for row in group], dtype=float
        )
        collapse_flags = np.asarray(
            [int(row["class_collapse"]) for row in group], dtype=int
        )
        chance_flags = np.asarray(
            [int(row["chance_or_worse"]) for row in group], dtype=int
        )
        n_classes = int(group[0]["n_classes"])
        output.append(
            {
                "scenario": group[0]["scenario"], "regime": key[0],
                "M": key[1], "N_train_per_class": key[2],
                "N_validation_per_class": int(group[0]["N_validation_per_class"]),
                "repetitions": len(group),
                **{f"direct_nn_accuracy_{name}": value for name, value in stats.items()},
                "direct_nn_accuracy_median": float(np.median(accuracies)),
                "direct_nn_accuracy_q25": float(np.quantile(accuracies, 0.25)),
                "direct_nn_accuracy_q75": float(np.quantile(accuracies, 0.75)),
                "direct_nn_accuracy_minimum": float(accuracies.min()),
                "direct_nn_accuracy_maximum": float(accuracies.max()),
                "class_collapse_count": int(collapse_flags.sum()),
                "class_collapse_rate": float(collapse_flags.mean()),
                "chance_or_worse_count": int(chance_flags.sum()),
                "chance_or_worse_rate": float(chance_flags.mean()),
                **{
                    f"class_{class_index}_recall_mean": float(np.mean([
                        float(row[f"class_{class_index}_recall"]) for row in group
                    ]))
                    for class_index in range(n_classes)
                },
            }
        )
    return output


def _test_diagnostics(
    labels: np.ndarray,
    predictions: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """Pre-specified diagnostics; they never affect model selection or reruns."""
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    denominators = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion), denominators,
        out=np.zeros(n_classes, dtype=float), where=denominators > 0,
    )
    predicted_classes = np.unique(predictions)
    collapse = len(predicted_classes) < n_classes
    return {
        "n_classes": n_classes,
        "predicted_class_count": len(predicted_classes),
        "missing_predicted_classes": ";".join(
            str(index) for index in range(n_classes) if index not in predicted_classes
        ),
        "class_collapse": int(collapse),
        "chance_or_worse": int(float(np.mean(predictions == labels)) <= 1.0 / n_classes),
        "minimum_class_recall": float(recalls.min()),
        "zero_recall_class_count": int(np.sum(recalls == 0.0)),
        "confusion_matrix": json.dumps(confusion.tolist(), separators=(",", ":")),
        **{
            f"class_{class_index}_recall": float(recalls[class_index])
            for class_index in range(n_classes)
        },
    }


def run_neural_baseline(
    config: NeuralBaselineConfig,
    output_dir: Path,
) -> list[dict[str, Any]]:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")
    checkpoint = output_dir / "repetition_results.csv"
    rows: list[dict[str, Any]] = list(read_rows(checkpoint))
    done = {
        (str(row["regime"]), int(row["M"]), int(row["N_train_per_class"]), int(row["repetition"]))
        for row in rows
    }
    regimes = (
        [("A(i)", 1.0), ("A(ii)", 0.5)]
        if config.scenario == "A" else [("B", None)]
    )
    n_classes = 2 if config.scenario == "A" else len(config.theta_values)
    started = time.perf_counter()

    for regime_index, (regime, regime_parameter) in enumerate(regimes):
        for M in config.observation_steps:
            for n_train in config.train_particles_per_class:
                for repetition in range(config.repetitions):
                    key = (regime, M, n_train, repetition)
                    if key in done:
                        continue
                    banks: dict[str, list[np.ndarray]] = {}
                    for name, size, bank_id in (
                        ("train_paths", n_train, 10),
                        ("validation_measures", n_train, 20),
                        ("test_paths", config.test_particles_per_class, 30),
                    ):
                        banks[name] = [
                            _simulate(
                                config, class_index=class_index,
                                regime_parameter=regime_parameter,
                                n_particles=size, M=M,
                                stream_ids=(
                                    1 if config.scenario == "A" else 2,
                                    regime_index, M, n_train, repetition, bank_id, class_index,
                                ),
                            )
                            for class_index in range(n_classes)
                        ]

                    def rng_factory(stream: int) -> np.random.Generator:
                        return make_rng(
                            config.base_seed, 99, regime_index, M, n_train,
                            repetition, stream,
                        )

                    evaluation = tune_and_evaluate(
                        banks["train_paths"], banks["validation_measures"], banks["test_paths"],
                        rng_factory=rng_factory, trials=config.trials,
                        tune_epochs=config.tune_epochs, max_epochs=config.max_epochs,
                        batch_size=config.batch_size, patience=config.patience,
                        sparsity_budget=config.sparsity_budget,
                    )
                    rows.append(
                        {
                            "scenario": config.scenario, "regime": regime,
                            "M": M, "N_train_per_class": n_train,
                            "N_validation_per_class": n_train,
                            "N_test_per_class": config.test_particles_per_class,
                            "repetition": repetition,
                            "direct_nn_accuracy": evaluation.accuracy,
                            "validation_accuracy": evaluation.validation_accuracy,
                            "validation_cross_entropy": evaluation.validation_loss,
                            "hidden_layers": evaluation.parameters.hidden_layers,
                            "hidden_width": evaluation.parameters.hidden_width,
                            "learning_rate": evaluation.parameters.learning_rate,
                            "selected_trial": evaluation.selected_trial,
                            "finalists_evaluated": 1,
                            "best_epoch": evaluation.best_epoch,
                            **_test_diagnostics(
                                evaluation.test_labels,
                                evaluation.test_predictions,
                                n_classes,
                            ),
                        }
                    )
                    write_rows_atomic(checkpoint, rows)
                    print(
                        f"Scenario {config.scenario} {regime} M={M} N={n_train} "
                        f"rep={repetition}: NN accuracy={evaluation.accuracy:.4f} "
                        f"collapse={rows[-1]['class_collapse']}"
                    )
    elapsed = time.perf_counter() - started
    write_rows_atomic(output_dir / "summary.csv", _summary(rows))
    write_manifest(output_dir / "manifest.json", config=asdict(config), elapsed_seconds=elapsed)
    return rows
