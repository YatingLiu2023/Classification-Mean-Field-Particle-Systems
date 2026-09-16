#!/usr/bin/env python3
"""Reproducible finite-particle oracle proxies for Scenarios A and B.

The experiment uses the true class drift functions b_k^*, approximates each
class-conditional law mu_t^[k] with an independent N_ref-particle reference
system, and evaluates classification on an independent N_test-particle test
system.

Examples (run from repository root):

    PYTHONPATH=src python scripts/run_oracle_proxy.py --scenario all \
        --output results/oracle_proxy_paper

    PYTHONPATH=src python scripts/run_oracle_proxy.py --scenario B \
        --output results/oracle_proxy_paper

By default the paper configuration is used:
N_ref=5000, N_test=1000, M=100, 50 repetitions, base seed 20260904.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads

from mkv_classification.randomness import make_rng
from mkv_classification.simulation import mean_interaction_drift
from mkv_classification.scenario_b import simulate_ips as simulate_ips_b
from mkv_classification.scenario_b import mean_drift as mean_drift_b


BASE_SEED = 20260904
N_REF = 5000
N_TEST = 1000
M = 100
TOTAL_TIME = 1.0
THETA0_VALUES = (1.0, 0.5)
THETA_B_VALUES = np.asarray((-6.0, -3.0, 0.0, 3.0, 6.0), dtype=float)
NUMBA_THREADS = 5


# -----------------------------------------------------------------------------
# Generic I/O
# -----------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {
        "mean_accuracy": float(values.mean()),
        "std_accuracy": std,
        "standard_error": std / np.sqrt(values.size),
        "min_accuracy": float(values.min()),
        "max_accuracy": float(values.max()),
    }


# =============================================================================
# Scenario A
# =============================================================================

# The class-2 interaction is O(N^2).  The Numba implementation below computes
# exactly the same empirical average as the repository function, but exploits
# antisymmetry for the self-interaction case and parallelizes the pair sum.

def _pair_bounds(n: int, n_threads: int) -> np.ndarray:
    total_pairs = n * (n - 1) // 2
    bounds = [0]
    for thread in range(1, n_threads):
        target = total_pairs * thread / n_threads
        k = int(((2 * n - 1) - math.sqrt((2 * n - 1) ** 2 - 8 * target)) / 2)
        bounds.append(k)
    bounds.append(n - 1)
    return np.asarray(bounds, dtype=np.int64)


PAIR_BOUNDS_5000 = _pair_bounds(N_REF, NUMBA_THREADS)
PAIR_BOUNDS_1000 = _pair_bounds(N_TEST, NUMBA_THREADS)


@njit(parallel=True)
def _drift2_self_bounded(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    n = x.size
    n_threads = bounds.size - 1
    buffer = np.zeros((n_threads, n))
    for thread in prange(n_threads):
        for i in range(bounds[thread], bounds[thread + 1]):
            xi = x[i]
            row_sum = 0.0
            for j in range(i + 1, n):
                delta = xi - x[j]
                delta2 = delta * delta
                value = (
                    2.0 * delta * np.exp(-delta2)
                    - 8.0 * delta * np.exp(-2.0 * delta2)
                )
                row_sum += value
                buffer[thread, j] -= value
            buffer[thread, i] += row_sum
    output = np.zeros(n)
    for thread in range(n_threads):
        for i in range(n):
            output[i] += buffer[thread, i]
    return output / n


def _drift2_self(x: np.ndarray) -> np.ndarray:
    if x.size == N_REF:
        bounds = PAIR_BOUNDS_5000
    elif x.size == N_TEST:
        bounds = PAIR_BOUNDS_1000
    else:
        bounds = _pair_bounds(int(x.size), NUMBA_THREADS)
    return _drift2_self_bounded(x, bounds)


@njit(parallel=True)
def _drift2_cross(x: np.ndarray, measure: np.ndarray) -> np.ndarray:
    n_x = x.size
    n_measure = measure.size
    output = np.empty(n_x)
    for i in prange(n_x):
        xi = x[i]
        total = 0.0
        for j in range(n_measure):
            delta = xi - measure[j]
            delta2 = delta * delta
            total += (
                2.0 * delta * np.exp(-delta2)
                - 8.0 * delta * np.exp(-2.0 * delta2)
            )
        output[i] = total / n_measure
    return output


def _simulate_a_class1(n_particles: int, theta0: float, rng: np.random.Generator) -> np.ndarray:
    dt = TOTAL_TIME / M
    paths = np.empty((n_particles, M + 1), dtype=float)
    paths[:, 0] = 0.0
    noise_scale = np.sqrt(dt)
    for m in range(M):
        x = paths[:, m]
        drift = theta0 * (0.25 + 0.75 * np.cos(x) ** 2 + np.mean(np.cos(x) ** 2))
        paths[:, m + 1] = x + dt * drift + noise_scale * rng.standard_normal(n_particles)
    return paths


def _simulate_a_class2(n_particles: int, rng: np.random.Generator) -> np.ndarray:
    dt = TOTAL_TIME / M
    paths = np.empty((n_particles, M + 1), dtype=float)
    paths[:, 0] = 0.0
    noise_scale = np.sqrt(dt)
    for m in range(M):
        x = paths[:, m]
        drift = _drift2_self(x)
        paths[:, m + 1] = x + dt * drift + noise_scale * rng.standard_normal(n_particles)
    return paths


def _score_a_class1(paths: np.ndarray, reference_paths: np.ndarray, theta0: float) -> np.ndarray:
    dt = TOTAL_TIME / M
    increments = np.diff(paths, axis=1)
    score = np.zeros(paths.shape[0], dtype=float)
    for m in range(M):
        drift = theta0 * (
            0.25
            + 0.75 * np.cos(paths[:, m]) ** 2
            + np.mean(np.cos(reference_paths[:, m]) ** 2)
        )
        score += drift * increments[:, m] - 0.5 * dt * drift**2
    return score


def _score_a_class2(paths: np.ndarray, reference_paths: np.ndarray) -> np.ndarray:
    dt = TOTAL_TIME / M
    increments = np.diff(paths, axis=1)
    score = np.zeros(paths.shape[0], dtype=float)
    for m in range(M):
        drift = _drift2_cross(paths[:, m], reference_paths[:, m])
        score += drift * increments[:, m] - 0.5 * dt * drift**2
    return score


def _validate_a_acceleration() -> float:
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    reference = mean_interaction_drift(x, x, 2, 1.0)
    accelerated = _drift2_self(x)
    error = float(np.max(np.abs(reference - accelerated)))
    if error >= 1e-12:
        raise RuntimeError(f"Scenario-A class-2 acceleration failed validation: {error:.3e}")
    return error


def run_scenario_a(repetitions: int, output_dir: Path) -> list[dict[str, object]]:
    set_num_threads(NUMBA_THREADS)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_error = _validate_a_acceleration()
    print(f"Scenario A class-2 acceleration validation max error: {validation_error:.3e}")

    config = {
        "scenario": "A",
        "base_seed": BASE_SEED,
        "N_ref_per_class": N_REF,
        "N_test_per_class": N_TEST,
        "M": M,
        "total_time": TOTAL_TIME,
        "repetitions": repetitions,
        "theta0_values": list(THETA0_VALUES),
        "independent_reference_and_test": True,
        "equal_priors": True,
        "numba_threads": NUMBA_THREADS,
        "stream_scheme": (
            "make_rng(base_seed, 90, regime_id, repetition, bank_id, label); "
            "regime_id=99 for class 2 shared across regimes; "
            "bank_id=10 reference, bank_id=20 test"
        ),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    for repetition in range(repetitions):
        reference_class2 = _simulate_a_class2(
            N_REF, make_rng(BASE_SEED, 90, 99, repetition, 10, 2)
        )
        test_class2 = _simulate_a_class2(
            N_TEST, make_rng(BASE_SEED, 90, 99, repetition, 20, 2)
        )
        class2_score_on_class2 = _score_a_class2(test_class2, reference_class2)

        for regime_id, theta0 in enumerate(THETA0_VALUES):
            regime = "A(i)" if regime_id == 0 else "A(ii)"
            reference_class1 = _simulate_a_class1(
                N_REF, theta0, make_rng(BASE_SEED, 90, regime_id, repetition, 10, 1)
            )
            test_class1 = _simulate_a_class1(
                N_TEST, theta0, make_rng(BASE_SEED, 90, regime_id, repetition, 20, 1)
            )
            s11 = _score_a_class1(test_class1, reference_class1, theta0)
            s21 = _score_a_class2(test_class1, reference_class2)
            s12 = _score_a_class1(test_class2, reference_class1, theta0)

            c11 = int(np.sum(s11 >= s21))
            c12 = N_TEST - c11
            c22 = int(np.sum(class2_score_on_class2 > s12))
            c21 = N_TEST - c22
            accuracy = (c11 + c22) / (2 * N_TEST)
            rows.append({
                "scenario": "A",
                "regime": regime,
                "theta0": theta0,
                "repetition": repetition,
                "accuracy": accuracy,
                "class1_accuracy": c11 / N_TEST,
                "class2_accuracy": c22 / N_TEST,
                "c11": c11,
                "c12": c12,
                "c21": c21,
                "c22": c22,
            })
        _write_csv(output_dir / "repetition_results.csv", rows)
        print(f"Scenario A rep {repetition:02d}: A(i)={rows[-2]['accuracy']:.4f}, A(ii)={rows[-1]['accuracy']:.4f}")

    summary_rows: list[dict[str, object]] = []
    for regime in ("A(i)", "A(ii)"):
        group = [row for row in rows if row["regime"] == regime]
        accuracy = np.asarray([row["accuracy"] for row in group], dtype=float)
        class1_accuracy = np.asarray([row["class1_accuracy"] for row in group], dtype=float)
        class2_accuracy = np.asarray([row["class2_accuracy"] for row in group], dtype=float)
        summary_rows.append({
            "scenario": "A",
            "regime": regime,
            "repetitions": int(accuracy.size),
            **_summary_stats(accuracy),
            "mean_class1_accuracy": float(class1_accuracy.mean()),
            "mean_class2_accuracy": float(class2_accuracy.mean()),
        })
    _write_csv(output_dir / "summary.csv", summary_rows)
    return summary_rows


# =============================================================================
# Scenario B
# =============================================================================

def _scores_b_all_classes(paths: np.ndarray, reference_paths: list[np.ndarray]) -> np.ndarray:
    """Oracle scores for all five Scenario-B classes, vectorized exactly."""
    dt = TOTAL_TIME / M
    x = paths[:, :-1]
    increments = np.diff(paths, axis=1)
    x_term = 0.25 + 0.75 * np.cos(x) ** 2
    reference_means = np.stack(
        [np.mean(np.cos(ref[:, :-1]) ** 2, axis=0) for ref in reference_paths],
        axis=0,
    )
    base = x_term[:, None, :] + reference_means[None, :, :]
    drift = base * THETA_B_VALUES[None, :, None]
    # Equal priors: the common log(1/5) term cancels in argmax.
    return np.sum(
        drift * increments[:, None, :] - 0.5 * dt * drift**2,
        axis=2,
    )


def _validate_b_scores() -> float:
    refs = [
        simulate_ips_b(float(theta), 50, make_rng(BASE_SEED, 191, 10, k), total_time=TOTAL_TIME, n_steps=M)
        for k, theta in enumerate(THETA_B_VALUES)
    ]
    test = simulate_ips_b(
        float(THETA_B_VALUES[2]), 30, make_rng(BASE_SEED, 191, 20, 2),
        total_time=TOTAL_TIME, n_steps=M,
    )
    fast = _scores_b_all_classes(test, refs)
    dt = TOTAL_TIME / M
    increments = np.diff(test, axis=1)
    slow_columns = []
    for k, theta in enumerate(THETA_B_VALUES):
        score = np.zeros(test.shape[0], dtype=float)
        for m in range(M):
            drift = mean_drift_b(float(theta), test[:, m], refs[k][:, m])
            score += drift * increments[:, m] - 0.5 * dt * drift**2
        slow_columns.append(score)
    slow = np.column_stack(slow_columns)
    error = float(np.max(np.abs(fast - slow)))
    if error >= 1e-12:
        raise RuntimeError(f"Scenario-B vectorized score failed validation: {error:.3e}")
    return error


def run_scenario_b(repetitions: int, output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_error = _validate_b_scores()
    print(f"Scenario B score validation max error: {validation_error:.3e}")

    config = {
        "scenario": "B",
        "base_seed": BASE_SEED,
        "N_ref_per_class": N_REF,
        "N_test_per_class": N_TEST,
        "M": M,
        "total_time": TOTAL_TIME,
        "repetitions": repetitions,
        "theta_values": THETA_B_VALUES.tolist(),
        "independent_reference_and_test": True,
        "equal_priors": True,
        "stream_scheme": (
            "make_rng(base_seed, 91, repetition, bank_id, class_index); "
            "bank_id=10 reference, bank_id=20 test"
        ),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    n_classes = len(THETA_B_VALUES)
    for repetition in range(repetitions):
        references = [
            simulate_ips_b(
                float(theta), N_REF, make_rng(BASE_SEED, 91, repetition, 10, k),
                total_time=TOTAL_TIME, n_steps=M,
            )
            for k, theta in enumerate(THETA_B_VALUES)
        ]
        tests = [
            simulate_ips_b(
                float(theta), N_TEST, make_rng(BASE_SEED, 91, repetition, 20, k),
                total_time=TOTAL_TIME, n_steps=M,
            )
            for k, theta in enumerate(THETA_B_VALUES)
        ]

        confusion = np.zeros((n_classes, n_classes), dtype=int)
        for true_k, paths in enumerate(tests):
            predictions = np.argmax(_scores_b_all_classes(paths, references), axis=1)
            np.add.at(confusion, (np.full(N_TEST, true_k, dtype=int), predictions), 1)

        accuracy = float(np.trace(confusion) / (n_classes * N_TEST))
        row: dict[str, object] = {
            "scenario": "B",
            "repetition": repetition,
            "accuracy": accuracy,
        }
        for k in range(n_classes):
            row[f"class{k + 1}_accuracy"] = float(confusion[k, k] / N_TEST)
        for i in range(n_classes):
            for j in range(n_classes):
                row[f"c{i + 1}{j + 1}"] = int(confusion[i, j])
        rows.append(row)
        _write_csv(output_dir / "repetition_results.csv", rows)
        print(f"Scenario B rep {repetition:02d}: accuracy={accuracy:.4f}")

    accuracy = np.asarray([row["accuracy"] for row in rows], dtype=float)
    summary_row: dict[str, object] = {
        "scenario": "B",
        "regime": "B",
        "repetitions": int(accuracy.size),
        **_summary_stats(accuracy),
    }
    for k in range(n_classes):
        summary_row[f"mean_class{k + 1}_accuracy"] = float(
            np.mean([float(row[f"class{k + 1}_accuracy"]) for row in rows])
        )
    _write_csv(output_dir / "summary.csv", [summary_row])
    return [summary_row]


# =============================================================================
# Driver
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("A", "B", "all"), default="all")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/oracle_proxy_paper"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive.")
    started = time.perf_counter()
    all_summary: list[dict[str, object]] = []
    if args.scenario in ("A", "all"):
        all_summary.extend(run_scenario_a(args.repetitions, args.output / "scenario_a"))
    if args.scenario in ("B", "all"):
        all_summary.extend(run_scenario_b(args.repetitions, args.output / "scenario_b"))
    if all_summary:
        _write_csv(args.output / "summary.csv", all_summary)
    print(json.dumps(all_summary, indent=2))
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
