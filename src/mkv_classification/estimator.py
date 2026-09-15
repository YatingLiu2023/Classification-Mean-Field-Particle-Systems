"""Streaming spline drift estimator and paper-consistent thresholding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .splines import BSplineBasis


@dataclass(frozen=True)
class SolverDiagnostics:
    lambda_value: float
    constraint_active: bool
    coefficient_norm_squared: float
    radius_squared: float
    numerical_rank: int
    dimension: int
    condition_number: float
    kkt_residual: float


@dataclass(frozen=True)
class DriftModel:
    basis: BSplineBasis
    coefficients: np.ndarray
    measure_basis: np.ndarray
    n_train: int
    diagnostics: SolverDiagnostics
    train_outside_rate: float

    @property
    def threshold(self) -> float:
        return float(np.sqrt(np.log(self.n_train)))


def solve_ball_constrained_quadratic(
    gram: np.ndarray,
    rhs: np.ndarray,
    radius_squared: float,
) -> tuple[np.ndarray, SolverDiagnostics]:
    """Solve min a'Ga - 2 rhs'a subject to ||a||^2 <= radius_squared.

    The unconstrained minimum-norm solution is checked first.  A nonnegative
    Lagrange multiplier is found by monotone bisection only when necessary.
    """
    gram = np.asarray(gram, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if gram.shape != (rhs.size, rhs.size) or radius_squared <= 0:
        raise ValueError("Incompatible Gram/rhs shapes or nonpositive radius.")
    symmetric = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = np.finfo(float).eps * rhs.size * scale * 10.0
    if float(np.min(eigenvalues)) < -tolerance:
        raise np.linalg.LinAlgError("Gram matrix is not positive semidefinite.")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rotated_rhs = eigenvectors.T @ rhs

    def solution(lagrange: float) -> np.ndarray:
        denominator = eigenvalues + lagrange
        rotated_solution = np.zeros_like(rotated_rhs)
        identifiable = denominator > tolerance
        rotated_solution[identifiable] = rotated_rhs[identifiable] / denominator[identifiable]
        return eigenvectors @ rotated_solution

    coefficients = solution(0.0)
    lagrange = 0.0
    constraint_active = float(coefficients @ coefficients) > radius_squared * (1.0 + 1e-10)
    if constraint_active:
        lower, upper = 0.0, scale
        while float(solution(upper) @ solution(upper)) > radius_squared:
            upper *= 2.0
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            if float(solution(midpoint) @ solution(midpoint)) > radius_squared:
                lower = midpoint
            else:
                upper = midpoint
        lagrange = 0.5 * (lower + upper)
        coefficients = solution(lagrange)

    positive = eigenvalues[eigenvalues > tolerance]
    condition_number = float(positive.max() / positive.min()) if positive.size else float("inf")
    stationarity = symmetric @ coefficients - rhs + lagrange * coefficients
    kkt_residual = float(np.linalg.norm(stationarity) / (1.0 + np.linalg.norm(rhs)))
    diagnostics = SolverDiagnostics(
        lambda_value=float(lagrange),
        constraint_active=constraint_active,
        coefficient_norm_squared=float(coefficients @ coefficients),
        radius_squared=float(radius_squared),
        numerical_rank=int(positive.size),
        dimension=int(rhs.size),
        condition_number=condition_number,
        kkt_residual=kkt_residual,
    )
    return coefficients, diagnostics


def fit_drift(
    path_system: np.ndarray,
    measure_system: np.ndarray,
    *,
    D: int = 2,
    degree: int = 2,
    A: float = 6.0,
    total_time: float = 1.0,
) -> DriftModel:
    """Fit the tensor-product estimator without materializing the full design matrix."""
    paths = np.asarray(path_system, dtype=float)
    measures = np.asarray(measure_system, dtype=float)
    if paths.ndim != 2 or measures.ndim != 2 or paths.shape[1] != measures.shape[1]:
        raise ValueError("Training systems must be 2D arrays on the same observation grid.")
    if paths.shape[0] < 2 or measures.shape[0] < 2:
        raise ValueError("Each training system must contain at least two particles.")
    n_train, n_grid_points = paths.shape
    n_times = n_grid_points - 1
    dt = total_time / n_times
    basis = BSplineBasis(D=D, degree=degree, A=A)
    J = basis.dimension
    gram = np.zeros((J * J, J * J), dtype=float)
    rhs = np.zeros(J * J, dtype=float)
    measure_basis = np.empty((n_times, measures.shape[0], J), dtype=float)
    outside_count = 0
    outside_total = 0

    for m in range(n_times):
        path_basis, path_outside = basis.transform(paths[:, m])
        current_measure_basis, measure_outside = basis.transform(measures[:, m])
        measure_basis[m] = current_measure_basis
        measure_mean = current_measure_basis.mean(axis=0)
        velocity = (paths[:, m + 1] - paths[:, m]) / dt
        gram += np.kron(path_basis.T @ path_basis, np.outer(measure_mean, measure_mean))
        rhs += np.outer(path_basis.T @ velocity, measure_mean).reshape(-1)
        outside_count += int(path_outside.sum() + measure_outside.sum())
        outside_total += path_outside.size + measure_outside.size

    normalizer = n_train * n_times
    gram /= normalizer
    rhs /= normalizer
    radius_squared = (J * A) ** 2 * np.log(n_train)
    flat_coefficients, diagnostics = solve_ball_constrained_quadratic(
        gram, rhs, radius_squared
    )
    return DriftModel(
        basis=basis,
        coefficients=flat_coefficients.reshape(J, J),
        measure_basis=measure_basis,
        n_train=n_train,
        diagnostics=diagnostics,
        train_outside_rate=outside_count / outside_total,
    )


def mean_estimated_drift(
    paths: np.ndarray,
    model: DriftModel,
    *,
    measure_paths: np.ndarray | None = None,
    clip: bool = True,
    path_chunk_size: int = 128,
    measure_chunk_size: int = 256,
) -> tuple[np.ndarray, float]:
    """Evaluate the empirical-measure estimator along paths.

    Clipping is applied to b_hat(x, y) before the empirical-measure average,
    exactly as in the paper.  Chunking bounds memory without changing results.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] - 1 != model.measure_basis.shape[0]:
        raise ValueError("Evaluation paths and fitted model must use the same grid.")
    evaluation_measure_basis: np.ndarray | None = None
    if measure_paths is not None:
        measure_paths = np.asarray(measure_paths, dtype=float)
        if measure_paths.ndim != 2 or measure_paths.shape[1] != paths.shape[1]:
            raise ValueError("Evaluation path and measure systems must share the same grid.")
        evaluation_measure_basis = np.empty(
            (paths.shape[1] - 1, measure_paths.shape[0], model.basis.dimension),
            dtype=float,
        )
        for m in range(paths.shape[1] - 1):
            evaluation_measure_basis[m], _ = model.basis.transform(measure_paths[:, m])
    n_paths = paths.shape[0]
    n_times = paths.shape[1] - 1
    output = np.empty((n_paths, n_times), dtype=float)
    outside_count = 0
    threshold = model.threshold

    for m in range(n_times):
        path_basis, outside = model.basis.transform(paths[:, m])
        outside_count += int(outside.sum())
        current_measure_basis = (
            model.measure_basis[m]
            if evaluation_measure_basis is None
            else evaluation_measure_basis[m]
        )
        if not clip:
            output[:, m] = (
                path_basis @ model.coefficients @ current_measure_basis.mean(axis=0)
            )
            continue
        for path_start in range(0, n_paths, path_chunk_size):
            path_stop = min(path_start + path_chunk_size, n_paths)
            left = path_basis[path_start:path_stop] @ model.coefficients
            accumulated = np.zeros(path_stop - path_start, dtype=float)
            for measure_start in range(0, current_measure_basis.shape[0], measure_chunk_size):
                measure_stop = min(
                    measure_start + measure_chunk_size, current_measure_basis.shape[0]
                )
                pair_values = left @ current_measure_basis[measure_start:measure_stop].T
                np.clip(pair_values, -threshold, threshold, out=pair_values)
                accumulated += pair_values.sum(axis=1)
            output[path_start:path_stop, m] = accumulated / current_measure_basis.shape[0]
    return output, outside_count / (n_paths * n_times)
