import numpy as np

from mkv_classification.estimator import solve_ball_constrained_quadratic


def test_unconstrained_solution_keeps_lambda_zero() -> None:
    gram = np.diag([2.0, 4.0])
    rhs = np.array([2.0, 8.0])
    solution, diagnostics = solve_ball_constrained_quadratic(gram, rhs, 100.0)
    assert np.allclose(solution, [1.0, 2.0])
    assert diagnostics.lambda_value == 0.0
    assert not diagnostics.constraint_active
    assert diagnostics.kkt_residual < 1e-12


def test_active_constraint_satisfies_kkt_conditions() -> None:
    gram = np.eye(3)
    rhs = np.array([3.0, 4.0, 0.0])
    solution, diagnostics = solve_ball_constrained_quadratic(gram, rhs, 1.0)
    assert diagnostics.constraint_active
    assert diagnostics.lambda_value > 0.0
    assert np.isclose(solution @ solution, 1.0, atol=1e-9)
    assert diagnostics.kkt_residual < 1e-10


def test_singular_gram_uses_minimum_norm_solution() -> None:
    gram = np.diag([1.0, 0.0])
    rhs = np.array([2.0, 0.0])
    solution, diagnostics = solve_ball_constrained_quadratic(gram, rhs, 100.0)
    assert np.allclose(solution, [2.0, 0.0])
    assert diagnostics.numerical_rank == 1

