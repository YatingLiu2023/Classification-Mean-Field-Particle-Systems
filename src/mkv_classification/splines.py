"""Compact B-spline basis with explicit domain-boundary semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BSplineBasis:
    """Uniform clamped B-spline basis on [-A, A].

    D is the number of intervals and degree is the polynomial degree.  The
    number of one-dimensional basis functions is J = D + degree.
    """

    D: int = 2
    degree: int = 2
    A: float = 6.0

    def __post_init__(self) -> None:
        if self.D < 1 or self.degree < 0 or self.A <= 0:
            raise ValueError("D >= 1, degree >= 0, and A > 0 are required.")

    @property
    def dimension(self) -> int:
        return self.D + self.degree

    @property
    def knots(self) -> np.ndarray:
        internal = np.linspace(-self.A, self.A, self.D + 1)[1:-1]
        return np.concatenate(
            [
                np.repeat(-self.A, self.degree + 1),
                internal,
                np.repeat(self.A, self.degree + 1),
            ]
        )

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return basis values and an elementwise outside-domain mask."""
        original_shape = np.asarray(values).shape
        x = np.asarray(values, dtype=float).reshape(-1)
        knots = self.knots
        outside = (x < -self.A) | (x > self.A) | ~np.isfinite(x)
        basis = ((x[:, None] >= knots[:-1]) & (x[:, None] < knots[1:])).astype(float)
        for order in range(1, self.degree + 1):
            updated = np.zeros((x.size, len(knots) - order - 1), dtype=float)
            for column in range(updated.shape[1]):
                left_denominator = knots[column + order] - knots[column]
                right_denominator = knots[column + order + 1] - knots[column + 1]
                if left_denominator > 0:
                    updated[:, column] += (
                        (x - knots[column]) / left_denominator * basis[:, column]
                    )
                if right_denominator > 0:
                    updated[:, column] += (
                        (knots[column + order + 1] - x)
                        / right_denominator
                        * basis[:, column + 1]
                    )
            basis = updated
        basis = basis[:, : self.dimension]
        right_boundary = np.isclose(x, self.A, rtol=0.0, atol=1e-12)
        basis[right_boundary] = 0.0
        basis[right_boundary, -1] = 1.0
        basis[outside] = 0.0
        return basis.reshape((*original_shape, self.dimension)), outside.reshape(original_shape)

