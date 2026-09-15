import numpy as np

from mkv_classification.splines import BSplineBasis


def test_partition_of_unity_and_boundary_handling() -> None:
    basis = BSplineBasis(D=2, degree=2, A=6.0)
    values = np.array([-7.0, -6.0, -1.0, 0.0, 5.0, 6.0, 7.0])
    transformed, outside = basis.transform(values)
    assert transformed.shape == (7, 4)
    assert np.array_equal(outside, [True, False, False, False, False, False, True])
    assert np.allclose(transformed[~outside].sum(axis=1), 1.0)
    assert np.all(transformed[outside] == 0.0)
    assert transformed[-2, -1] == 1.0


def test_single_outlier_does_not_zero_other_rows() -> None:
    basis = BSplineBasis(D=5, degree=2, A=2.0)
    transformed, outside = basis.transform(np.array([0.0, 10.0]))
    assert not outside[0] and outside[1]
    assert np.isclose(transformed[0].sum(), 1.0)
    assert np.all(transformed[1] == 0.0)

