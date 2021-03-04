import numpy as np
import pytest

import pennylane as qml


def f(x):
    return np.sin(x)


x1 = 0.5
delta1 = 0.001
df1 = 0.877582525324383

x2 = np.array([0.5, -0.3])
delta2 = 0.002
df2 = np.array([0.87758242, 0.0])


def g(x):
    return np.array([np.sin(x), 1 / x])


x3 = -0.25
delta3 = 0.003
dg3 = np.array([0.96891206, -16.00057602])

x4 = np.array([-0.25, 1.975])
delta4 = 0.004
dg4 = np.array([[0.0, -0.39328647], [0.0, -0.25636943]])


@pytest.mark.parametrize(
    ("func", "x", "i", "delta", "deriv_ref"),
    [
        (f, x1, None, delta1, df1),
        (f, x2, 0, delta2, df2),
        (g, x3, None, delta3, dg3),
        (g, x4, 1, delta4, dg4),
    ],
)
def test_finit_diff(func, x, i, delta, deriv_ref):    
    r"""Tests the correctness of the derivative evaluated by the 'finite_diff' function."""

    deriv = qml.finite_diff(func, x, i, delta)

    assert np.allclose(deriv_ref, deriv)


def test_not_callable_func():
    r"""Test that an error is raised if the function to be differentiated
    is not a callable object"""

    with pytest.raises(TypeError, match="'F' should be a callable function"):
        qml.finite_diff(f(x1), x1, 0)


@pytest.mark.parametrize(
    ("x", "i"),
    [(x2, None), (x2, 4)],
)
def test_exceptions(x, i):
    r"""Test that an error is raised if the index 'i' of the variable we are differentiating
    with respect to is not given or is out of bounds for multivariable functions"""

    with pytest.raises(ValueError, match="'i' must be an integer between"):
        qml.finite_diff(f, x2, i)
