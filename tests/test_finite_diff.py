import numpy as np
import pytest

import pennylane as qml


x1 = 0.5
y1 = -0.3

x2 = np.array([0.5, -0.1975])
y2 = -0.3

x3 = np.array([[1.1, 2.2], [3.3, 4.4]])
y3 = -0.54
grad3 = np.array(
    [
        np.array(
            [np.array([[-1.90380464, 0.0], [0.0, 0.0]]), np.array([[0.0, -0.08787575], [0.0, 0.0]])]
        ),
        np.array(
            [np.array([[0.0, 0.0], [-0.00671454, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.00148809]])]
        ),
    ]
)


def catch_warn_finite_diff(f, N=1, argnum=0, idx=None, delta=0.01):
    """Computes the finite diff and catches the initial deprecation warning."""

    with pytest.warns(UserWarning, match="The black-box finite_diff function is deprecated"):
        res = qml.finite_diff(f, N, argnum, idx, delta)
    return res


@pytest.mark.parametrize(
    ("x", "y", "argnum", "idx", "delta", "exp_grad"),
    [
        (x1, y1, 0, None, 0.01, -47.341084643123565),
        (x1, y1, 1, None, 0.02, -371.7453193214464),
        (x2, y2, 0, [0], 0.01, np.array([np.array([-47.34108464, 0.0]), 0])),
        (
            x2,
            y2,
            0,
            None,
            0.01,
            np.array([np.array([-47.34108464, 0.0]), np.array([0.0, -1971.68884622])]),
        ),
        (x3, y3, 0, None, 0.01, grad3),
    ],
)
def test_first_finit_diff(x, y, argnum, idx, delta, exp_grad):
    r"""Tests the correctness of the first-order finite difference function."""

    def f(x, y):
        return np.sin(x) / x**4 + y**-3

    grad = catch_warn_finite_diff(f, argnum=argnum, idx=idx, delta=delta)(x, y)

    if grad.ndim != 0:
        idx = list(np.ndindex(*grad.shape))
        for i in idx:
            assert np.allclose(grad[i], exp_grad[i])
    else:
        np.allclose(np.array(grad, dtype=float), exp_grad)


x1 = 1.975
y1 = 0.33

x2 = np.array([1.975, 0.33])
y2 = 0.376

x3 = np.array([[1.1, 2.2], [3.3, 4.4]])
y3 = -0.54


@pytest.mark.parametrize(
    ("x", "y", "argnum", "idx", "delta", "exp_deriv2"),
    [
        (x1, y1, 0, None, 0.01, 0.3541412270280375),
        (x1, y1, 1, None, 0.02, 3094.62916408922),
        (x2, y2, 0, [0, 0], 0.01, np.array([0.35414123, 0.0])),
        (x2, y2, 0, [1, 1], 0.01, np.array([0.0, 3064.04502087])),
        (x2, y2, 0, [0, 1], 0.01, np.array([0.0, 0.0])),
        (x2, y2, 1, None, 0.02, np.array([1608.12314415, 1608.12314415])),
        (x3, y3, 0, [[0, 1], [1, 1]], 0.01, np.array([[0.0, 0.0], [0.0190738, 0.001407]])),
    ],
)
def test_second_order_finite_diff(x, y, argnum, idx, delta, exp_deriv2):
    r"""Test correctness of the second derivative calculated with the
    function '_fd_second_order_centered'"""

    def f(x, y):
        return np.sin(x) / x**4 + y**-3

    deriv2 = catch_warn_finite_diff(f, N=2, argnum=argnum, idx=idx, delta=delta)(x, y)

    assert np.allclose(deriv2, exp_deriv2)


def f(x):
    return np.sin(x)


@pytest.mark.parametrize(
    ("N", "delta", "func", "msg_match"),
    [
        (3, 0.01, f, "finite-difference approximations are supported up to second-order"),
        (1, 0.0, f, "The value of the step size 'delta' has to be greater than 0"),
        (0, 0.01, f(0.5), "'f' should be a callable function"),
    ],
)
def test_exceptions_finite_diff(N, delta, func, msg_match):
    r"""Test exceptions of the 'finite_diff' function"""

    if N != 0:
        with pytest.raises(ValueError, match=msg_match):
            catch_warn_finite_diff(f, N=N, delta=delta)
    elif N == 0:
        with pytest.raises(TypeError, match=msg_match):
            catch_warn_finite_diff(f(0.5), N=N, delta=delta)


@pytest.mark.parametrize(
    ("which", "argnum", "idx", "msg_match"),
    [
        ("both", 2, None, "The value of 'argnum' has to be between 0 and "),
        ("both", 1, [0, 1], "is not an array, 'idx' should be set to 'None'"),
        (
            "second",
            0,
            [0, 2, 3],
            "The number of indices given in 'idx' can not be greater than two",
        ),
        ("second", 0, None, "'idx' should contain the indices of the arguments"),
    ],
)
def test_exceptions(which, argnum, idx, msg_match):
    r"""Test exceptions raised by the internal functions
    '_fd_first_order_centered' and '_fd_second_order_centered'"""

    def f(x, y):
        return np.sin(x) + 1 / y

    x = np.array([0.5, -0.25])
    y = 0.3

    if which == "both":
        with pytest.raises(ValueError, match=msg_match):
            catch_warn_finite_diff(f, argnum=argnum, idx=idx)(x, y)

        with pytest.raises(ValueError, match=msg_match):
            catch_warn_finite_diff(f, N=2, argnum=argnum, idx=idx)(x, y)

    if which == "second":
        with pytest.raises(ValueError, match=msg_match):
            catch_warn_finite_diff(f, N=2, argnum=argnum, idx=idx)(x, y)
