import numpy as np
import pytest

import pennylane as qml


# x1 = 0.5
# grad1 = 0.8775789053009353

# x2 = np.array([0.5, 0.25])
# grad2 = np.array([np.array([0.87757891, 0.        ]), np.array([0.        , 0.96890838])])


# @pytest.mark.parametrize(
#     ("argnum", "idx", "delta", "x", "exp_grad"),
#     [
#         # (0, None, 0.01, x1, grad1),
#         (0, None, 0.01, x2, grad2),
#     ],
# )
# def test_first_finit_diff(argnum, idx, delta, x, exp_grad):
#     r"""Tests the correctness of the first-order finite difference function."""

#     def f(x):
#         return np.sin(x)

#     grad = qml.finite_diff(f, N=1, argnum=argnum, idx=idx, delta=delta)(x)
#     for i, _grad in enumerate(grad):
#         grad[i] = np.array(_grad, dtype=float)

#     print("computed", grad, type(grad))
#     print("expected", exp_grad, type(exp_grad))

#     assert np.allclose(grad, exp_grad)


x1 = 1.975
y1 = 0.33

x2 = np.array([1.975, 0.33])
y2 = 0.376


@pytest.mark.parametrize(
    ("x", "y", "argnum", "idx", "delta", "exp_deriv2"),
    [
        (x1, y1, 0, None, 0.01, 0.3541412270280375),
        (x1, y1, 1, None, 0.02, 3094.62916408922),
        (x2, y2, 0, [0, 0], 0.01, np.array([0.35414123, 0.0])),
        (x2, y2, 0, [1, 1], 0.01, np.array([0.0, 3064.04502087])),
        (x2, y2, 0, [0, 1], 0.01, np.array([0.0, 0.0])),
        (x2, y2, 1, None, 0.02, np.array([1608.12314415, 1608.12314415])),
    ],
)
def test_second_order_finite_diff(x, y, argnum, idx, delta, exp_deriv2):
    r"""Test correctness of the second derivative calculated with the
    function '_fd_second_order_centered'"""

    def f(x, y):
        return np.sin(x) / x ** 4 + y ** -3

    deriv2 = qml.finite_diff(f, N=2, argnum=argnum, idx=idx, delta=delta)(x, y)

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
            qml.finite_diff(f, N=N, delta=delta)
    elif N == 0:
        with pytest.raises(TypeError, match=msg_match):
            qml.finite_diff(f(0.5), N=N, delta=delta)


@pytest.mark.parametrize(
    ("which", "argnum", "idx", "msg_match"),
    [
        ("both", 2, None, "The value of 'argnum' has to be between 0 and"),
        ("both", 1, [0, 1], "is not an array, 'idx' should be set to 'None'"),
        ("both", 0, [0, 2], "Indices in 'idx' can not be greater than"),
        (
            "second",
            0,
            [0, 2, 3],
            "The number of indices given in 'idx' can not be greater than two",
        ),
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
            qml.finite_diff(f, argnum=argnum, idx=idx)(x, y)

        with pytest.raises(ValueError, match=msg_match):
            qml.finite_diff(f, N=2, argnum=argnum, idx=idx)(x, y)

    if which == "second":
        with pytest.raises(ValueError, match=msg_match):
            qml.finite_diff(f, N=2, argnum=argnum, idx=idx)(x, y)
