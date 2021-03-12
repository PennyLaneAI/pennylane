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
