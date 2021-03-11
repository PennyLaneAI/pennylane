import numpy as np
import pytest

import pennylane as qml



x1 = 0.5
grad1 = 0.8775789053009353

x2 = np.array([0.5, 0.25])
grad2 = np.array([np.array([0.87757891, 0.        ]), np.array([0.        , 0.96890838])])


@pytest.mark.parametrize(
    ("argnum", "idx", "delta", "x", "exp_grad"),
    [
        (0, None, 0.01, x1, grad1),
        (0, None, 0.01, x2, grad2),
    ],
)
def test_first_finit_diff(argnum, idx, delta, x, exp_grad):
    r"""Tests the correctness of the first-order finite difference function."""

    def f(x):
        return np.sin(x)

    grad = qml.finite_diff(f, N=1, argnum=argnum, idx=idx, delta=delta)(x)

    print("computed", grad, type(grad))
    print("expected", exp_grad, type(exp_grad))

    assert np.allclose(grad, exp_grad)


# def test_not_callable_func():
#     r"""Test that an error is raised if the function to be differentiated
#     is not a callable object"""

#     with pytest.raises(TypeError, match="'F' should be a callable function"):
#         qml.finite_diff(f(x1), x1, 0)


# @pytest.mark.parametrize(
#     ("x", "i"),
#     [(x2, None), (x2, 4)],
# )
# def test_exceptions(x, i):
#     r"""Test that an error is raised if the index 'i' of the variable we are differentiating
#     with respect to is not given or is out of bounds for multivariable functions"""

#     with pytest.raises(ValueError, match="'i' must be an integer between"):
#         qml.finite_diff(f, x2, i)
