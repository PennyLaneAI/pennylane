# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for qp.grad
"""

import pytest

import pennylane as qp


@pytest.mark.parametrize("grad_fn", (qp.grad, qp.jacobian))
def test_kwarg_errors_without_qjit(grad_fn):
    """Test that errors are raised with method and h when qjit is not active."""

    def f(x):
        return x**2

    x = qp.numpy.array(0.5)

    with pytest.raises(ValueError, match="method = 'fd' unsupported without QJIT."):
        grad_fn(f, method="fd")(x)

    with pytest.raises(ValueError, match="unsupported without QJIT. "):
        grad_fn(f, h=1e-6)(0.5)


@pytest.mark.parametrize("grad_fn", (qp.grad, qp.jacobian))
def test_argnum_deprecation(grad_fn):
    """Test that using argnum raises a deprecation warning."""

    def f(x, y):
        return y * x**2

    with pytest.warns(
        qp.exceptions.PennyLaneDeprecationWarning, match="has been renamed to argnums"
    ):
        g = grad_fn(f, argnum=1)

    x = 0.5
    y = qp.numpy.array(0.5, requires_grad=True)
    r = g(x, y)
    assert qp.math.allclose(r, 0.25)


def test_grad_name():
    """Test that grad has name associated with it for the later mlir op."""

    def f(x):
        return x**2

    assert qp.grad(f).__name__ == "<grad: f>"

    class A:

        def __repr__(self):
            return "A"

        def __call__(self, x):
            return x**2

    assert qp.grad(A()).__name__ == "<grad: A>"


def test_jacobian_name():
    """Test that jacobian has name associated with it for the later mlir op."""

    def f(x):
        return x**2

    assert qp.jacobian(f).__name__ == "<jacobian: f>"

    class A:

        def __repr__(self):
            return "A"

        def __call__(self, x):
            return x**2

    assert qp.jacobian(A()).__name__ == "<jacobian: A>"
