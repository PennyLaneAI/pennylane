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
Unit tests for qml.grad
"""

import pytest

import pennylane as qml


@pytest.mark.parametrize("grad_fn", (qml.grad, qml.jacobian))
def test_kwarg_errors_without_qjit(grad_fn):
    """Test that errors are raised with method and h when qjit is not active."""

    def f(x):
        return x**2

    x = qml.numpy.array(0.5)

    with pytest.raises(ValueError, match="method = 'fd' unsupported without QJIT."):
        grad_fn(f, method="fd")(x)

    with pytest.raises(ValueError, match="unsupported without QJIT. "):
        grad_fn(f, h=1e-6)(0.5)


def test_grad_name():
    """Test that grad has name associated with it for the later mlir op."""

    def f(x):
        return x**2

    assert qml.grad(f).__name__ == "<grad: f>"

    class A:

        def __repr__(self):
            return "A"

        def __call__(self, x):
            return x**2

    assert qml.grad(A()).__name__ == "<grad: A>"


def test_jacobian_name():
    """Test that jacobian has name associated with it for the later mlir op."""

    def f(x):
        return x**2

    assert qml.jacobian(f).__name__ == "<jacobian: f>"

    class A:

        def __repr__(self):
            return "A"

        def __call__(self, x):
            return x**2

    assert qml.jacobian(A()).__name__ == "<jacobian: A>"


def test_vjp_without_qjit():
    """Test that an error is raised when using VJP without QJIT."""

    def vjp(params, cotangent):
        def f(x):
            y = [qml.math.sin(x[0]), x[1] ** 2, x[0] * x[1]]
            return qml.math.stack(y)

        return qml.vjp(f, [params], [cotangent])

    x = qml.numpy.array([0.1, 0.2])
    dy = qml.numpy.array([-0.5, 0.1, 0.3])

    with pytest.raises(
        qml.exceptions.CompileError,
        match="Pennylane does not support the VJP function without QJIT.",
    ):
        vjp(x, dy)
