# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Test the qml.math.grad and qml.math.jacobian functions.
"""

import pytest

from pennylane import math

pytestmark = pytest.mark.all_interfaces


@pytest.mark.parametrize("interface", ("autograd", "jax", "tensorflow", "torch"))
class TestGrad:
    """Tests for qml.math.grad"""

    def test_differentiate_first_arg(self, interface):
        """Test that we just differentiate the first argument by default."""

        def f(x, y):
            return x * y

        x = math.asarray(2.0, like=interface, requires_grad=True)
        y = math.asarray(3.0, like=interface, requires_grad=True)

        g = math.grad(f)(x, y)
        if interface != "autograd":
            assert math.get_interface(g) == interface
        assert math.allclose(g, 3.0)


    def test_multiple_argnums(self, interface):
        """Test that we can differentiate multiple arguments."""

        def g(x, y):
            return 2 * x + 3 * y

        x = math.asarray(0.5, like=interface, requires_grad=True)
        y = math.asarray(2.5, like=interface, requires_grad=True)

        g1, g2 = math.grad(g, argnums=(0, 1))(x, y)
        if interface != "autograd":
            assert math.get_interface(g1) == interface
            assert math.get_interface(g2) == interface

        assert math.allclose(g1, 2)
        assert math.allclose(g2, 3)


    def test_keyword_arguments(self, interface):
        """Test that keyword arguments are considered."""

        def f(x, *, constant):
            return constant * x

        x = math.asarray(2.0, like=interface, requires_grad=True)

        g = math.grad(f)(x, constant=2.0)
        assert math.allclose(g, 2.0)


@pytest.mark.parametrize("interface", ("autograd", "jax", "tensorflow", "torch"))
class TestJacobian:
    """Tests for the math.jacobian function."""

    def test_jac_first_arg(self, interface):
        """Test taking the jacobian of the first argument."""

        def f(x, y):
            return x * y

        x = math.asarray([2.0, 3.0], like=interface, requires_grad=True)
        y = math.asarray(3.0, like=interface, requires_grad=True)

        g = math.jacobian(f)(x, y)
        if interface != "autograd":
            assert math.get_interface(g) == interface
        expected = math.asarray([[3.0, 0.0], [0.0, 3.0]])
        assert math.allclose(g, expected)

    def test_multiple_argnums(self, interface):
        """Test that we can differentiate multiple arguments."""

        def g(x, y):
            return 2 * x + 3 * y

        x = math.asarray([0.5, 1.2], like=interface, requires_grad=True)
        y = math.asarray([2.5, 4.8], like=interface, requires_grad=True)

        g1, g2 = math.jacobian(g, argnums=(0, 1))(x, y)
        if interface != "autograd":
            assert math.get_interface(g1) == interface
            assert math.get_interface(g2) == interface

        assert math.allclose(g1, 2 * math.eye(2))
        assert math.allclose(g2, 3 * math.eye(2))


    def test_keyword_arguments(self, interface):
        """Test that keyword arguments are considered."""

        def f(x, *, constant):
            return constant * x

        x = math.asarray([2.0, 3.0], like=interface, requires_grad=True)

        g = math.jacobian(f)(x, constant=2.0)
        assert math.allclose(g, 2 * math.eye(2))