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

import numpy as np
import pytest

from pennylane import math

pytestmark = pytest.mark.all_interfaces


@pytest.mark.parametrize("grad_fn", (math.grad, math.jacobian))
def test_no_interface_error_numpy(grad_fn):
    """Test that an error is raised for an unknown interface."""

    with pytest.raises(ValueError, match="Interface numpy is not differentiable"):
        grad_fn(lambda x: x**2)(np.array(2.0))


@pytest.mark.parametrize("interface", ("autograd", "jax", "torch"))
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

        assert math.allclose(g1, 2.0)
        assert math.allclose(g2, 3.0)

    def test_keyword_arguments(self, interface):
        """Test that keyword arguments are considered."""

        def f(x, *, constant):
            return constant * x

        x = math.asarray(2.0, like=interface, requires_grad=True)

        g = math.grad(f)(x, constant=2.0)
        assert math.allclose(g, 2.0)


@pytest.mark.parametrize("interface", ("autograd", "jax", "torch"))
class TestJacobian:
    """Tests for the math.jacobian function."""

    def test_jac_first_arg(self, interface):
        """Test taking the jacobian of the first argument."""

        def f(x, y):
            return x * y

        x = math.asarray([2.0, 3.0], like=interface, requires_grad=True)
        y = math.asarray(3.0, like=interface, requires_grad=True)

        g = math.jacobian(f)(x, y)
        print(g)
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


class TestJacobianPytreeOutput:
    """Test jacobians of non-array outputs."""

    def test_jacobian_autograd_error(self):
        """Test that an informative error if the output of a function isnt an array."""

        def f(x):
            return (x,)

        with pytest.raises(
            ValueError, match="autograd can only differentiate with respect to arrays,"
        ):
            math.jacobian(f)(math.asarray(2.0, like="autograd", requires_grad=True))

    @pytest.mark.parametrize("interface", ("torch", "jax"))
    def test_tuple_output_scalar_argnum(self, interface):
        """Test the shape outputted for a tuple valued function."""

        def f(x):
            return (5 * x, x**2)

        jac = math.jacobian(f)(math.asarray(2.0, like=interface, requires_grad=True))
        assert isinstance(jac, tuple)
        assert len(jac) == 2
        for j in jac:
            assert math.get_interface(j) == interface

        assert math.allclose(jac[0], 5)
        assert math.allclose(jac[1], 4)

    @pytest.mark.parametrize("interface", ("torch", "jax"))
    def test_tuple_output_tuple_argnum(self, interface):
        """Test the shape outputted for a tuple valued function with multiple traianble arguments."""

        def f(x, y):
            return (2 * y**2, 3 * x**2)

        x = math.asarray(1.5, like=interface, requires_grad=True)
        y = math.asarray(2.5, like=interface, requires_grad=True)

        jac = math.jacobian(f, argnums=(0, 1))(x, y)

        assert len(jac) == 2
        assert len(jac[0]) == 2
        assert len(jac[1]) == 2
        dy0_dx0_expected = 0
        assert math.allclose(jac[0][0], dy0_dx0_expected)
        dy0_dx1_expected = 4 * y
        assert math.allclose(jac[0][1], dy0_dx1_expected)
        dy1_dx0_expected = 6 * x
        assert math.allclose(jac[1][0], dy1_dx0_expected)
        dy1_dx1_expected = 0
        assert math.allclose(jac[1][1], dy1_dx1_expected)
