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
This file tests qml.gradients.finite_diff_jvp
"""
import numpy as np
import pytest

import pennylane as qml

pytestmark = pytest.mark.all_interfaces


class TestValidation:

    def test_approx_order_validation(self):
        """Test that a NotImplementedError is thrown for higher order approx_order."""

        def f(x):
            return x**2

        with pytest.raises(NotImplementedError, match="only approx_order=1 is currently"):
            qml.gradients.finite_diff_jvp(f, (0.5,), (1.0,), approx_order=2)

    def test_strategy_validation(self):
        """Test that a NotImplementedError is thrown for strategies other than forward."""

        with pytest.raises(NotImplementedError, match="only strategy='forward'"):
            qml.gradients.finite_diff_jvp(lambda x: (x**2,), (0.5,), (1.0,), strategy="backward")

    @pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "tensorflow", "autograd"))
    def test_float32_warning(self, interface):
        """Test that a warning is raised with float32 parameters."""

        x = qml.math.asarray(0.5, dtype=np.float32, like=interface)

        with pytest.warns(UserWarning, match="Detected float32 parameter"):
            _ = qml.gradients.finite_diff_jvp(lambda x: (x**2,), (x,), (1.0,))

    def test_must_return_list_tuple(self):
        """Test that an error is raised if the output is not a list or tuple."""

        with pytest.raises(ValueError, match="Input function f must return either a list or tuple"):
            _ = qml.gradients.finite_diff_jvp(lambda x: x**2, (0.5,), (1.0,))


def test_builtins():
    """Test that the function can be used with builtins."""

    def f(x, y):
        return (x + y, x * y)

    res, dres = qml.gradients.finite_diff_jvp(f, (0.5, 0.5), (2.0, 1.0))

    assert qml.math.allclose(res, (1.0, 0.25))
    assert qml.math.allclose(dres, (3.0, 0.5 * 2 + 0.5 * 1.0))


@pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "tensorflow", "autograd"))
def test_scalar_in_scalar_out(interface):
    """Test using finite_diff_jvp with scalars in and scalars out."""

    def f(x):
        return (3 * x**2,)

    x = qml.math.asarray(0.5, like=interface, dtype=np.float64)
    dx = qml.math.asarray(2.0, like=interface, dtype=np.float64)

    res, dres = qml.gradients.finite_diff_jvp(f, (x,), (dx,))

    assert len(res) == 1
    assert len(dres) == 1
    assert qml.math.allclose(res[0], 3 * x**2)
    expected_dres = dx * 6 * x
    assert qml.math.allclose(dres[0], expected_dres)


@pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "tensorflow", "autograd"))
def test_mutliple_args(interface):
    """Test using finite_diff_jvp with muliple arguments."""

    counter = []

    def f(x, y, z):
        nonlocal counter
        counter.append(1)
        return (x * y * z,)

    x = qml.math.asarray(1.0, like=interface, dtype=np.float64)
    y = qml.math.asarray(2.0, like=interface, dtype=np.float64)
    z = qml.math.asarray(0.5, like=interface, dtype=np.float64)
    dx = qml.math.asarray(1.0, like=interface, dtype=np.float64)
    dy = qml.math.asarray(2.0, like=interface, dtype=np.float64)
    dz = qml.math.asarray(0.0, like=interface, dtype=np.float64)

    res, dres = qml.gradients.finite_diff_jvp(f, (x, y, z), (dx, dy, dz))

    assert len(res) == 1
    assert len(dres) == 1
    assert sum(counter) == 3  # forward pass plus x and y

    assert qml.math.allclose(res, x * y * z)
    dres_expected = dx * y * z + x * dy * z
    assert qml.math.allclose(dres, dres_expected)


@pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "tensorflow", "autograd"))
def test_array_input(interface):
    """Test that the function can accept multi-dimensional arrays."""

    counter = []

    def f(x):
        nonlocal counter
        counter.append(1)
        return (qml.math.sin(x),)

    x = qml.math.asarray(np.arange(4), like=interface, dtype=np.float64)
    dx = qml.math.asarray(np.ones(4), like=interface, dtype=np.float64)

    res, dres = qml.gradients.finite_diff_jvp(f, (x,), (dx,))

    assert len(res) == 1
    assert len(dres) == 1
    assert sum(counter) == 5

    assert qml.math.allclose(res[0], qml.math.sin(x))
    assert qml.math.allclose(dres[0], qml.math.cos(x))


@pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "tensorflow", "autograd"))
def test_multiple_outputs(interface):
    """Test that finite_diff_jvp can handle multiple outputs."""

    counter = []

    def f(x):
        nonlocal counter
        counter.append(1)
        return (x**2, x**3, x**4)

    x = qml.math.asarray([0.5, 0.6, 0.7], like=interface, dtype=np.float64)
    dx = qml.math.asarray([1.0, 1.0, 1.0], like=interface, dtype=np.float64)

    res, dres = qml.gradients.finite_diff_jvp(f, (x,), (dx,))
    assert len(res) == 3
    assert len(dres) == 3
    assert sum(counter) == 4

    assert qml.math.allclose(res, f(x))
    assert qml.math.allclose(dres[0], 2 * x)
    assert qml.math.allclose(dres[1], 3 * x**2)
    assert qml.math.allclose(dres[2], 4 * x**3)
