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


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "autograd"))
def test_float32_warning(interface):
    """Test that a warning is raised with float32 parameters."""

    x = qml.math.asarray(0.5, dtype=np.float32, like=interface)

    with pytest.warns(
        UserWarning, match="Detected 32 bits precision parameter with finite differences."
    ):
        _ = qml.gradients.finite_diff_jvp(lambda x: (x**2,), (x,), (1.0,))


def test_builtins():
    """Test that the function can be used with builtins."""

    def f(x, y):
        return (x + y, x * y)

    res, dres = qml.gradients.finite_diff_jvp(f, (0.5, 0.5), (2.0, 1.0))

    assert qml.math.allclose(res, (1.0, 0.25))
    assert qml.math.allclose(dres, (3.0, 0.5 * 2 + 0.5 * 1.0))


def test_effect_of_h():
    """Test that setting h does change the value used when computing the derivative."""

    def f(x):
        return x**3

    _, dres = qml.gradients.finite_diff_jvp(f, (2.0,), (1.0,), h=1e-1)

    expected_dres = (f(2.1) - f(2.0)) / 0.1  # 12.61, quite a bit off from 12
    assert qml.math.allclose(dres, expected_dres)


@pytest.mark.parametrize(
    "kwargs",
    ({"approx_order": 2}, {"strategy": "backward"}, {"approx_order": 2, "strategy": "center"}),
)
@pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "autograd"))
def test_scalar_in_scalar_out(interface, kwargs):
    """Test using finite_diff_jvp with scalars in and scalars out."""

    def f(x):
        return 3 * x**2

    x = qml.math.asarray(0.5, like=interface, dtype=np.float64)
    dx = qml.math.asarray(2.0, like=interface, dtype=np.float64)

    res, dres = qml.gradients.finite_diff_jvp(f, (x,), (dx,), **kwargs)

    assert qml.math.allclose(res, 3 * x**2)
    expected_dres = dx * 6 * x
    assert qml.math.allclose(dres, expected_dres)


@pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "autograd"))
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


@pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "autograd"))
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


@pytest.mark.parametrize("interface", ("numpy", "torch", "jax", "autograd"))
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


def test_approx_order():
    """Test that setting approx order changes the executions used."""

    args = []

    def f(x):
        nonlocal args
        args.append(x)
        return x**2

    res, dres = qml.gradients.finite_diff_jvp(f, (1.0,), (1.0,), approx_order=2)
    assert qml.math.allclose(res, 1)
    assert qml.math.allclose(dres, 2)

    assert args == [1, 1 + 1e-7, 1 + 2e-7]


def test_backward_strategy():
    """Test the backward stategy gets evaluated at different points."""

    args = []

    def f(x):
        nonlocal args
        args.append(x)
        return {"a": x**2}

    res, dres = qml.gradients.finite_diff_jvp(f, (1.0,), (1.0,), strategy="backward")
    assert res == {"a": 1.0}
    assert qml.math.allclose(dres["a"], 2.0)

    assert args == [1.0, 1 - 1e-7]


def test_center_strategy():
    """Test the strategy with approx_order 2 is evaluated at the correct points."""

    args = []

    def f(x):
        nonlocal args
        args.append(x)
        return {"key": 4 * x**3}

    res, dres = qml.gradients.finite_diff_jvp(f, (1.0,), (1.0,), approx_order=2, strategy="center")
    assert res == {"key": 4}
    assert qml.math.allclose(dres["key"], 12)

    assert args == [1.0, 1 - 1e-7, 1 + 1e-7]
