# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test autograph support for standard Python item assignment with JAX Arrays."""

import pytest

pytestmark = pytest.mark.capture
jax = pytest.importorskip("jax")

# pylint: disable = wrong-import-position
import jax.numpy as jnp
from jax import make_jaxpr
from jax.core import eval_jaxpr

import pennylane as qml
from pennylane.capture.autograph import run_autograph


@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize(
    "array_in, index, new_value, array_out",
    [
        (jnp.array([1, 2, 3]), 0, 10, jnp.array([10, 2, 3])),
        (jnp.array([1, 2, 3]), -1, 20, jnp.array([1, 2, 20])),
    ],
)
def test_single_integer_indexing(array_in, index, new_value, array_out):
    """Tests single integer indexing like `x[index] = new_value`."""

    def fn(x):
        x[index] = new_value
        return x

    ag_fn = run_autograph(fn)
    args = (array_in,)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], array_out)


@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize(
    "array_in, index, new_value, array_out",
    [
        (jnp.array([1, 2, 3]), slice(0, 2), 10, jnp.array([10, 10, 3])),
        (jnp.array([1, 2, 3]), slice(1, None), 20, jnp.array([1, 20, 20])),
        (jnp.array([1, 2, 3]), slice(None), 5, jnp.array([5, 5, 5])),
        (jnp.array([1, 2, 3, 4, 5]), slice(0, None, 2), 9, jnp.array([9, 2, 9, 4, 9])),
    ],
)
def test_slicing(array_in, index, new_value, array_out):
    """Tests slicing assignment like `x[slice] = new_value`."""

    def fn(x):
        x[index] = new_value
        return x

    ag_fn = run_autograph(fn)
    args = (array_in,)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], array_out)


@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize(
    "array_in, index, new_value, array_out",
    [
        # Slice and set to non singleton value
        (jnp.array([1, 2, 3, 4]), slice(1, 3), jnp.array([99, 88]), jnp.array([1, 99, 88, 4])),
        # Use array for indexing
        (jnp.array([1, 2, 3, 4]), jnp.array([0, 3]), 7, jnp.array([7, 2, 3, 7])),
        # Use boolean mask
        (
            jnp.array([1, 5, 2, 6]),
            jnp.array([False, True, False, True]),
            0,
            jnp.array([1, 0, 2, 0]),
        ),
        # Index a two dimensional array
        (jnp.array([[1, 2], [3, 4]]), 0, 9, jnp.array([[9, 9], [3, 4]])),
        # Index with tuple
        (jnp.array([[1, 2], [3, 4]]), (1, 0), 9, jnp.array([[1, 2], [9, 4]])),
        # 3D array assignment
        (
            jnp.zeros((2, 2, 2)),
            (0, 1, 0),
            5.0,
            jnp.array([[[0.0, 0.0], [5.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        ),
        # Ellipsis to select the last column
        (
            jnp.ones((3, 4)),
            (..., -1),
            99.0,
            jnp.array([[1.0, 1.0, 1.0, 99.0], [1.0, 1.0, 1.0, 99.0], [1.0, 1.0, 1.0, 99.0]]),
        ),
        # Complex numbers
        (
            jnp.array([1 + 1j, 2 + 2j]),
            0,
            3 - 3j,
            jnp.array([3 - 3j, 2 + 2j]),
        ),
    ],
)
def test_non_trivial_indexing(array_in, index, new_value, array_out):
    """Tests non-trivial indexing like boolean masks or arrays."""

    def fn(x):
        x[index] = new_value
        return x

    ag_fn = run_autograph(fn)
    args = (array_in,)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], array_out)


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_non_tracing_assignment():
    """Tests item assignment if the list is not a tracer."""

    def fn():
        x = [0] * 5
        x[2] = 1
        return x

    ag_fn = run_autograph(fn)
    ag_fn_jaxpr = make_jaxpr(ag_fn)()
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts)
    expected = jnp.array([0, 0, 1, 0, 0])
    assert jnp.array_equal(result, expected)


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_while_loop_integration():
    """Tests item assignment within a while loop."""

    def fn():
        x = jnp.zeros(5)
        i = 0
        while i < 5:
            x[i] = i
            i += 1
        return x

    ag_fn = run_autograph(fn)
    ag_fn_jaxpr = make_jaxpr(ag_fn)()
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts)
    expected = jnp.array([0, 1, 2, 3, 4])
    assert jnp.array_equal(result[0], expected)


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_for_loop_integration():
    """Tests item assignment within a for loop."""

    def fn():
        x = jnp.zeros(5)
        for i in range(5):
            x[i] = i
        return x

    ag_fn = run_autograph(fn)
    ag_fn_jaxpr = make_jaxpr(ag_fn)()
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts)
    expected = jnp.array([0, 1, 2, 3, 4])
    assert jnp.array_equal(result[0], expected)


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_qnode_with_python_array_assignment():
    """Test a QNode where a python array argument is modified."""

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(new_val):
        angles = [0.1, 0.2, 0.3]
        angles[0] = new_val
        qml.RX(angles[0], wires=0)
        return qml.expval(qml.Z(0))

    ag_circuit = run_autograph(circuit)
    new_angle = jnp.pi

    # Test forward pass
    res = ag_circuit(new_angle)
    assert jnp.allclose(res, -1.0)

    # Test gradient
    grad = jax.grad(ag_circuit, argnums=0)(new_angle)
    # d/dx cos(x) = -sin(x), at x=pi, -sin(pi) = 0
    assert jnp.allclose(grad, 0.0)


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_qnode_with_jax_array_assignment():
    """Test a QNode where a JAX array argument is modified."""

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(angles, new_val):
        angles[0] = new_val
        qml.RX(angles[0], wires=0)
        return qml.expval(qml.Z(0))

    ag_circuit = run_autograph(circuit)
    angles_in = jnp.array([0.1, 0.2, 0.3])
    new_angle = jnp.pi

    # Test forward pass
    res = ag_circuit(angles_in, new_angle)
    assert jnp.allclose(res, -1.0)

    # Test gradient
    grad = jax.grad(ag_circuit, argnums=1)(angles_in, new_angle)
    # d/dx cos(x) = -sin(x), at x=pi, -sin(pi) = 0
    assert jnp.allclose(grad, 0.0)


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_item_assignment_is_differentiable():
    """Test that item assignment is differentiable."""

    def fn(x, val):
        x[0] = val
        return jnp.sum(x)

    ag_fn = run_autograph(fn)
    array_in = jnp.ones(5)
    value_in = 5.0
    args = (array_in, value_in)
    grad_jaxpr = make_jaxpr(jax.grad(ag_fn, argnums=1))(*args)
    result = eval_jaxpr(grad_jaxpr.jaxpr, grad_jaxpr.consts, *args)

    assert jnp.allclose(result[0], 1.0)


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_shape_mismatch_raises_error():
    """Test that assigning an array of the wrong shape raises an error."""

    def fn(x):
        x[0:2] = jnp.array([1, 2, 3])
        return x

    ag_fn = run_autograph(fn)
    array_in = jnp.zeros(5)
    with pytest.raises(ValueError, match="Incompatible shapes"):
        _ = make_jaxpr(ag_fn)(array_in)
