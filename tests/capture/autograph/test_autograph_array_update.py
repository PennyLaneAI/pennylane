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
"""Test autograph support for standard Python item array updates with operations."""
import pytest

pytestmark = pytest.mark.capture
jax = pytest.importorskip("jax")

# pylint: disable = wrong-import-position
import jax.numpy as jnp
from jax import make_jaxpr
from jax.core import eval_jaxpr

from pennylane.capture.autograph import run_autograph


@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize(
    "op,expected",
    [
        ("add", jnp.array([6, 2, 1])),
        ("sub", jnp.array([2, 2, 1])),
        ("mult", jnp.array([8, 2, 1])),
        ("pow", jnp.array([16, 2, 1])),
    ],
)
def test_update_array_with_operations(op, expected):
    """Tests updating an array in-place with various operations."""

    def fn(x):
        if op == "add":
            x[0] += 2
        elif op == "sub":
            x[0] -= 2
        elif op == "mult":
            x[0] *= 2
        elif op == "pow":
            x[0] **= 2
        else:
            x[0] = -1
        return x

    array_in = jnp.array([4, 2, 1], dtype=int)
    ag_fn = run_autograph(fn)
    args = (array_in,)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], expected)


@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize(
    "array_in",
    [
        # Set-up array to get int
        jnp.array([4, 2, 1], dtype=int),
        # Set-up array to get a float
        jnp.array([3, 2, 1], dtype=int),
    ],
)
def test_update_array_with_div_operation(array_in):
    """Tests that the /= operator works with arrays."""

    def fn(x):
        x[0] /= 2
        return x

    ag_fn = run_autograph(fn)
    args = (array_in,)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], jnp.array([array_in[0] / 2, 2, 1], dtype=result[0].dtype))


def test_slicing_update():
    """Test that slicing indices can be used to update arrays."""

    def fn(x):
        # For some reason slicing is allowed if you don't directly use an object
        index = slice(0, 3, 1)
        x[index] += 2
        x[0:3:1] += 2
        return x

    array_in = jnp.array([4, 2, 1], dtype=int)
    ag_fn = run_autograph(fn)
    args = (array_in,)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], jnp.array([8, 6, 5], dtype=result[0].dtype))


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_static_array_update():
    """Test that static arrays can be updated."""

    def f():
        my_list = [0, 1]
        my_list[1] += 10
        return my_list

    ag_fn = run_autograph(f)
    ag_fn_jaxpr = make_jaxpr(ag_fn)()
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts)
    assert jnp.array_equal(result, [0, 11])


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_dynamic_index():
    """Tests that a dynamic index can be used."""

    def fn(x, i):
        x[i] += 2
        return x

    ag_fn = run_autograph(fn)
    args = (jnp.array([4, 2, 1], dtype=int), 0)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], jnp.array([6, 2, 1], dtype=result[0].dtype))


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestUnsupportedArrayUpdates:
    """Test that errors are thrown for unsupported cases"""

    def test_modulo_op_update(self):
        """Tests that an unsupported operator works with arrays."""

        def fn(x):
            x[0] %= 2
            return x

        ag_fn = run_autograph(fn)
        args = (jnp.array([4, 2, 1], dtype=int),)
        with pytest.raises(TypeError, match="JAX arrays are immutable"):
            _ = make_jaxpr(ag_fn)(*args)

    def test_update_with_slice_index(self):
        """Tests that an unsupported operator works with arrays."""

        def fn(x):
            x[slice(1, 2)] += 2
            return x

        ag_fn = run_autograph(fn)
        args = (jnp.array([4, 2, 1], dtype=int),)
        with pytest.raises(TypeError, match="JAX arrays are immutable"):
            _ = make_jaxpr(ag_fn)(*args)

    def test_multi_dimensional_index(self):
        """Test that TypeError is raised when using multi-dim indexing."""

        def fn(x):
            x[0, 1] += 5
            return x

        args = (jnp.array([[1, 2], [3, 4]]),)
        ag_fn = run_autograph(fn)
        with pytest.raises(TypeError, match="JAX arrays are immutable"):
            _ = make_jaxpr(ag_fn)(*args)
