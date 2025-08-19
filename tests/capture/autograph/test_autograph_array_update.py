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
    "array_in", [jnp.array([4, 2, 1], dtype=int), jnp.array([3, 2, 1], dtype=int)]
)
def test_div_update(array_in):
    """Tests that the /= operator works with arrays."""

    def fn(x):
        x[0] /= 2
        return x

    ag_fn = run_autograph(fn)
    args = (array_in,)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], jnp.array([array_in[0] / 2, 2, 1], dtype=result[0].dtype))


@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize("index", (slice(0, 1),))
def test_slicing_update(index):
    """Test that slicing indices can be used to update arrays."""

    def fn(x):
        x[index] += 2
        return x

    array_in = jnp.array([4, 2, 1], dtype=int)
    ag_fn = run_autograph(fn)
    args = (array_in,)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], jnp.array([6, 2, 1], dtype=result[0].dtype))


@pytest.mark.usefixtures("enable_disable_plxpr")
def test_static_array_update():
    """Test that static arrays can be updated."""

    def f():
        my_list = [0, 0]
        for i in range(2):
            my_list[i] = i
        my_list[1] += 10
        return my_list

    ag_fn = run_autograph(f)
    ag_fn_jaxpr = make_jaxpr(ag_fn)()
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts)
    assert jnp.array_equal(result[0], [0, 11])
