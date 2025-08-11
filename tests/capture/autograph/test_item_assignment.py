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

"""Test autograph support for standard Python item assignment with JAX Arrays."""

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
    ("array_in", "index", "new_value", "array_out"),
    (
        # --- Single Integer Indexing ---
        (jnp.array([1, 2, 3]), 0, 10, jnp.array([10, 2, 3])),
        (jnp.array([1, 2, 3]), -1, 20, jnp.array([1, 2, 20])),
        # --- Slicing ---
        (jnp.array([1, 2, 3]), slice(0, 2), 10, jnp.array([10, 10, 3])),
        (jnp.array([1, 2, 3]), slice(1, None), 20, jnp.array([1, 20, 20])),
        (jnp.array([1, 2, 3]), slice(None), 5, jnp.array([5, 5, 5])),
        (jnp.array([1, 2, 3, 4, 5]), slice(0, None, 2), 9, jnp.array([9, 2, 9, 4, 9])),
        (jnp.array([1, 2, 3, 4]), slice(1, 3), jnp.array([99, 88]), jnp.array([1, 99, 88, 4])),
        # --- Non-Trivial Indexing ---
        (jnp.array([1, 2, 3, 4]), jnp.array([0, 3]), 7, jnp.array([7, 2, 3, 7])),
        (
            jnp.array([1, 5, 2, 6]),
            jnp.array([False, True, False, True]),
            0,
            jnp.array([1, 0, 2, 0]),
        ),
        (jnp.array([[1, 2], [3, 4]]), 0, 9, jnp.array([[9, 9], [3, 4]])),
        (jnp.array([[1, 2], [3, 4]]), (1, 0), 9, jnp.array([[1, 2], [9, 4]])),
    ),
)
def test_jaxpr_generation(array_in, index, new_value, array_out):
    """Test JAXPR generation for various pythonic index assignments."""

    def fn(x):
        x[index] = new_value
        return x

    ag_fn = run_autograph(fn)
    args = (array_in,)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)

    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)
    assert jnp.array_equal(result[0], array_out)
