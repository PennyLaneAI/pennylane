# Copyright 2026 Xanadu Quantum Technologies Inc.
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
Tests for estimation_array
"""

import numpy as np
import pytest

import pennylane as qp

pytestmark = pytest.mark.capture

jax = pytest.importorskip("jax")
jnp = jax.numpy
from pennylane.capture.primitives import (  # pylint: disable=wrong-import-position
    estimation_array_prim,
)


def test_error_without_capture():
    """Test a ``NotImplementedError`` is raised if capture is not turned on."""
    qp.capture.disable()

    with pytest.raises(NotImplementedError, match="estimation_array requires program capture"):
        qp.capture.estimation_array((), float)


def test_error_if_execute():
    """Test that a NotImplementedError is raised if we try and execute estimation_array."""

    with pytest.raises(
        NotImplementedError, match="estimation_arrays can only be produced for abstract evaluation"
    ):
        qp.capture.estimation_array((), float)


@pytest.mark.parametrize("bad_dimension", (..., -1, 0, 3.0))
def test_error_if_bad_dimesion(bad_dimension):
    """Test that a ValueError is raised if a dimension is invalid."""

    def f():
        qp.capture.estimation_array((2, bad_dimension), float)

    with pytest.raises(ValueError, match="must be integers greater than zero"):
        jax.make_jaxpr(f)()


@pytest.mark.parametrize("shape", [(), (4, 3), (5, 2, 1)])
@pytest.mark.parametrize(
    "dtype, converted_dtype",
    [
        (float, jnp.float64),
        (jnp.float32, jnp.float32),
        (bool, jnp.bool),
        (np.complex128, jnp.complex128),
    ],
)
def test_capturing_estimation_array(shape, dtype, converted_dtype):
    """Test capturing estimation_array's into jaxpr."""

    def f():
        return qp.capture.estimation_array(shape, dtype)

    jaxpr = jax.make_jaxpr(f)()

    assert jaxpr.eqns[0].primitive == estimation_array_prim
    assert jaxpr.eqns[0].params["shape"] == shape
    assert jaxpr.eqns[0].params["dtype"] == converted_dtype

    assert jaxpr.eqns[0].outvars[0].aval.shape == shape
    assert jaxpr.eqns[0].outvars[0].aval.dtype == converted_dtype
