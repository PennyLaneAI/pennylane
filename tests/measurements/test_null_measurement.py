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
"""Unit tests for the mutual_info module"""
# pylint: disable=use-implicit-booleaness-not-comparison
import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements.null_measurement import NullMeasurement


@pytest.mark.parametrize(
    "method", ("process_density_matrix", "process_state", "process_counts", "process_samples")
)
def test_null_measurement_process_methods(method):
    """Test that all process_methods return `np.nan`"""

    mp = NullMeasurement()
    out = getattr(mp, method)()
    assert np.isnan(out)
    assert out.shape == ()
    assert out.dtype == np.float64


def test_shape_and_dtype_null_measurement():
    """Test that the shape of the shape and dtype of a null measurement are correct."""

    mp = NullMeasurement()

    assert mp.shape() == ()
    assert mp.numeric_type == float
    assert mp._abstract_eval() == ((), float)  # pylint: disable=protected-access


@pytest.mark.jax
def test_integration_jax_jit():
    """Test that execution of the null measurement works with jitting."""
    import jax

    @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
    def c(x):
        qml.RX(x, 0)
        return NullMeasurement()

    r = jax.jit(c)(jax.numpy.array(0.5))
    assert np.isnan(r)
    assert r.shape == ()
    assert r.dtype == np.float64


@pytest.mark.capture
def test_capture():
    """Test that null measurement works with plxpr."""

    @qml.qnode(qml.device("default.qubit", wires=1))
    def c(x):
        qml.RX(x, 0)
        return NullMeasurement()

    out = c(0.5)
    assert np.isnan(out)
    assert out.shape == ()
    assert out.dtype == np.float64
