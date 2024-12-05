# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for exeuction with default qubit 2 independent of any interface."""

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import DefaultQubit


@pytest.mark.parametrize("diff_method", (None, "backprop", qml.gradients.param_shift))
def test_caching(diff_method):
    """Test that cache execute returns the cached result if the same script is executed
    multiple times, both in multiple times in a batch and in separate batches."""
    dev = DefaultQubit()

    qs = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))])

    cache = {}

    with qml.Tracker(dev) as tracker:
        results = qml.execute([qs, qs], dev, cache=cache, diff_method=diff_method)
        results2 = qml.execute([qs, qs], dev, cache=cache, diff_method=diff_method)

    assert len(cache) == 1
    assert cache[qs.hash] == -1.0

    assert list(results) == [-1.0, -1.0]
    assert list(results2) == [-1.0, -1.0]

    assert tracker.totals["batches"] == 1
    assert tracker.totals["executions"] == 1
    assert cache[qs.hash] == -1.0


def test_execute_legacy_device():
    """Test that qml.execute works when passed a legacy device class."""

    dev = qml.devices.DefaultMixed(wires=2)

    tape = qml.tape.QuantumScript([qml.RX(0.1, 0)], [qml.expval(qml.Z(0))])

    res = qml.execute((tape,), dev)

    assert qml.math.allclose(res[0], np.cos(0.1))


def test_gradient_fn_deprecation():
    """Test that gradient_fn has been renamed to diff_method."""

    tape = qml.tape.QuantumScript([qml.RX(qml.numpy.array(1.0), 0)], [qml.expval(qml.Z(0))])
    dev = qml.device("default.qubit")

    with dev.tracker:
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match=r"gradient_fn has been renamed to diff_method"
        ):
            qml.execute((tape,), dev, gradient_fn="adjoint")

    assert dev.tracker.totals["execute_and_derivative_batches"] == 1  # uses adjoint diff
