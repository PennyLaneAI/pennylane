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

import pennylane as qp
from pennylane.devices import DefaultQubit


@pytest.mark.parametrize("diff_method", (None, "backprop", qp.gradients.param_shift))
def test_caching(diff_method):
    """Test that cache execute returns the cached result if the same script is executed
    multiple times, both in multiple times in a batch and in separate batches."""
    dev = DefaultQubit()

    qs = qp.tape.QuantumScript([qp.PauliX(0)], [qp.expval(qp.PauliZ(0))])

    cache = {}

    with qp.Tracker(dev) as tracker:
        results = qp.execute([qs, qs], dev, cache=cache, diff_method=diff_method)
        results2 = qp.execute([qs, qs], dev, cache=cache, diff_method=diff_method)

    assert len(cache) == 1
    assert cache[qs.hash] == -1.0

    assert list(results) == [-1.0, -1.0]
    assert list(results2) == [-1.0, -1.0]

    assert tracker.totals["batches"] == 1
    assert tracker.totals["executions"] == 1
    assert cache[qs.hash] == -1.0


def test_execute_legacy_device():
    """Test that qp.execute works when passed a legacy device class."""

    dev = qp.devices.DefaultMixed(wires=2)

    tape = qp.tape.QuantumScript([qp.RX(0.1, 0)], [qp.expval(qp.Z(0))])

    res = qp.execute((tape,), dev)

    assert qp.math.allclose(res[0], np.cos(0.1))


def test_execution_with_empty_batch():
    """Test that qp.execute can be used with an empty batch."""

    dev = qp.device("default.qubit")
    res = qp.execute((), dev)
    assert res == ()
