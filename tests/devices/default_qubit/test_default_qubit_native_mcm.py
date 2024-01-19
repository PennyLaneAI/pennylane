# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qubit preprocessing."""

import numpy as np

import pennylane as qml


def test_single_mcm_analytic():
    """DefaultQubit uses deferred measurements in analytic mode."""
    dev = qml.device("default.qubit", shots=None)

    @qml.qnode(dev)
    def func1(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return qml.probs(wires=1), qml.probs(op=m0)

    results1 = func1(np.pi / 2, np.pi / 4)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return qml.probs(wires=1), qml.probs(op=m0)

    results2 = func2(np.pi / 2, np.pi / 4)

    assert np.allclose(results1, results2)


def test_single_mcm_shots():
    """Deferred measurements yields equivalent results to DefaultQubit's native mid-circuit measurements."""
    dev = qml.device("default.qubit", shots=10000)

    @qml.qnode(dev)
    def func1(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return qml.probs(wires=1), qml.probs(op=m0)

    results1 = func1(np.pi / 2, np.pi / 4)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return qml.probs(wires=1), qml.probs(op=m0)

    results2 = func2(np.pi / 2, np.pi / 4)

    assert np.allclose(results1, results2, atol=0, rtol=5e-2)
