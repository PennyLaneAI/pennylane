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
"""
This module contains the tests for the clifford simulator based on stim
"""
import sys
import math
import pytest
import pennylane as qml
from pennylane import numpy as np

stim = pytest.importorskip("stim")

INVSQ2 = 1 / math.sqrt(2)
PI = math.pi


def circuit_1():
    """Circuit 1 with Clifford gates"""
    qml.GlobalPhase(PI)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=[1])
    qml.ISWAP(wires=[0, 1])
    qml.Hadamard(wires=[0])
    qml.WireCut(wires=[1])
    qml.Snapshot(tag="cliffy")


@pytest.mark.parametrize("circuit", [circuit_1])
@pytest.mark.parametrize(
    "expec_op",
    [
        qml.PauliZ(0),
        qml.PauliZ(0) @ qml.PauliX(1),
        qml.Hamiltonian([0.42], [qml.PauliZ(0) @ qml.PauliX(1)]),
    ],
)
def test_expectation_clifford(circuit, expec_op):
    """Test that the execution of default.clifford is possible and agrees with default.qubit"""
    dev_c = qml.device("default.clifford")
    dev_d = qml.device("default.qubit")

    def circuit_fn():
        circuit()
        return qml.expval(expec_op)

    qn_c = qml.QNode(circuit_fn, dev_c)
    qn_d = qml.QNode(circuit_fn, dev_d)

    assert np.allclose(qn_c(), qn_d())
    assert np.allclose(qml.grad(qn_c)(), qml.grad(qn_d)())


@pytest.mark.parametrize("circuit", [circuit_1])
@pytest.mark.parametrize("state", ["tableau", "state_vector"])
def test_state_clifford(circuit, state):
    """Test that the execution of default.clifford is possible and agrees with default.qubit"""
    dev_c = qml.device("default.clifford", state=state)
    dev_d = qml.device("default.qubit")

    def circuit_fn():
        circuit()
        return qml.state()

    qn_c = qml.QNode(circuit_fn, dev_c)
    qn_d = qml.QNode(circuit_fn, dev_d)

    if state == "state_vector":
        st1, st2 = qn_c(), qn_c()
        phase = qml.math.divide(
            st1, st2, out=qml.math.zeros_like(st1, dtype=complex), where=st1 != 0
        )[qml.math.nonzero(np.round(st1, 10))]
        assert qml.math.allclose(phase / phase[0], qml.math.ones(len(phase)))

    else:
        tableau = np.array([[0, 1, 1, 0, 0], [1, 0, 1, 1, 1], [0, 0, 0, 1, 0], [1, 0, 0, 1, 1]])
        assert qml.math.allclose(tableau, qn_c())

    assert np.allclose(qml.grad(qn_c)(), qml.grad(qn_d)())


@pytest.mark.parametrize("circuit", [circuit_1])
@pytest.mark.parametrize(
    "meas_type",
    ["dm", "purity"],
)
def test_meas_clifford(circuit, meas_type):
    """Test that the execution of default.clifford is possible and agrees with default.qubit"""
    dev_c = qml.device("default.clifford")
    dev_d = qml.device("default.qubit")

    def circuit_fn():
        circuit()
        return qml.density_matrix([0, 1]) if meas_type == "dm" else qml.purity([0, 1])

    qn_c = qml.QNode(circuit_fn, dev_c)
    qn_d = qml.QNode(circuit_fn, dev_d)

    assert np.allclose(qn_c(), qn_d())

@pytest.mark.parametrize("circuit", [circuit_1])
def test_prep_clifford(circuit):
    """Test that the execution of default.clifford is possible and agrees with default.qubit"""
    dev_c = qml.device("default.clifford")
    dev_d = qml.device("default.qubit")

    def circuit_fn():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        circuit()
        return qml.expval(qml.PauliZ(0))

    qn_c = qml.QNode(circuit_fn, dev_c)
    qn_d = qml.QNode(circuit_fn, dev_d)

    assert np.allclose(qn_c(), qn_d())
    assert np.allclose(qml.grad(qn_c)(), qml.grad(qn_d)())


def test_fail_import_stim(monkeypatch):
    """Test if an ImportError is raised when stim is requested but not installed"""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "stim", None)
        with pytest.raises(ImportError, match="This feature requires stim"):
            qml.device("default.clifford")
