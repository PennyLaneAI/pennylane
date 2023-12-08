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
import os
import sys
import math
import pytest
import pennylane as qml
from pennylane import numpy as np

from pennylane.devices.default_clifford import _import_stim

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

    assert np.allclose(qn_c(), qn_d(), atol=1e-2)


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
        st1, st2 = qn_c(), qn_d()
        phase = qml.math.divide(
            st1, st2, out=qml.math.zeros_like(st1, dtype=complex), where=st1 != 0
        )[qml.math.nonzero(np.round(st1, 10))]
        assert qml.math.allclose(phase / phase[0], qml.math.ones(len(phase)))

    else:
        tableau = np.array([[0, 1, 1, 0, 0], [1, 0, 1, 1, 1], [0, 0, 0, 1, 0], [1, 0, 0, 1, 1]])
        assert qml.math.allclose(tableau, qn_c())


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
def test_prep_snap_clifford(circuit):
    """Test that the execution of default.clifford is possible and agrees with default.qubit"""
    dev_c = qml.device(
        "default.clifford",
    )
    dev_d = qml.device("default.qubit")

    def circuit_fn():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        circuit()
        qml.Snapshot(tag="cliffy")
        return qml.expval(qml.PauliZ(0))

    qn_c = qml.QNode(circuit_fn, dev_c)
    qn_d = qml.QNode(circuit_fn, dev_d)

    assert np.allclose(qn_c(), qn_d())
    assert np.allclose(qml.grad(qn_c)(), qml.grad(qn_d)())


def test_max_worker_clifford():
    """Test that the execution of default.clifford is possible and agrees with default.qubit"""

    os.environ["OMP_NUM_THREADS"] = "4"

    dev_c = qml.device("default.clifford", max_workers=2)
    dev_d = qml.device("default.qubit", max_workers=2)

    qscript = qml.tape.QuantumScript(
        [qml.Hadamard(wires=[0]), qml.CNOT([0, 1])],
        [qml.expval(qml.PauliZ(0))],
    )
    tapes = tuple([qscript])

    program, config = dev_d.preprocess()
    tapes, _ = program(tapes)
    res_d = dev_d.execute(tapes, config)
    program, config = dev_c.preprocess()
    tapes, _ = program(tapes)
    res_c = dev_c.execute(tapes, config)
    assert np.allclose(res_d, res_c)

    res_q = dev_c.execute(qscript, config)
    assert np.allclose(res_q, res_c[0])

    grad_t = dev_c.compute_derivatives(tapes, config)
    grad_q = dev_c.compute_derivatives(qscript, config)
    assert qml.math.allclose(grad_q, grad_t[0])

    (res_t, grad_t) = dev_c.execute_and_compute_derivatives(tapes, config)
    (res_q, grad_q) = dev_c.execute_and_compute_derivatives(qscript, config)
    assert qml.math.allclose(res_q, res_t[0]) and qml.math.allclose(grad_q, grad_t[0])


def test_tracker():
    """Test that the tracker works for this device"""

    dev_c = qml.device("default.clifford")
    dev_d = qml.device("default.qubit")

    assert dev_c.supports_derivatives()
    assert not dev_c.supports_jvp()
    assert not dev_c.supports_vjp()

    qscript = qml.tape.QuantumScript(
        [qml.Hadamard(wires=[0]), qml.CNOT([0, 1])],
        [qml.expval(qml.PauliZ(0))],
    )
    tapes = tuple([qscript])

    with qml.Tracker(dev_c) as tracker:
        program, config = dev_d.preprocess()
        tapes, _ = program(tapes)
        res_d = dev_d.execute(tapes, config)
        program, config = dev_c.preprocess()
        tapes, _ = program(tapes)
        res_c = dev_c.execute(tapes, config)
        assert np.allclose(res_d, res_c)

        res_q = dev_c.execute(qscript, config)
        assert np.allclose(res_q, res_c[0])

        grad_t = dev_c.compute_derivatives(tapes, config)
        grad_q = dev_c.compute_derivatives(qscript, config)
        assert qml.math.allclose(grad_q, grad_t[0])

        (res_t, grad_t) = dev_c.execute_and_compute_derivatives(tapes, config)
        (res_q, grad_q) = dev_c.execute_and_compute_derivatives(qscript, config)
        assert qml.math.allclose(res_q, res_t[0]) and qml.math.allclose(grad_q, grad_t[0])

    assert tracker.totals


def test_debugger():
    """Test that the debugger works for a simple circuit"""

    class Debugger:
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    dev = qml.device("default.clifford")
    ops = [qml.Snapshot(), qml.Hadamard(wires=0), qml.Snapshot("final_state"), qml.WireCut(0)]
    qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))])

    debugger = Debugger()
    result = dev.simulate(qs, debugger=debugger)

    assert isinstance(result, tuple)
    assert len(result) == 2

    assert list(debugger.snapshots.keys()) == [0, "final_state"]
    assert qml.math.allclose(debugger.snapshots[0], qml.math.array([1, 0]))
    assert qml.math.allclose(
        debugger.snapshots["final_state"], qml.math.array([1, 1]) / qml.math.sqrt(2)
    )

    assert qml.math.allclose(result[0], 1.0)
    assert qml.math.allclose(result[1], 0.0)


def test_shot_error():
    """Test if an NotImplementedError is raised when shots are requested"""

    @qml.qnode(qml.device("default.clifford", shots=1024))
    def circuit_fn():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(
        NotImplementedError,
        match="default.clifford currently doesn't support computation with shots.",
    ):
        circuit_fn()


def test_pauli_sentence_error():
    """Test if an NotImplementedError is raised when taking expectation value of op_math objects"""

    @qml.qnode(qml.device("default.clifford"))
    def circuit_fn():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        return qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(1)))

    with pytest.raises(
        NotImplementedError,
        match="default.clifford doesn't support expectation value calculation with",
    ):
        circuit_fn()


def test_state_error():
    """Test if an ValueError is raised when state is invalid"""

    with pytest.raises(
        ValueError, match="Keyword state only accepts two options: 'tableau' and 'state_vector'"
    ):
        qml.device("default.clifford", state="dm")


def test_fail_import_stim(monkeypatch):
    """Test if an ImportError is raised when stim is requested but not installed"""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "stim", None)
        with pytest.raises(ImportError, match="This feature requires stim"):
            _import_stim()
