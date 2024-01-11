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
import pytest
import numpy as np
import pennylane as qml

from pennylane.devices.default_clifford import _import_stim

stim = pytest.importorskip("stim")

pytestmark = pytest.mark.external


def circuit_1():
    """Circuit 1 with Clifford gates."""
    qml.GlobalPhase(np.pi)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=[1])
    qml.Barrier()
    qml.ISWAP(wires=[0, 1])
    qml.Hadamard(wires=[0])


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
    """Test that execution of default.clifford is possible and agrees with default.qubit."""
    dev_c = qml.device("default.clifford")
    dev_q = qml.device("default.qubit")

    def circuit_fn():
        circuit()
        return qml.expval(expec_op)

    qnode_clfrd = qml.QNode(circuit_fn, dev_c)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)
    assert np.allclose(qnode_clfrd(), qnode_qubit())


@pytest.mark.parametrize("circuit", [circuit_1])
def test_grad_clifford(circuit):
    """Test that gradients of default.clifford agrees with default.qubit."""
    dev_c = qml.device("default.clifford")
    dev_q = qml.device("default.qubit")
    assert dev_c.supports_derivatives()
    assert not dev_c.supports_jvp()
    assert not dev_c.supports_vjp()

    def circuit_fn():
        circuit()
        return qml.expval(qml.PauliZ(0))

    qnode_clfrd = qml.QNode(circuit_fn, dev_c)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)
    res_q, grad_q = qnode_qubit(), qml.grad(qnode_qubit)()
    assert np.allclose(qml.grad(qnode_clfrd)(), grad_q)

    conf_c = dev_c.preprocess()[1]
    tape_c, tape_q = qnode_clfrd.tape, qnode_qubit.tape
    assert tape_c.operations == tape_q.operations

    grad_c = dev_c.compute_derivatives(tape_c, conf_c)
    assert qml.math.allclose(grad_c, grad_q)

    (res_c, grad_c) = dev_c.execute_and_compute_derivatives(tape_c, conf_c)
    assert qml.math.allclose(res_q, res_c) and qml.math.allclose(grad_q, grad_c)


@pytest.mark.parametrize("circuit", [circuit_1])
@pytest.mark.parametrize("tableau", [True, False])
def test_state_clifford(circuit, tableau):
    """Test that state computation with default.clifford is possible and agrees with default.qubit."""
    dev_c = qml.device("default.clifford", tableau=tableau, wires=2)
    dev_q = qml.device("default.qubit", wires=2)

    def circuit_fn():
        circuit()
        return qml.state()

    # Tableau for the circuit define above
    circuit_tableau = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
        ]
    )

    qnode_clfrd = qml.QNode(circuit_fn, dev_c)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)

    if not tableau:
        st1, st2 = qnode_clfrd(), qnode_qubit()
        phase = qml.math.divide(
            st1, st2, out=qml.math.zeros_like(st1, dtype=complex), where=st1 != 0
        )[qml.math.nonzero(np.round(st1, 10))]
        if not qml.math.allclose(phase, 0.0):
            assert qml.math.allclose(phase / phase[0], qml.math.ones(len(phase)))
    else:
        assert qml.math.allclose(circuit_tableau, qnode_clfrd())

    @qml.qnode(dev_c)
    def circuit_empty():
        return qml.state()

    # Tableau for the circuit define above
    circuit_tableau = np.array([[1, 0, 0], [0, 1, 0]])
    if not tableau:
        assert qml.math.allclose(circuit_empty(), qml.math.array([1.0, 0.0, 0.0, 0.0]))
    else:
        assert qml.math.allclose(circuit_empty(), circuit_tableau)


@pytest.mark.parametrize("circuit", [circuit_1])
@pytest.mark.parametrize(
    "meas_type",
    ["dm", "purity"],
)
def test_meas_clifford(circuit, meas_type):
    """Test that measurements with default.clifford is possible and agrees with default.qubit."""
    dev_c = qml.device("default.clifford")
    dev_q = qml.device("default.qubit")

    def circuit_fn():
        circuit()
        return qml.density_matrix([0, 1]) if meas_type == "dm" else qml.purity([0, 1])

    qnode_clfrd = qml.QNode(circuit_fn, dev_c)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit())


@pytest.mark.parametrize("circuit", [circuit_1])
def test_prep_snap_clifford(circuit):
    """Test that state preparation with default.clifford is possible and agrees with default.qubit."""
    dev_c = qml.device("default.clifford", check_clifford=False)
    dev_q = qml.device("default.qubit")

    def circuit_fn():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        circuit()
        qml.Snapshot(tag="cliffy")
        return qml.expval(qml.PauliZ(0))

    qnode_clfrd = qml.QNode(circuit_fn, dev_c)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit())
    assert np.allclose(qml.grad(qnode_clfrd)(), qml.grad(qnode_qubit)())


@pytest.mark.parametrize(
    "pl_op,stim_op",
    [
        (qml.PauliX(0), ("X", [0])),
        (qml.CNOT(["a", "b"]), ("CNOT", ["a", "b"])),
        (qml.GlobalPhase(1.0), (None, [])),
        (qml.Snapshot(), (None, [])),
    ],
)
def test_pl_to_stim(pl_op, stim_op):
    """Test that the PennyLane operation get converted to Stim operation"""
    dev_c = qml.device("default.clifford")
    op, wires = dev_c.pl_to_stim(pl_op)
    assert op == stim_op[0]
    assert wires == qml.wires.Wires(stim_op[1])


def test_max_worker_clifford():
    """Test that the execution of multiple tapes is possible with multiprocessing on this device."""

    os.environ["OMP_NUM_THREADS"] = "4"

    dev_c = qml.device("default.clifford", max_workers=2)
    dev_q = qml.device("default.qubit", max_workers=2)

    qscript = qml.tape.QuantumScript(
        [qml.Hadamard(wires=[0]), qml.CNOT(wires=[0, 1])],
        [qml.expval(qml.PauliX(0))],
    )
    tapes = (qscript, qscript)

    program, conf_d = dev_c.preprocess()
    tapes_clrfd, _ = program(tapes)
    res_c = dev_c.execute(tapes, conf_d)
    program, conf_q = dev_q.preprocess()
    tapes_qubit, _ = program(tapes)
    res_q = dev_q.execute(tapes, conf_q)
    assert np.allclose(res_q, res_c)

    grad_c = dev_c.compute_derivatives(tapes_clrfd, conf_q)
    grad_q = dev_q.compute_derivatives(tapes_qubit, conf_d)
    assert qml.math.allclose(grad_q, grad_c)

    (res_c, grad_c) = dev_c.execute_and_compute_derivatives(tapes_clrfd, conf_q)
    assert qml.math.allclose(res_q, res_c) and qml.math.allclose(grad_q, grad_c)


def test_tracker():
    """Test that the tracker works for this device."""

    dev_c = qml.device("default.clifford")
    dev_q = qml.device("default.qubit")

    qscript = qml.tape.QuantumScript(
        [qml.Hadamard(wires=[0]), qml.CNOT([0, 1])],
        [qml.expval(qml.PauliZ(0))],
    )
    tapes = tuple([qscript])

    with qml.Tracker(dev_c) as tracker:
        program, conf_d = dev_c.preprocess()
        tapes_clrfd, _ = program(tapes)
        res_c = dev_c.execute(tapes, conf_d)
        program, conf_q = dev_q.preprocess()
        tapes_qubit, _ = program(tapes)
        res_q = dev_q.execute(tapes, conf_q)
        assert np.allclose(res_q, res_c)

        res_s = dev_c.execute(qscript, conf_d)
        assert np.allclose(res_c, res_s)

        grad_c = dev_c.compute_derivatives(tapes_clrfd, conf_q)
        grad_q = dev_q.compute_derivatives(tapes_qubit, conf_d)
        assert qml.math.allclose(grad_q, grad_c)

        (res_c, grad_c) = dev_c.execute_and_compute_derivatives(tapes_clrfd, conf_q)
        assert qml.math.allclose(res_q, res_c) and qml.math.allclose(grad_q, grad_c)

    assert tracker.totals == {
        "batches": 3,
        "simulations": 3,
        "executions": 3,
        "derivative_batches": 2,
        "derivatives": 2,
    }
    assert np.allclose(tracker.history.pop("results")[0], 0.0)
    assert tracker.history.pop("resources")[0] == qml.resource.Resources(
        num_wires=2,
        num_gates=2,
        gate_types={"Hadamard": 1, "CNOT": 1},
        gate_sizes={1: 1, 2: 1},
        depth=2,
    )
    assert tracker.history == {
        "batches": [1, 1, 1],
        "simulations": [1, 1, 1],
        "executions": [1, 1, 1],
        "derivative_batches": [1, 1],
        "derivatives": [1, 1],
    }


def test_debugger():
    """Test that the debugger works for a simple circuit."""

    # pylint: disable=too-few-public-methods
    class Debugger:
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    dev = qml.device("default.clifford")
    ops = [qml.Snapshot(), qml.Hadamard(wires=0), qml.Snapshot("final_state")]
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

    @qml.qnode(dev)
    def circuit():
        for op in ops:
            qml.apply(op)
        return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]

    assert getattr(dev, "_debugger") is None

    result = qml.snapshots(circuit)()
    expected = {
        0: qml.math.array([1, 0]),
        "final_state": qml.math.array([1, 1]) / qml.math.sqrt(2),
        "execution_results": [qml.math.array(1.0), qml.math.array(0.0)],
    }

    assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
    assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))


def test_shot_error():
    """Test if an NotImplementedError is raised when shots are requested."""

    @qml.qnode(qml.device("default.clifford", shots=1024))
    def circuit_fn():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(
        NotImplementedError,
        match="default.clifford currently doesn't support computation with shots.",
    ):
        circuit_fn()


def test_meas_error():
    """Test if an NotImplementedError is raised when taking expectation value of op_math objects."""

    @qml.qnode(qml.device("default.clifford"))
    def circuit_exp():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        return qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(1)))

    with pytest.raises(
        NotImplementedError,
        match="default.clifford doesn't support expectation value calculation with",
    ):
        circuit_exp()

    @qml.qnode(qml.device("default.clifford"))
    def circuit_ent():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        return qml.vn_entropy(wires=[0, 1])

    with pytest.raises(NotImplementedError, match="default.clifford doesn't support the"):
        circuit_ent()

    @qml.qnode(qml.device("default.clifford"))
    def circuit_snap():
        qml.Snapshot(measurement=qml.expval(qml.PauliZ(0)))
        return qml.state()

    with pytest.raises(
        ValueError,
        match="default.clifford does not support arbitrary measurements of a state with snapshots.",
    ):
        qml.snapshots(circuit_snap)()


def test_purity_error_not_all_wires():
    """Test that a NotImplementedError is raised when purity of not all wires is measured."""

    @qml.qnode(qml.device("default.clifford"))
    def circuit():
        qml.PauliX(0)
        qml.CNOT([0, 1])
        return qml.purity([0])

    with pytest.raises(
        NotImplementedError,
        match="default.clifford doesn't support measuring the purity of a subset of wires at the moment",
    ):
        circuit()


@pytest.mark.parametrize("check", [True, False])
def test_clifford_error(check):
    """Test if an QuantumFunctionError is raised when one of the operations is not Clifford."""

    dev = qml.device("default.clifford", tableau="tableau", check_clifford=check)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=[0])
        qml.PauliX(wires=[0])
        qml.RX(1.0, wires=[0])
        return qml.state()

    with pytest.raises(
        qml.DeviceError,
        match=r"Operator RX\(1.0, wires=\[0\]\) not supported on default.clifford and does not provide a decomposition",
    ):
        circuit()


def test_fail_import_stim(monkeypatch):
    """Test if an ImportError is raised when stim is requested but not installed."""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "stim", None)
        with pytest.raises(ImportError, match="This feature requires stim"):
            _import_stim()
