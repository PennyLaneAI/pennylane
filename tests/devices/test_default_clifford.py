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

import numpy as np
import pytest
import scipy as sp
from dummy_debugger import Debugger

import pennylane as qml
from pennylane.devices.default_clifford import _pl_op_to_stim
from pennylane.exceptions import DeviceError, QuantumFunctionError

stim = pytest.importorskip("stim")

pytestmark = pytest.mark.external


# pylint: disable=protected-access
def test_applied_modifiers():
    """Test that default qubit has the `single_tape_support` and `simulator_tracking`
    modifiers applied.
    """
    dev = qml.device("default.clifford")
    assert dev._applied_modifiers == [
        qml.devices.modifiers.single_tape_support,
        qml.devices.modifiers.simulator_tracking,
    ]


def circuit_1():
    """Circuit 1 with Clifford gates."""
    qml.GlobalPhase(np.pi)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=[1])
    qml.Barrier()
    qml.ISWAP(wires=[0, 1])
    qml.Hadamard(wires=[0])


def circuit_2():
    """Circuit 2 with error channels."""
    qml.GlobalPhase(np.pi)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=[1])
    qml.PauliError("YZ", 0.2, wires=[0, 1])
    qml.Barrier()
    qml.BitFlip(0.01, wires=[0])
    qml.PhaseFlip(0.01, wires=[0])
    qml.Barrier()
    qml.Hadamard(wires=[0])
    qml.DepolarizingChannel(0.2, wires=[0])


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


def test_execution_with_no_execution_config():
    """Test execution of a tape with no execution config."""
    dev = qml.device("default.clifford")
    qs = qml.tape.QuantumScript([qml.X(0)], [qml.expval(qml.PauliZ(0))])
    result = dev.execute(qs)
    assert qml.math.allclose(result, -1.0)


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

    dev_c = qml.device("default.clifford", tableau=tableau, wires=1)

    @qml.qnode(dev_c)
    def circuit_empty():
        return qml.state()

    # Tableau for the circuit define above
    circuit_tableau = np.array([[1, 0, 0], [0, 1, 0]])
    if not tableau:
        assert qml.math.allclose(circuit_empty(), qml.math.array([1.0, 0.0]))
    else:
        assert qml.math.allclose(circuit_empty(), circuit_tableau)


@pytest.mark.parametrize(
    "meas_op",
    [
        qml.density_matrix([1]),
        qml.density_matrix([1, 2]),
        qml.density_matrix([0, 2]),
        qml.purity([1]),
        qml.purity([0, 2]),
        qml.purity([1, 2]),
        qml.purity([0, 1, 2]),
        qml.vn_entropy([1]),
        qml.vn_entropy([0, 2]),
        qml.vn_entropy([0, 1, 2]),
        qml.mutual_info([0], [1]),
        qml.mutual_info([1], [0, 2]),
        qml.mutual_info([0], [1, 2]),
    ],
)
def test_meas_qinfo_clifford(meas_op):
    """Test that quantum information measurements with `default.clifford` is possible
    and agrees with `default.qubit`."""
    dev_c = qml.device("default.clifford")
    dev_q = qml.device("default.qubit")

    def circuit_fn():
        circuit_1()
        qml.PauliX(wires=[2])
        qml.Hadamard(wires=[2])
        qml.CNOT(wires=[1, 2])
        return qml.apply(meas_op)

    qnode_clfrd = qml.QNode(circuit_fn, dev_c)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit())


@pytest.mark.parametrize("shots", [None, 1_000_000])
@pytest.mark.parametrize(
    "ops",
    [
        qml.PauliX(0),
        qml.PauliX(0) @ qml.PauliY(1),
        qml.Hamiltonian([3.0, 2.0], [qml.PauliX(0), qml.PauliY(1)]),
        qml.Hamiltonian([0.42], [qml.PauliZ(0) @ qml.PauliX(1)]),
        qml.sum(qml.PauliX(0), qml.PauliY(1)),
        qml.prod(qml.PauliX(0), qml.PauliY(1)),
        qml.s_prod(3.0, qml.PauliX(0)),
        qml.sum(qml.PauliX(0), qml.s_prod(2.0, qml.PauliY(1))),
        qml.prod(qml.sum(qml.PauliZ(0), qml.PauliY(1)), qml.s_prod(3, qml.PauliX(2))),
        qml.Hermitian(qml.sum(qml.PauliX(0), qml.PauliY(1)).matrix(), [0, 1]),
        qml.Projector([1, 0], [0, 1]),
    ],
)
def test_meas_expval(shots, ops, seed):
    """Test that expectation value measurements with `default.clifford` is possible
    and agrees with `default.qubit`."""
    dev_c = qml.device("default.clifford", seed=seed)
    dev_q = qml.device("default.qubit")

    def circuit_fn():
        circuit_1()
        return qml.expval(ops)

    qnode_clfrd = qml.set_shots(qml.QNode(circuit_fn, dev_c), shots=shots)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit(), atol=1e-2 if shots else 1e-8)


@pytest.mark.parametrize("shots", [None, int(2e6)])
@pytest.mark.parametrize(
    "ops",
    [
        qml.PauliZ(0),
        qml.PauliZ(0) @ qml.PauliY(1),
        qml.sum(qml.PauliZ(0), qml.PauliY(1)),
        qml.prod(qml.PauliZ(0), qml.PauliY(1)),
        qml.s_prod(3.0, qml.PauliX(0)),
        qml.sum(qml.PauliZ(0), qml.s_prod(2.0, qml.PauliY(1))),
    ],
)
def test_meas_var(shots, ops, seed):
    """Test that variance measurements with `default.clifford` is possible
    and agrees with `default.qubit`."""
    dev_c = qml.device("default.clifford", seed=seed)
    dev_q = qml.device("default.qubit")

    def circuit_fn():
        circuit_1()
        return qml.var(ops)

    qnode_clfrd = qml.set_shots(qml.QNode(circuit_fn, dev_c), shots=shots)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit(), atol=1e-2 if shots else 1e-8)


@pytest.mark.parametrize("circuit", [circuit_1, circuit_2])
@pytest.mark.parametrize("shots", [1024, 8192, 16384])
def test_meas_samples(circuit, shots):
    """Test if samples are returned with shots given in the clifford device."""

    @qml.qnode(qml.device("default.clifford"), shots=shots)
    def circuit_fn():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        circuit()
        return [
            qml.sample(wires=[1]),
            qml.sample(qml.PauliZ(0)),
            qml.sample(qml.PauliX(0) @ qml.PauliY(1)),
        ]

    samples = circuit_fn()
    assert len(samples) == 3
    assert qml.math.shape(samples[0]) == (shots,)
    assert qml.math.shape(samples[1]) == (shots,)
    assert qml.math.shape(samples[2]) == (shots,)


@pytest.mark.parametrize("tableau", [True, False])
@pytest.mark.parametrize("shots", [None, 50000])
@pytest.mark.parametrize(
    "ops",
    [
        None,
        qml.PauliY(0),
        qml.PauliX(0) @ qml.PauliY(1),
        qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2),
        qml.Projector([0, 1], wires=[0, 1]),
    ],
)
def test_meas_probs(tableau, shots, ops, seed):
    """Test if probabilities are returned in the clifford device."""

    dev_c = qml.device("default.clifford", tableau=tableau, seed=seed)
    dev_q = qml.device("default.qubit")

    def circuit_fn():
        for wire in range(3):
            qml.PauliX(wire)
            qml.PauliY(wire)
            qml.PauliZ(wire)
        return qml.probs(op=ops) if ops else qml.probs(wires=[0, 1])

    qnode_clfrd = qml.set_shots(qml.QNode(circuit_fn, dev_c), shots=shots)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)

    gotten_probs, target_probs = qnode_clfrd(), qnode_qubit()

    assert qml.math.allclose(gotten_probs, target_probs, atol=5e-2 if shots else 1e-8)


def test_meas_probs_large(seed):
    """Test if probabilities are returned in the clifford device with target basis states"""

    def single_op(idx):
        return [qml.PauliX, qml.PauliY, qml.Hadamard, qml.PauliZ][idx]

    def circuit_fn2(meas):
        for wire in range(16):
            single_op(wire % 4)(wires=wire)
            qml.CNOT([wire, wire + 1])
        return qml.apply(meas)

    dev_c = qml.device("default.clifford", seed=seed)
    qnode_clfrd = qml.QNode(circuit_fn2, dev_c)

    meas1 = qml.probs(op=qml.Projector([1, 1, 0], wires=[0, 6, 14]))
    assert qnode_clfrd(meas1).shape == (8,)
    assert qml.math.allclose(qnode_clfrd(meas1), [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25])

    for basis_state in np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1]]):
        meas_b = qml.expval(op=qml.Projector(basis_state, wires=[0, 6, 14]))
        assert qnode_clfrd(meas_b) == [0.0, 0.25][basis_state[0]]


@pytest.mark.parametrize("shots", [1024, 4096])
@pytest.mark.parametrize(
    "ops",
    [None, qml.PauliY(0), qml.PauliX(0) @ qml.PauliY(1)],
)
def test_meas_counts(shots, ops, seed):
    """Test if counts are returned with shots given in the clifford device."""

    dev_c = qml.device("default.clifford", seed=seed)
    dev_q = qml.device("default.qubit", seed=seed)

    def circuit_fn():
        qml.PauliX(0)
        qml.PauliX(1)
        return qml.counts(op=ops) if ops else qml.counts(wires=[0, 1])

    qnode_clfrd = qml.set_shots(qml.QNode(circuit_fn, dev_c), shots=shots)
    qnode_qubit = qml.set_shots(qml.QNode(circuit_fn, dev_q), shots=shots)

    counts_clfrd = qnode_clfrd()
    counts_qubit = qnode_qubit()

    assert list(counts_clfrd.keys()) == list(counts_qubit.keys())

    for k1, k2 in zip(counts_clfrd, counts_clfrd):
        assert qml.math.abs(counts_clfrd[k1] - counts_qubit[k2]) / shots < 0.1  # 10% threshold


@pytest.mark.parametrize("shots", [1024, 10240])
@pytest.mark.parametrize(
    "ops",
    [
        qml.PauliZ(0),
        qml.PauliZ(0) @ qml.PauliY(1),
    ],
)
def test_meas_classical_shadows(shots, ops, seed):
    """Test if classical shadows measurements are returned with shots
    given in the clifford device."""

    def circuit():
        qml.PauliX(0)
        qml.PauliX(1)
        qml.Hadamard(0)
        qml.CNOT((0, 1))

    dev_c = qml.device("default.clifford")
    dev_q = qml.device("default.qubit")

    def circuit_shadow():
        circuit()
        return qml.classical_shadow(wires=[0, 1], seed=seed)

    qnode_clfrd_shadow = qml.set_shots(qml.QNode(circuit_shadow, dev_c), shots=shots)
    qnode_qubit_shadow = qml.set_shots(qml.QNode(circuit_shadow, dev_q), shots=shots)

    bits1, recipes1 = qnode_clfrd_shadow()
    bits2, recipes2 = qnode_qubit_shadow()

    assert bits1.shape == bits2.shape
    assert recipes1.shape == recipes2.shape
    assert np.allclose(np.sort(np.unique(recipes1.reshape(shots * 2, 1))), np.array([0, 1, 2]))

    def circuit_expval():
        circuit()
        return qml.shadow_expval(ops, seed=seed)

    qnode_clfrd_expval = qml.set_shots(qml.QNode(circuit_expval, dev_c), shots=shots)
    expval = qnode_clfrd_expval()

    assert -1.0 <= expval <= 1.0


@pytest.mark.parametrize("circuit", [circuit_1])
def test_prep_snap_clifford(circuit):
    """Test that state preparation with default.clifford is possible and agrees with default.qubit."""
    dev_c = qml.device("default.clifford", wires=2, check_clifford=False)
    dev_q = qml.device("default.qubit", wires=2)

    def circuit_fn():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        circuit()
        qml.Snapshot(tag="cliffy")
        return qml.expval(qml.PauliZ(0))

    qnode_clfrd = qml.QNode(circuit_fn, dev_c)
    qnode_qubit = qml.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit())


@pytest.mark.parametrize(
    "pl_op,stim_op",
    [
        (qml.PauliX(0), ("X", [0])),
        (qml.CNOT(["a", "b"]), ("CNOT", ["a", "b"])),
        (qml.GlobalPhase(1.0), (None, [])),
        (qml.Snapshot(), (None, [])),
        (qml.DepolarizingChannel(0.2, [2]), ("DEPOLARIZE1(0.2)", [2])),
        (qml.PauliError("XYZ", 0.2, [0, 1, 2]), ("CORRELATED_ERROR(0.2)", ["X0", "Y1", "Z2"])),
    ],
)
def test_pl_to_stim(pl_op, stim_op):
    """Test that the PennyLane operation get converted to Stim operation"""
    op, wires = _pl_op_to_stim(pl_op)  # pylint:disable=protected-access
    assert op == stim_op[0]
    assert wires == " ".join(map(str, stim_op[1]))


@pytest.mark.parametrize(
    ["measurement", "tag"],
    [
        (qml.expval(op=qml.Z(1)), None),
        (qml.expval(op=qml.Y(0) @ qml.X(1)), "expval"),
        (qml.var(op=qml.X(0)), None),
        (qml.var(op=qml.X(0) @ qml.Z(1)), "var"),
        (qml.density_matrix(wires=[1]), None),
        (qml.density_matrix(wires=[0, 1]), "dm"),
        (qml.probs(op=qml.Y(0)), None),
        (qml.probs(op=qml.X(0) @ qml.Y(1)), "probs"),
        (qml.vn_entropy(wires=[0]), None),
        (qml.vn_entropy(wires=[1]), "vn_entropy"),
        (qml.mutual_info(wires0=[1], wires1=[0]), None),
        (qml.mutual_info(wires0=[0], wires1=[1]), "mi"),
        (qml.purity(wires=[0]), None),
        (qml.purity(wires=[1]), "purity"),
    ],
)
def test_snapshot_supported(measurement, tag):
    """Tests that applying snapshot of measurements is done correctly"""

    def circuit():
        """Snapshot circuit"""
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Snapshot(measurement=qml.expval(qml.Z(0) @ qml.Z(1)))
        qml.CNOT(wires=[0, 1])
        qml.Snapshot(measurement=measurement, tag=tag)
        qml.CZ(wires=[1, 0])
        qml.Snapshot("meas2", measurement=measurement)
        return qml.probs(op=qml.Y(1) @ qml.Z(0))

    dev_qubit = qml.device("default.qubit", wires=2)
    dev_clifford = qml.device("default.clifford", wires=2)

    qnode_qubit = qml.QNode(circuit, device=dev_qubit)
    qnode_clifford = qml.QNode(circuit, device=dev_clifford)

    snaps_qubit = qml.snapshots(qnode_qubit)()
    snaps_clifford = qml.snapshots(qnode_clifford)()

    assert len(snaps_qubit) == len(snaps_clifford)
    for key1, key2 in zip(snaps_qubit, snaps_clifford):
        assert key1 == key2
        assert qml.math.allclose(snaps_qubit[key1], snaps_clifford[key2])


def test_max_worker_clifford(monkeypatch):
    """Test that the execution of multiple tapes is possible with multiprocessing on this device."""

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    dev_c = qml.device("default.clifford", max_workers=2)
    dev_q = qml.device("default.qubit", max_workers=2)

    qscript = qml.tape.QuantumScript(
        [qml.Hadamard(wires=[0]), qml.CNOT(wires=[0, 1])],
        [qml.expval(qml.PauliX(0))],
    )
    tapes = (qscript, qscript)

    conf_d = dev_c.setup_execution_config()
    res_c = dev_c.execute(tapes, conf_d)
    conf_q = dev_q.setup_execution_config()
    res_q = dev_q.execute(tapes, conf_q)
    assert np.allclose(res_q, res_c)


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
        conf_d = dev_c.setup_execution_config()
        res_c = dev_c.execute(tapes, conf_d)
        conf_q = dev_q.setup_execution_config()
        res_q = dev_q.execute(tapes, conf_q)
        assert np.allclose(res_q, res_c)

        res_s = dev_c.execute(qscript, conf_d)
        assert np.allclose(res_c, res_s)

    assert tracker.totals == {
        "batches": 2,
        "simulations": 2,
        "executions": 2,
        "results": 0.0,
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
        "batches": [1, 1],
        "simulations": [1, 1],
        "executions": [1, 1],
        "errors": [{}, {}],
    }


def test_debugger():
    """Test that the debugger works for a simple circuit."""

    dev = qml.device("default.clifford")
    ops = [qml.Snapshot(), qml.Hadamard(wires=0), qml.Snapshot("final_state")]
    qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))])

    debugger = Debugger()
    result = dev.simulate(qs, debugger=debugger)

    assert isinstance(result, tuple)
    assert len(result) == 2

    assert list(debugger.snapshots.keys()) == [0, "final_state"]
    assert qml.math.allclose(
        debugger.snapshots[0], qml.math.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    assert qml.math.allclose(
        debugger.snapshots["final_state"], qml.math.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    )

    assert qml.math.allclose(result[0], 1.0)
    assert qml.math.allclose(result[1], 0.0)

    dev2 = qml.device("default.clifford", wires=1)
    result = dev2.simulate(qs, debugger=debugger)
    assert qml.math.allclose(
        debugger.snapshots[0], qml.math.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )

    dev = qml.device("default.clifford", tableau=False)

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


@pytest.mark.parametrize("circuit", [circuit_1])
def test_grad_error(circuit):
    """Test that computing gradients with `default.clifford` raises error."""
    dev_c = qml.device("default.clifford")
    assert not dev_c.supports_derivatives()
    assert not dev_c.supports_jvp()
    assert not dev_c.supports_vjp()

    def circuit_fn():
        circuit()
        return qml.expval(qml.PauliZ(0))

    qnode_clfrd = qml.QNode(circuit_fn, dev_c)
    tape = qml.workflow.construct_tape(qnode_clfrd)()

    conf_c, tape_c = dev_c.setup_execution_config(), tape

    with pytest.raises(
        NotImplementedError,
        match="default.clifford does not support differentiable workflows.",
    ):
        dev_c.compute_derivatives(tape_c, conf_c)

    with pytest.raises(
        NotImplementedError,
        match="default.clifford does not support differentiable workflows.",
    ):
        dev_c.execute_and_compute_derivatives(tape_c, conf_c)


def test_meas_error():
    """Test if an NotImplementedError is raised when taking expectation value of op_math objects."""

    @qml.qnode(qml.device("default.clifford"))
    def circuit_exp():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        return qml.expval(qml.sum(qml.exp(qml.PauliX(0), coeff=-3.2j), qml.PauliZ(1)))

    with pytest.raises(
        NotImplementedError,
        match="default.clifford doesn't support expectation value calculation with",
    ):
        circuit_exp()

    @qml.qnode(qml.device("default.clifford", wires=3), shots=10)
    def circuit_herm():
        qml.Hadamard(wires=[0])
        qml.CNOT(wires=[0, 1])
        Amat = np.random.rand(4, 4)
        return qml.probs(op=qml.Hermitian(Amat + Amat.conj().T, wires=[0, 1]))

    with pytest.raises(
        QuantumFunctionError,
        match="Hermitian is not supported for rotating probabilities on default.clifford.",
    ):
        circuit_herm()


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
        DeviceError,
        match=r"Operator RX\(1.0, wires=\[0\]\) not supported with default.clifford and does not provide a decomposition",
    ):
        circuit()


def test_meas_error_noisy():
    """Test error is raised when noisy circuit are executed on Clifford device analytically."""

    @qml.qnode(qml.device("default.clifford"))
    def circ_1():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        circuit_2()
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(
        DeviceError,
        match="Channel not supported on default.clifford without finite shots.",
    ):
        circ_1()

    @qml.qnode(qml.device("default.clifford"))
    def circ_2():
        qml.BasisState(np.array([1, 1]), wires=range(2))
        qml.AmplitudeDamping(0.2, [0])
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(
        DeviceError,
        match=r"Operator AmplitudeDamping\(0.2, wires=\[0\]\) not supported with default.clifford",
    ):
        circ_2()


@pytest.mark.parametrize(
    "channel_op",
    [
        qml.PauliError("YZ", 0.2, wires=[0, 1]),
        qml.PauliError("YZ", 0.3, wires=[1, 3]),
        qml.BitFlip(0.1, wires=[0]),
        qml.BitFlip(0.5, wires=[2]),
        qml.PhaseFlip(0.3, wires=[1]),
        qml.PhaseFlip(0.5, wires=[3]),
        qml.DepolarizingChannel(0.2, wires=[0]),
        qml.DepolarizingChannel(0.75, wires=[2]),
    ],
)
def test_meas_noisy_distribution(channel_op):
    """Test error distribution of samples matches with that from `default.mixed` device."""

    dev_c = qml.device("default.clifford", wires=4)
    dev_q = qml.device("default.mixed", wires=4)

    def circuit():
        qml.Hadamard(wires=[0])
        for idx in range(3):
            qml.CNOT(wires=[idx, idx + 1])
        qml.apply(channel_op)
        return qml.probs()

    qnode_clfrd = qml.set_shots(qml.QNode(circuit, dev_c), shots=10000)
    qnode_qubit = qml.set_shots(qml.QNode(circuit, dev_q), shots=10000)

    kl_d = np.ma.masked_invalid(sp.special.rel_entr(qnode_clfrd(), qnode_qubit()))
    assert qml.math.allclose(np.abs(kl_d.sum()), 0.0, atol=1e-1)


def test_fail_import_stim(monkeypatch):
    """Test if an ImportError is raised when stim is requested but not installed."""

    with monkeypatch.context() as m:
        m.setattr(qml.devices.default_clifford, "has_stim", False)
        with pytest.raises(ImportError, match="This feature requires stim"):
            qml.device("default.clifford")
