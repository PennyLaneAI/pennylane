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

import pennylane as qp
from pennylane.devices.default_clifford import _pl_op_to_stim
from pennylane.exceptions import DeviceError, QuantumFunctionError

stim = pytest.importorskip("stim")

pytestmark = pytest.mark.external


# pylint: disable=protected-access
def test_applied_modifiers():
    """Test that default qubit has the `single_tape_support` and `simulator_tracking`
    modifiers applied.
    """
    dev = qp.device("default.clifford")
    assert dev._applied_modifiers == [
        qp.devices.modifiers.single_tape_support,
        qp.devices.modifiers.simulator_tracking,
    ]


def circuit_1():
    """Circuit 1 with Clifford gates."""
    qp.GlobalPhase(np.pi)
    qp.CNOT(wires=[0, 1])
    qp.PauliX(wires=[1])
    qp.Barrier()
    qp.ISWAP(wires=[0, 1])
    qp.Hadamard(wires=[0])


def circuit_2():
    """Circuit 2 with error channels."""
    qp.GlobalPhase(np.pi)
    qp.CNOT(wires=[0, 1])
    qp.PauliX(wires=[1])
    qp.PauliError("YZ", 0.2, wires=[0, 1])
    qp.Barrier()
    qp.BitFlip(0.01, wires=[0])
    qp.PhaseFlip(0.01, wires=[0])
    qp.Barrier()
    qp.Hadamard(wires=[0])
    qp.DepolarizingChannel(0.2, wires=[0])


@pytest.mark.parametrize("circuit", [circuit_1])
@pytest.mark.parametrize(
    "expec_op",
    [
        qp.PauliZ(0),
        qp.PauliZ(0) @ qp.PauliX(1),
        qp.Hamiltonian([0.42], [qp.PauliZ(0) @ qp.PauliX(1)]),
    ],
)
def test_expectation_clifford(circuit, expec_op):
    """Test that execution of default.clifford is possible and agrees with default.qubit."""
    dev_c = qp.device("default.clifford")
    dev_q = qp.device("default.qubit")

    def circuit_fn():
        circuit()
        return qp.expval(expec_op)

    qnode_clfrd = qp.QNode(circuit_fn, dev_c)
    qnode_qubit = qp.QNode(circuit_fn, dev_q)
    assert np.allclose(qnode_clfrd(), qnode_qubit())


def test_execution_with_no_execution_config():
    """Test execution of a tape with no execution config."""
    dev = qp.device("default.clifford")
    qs = qp.tape.QuantumScript([qp.X(0)], [qp.expval(qp.PauliZ(0))])
    result = dev.execute(qs)
    assert qp.math.allclose(result, -1.0)


@pytest.mark.parametrize("circuit", [circuit_1])
@pytest.mark.parametrize("tableau", [True, False])
def test_state_clifford(circuit, tableau):
    """Test that state computation with default.clifford is possible and agrees with default.qubit."""
    dev_c = qp.device("default.clifford", tableau=tableau, wires=2)
    dev_q = qp.device("default.qubit", wires=2)

    def circuit_fn():
        circuit()
        return qp.state()

    # Tableau for the circuit define above
    circuit_tableau = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
        ]
    )

    qnode_clfrd = qp.QNode(circuit_fn, dev_c)
    qnode_qubit = qp.QNode(circuit_fn, dev_q)

    if not tableau:
        st1, st2 = qnode_clfrd(), qnode_qubit()
        phase = qp.math.divide(
            st1, st2, out=qp.math.zeros_like(st1, dtype=complex), where=st1 != 0
        )[qp.math.nonzero(np.round(st1, 10))]
        if not qp.math.allclose(phase, 0.0):
            assert qp.math.allclose(phase / phase[0], qp.math.ones(len(phase)))
    else:
        assert qp.math.allclose(circuit_tableau, qnode_clfrd())

    dev_c = qp.device("default.clifford", tableau=tableau, wires=1)

    @qp.qnode(dev_c)
    def circuit_empty():
        return qp.state()

    # Tableau for the circuit define above
    circuit_tableau = np.array([[1, 0, 0], [0, 1, 0]])
    if not tableau:
        assert qp.math.allclose(circuit_empty(), qp.math.array([1.0, 0.0]))
    else:
        assert qp.math.allclose(circuit_empty(), circuit_tableau)


@pytest.mark.parametrize(
    "meas_op",
    [
        qp.density_matrix([1]),
        qp.density_matrix([1, 2]),
        qp.density_matrix([0, 2]),
        qp.purity([1]),
        qp.purity([0, 2]),
        qp.purity([1, 2]),
        qp.purity([0, 1, 2]),
        qp.vn_entropy([1]),
        qp.vn_entropy([0, 2]),
        qp.vn_entropy([0, 1, 2]),
        qp.mutual_info([0], [1]),
        qp.mutual_info([1], [0, 2]),
        qp.mutual_info([0], [1, 2]),
    ],
)
def test_meas_qinfo_clifford(meas_op):
    """Test that quantum information measurements with `default.clifford` is possible
    and agrees with `default.qubit`."""
    dev_c = qp.device("default.clifford")
    dev_q = qp.device("default.qubit")

    def circuit_fn():
        circuit_1()
        qp.PauliX(wires=[2])
        qp.Hadamard(wires=[2])
        qp.CNOT(wires=[1, 2])
        return qp.apply(meas_op)

    qnode_clfrd = qp.QNode(circuit_fn, dev_c)
    qnode_qubit = qp.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit())


@pytest.mark.parametrize("shots", [None, 1_000_000])
@pytest.mark.parametrize(
    "ops",
    [
        qp.PauliX(0),
        qp.PauliX(0) @ qp.PauliY(1),
        qp.Hamiltonian([3.0, 2.0], [qp.PauliX(0), qp.PauliY(1)]),
        qp.Hamiltonian([0.42], [qp.PauliZ(0) @ qp.PauliX(1)]),
        qp.sum(qp.PauliX(0), qp.PauliY(1)),
        qp.prod(qp.PauliX(0), qp.PauliY(1)),
        qp.s_prod(3.0, qp.PauliX(0)),
        qp.sum(qp.PauliX(0), qp.s_prod(2.0, qp.PauliY(1))),
        qp.prod(qp.sum(qp.PauliZ(0), qp.PauliY(1)), qp.s_prod(3, qp.PauliX(2))),
        qp.Hermitian(qp.sum(qp.PauliX(0), qp.PauliY(1)).matrix(), [0, 1]),
        qp.Projector([1, 0], [0, 1]),
    ],
)
def test_meas_expval(shots, ops, seed):
    """Test that expectation value measurements with `default.clifford` is possible
    and agrees with `default.qubit`."""
    dev_c = qp.device("default.clifford", seed=seed)
    dev_q = qp.device("default.qubit")

    def circuit_fn():
        circuit_1()
        return qp.expval(ops)

    qnode_clfrd = qp.set_shots(qp.QNode(circuit_fn, dev_c), shots=shots)
    qnode_qubit = qp.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit(), atol=1e-2 if shots else 1e-8)


@pytest.mark.parametrize("shots", [None, int(2e6)])
@pytest.mark.parametrize(
    "ops",
    [
        qp.PauliZ(0),
        qp.PauliZ(0) @ qp.PauliY(1),
        qp.sum(qp.PauliZ(0), qp.PauliY(1)),
        qp.prod(qp.PauliZ(0), qp.PauliY(1)),
        qp.s_prod(3.0, qp.PauliX(0)),
        qp.sum(qp.PauliZ(0), qp.s_prod(2.0, qp.PauliY(1))),
    ],
)
def test_meas_var(shots, ops, seed):
    """Test that variance measurements with `default.clifford` is possible
    and agrees with `default.qubit`."""
    dev_c = qp.device("default.clifford", seed=seed)
    dev_q = qp.device("default.qubit")

    def circuit_fn():
        circuit_1()
        return qp.var(ops)

    qnode_clfrd = qp.set_shots(qp.QNode(circuit_fn, dev_c), shots=shots)
    qnode_qubit = qp.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit(), atol=1e-2 if shots else 1e-8)


@pytest.mark.parametrize("circuit", [circuit_1, circuit_2])
@pytest.mark.parametrize("shots", [1024, 8192, 16384])
def test_meas_samples(circuit, shots):
    """Test if samples are returned with shots given in the clifford device."""

    @qp.qnode(qp.device("default.clifford"), shots=shots)
    def circuit_fn():
        qp.BasisState(np.array([1, 1]), wires=range(2))
        circuit()
        return [
            qp.sample(wires=[1]),
            qp.sample(qp.PauliZ(0)),
            qp.sample(qp.PauliX(0) @ qp.PauliY(1)),
        ]

    samples = circuit_fn()
    assert len(samples) == 3
    assert qp.math.shape(samples[0]) == (shots,)
    assert qp.math.shape(samples[1]) == (shots,)
    assert qp.math.shape(samples[2]) == (shots,)


@pytest.mark.parametrize("tableau", [True, False])
@pytest.mark.parametrize("shots", [None, 50000])
@pytest.mark.parametrize(
    "ops",
    [
        None,
        qp.PauliY(0),
        qp.PauliX(0) @ qp.PauliY(1),
        qp.PauliX(0) @ qp.PauliY(1) @ qp.PauliZ(2),
        qp.Projector([0, 1], wires=[0, 1]),
    ],
)
def test_meas_probs(tableau, shots, ops, seed):
    """Test if probabilities are returned in the clifford device."""

    dev_c = qp.device("default.clifford", tableau=tableau, seed=seed)
    dev_q = qp.device("default.qubit")

    def circuit_fn():
        for wire in range(3):
            qp.PauliX(wire)
            qp.PauliY(wire)
            qp.PauliZ(wire)
        return qp.probs(op=ops) if ops else qp.probs(wires=[0, 1])

    qnode_clfrd = qp.set_shots(qp.QNode(circuit_fn, dev_c), shots=shots)
    qnode_qubit = qp.QNode(circuit_fn, dev_q)

    gotten_probs, target_probs = qnode_clfrd(), qnode_qubit()

    assert qp.math.allclose(gotten_probs, target_probs, atol=5e-2 if shots else 1e-8)


def test_meas_probs_large(seed):
    """Test if probabilities are returned in the clifford device with target basis states"""

    def single_op(idx):
        return [qp.PauliX, qp.PauliY, qp.Hadamard, qp.PauliZ][idx]

    def circuit_fn2(meas):
        for wire in range(16):
            single_op(wire % 4)(wires=wire)
            qp.CNOT([wire, wire + 1])
        return qp.apply(meas)

    dev_c = qp.device("default.clifford", seed=seed)
    qnode_clfrd = qp.QNode(circuit_fn2, dev_c)

    meas1 = qp.probs(op=qp.Projector([1, 1, 0], wires=[0, 6, 14]))
    assert qnode_clfrd(meas1).shape == (8,)
    assert qp.math.allclose(qnode_clfrd(meas1), [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25])

    for basis_state in np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1]]):
        meas_b = qp.expval(op=qp.Projector(basis_state, wires=[0, 6, 14]))
        assert qnode_clfrd(meas_b) == [0.0, 0.25][basis_state[0]]


@pytest.mark.parametrize("shots", [1024, 4096])
@pytest.mark.parametrize(
    "ops",
    [None, qp.PauliY(0), qp.PauliX(0) @ qp.PauliY(1)],
)
def test_meas_counts(shots, ops, seed):
    """Test if counts are returned with shots given in the clifford device."""

    dev_c = qp.device("default.clifford", seed=seed)
    dev_q = qp.device("default.qubit", seed=seed)

    def circuit_fn():
        qp.PauliX(0)
        qp.PauliX(1)
        return qp.counts(op=ops) if ops else qp.counts(wires=[0, 1])

    qnode_clfrd = qp.set_shots(qp.QNode(circuit_fn, dev_c), shots=shots)
    qnode_qubit = qp.set_shots(qp.QNode(circuit_fn, dev_q), shots=shots)

    counts_clfrd = qnode_clfrd()
    counts_qubit = qnode_qubit()

    assert list(counts_clfrd.keys()) == list(counts_qubit.keys())

    for k1, k2 in zip(counts_clfrd, counts_clfrd):
        assert qp.math.abs(counts_clfrd[k1] - counts_qubit[k2]) / shots < 0.1  # 10% threshold


@pytest.mark.parametrize("shots", [1024, 10240])
@pytest.mark.parametrize(
    "ops",
    [
        qp.PauliZ(0),
        qp.PauliZ(0) @ qp.PauliY(1),
    ],
)
def test_meas_classical_shadows(shots, ops, seed):
    """Test if classical shadows measurements are returned with shots
    given in the clifford device."""

    def circuit():
        qp.PauliX(0)
        qp.PauliX(1)
        qp.Hadamard(0)
        qp.CNOT((0, 1))

    dev_c = qp.device("default.clifford")
    dev_q = qp.device("default.qubit")

    def circuit_shadow():
        circuit()
        return qp.classical_shadow(wires=[0, 1], seed=seed)

    qnode_clfrd_shadow = qp.set_shots(qp.QNode(circuit_shadow, dev_c), shots=shots)
    qnode_qubit_shadow = qp.set_shots(qp.QNode(circuit_shadow, dev_q), shots=shots)

    bits1, recipes1 = qnode_clfrd_shadow()
    bits2, recipes2 = qnode_qubit_shadow()

    assert bits1.shape == bits2.shape
    assert recipes1.shape == recipes2.shape
    assert np.allclose(np.sort(np.unique(recipes1.reshape(shots * 2, 1))), np.array([0, 1, 2]))

    def circuit_expval():
        circuit()
        return qp.shadow_expval(ops, seed=seed)

    qnode_clfrd_expval = qp.set_shots(qp.QNode(circuit_expval, dev_c), shots=shots)
    expval = qnode_clfrd_expval()

    assert -1.0 <= expval <= 1.0


@pytest.mark.parametrize("circuit", [circuit_1])
def test_prep_snap_clifford(circuit):
    """Test that state preparation with default.clifford is possible and agrees with default.qubit."""
    dev_c = qp.device("default.clifford", wires=2, check_clifford=False)
    dev_q = qp.device("default.qubit", wires=2)

    def circuit_fn():
        qp.BasisState(np.array([1, 1]), wires=range(2))
        circuit()
        qp.Snapshot(tag="cliffy")
        return qp.expval(qp.PauliZ(0))

    qnode_clfrd = qp.QNode(circuit_fn, dev_c)
    qnode_qubit = qp.QNode(circuit_fn, dev_q)

    assert np.allclose(qnode_clfrd(), qnode_qubit())


@pytest.mark.parametrize(
    "pl_op,stim_op",
    [
        (qp.PauliX(0), ("X", [0])),
        (qp.CNOT(["a", "b"]), ("CNOT", ["a", "b"])),
        (qp.GlobalPhase(1.0), (None, [])),
        (qp.Snapshot(), (None, [])),
        (qp.DepolarizingChannel(0.2, [2]), ("DEPOLARIZE1(0.2)", [2])),
        (qp.PauliError("XYZ", 0.2, [0, 1, 2]), ("CORRELATED_ERROR(0.2)", ["X0", "Y1", "Z2"])),
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
        (qp.expval(op=qp.Z(1)), None),
        (qp.expval(op=qp.Y(0) @ qp.X(1)), "expval"),
        (qp.var(op=qp.X(0)), None),
        (qp.var(op=qp.X(0) @ qp.Z(1)), "var"),
        (qp.density_matrix(wires=[1]), None),
        (qp.density_matrix(wires=[0, 1]), "dm"),
        (qp.probs(op=qp.Y(0)), None),
        (qp.probs(op=qp.X(0) @ qp.Y(1)), "probs"),
        (qp.vn_entropy(wires=[0]), None),
        (qp.vn_entropy(wires=[1]), "vn_entropy"),
        (qp.mutual_info(wires0=[1], wires1=[0]), None),
        (qp.mutual_info(wires0=[0], wires1=[1]), "mi"),
        (qp.purity(wires=[0]), None),
        (qp.purity(wires=[1]), "purity"),
    ],
)
def test_snapshot_supported(measurement, tag):
    """Tests that applying snapshot of measurements is done correctly"""

    def circuit():
        """Snapshot circuit"""
        qp.Hadamard(wires=0)
        qp.Hadamard(wires=1)
        qp.Snapshot(measurement=qp.expval(qp.Z(0) @ qp.Z(1)))
        qp.CNOT(wires=[0, 1])
        qp.Snapshot(measurement=measurement, tag=tag)
        qp.CZ(wires=[1, 0])
        qp.Snapshot("meas2", measurement=measurement)
        return qp.probs(op=qp.Y(1) @ qp.Z(0))

    dev_qubit = qp.device("default.qubit", wires=2)
    dev_clifford = qp.device("default.clifford", wires=2)

    qnode_qubit = qp.QNode(circuit, device=dev_qubit)
    qnode_clifford = qp.QNode(circuit, device=dev_clifford)

    snaps_qubit = qp.snapshots(qnode_qubit)()
    snaps_clifford = qp.snapshots(qnode_clifford)()

    assert len(snaps_qubit) == len(snaps_clifford)
    for key1, key2 in zip(snaps_qubit, snaps_clifford):
        assert key1 == key2
        assert qp.math.allclose(snaps_qubit[key1], snaps_clifford[key2])


def test_max_worker_clifford(monkeypatch):
    """Test that the execution of multiple tapes is possible with multiprocessing on this device."""

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    dev_c = qp.device("default.clifford", max_workers=2)
    dev_q = qp.device("default.qubit", max_workers=2)

    qscript = qp.tape.QuantumScript(
        [qp.Hadamard(wires=[0]), qp.CNOT(wires=[0, 1])],
        [qp.expval(qp.PauliX(0))],
    )
    tapes = (qscript, qscript)

    conf_d = dev_c.setup_execution_config()
    res_c = dev_c.execute(tapes, conf_d)
    conf_q = dev_q.setup_execution_config()
    res_q = dev_q.execute(tapes, conf_q)
    assert np.allclose(res_q, res_c)


def test_tracker():
    """Test that the tracker works for this device."""

    dev_c = qp.device("default.clifford")
    dev_q = qp.device("default.qubit")

    qscript = qp.tape.QuantumScript(
        [qp.Hadamard(wires=[0]), qp.CNOT([0, 1])],
        [qp.expval(qp.PauliZ(0))],
    )
    tapes = tuple([qscript])

    with qp.Tracker(dev_c) as tracker:
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
    assert tracker.history.pop("resources")[0] == qp.resource.SpecsResources(
        num_allocs=2,
        gate_types={"Hadamard": 1, "CNOT": 1},
        gate_sizes={1: 1, 2: 1},
        measurements={"expval(PauliZ)": 1},
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

    dev = qp.device("default.clifford")
    ops = [qp.Snapshot(), qp.Hadamard(wires=0), qp.Snapshot("final_state")]
    qs = qp.tape.QuantumScript(ops, [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliZ(0))])

    debugger = Debugger()
    result = dev.simulate(qs, debugger=debugger)

    assert isinstance(result, tuple)
    assert len(result) == 2

    assert list(debugger.snapshots.keys()) == [0, "final_state"]
    assert qp.math.allclose(
        debugger.snapshots[0], qp.math.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    assert qp.math.allclose(
        debugger.snapshots["final_state"], qp.math.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    )

    assert qp.math.allclose(result[0], 1.0)
    assert qp.math.allclose(result[1], 0.0)

    dev2 = qp.device("default.clifford", wires=1)
    result = dev2.simulate(qs, debugger=debugger)
    assert qp.math.allclose(
        debugger.snapshots[0], qp.math.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )

    dev = qp.device("default.clifford", tableau=False)

    @qp.qnode(dev)
    def circuit():
        for op in ops:
            qp.apply(op)
        return [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliZ(0))]

    assert getattr(dev, "_debugger") is None

    result = qp.snapshots(circuit)()
    expected = {
        0: qp.math.array([1, 0]),
        "final_state": qp.math.array([1, 1]) / qp.math.sqrt(2),
        "execution_results": [qp.math.array(1.0), qp.math.array(0.0)],
    }

    assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
    assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))


@pytest.mark.parametrize("circuit", [circuit_1])
def test_grad_error(circuit):
    """Test that computing gradients with `default.clifford` raises error."""
    dev_c = qp.device("default.clifford")
    assert not dev_c.supports_derivatives()
    assert not dev_c.supports_jvp()
    assert not dev_c.supports_vjp()

    def circuit_fn():
        circuit()
        return qp.expval(qp.PauliZ(0))

    qnode_clfrd = qp.QNode(circuit_fn, dev_c)
    tape = qp.workflow.construct_tape(qnode_clfrd)()

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

    @qp.qnode(qp.device("default.clifford"))
    def circuit_exp():
        qp.BasisState(np.array([1, 1]), wires=range(2))
        return qp.expval(qp.sum(qp.exp(qp.PauliX(0), coeff=-3.2j), qp.PauliZ(1)))

    with pytest.raises(
        NotImplementedError,
        match="default.clifford doesn't support expectation value calculation with",
    ):
        circuit_exp()

    @qp.qnode(qp.device("default.clifford", wires=3), shots=10)
    def circuit_herm():
        qp.Hadamard(wires=[0])
        qp.CNOT(wires=[0, 1])
        Amat = np.random.rand(4, 4)
        return qp.probs(op=qp.Hermitian(Amat + Amat.conj().T, wires=[0, 1]))

    with pytest.raises(
        QuantumFunctionError,
        match="Hermitian is not supported for rotating probabilities on default.clifford.",
    ):
        circuit_herm()


@pytest.mark.parametrize("check", [True, False])
def test_clifford_error(check):
    """Test if an QuantumFunctionError is raised when one of the operations is not Clifford."""

    dev = qp.device("default.clifford", tableau="tableau", check_clifford=check)

    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=[0])
        qp.PauliX(wires=[0])
        qp.RX(1.0, wires=[0])
        return qp.state()

    with pytest.raises(
        DeviceError,
        match=r"Operator RX\(1.0, wires=\[0\]\) not supported with default.clifford and does not provide a decomposition",
    ):
        circuit()


def test_meas_error_noisy():
    """Test error is raised when noisy circuit are executed on Clifford device analytically."""

    @qp.qnode(qp.device("default.clifford"))
    def circ_1():
        qp.BasisState(np.array([1, 1]), wires=range(2))
        circuit_2()
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(
        DeviceError,
        match="Channel not supported on default.clifford without finite shots.",
    ):
        circ_1()

    @qp.qnode(qp.device("default.clifford"))
    def circ_2():
        qp.BasisState(np.array([1, 1]), wires=range(2))
        qp.AmplitudeDamping(0.2, [0])
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(
        DeviceError,
        match=r"Operator AmplitudeDamping\(0.2, wires=\[0\]\) not supported with default.clifford",
    ):
        circ_2()


@pytest.mark.parametrize(
    "channel_op",
    [
        qp.PauliError("YZ", 0.2, wires=[0, 1]),
        qp.PauliError("YZ", 0.3, wires=[1, 3]),
        qp.BitFlip(0.1, wires=[0]),
        qp.BitFlip(0.5, wires=[2]),
        qp.PhaseFlip(0.3, wires=[1]),
        qp.PhaseFlip(0.5, wires=[3]),
        qp.DepolarizingChannel(0.2, wires=[0]),
        qp.DepolarizingChannel(0.75, wires=[2]),
    ],
)
def test_meas_noisy_distribution(channel_op):
    """Test error distribution of samples matches with that from `default.mixed` device."""

    dev_c = qp.device("default.clifford", wires=4)
    dev_q = qp.device("default.mixed", wires=4)

    def circuit():
        qp.Hadamard(wires=[0])
        for idx in range(3):
            qp.CNOT(wires=[idx, idx + 1])
        qp.apply(channel_op)
        return qp.probs()

    qnode_clfrd = qp.set_shots(qp.QNode(circuit, dev_c), shots=10000)
    qnode_qubit = qp.set_shots(qp.QNode(circuit, dev_q), shots=10000)

    kl_d = np.ma.masked_invalid(sp.special.rel_entr(qnode_clfrd(), qnode_qubit()))
    assert qp.math.allclose(np.abs(kl_d.sum()), 0.0, atol=1e-1)


def test_fail_import_stim(monkeypatch):
    """Test if an ImportError is raised when stim is requested but not installed."""

    with monkeypatch.context() as m:
        m.setattr(qp.devices.default_clifford, "has_stim", False)
        with pytest.raises(ImportError, match="This feature requires stim"):
            qp.device("default.clifford")
