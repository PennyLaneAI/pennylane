"""
Unit tests for transpiler function.
"""

from math import isclose

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.transpile import transpile


def build_qfunc_probs(wires):
    def qfunc(x, y, z):
        qml.Hadamard(wires=wires[0])
        qml.RZ(z, wires=wires[2])
        qml.CNOT(wires=[wires[2], wires[0]])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RX(x, wires=wires[0])
        qml.CNOT(wires=[wires[0], wires[2]])
        qml.RZ(-z, wires=wires[2])
        qml.RX(y, wires=wires[0])
        qml.PauliY(wires=wires[2])
        qml.CY(wires=[wires[1], wires[2]])
        return qml.probs(wires=[0, 1])

    return qfunc


def build_qfunc_pauli_z(wires):
    def qfunc(x, y, z):
        qml.Hadamard(wires=wires[0])
        qml.RZ(z, wires=wires[2])
        qml.CNOT(wires=[wires[2], wires[0]])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RX(x, wires=wires[0])
        qml.CNOT(wires=[wires[0], wires[2]])
        qml.RZ(-z, wires=wires[2])
        qml.RX(y, wires=wires[0])
        qml.PauliY(wires=wires[2])
        qml.CY(wires=[wires[1], wires[2]])
        return qml.expval(qml.PauliZ(wires=wires[0]))

    return qfunc


# pylint: disable=too-many-public-methods
class TestTranspile:
    """Unit tests for transpile function"""

    def test_transpile_invalid_coupling(self):
        """test that error is raised when coupling_map is invalid"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        # build circuit
        original_qfunc = build_qfunc_pauli_z([0, 1, 2])
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1)])
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        err_msg = (
            r"Not all wires present in coupling map! wires: \[0, 2, 1\], coupling map: \[0, 1\]"
        )
        with pytest.raises(ValueError, match=err_msg):
            transpiled_qnode(0.1, 0.2, 0.3)

    def test_transpile_raise_not_implemented_hamiltonian_mmt(self):
        """test that error is raised when measurement is expectation of a Hamiltonian"""
        dev = qml.device("default.qubit", wires=[0, 1, 2, 3])
        coeffs = [1]
        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        H = qml.Hamiltonian(coeffs, obs)

        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 3])
            return qml.expval(H)

        # build circuit
        transpiled_qfunc = transpile(circuit, coupling_map=[(0, 1), (1, 2), (2, 3)])
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        err_msg = (
            "Measuring expectation values of tensor products or Hamiltonians is not yet supported"
        )
        with pytest.raises(NotImplementedError, match=err_msg):
            transpiled_qnode()

    def test_transpile_raise_not_implemented_prod_mmt(self):
        """test that error is raised when measurement is expectation of a Prod"""
        dev = qml.device("default.qubit", wires=[0, 1, 2, 3])

        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 3])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        # build circuit
        transpiled_qfunc = transpile(circuit, coupling_map=[(0, 1), (1, 2), (2, 3)])
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        err_msg = (
            r"Measuring expectation values of tensor products or Hamiltonians is not yet supported"
        )
        with pytest.raises(NotImplementedError, match=err_msg):
            transpiled_qnode()

    def test_transpile_non_commuting_observables(self):
        """Test that transpile will work with non-commuting observables."""

        ops = (qml.CRX(0.1, wires=(0, 2)),)
        ms = (qml.expval(qml.X(0)), qml.expval(qml.Y(0)))
        tape = qml.tape.QuantumScript(ops, ms, shots=50)
        [out], _ = qml.transforms.transpile(tape, coupling_map=((0, 1), (1, 2)))

        expected = qml.tape.QuantumScript((qml.SWAP((1, 2)), qml.CRX(0.1, (0, 1))), ms, shots=50)
        qml.assert_equal(out, expected)

    def test_transpile_qfunc_transpiled_mmt_obs(self):
        """test that transpile does not alter output for expectation value of an observable"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        # build circuit without transpile
        original_qfunc = build_qfunc_pauli_z([0, 1, 2])
        original_qnode = qml.QNode(original_qfunc, dev)
        original_expectation = original_qnode(0.1, 0.2, 0.3)

        # build circuit with transpile
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        transpiled_expectation = transpiled_qnode(0.1, 0.2, 0.3)

        assert isclose(original_expectation, transpiled_expectation, abs_tol=np.finfo(float).eps)

    def test_transpile_qfunc_transpiled_mmt_probs(self):
        """test that transpile does not alter output for probs measurement"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        # build circuit without transpile
        original_qfunc = build_qfunc_probs([0, 1, 2])
        original_qnode = qml.QNode(original_qfunc, dev)
        original_probs = original_qnode(0.1, 0.2, 0.3)

        # build circuit with transpile
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        transpiled_probs = transpiled_qnode(0.1, 0.2, 0.3)

        assert all(
            isclose(po, pt, abs_tol=np.finfo(float).eps)
            for po, pt in zip(original_probs, transpiled_probs)
        )

    @pytest.mark.autograd
    def test_transpile_differentiable(self):
        """test that circuit remains differentiable after transpilation"""
        dev = qml.device("default.qubit", wires=3)

        def circuit(parameters):
            qml.RX(parameters[0], wires=0)
            qml.RY(parameters[1], wires=1)
            qml.CNOT(wires=[0, 2])
            qml.PhaseShift(parameters[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        transpiled_circ = transpile(circuit, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qml.QNode(transpiled_circ, dev)
        params = np.array([0.5, 0.1, 0.2], requires_grad=True)
        qml.gradients.param_shift(transpiled_qnode)(params)

    def test_more_than_2_qubits_raises_anywires(self):
        """test that transpile raises an error for an operation with num_wires=None that acts on more than 2 qubits"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        def circuit(param):
            qml.MultiRZ(param, wires=[0, 1, 2])
            return qml.probs(wires=[0, 1])

        param = 0.3

        transpiled_qfunc = transpile(circuit, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        with pytest.raises(
            NotImplementedError,
            match="transpile transform only supports gates acting on 1 or 2 qubits",
        ):
            transpiled_qnode(param)

    def test_more_than_2_qubits_raises_3_qubit_gate(self):
        """test that transpile raises an error for an operation that acts on more than 2 qubits"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        def circuit():
            qml.Toffoli(wires=[0, 1, 2])
            return qml.probs(wires=[0, 1])

        transpiled_qfunc = transpile(circuit, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        with pytest.raises(
            NotImplementedError,
            match="transpile transform only supports gates acting on 1 or 2 qubits",
        ):
            transpiled_qnode()

    def test_transpile_ops_anywires(self):
        """test that transpile does not alter output for expectation value of an observable if the qfunc contains
        operations that act on any number of wires"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        def circuit(param):
            qml.MultiRZ(param, wires=[0, 1])
            qml.PhaseShift(param, wires=2)
            qml.MultiRZ(param, wires=[0, 2])
            return qml.probs(wires=[0, 1])

        param = 0.3

        # build circuit without transpile
        original_qfunc = circuit
        original_qnode = qml.QNode(original_qfunc, dev)
        original_expectation = original_qnode(param)

        # build circuit with transpile
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        transpiled_expectation = transpiled_qnode(param)

        tape = qml.workflow.construct_tape(transpiled_qnode)(param)
        original_ops = list(tape)
        transpiled_ops = list(tape)
        qml.assert_equal(transpiled_ops[0], original_ops[0])
        qml.assert_equal(transpiled_ops[1], original_ops[1])

        # SWAP to ensure connectivity
        assert isinstance(transpiled_ops[2], qml.SWAP)
        assert transpiled_ops[2].wires == qml.wires.Wires([1, 2])

        assert isinstance(transpiled_ops[3], qml.MultiRZ)
        assert transpiled_ops[3].data == (param,)
        assert transpiled_ops[3].wires == qml.wires.Wires([0, 1])

        assert isinstance(transpiled_ops[4], qml.measurements.MeasurementProcess)
        assert transpiled_ops[4].wires == qml.wires.Wires([0, 2])

        assert qml.math.allclose(
            original_expectation, transpiled_expectation, atol=np.finfo(float).eps
        )

    def test_transpile_ops_anywires_1_qubit(self):
        """test that transpile does not alter output for expectation value of an observable if the qfunc contains
        1-qubit operations with num_wires=None defined for the operation"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        def circuit(param):
            qml.MultiRZ(param, wires=[0])
            qml.PhaseShift(param, wires=2)
            qml.MultiRZ(param, wires=[0, 2])
            return qml.probs(wires=[0, 1])

        param = 0.3

        # build circuit without transpile
        original_qfunc = circuit
        original_qnode = qml.QNode(original_qfunc, dev)
        original_expectation = original_qnode(param)

        # build circuit with transpile
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qml.QNode(transpiled_qfunc, dev)
        transpiled_expectation = transpiled_qnode(param)

        tape = qml.workflow.construct_tape(transpiled_qnode)(param)
        original_ops = list(tape)
        transpiled_ops = list(tape)
        qml.assert_equal(transpiled_ops[0], original_ops[0])
        qml.assert_equal(transpiled_ops[1], original_ops[1])

        # SWAP to ensure connectivity
        assert isinstance(transpiled_ops[2], qml.SWAP)
        assert transpiled_ops[2].wires == qml.wires.Wires([1, 2])

        assert isinstance(transpiled_ops[3], qml.MultiRZ)
        assert transpiled_ops[3].data == (param,)
        assert transpiled_ops[3].wires == qml.wires.Wires([0, 1])

        assert isinstance(transpiled_ops[4], qml.measurements.MeasurementProcess)
        assert transpiled_ops[4].wires == qml.wires.Wires([0, 2])

        assert qml.math.allclose(
            original_expectation, transpiled_expectation, atol=np.finfo(float).eps
        )

    def test_transpile_mcm(self):
        """Test that transpile can be used with mid circuit measurements."""

        m0 = qml.measure(0)
        ops = [qml.CNOT((0, 2)), *m0.measurements, qml.ops.Conditional(m0, qml.S(0))]
        tape = qml.tape.QuantumScript(ops, [qml.probs()], shots=50)

        [new_tape], _ = transpile(tape, [(0, 1), (1, 2)])
        expected_ops = [qml.SWAP((1, 2)), qml.CNOT((0, 1))] + ops[1:]
        assert new_tape.operations == expected_ops
        assert new_tape.shots == tape.shots

    def test_transpile_ops_anywires_1_qubit_qnode(self):
        """test that transpile does not alter output for expectation value of an observable if the qfunc contains
        1-qubit operations with num_wires=None defined for the operation"""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(device=dev)
        def circuit(param):
            qml.MultiRZ(param, wires=[0])
            qml.PhaseShift(param, wires=2)
            qml.MultiRZ(param, wires=[0, 2])
            return qml.probs(wires=[0, 1])

        param = 0.3

        # build circuit without transpile
        original_expectation = circuit(param)

        # build circuit with transpile
        transpiled_qnode = transpile(circuit, coupling_map=[(0, 1), (1, 2)])
        transpiled_expectation = transpiled_qnode(param)

        assert qml.math.allclose(
            original_expectation, transpiled_expectation, atol=np.finfo(float).eps
        )

    def test_transpile_state(self):
        """Test that transpile works with state measurement process."""

        tape = qml.tape.QuantumScript([qml.PauliX(0), qml.CNOT((0, 2))], [qml.state()], shots=100)
        batch, fn = qml.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)])

        assert len(batch) == 1
        assert fn(("a",)) == "a"

        assert batch[0][0] == qml.PauliX(0)
        assert batch[0][1] == qml.SWAP((1, 2))
        assert batch[0][2] == qml.CNOT((0, 1))
        assert batch[0][3] == qml.state()
        assert batch[0].shots == tape.shots

    def test_transpile_state_with_device(self):
        """Test that if a device is provided and a state is measured, then the state will be transposed during post processing."""

        dev = qml.device("default.qubit", wires=(0, 1, 2))

        tape = qml.tape.QuantumScript([qml.PauliX(0), qml.CNOT(wires=(0, 2))], [qml.state()])
        batch, fn = qml.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)], device=dev)

        original_mat = np.arange(8)
        new_mat = fn((original_mat,))
        expected_new_mat = np.swapaxes(np.reshape(original_mat, [2, 2, 2]), 1, 2).flatten()
        assert qml.math.allclose(new_mat, expected_new_mat)

        assert batch[0][0] == qml.PauliX(0)
        assert batch[0][1] == qml.SWAP((1, 2))
        assert batch[0][2] == qml.CNOT((0, 1))
        assert batch[0][3] == qml.state()

        pre, post = dev.preprocess_transforms()((tape,))
        original_results = post(dev.execute(pre))
        transformed_results = fn(dev.execute(batch))
        assert qml.math.allclose(original_results, transformed_results)

    def test_transpile_state_with_device_multiple_measurements(self):
        """Test that if a device is provided and a state is measured, then the state will be transposed during post processing."""

        dev = qml.device("default.qubit", wires=(0, 1, 2))

        tape = qml.tape.QuantumScript(
            [qml.PauliX(0), qml.CNOT(wires=(0, 2))], [qml.state(), qml.expval(qml.PauliZ(2))]
        )
        batch, fn = qml.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)], device=dev)

        original_mat = np.arange(8)
        new_mat, _ = fn(((original_mat, 2.0),))
        expected_new_mat = np.swapaxes(np.reshape(original_mat, [2, 2, 2]), 1, 2).flatten()
        assert qml.math.allclose(new_mat, expected_new_mat)

        assert batch[0][0] == qml.PauliX(0)
        assert batch[0][1] == qml.SWAP((1, 2))
        assert batch[0][2] == qml.CNOT((0, 1))
        assert batch[0][3] == qml.state()
        assert batch[0][4] == qml.expval(qml.PauliZ(1))

        pre, post = dev.preprocess_transforms()((tape,))
        original_results = post(dev.execute(pre))
        transformed_results = fn(dev.execute(batch))
        assert qml.math.allclose(original_results[0][0], transformed_results[0])
        assert qml.math.allclose(original_results[0][1], transformed_results[1])

    def test_transpile_with_state_default_mixed(self):
        """Test that if the state is default mixed, state measurements are converted in to density measurements with the device wires."""

        dev = qml.device("default.mixed", wires=(0, 1, 2))

        tape = qml.tape.QuantumScript([qml.PauliX(0), qml.CNOT(wires=(0, 2))], [qml.state()])
        batch, fn = qml.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)], device=dev)

        assert batch[0][-1] == qml.density_matrix(wires=(0, 2, 1))

        pre, post = dev.preprocess_transforms()((tape,))
        original_results = post(dev.execute(pre))
        transformed_results = fn(dev.execute(batch))
        assert qml.math.allclose(original_results, transformed_results)

    def test_transpile_probs_sample_filled_in_wires(self):
        """Test that if probs or sample are requested broadcasted over all wires, transpile fills in the device wires."""
        dev = qml.device("default.qubit", wires=(0, 1, 2))

        tape = qml.tape.QuantumScript(
            [qml.PauliX(0), qml.CNOT(wires=(0, 2))], [qml.probs(), qml.sample()], shots=100
        )
        batch, fn = qml.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)], device=dev)

        assert batch[0].measurements[0] == qml.probs(wires=(0, 2, 1))
        assert batch[0].measurements[1] == qml.sample(wires=(0, 2, 1))

        pre, post = dev.preprocess_transforms()((tape,))
        original_results = post(dev.execute(pre))[0]
        transformed_results = fn(dev.execute(batch))
        assert qml.math.allclose(original_results[0], transformed_results[0])
        assert qml.math.allclose(original_results[1], transformed_results[1])

    def test_custom_qnode_transform(self):
        """Test that applying the transform to a qnode adds the device to the transform kwargs."""

        dev = qml.device("default.qubit", wires=(0, 1, 2))

        def qfunc():
            return qml.state()

        original_qnode = qml.QNode(qfunc, dev)
        transformed_qnode = transpile(original_qnode, coupling_map=[(0, 1), (1, 2)])

        assert len(transformed_qnode.transform_program) == 1
        assert transformed_qnode.transform_program[0].kwargs["device"] is dev

    def test_qnode_transform_raises_if_device_kwarg(self):
        """Test an error is raised if a device is provided as a keyword argument to a qnode transform."""

        dev = qml.device("default.qubit", wires=[0, 1, 2, 3])

        @qml.qnode(dev)
        def circuit():
            return qml.state()

        with pytest.raises(ValueError, match=r"Cannot provide a "):
            qml.transforms.transpile(
                circuit, coupling_map=[(0, 1), (1, 3), (3, 2), (2, 0)], device=dev
            )
