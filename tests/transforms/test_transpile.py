"""
Unit tests for transpiler function.
"""

from math import isclose

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.transforms.transpile import transpile


def build_qfunc_probs(wires):
    def qfunc(x, y, z):
        qp.Hadamard(wires=wires[0])
        qp.RZ(z, wires=wires[2])
        qp.CNOT(wires=[wires[2], wires[0]])
        qp.CNOT(wires=[wires[1], wires[0]])
        qp.RX(x, wires=wires[0])
        qp.CNOT(wires=[wires[0], wires[2]])
        qp.RZ(-z, wires=wires[2])
        qp.RX(y, wires=wires[0])
        qp.PauliY(wires=wires[2])
        qp.CY(wires=[wires[1], wires[2]])
        return qp.probs(wires=[0, 1])

    return qfunc


def build_qfunc_pauli_z(wires):
    def qfunc(x, y, z):
        qp.Hadamard(wires=wires[0])
        qp.RZ(z, wires=wires[2])
        qp.CNOT(wires=[wires[2], wires[0]])
        qp.CNOT(wires=[wires[1], wires[0]])
        qp.RX(x, wires=wires[0])
        qp.CNOT(wires=[wires[0], wires[2]])
        qp.RZ(-z, wires=wires[2])
        qp.RX(y, wires=wires[0])
        qp.PauliY(wires=wires[2])
        qp.CY(wires=[wires[1], wires[2]])
        return qp.expval(qp.PauliZ(wires=wires[0]))

    return qfunc


# pylint: disable=too-many-public-methods
class TestTranspile:
    """Unit tests for transpile function"""

    def test_transpile_invalid_coupling(self):
        """test that error is raised when coupling_map is invalid"""
        dev = qp.device("default.qubit", wires=[0, 1, 2])

        # build circuit
        original_qfunc = build_qfunc_pauli_z([0, 1, 2])
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1)])
        transpiled_qnode = qp.QNode(transpiled_qfunc, dev)
        err_msg = (
            r"Not all wires present in coupling map! wires: \[0, 2, 1\], coupling map: \[0, 1\]"
        )
        with pytest.raises(ValueError, match=err_msg):
            transpiled_qnode(0.1, 0.2, 0.3)

    def test_transpile_raise_not_implemented_hamiltonian_mmt(self):
        """test that error is raised when measurement is expectation of a Hamiltonian"""
        dev = qp.device("default.qubit", wires=[0, 1, 2, 3])
        coeffs = [1]
        obs = [qp.PauliZ(0) @ qp.PauliZ(1)]
        H = qp.Hamiltonian(coeffs, obs)

        def circuit():
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[0, 3])
            return qp.expval(H)

        # build circuit
        transpiled_qfunc = transpile(circuit, coupling_map=[(0, 1), (1, 2), (2, 3)])
        transpiled_qnode = qp.QNode(transpiled_qfunc, dev)
        err_msg = (
            "Measuring expectation values of tensor products or Hamiltonians is not yet supported"
        )
        with pytest.raises(NotImplementedError, match=err_msg):
            transpiled_qnode()

    def test_transpile_raise_not_implemented_prod_mmt(self):
        """test that error is raised when measurement is expectation of a Prod"""
        dev = qp.device("default.qubit", wires=[0, 1, 2, 3])

        def circuit():
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[0, 3])
            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))

        # build circuit
        transpiled_qfunc = transpile(circuit, coupling_map=[(0, 1), (1, 2), (2, 3)])
        transpiled_qnode = qp.QNode(transpiled_qfunc, dev)
        err_msg = (
            r"Measuring expectation values of tensor products or Hamiltonians is not yet supported"
        )
        with pytest.raises(NotImplementedError, match=err_msg):
            transpiled_qnode()

    def test_transpile_non_commuting_observables(self):
        """Test that transpile will work with non-commuting observables."""

        ops = (qp.CRX(0.1, wires=(0, 2)),)
        ms = (qp.expval(qp.X(0)), qp.expval(qp.Y(0)))
        tape = qp.tape.QuantumScript(ops, ms, shots=50)
        [out], _ = qp.transforms.transpile(tape, coupling_map=((0, 1), (1, 2)))

        expected = qp.tape.QuantumScript((qp.SWAP((1, 2)), qp.CRX(0.1, (0, 1))), ms, shots=50)
        qp.assert_equal(out, expected)

    def test_transpile_qfunc_transpiled_mmt_obs(self):
        """test that transpile does not alter output for expectation value of an observable"""
        dev = qp.device("default.qubit", wires=[0, 1, 2])

        # build circuit without transpile
        original_qfunc = build_qfunc_pauli_z([0, 1, 2])
        original_qnode = qp.QNode(original_qfunc, dev)
        original_expectation = original_qnode(0.1, 0.2, 0.3)

        # build circuit with transpile
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qp.QNode(transpiled_qfunc, dev)
        transpiled_expectation = transpiled_qnode(0.1, 0.2, 0.3)

        assert isclose(original_expectation, transpiled_expectation, abs_tol=np.finfo(float).eps)

    def test_transpile_qfunc_transpiled_mmt_probs(self):
        """test that transpile does not alter output for probs measurement"""
        dev = qp.device("default.qubit", wires=[0, 1, 2])

        # build circuit without transpile
        original_qfunc = build_qfunc_probs([0, 1, 2])
        original_qnode = qp.QNode(original_qfunc, dev)
        original_probs = original_qnode(0.1, 0.2, 0.3)

        # build circuit with transpile
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qp.QNode(transpiled_qfunc, dev)
        transpiled_probs = transpiled_qnode(0.1, 0.2, 0.3)

        assert all(
            isclose(po, pt, abs_tol=np.finfo(float).eps)
            for po, pt in zip(original_probs, transpiled_probs)
        )

    @pytest.mark.autograd
    def test_transpile_differentiable(self):
        """test that circuit remains differentiable after transpilation"""
        dev = qp.device("default.qubit", wires=3)

        def circuit(parameters):
            qp.RX(parameters[0], wires=0)
            qp.RY(parameters[1], wires=1)
            qp.CNOT(wires=[0, 2])
            qp.PhaseShift(parameters[2], wires=0)
            return qp.expval(qp.PauliZ(0))

        transpiled_circ = transpile(circuit, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qp.QNode(transpiled_circ, dev)
        params = np.array([0.5, 0.1, 0.2], requires_grad=True)
        qp.gradients.param_shift(transpiled_qnode)(params)

    def test_more_than_2_qubits_raises_anywires(self):
        """test that transpile raises an error for an operation with num_wires=None that acts on more than 2 qubits"""
        dev = qp.device("default.qubit", wires=[0, 1, 2])

        def circuit(param):
            qp.MultiRZ(param, wires=[0, 1, 2])
            return qp.probs(wires=[0, 1])

        param = 0.3

        transpiled_qfunc = transpile(circuit, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qp.QNode(transpiled_qfunc, dev)
        with pytest.raises(
            NotImplementedError,
            match="transpile transform only supports gates acting on 1 or 2 qubits",
        ):
            transpiled_qnode(param)

    def test_more_than_2_qubits_raises_3_qubit_gate(self):
        """test that transpile raises an error for an operation that acts on more than 2 qubits"""
        dev = qp.device("default.qubit", wires=[0, 1, 2])

        def circuit():
            qp.Toffoli(wires=[0, 1, 2])
            return qp.probs(wires=[0, 1])

        transpiled_qfunc = transpile(circuit, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qp.QNode(transpiled_qfunc, dev)
        with pytest.raises(
            NotImplementedError,
            match="transpile transform only supports gates acting on 1 or 2 qubits",
        ):
            transpiled_qnode()

    def test_transpile_ops_anywires(self):
        """test that transpile does not alter output for expectation value of an observable if the qfunc contains
        operations that act on any number of wires"""
        dev = qp.device("default.qubit", wires=[0, 1, 2])

        def circuit(param):
            qp.MultiRZ(param, wires=[0, 1])
            qp.PhaseShift(param, wires=2)
            qp.MultiRZ(param, wires=[0, 2])
            return qp.probs(wires=[0, 1])

        param = 0.3

        # build circuit without transpile
        original_qfunc = circuit
        original_qnode = qp.QNode(original_qfunc, dev)
        original_expectation = original_qnode(param)

        # build circuit with transpile
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qp.QNode(transpiled_qfunc, dev)
        transpiled_expectation = transpiled_qnode(param)

        tape = qp.workflow.construct_tape(transpiled_qnode)(param)
        original_ops = list(tape)
        transpiled_ops = list(tape)
        qp.assert_equal(transpiled_ops[0], original_ops[0])
        qp.assert_equal(transpiled_ops[1], original_ops[1])

        # SWAP to ensure connectivity
        assert isinstance(transpiled_ops[2], qp.SWAP)
        assert transpiled_ops[2].wires == qp.wires.Wires([1, 2])

        assert isinstance(transpiled_ops[3], qp.MultiRZ)
        assert transpiled_ops[3].data == (param,)
        assert transpiled_ops[3].wires == qp.wires.Wires([0, 1])

        assert isinstance(transpiled_ops[4], qp.measurements.MeasurementProcess)
        assert transpiled_ops[4].wires == qp.wires.Wires([0, 2])

        assert qp.math.allclose(
            original_expectation, transpiled_expectation, atol=np.finfo(float).eps
        )

    def test_transpile_ops_anywires_1_qubit(self):
        """test that transpile does not alter output for expectation value of an observable if the qfunc contains
        1-qubit operations with num_wires=None defined for the operation"""
        dev = qp.device("default.qubit", wires=[0, 1, 2])

        def circuit(param):
            qp.MultiRZ(param, wires=[0])
            qp.PhaseShift(param, wires=2)
            qp.MultiRZ(param, wires=[0, 2])
            return qp.probs(wires=[0, 1])

        param = 0.3

        # build circuit without transpile
        original_qfunc = circuit
        original_qnode = qp.QNode(original_qfunc, dev)
        original_expectation = original_qnode(param)

        # build circuit with transpile
        transpiled_qfunc = transpile(original_qfunc, coupling_map=[(0, 1), (1, 2)])
        transpiled_qnode = qp.QNode(transpiled_qfunc, dev)
        transpiled_expectation = transpiled_qnode(param)

        tape = qp.workflow.construct_tape(transpiled_qnode)(param)
        original_ops = list(tape)
        transpiled_ops = list(tape)
        qp.assert_equal(transpiled_ops[0], original_ops[0])
        qp.assert_equal(transpiled_ops[1], original_ops[1])

        # SWAP to ensure connectivity
        assert isinstance(transpiled_ops[2], qp.SWAP)
        assert transpiled_ops[2].wires == qp.wires.Wires([1, 2])

        assert isinstance(transpiled_ops[3], qp.MultiRZ)
        assert transpiled_ops[3].data == (param,)
        assert transpiled_ops[3].wires == qp.wires.Wires([0, 1])

        assert isinstance(transpiled_ops[4], qp.measurements.MeasurementProcess)
        assert transpiled_ops[4].wires == qp.wires.Wires([0, 2])

        assert qp.math.allclose(
            original_expectation, transpiled_expectation, atol=np.finfo(float).eps
        )

    def test_transpile_mcm(self):
        """Test that transpile can be used with mid circuit measurements."""

        m0 = qp.measure(0)
        ops = [qp.CNOT((0, 2)), *m0.measurements, qp.ops.Conditional(m0, qp.S(0))]
        tape = qp.tape.QuantumScript(ops, [qp.probs()], shots=50)

        [new_tape], _ = transpile(tape, [(0, 1), (1, 2)])
        expected_ops = [qp.SWAP((1, 2)), qp.CNOT((0, 1))] + ops[1:]
        assert new_tape.operations == expected_ops
        assert new_tape.shots == tape.shots

    def test_transpile_ops_anywires_1_qubit_qnode(self):
        """test that transpile does not alter output for expectation value of an observable if the qfunc contains
        1-qubit operations with num_wires=None defined for the operation"""
        dev = qp.device("default.qubit", wires=[0, 1, 2])

        @qp.qnode(device=dev)
        def circuit(param):
            qp.MultiRZ(param, wires=[0])
            qp.PhaseShift(param, wires=2)
            qp.MultiRZ(param, wires=[0, 2])
            return qp.probs(wires=[0, 1])

        param = 0.3

        # build circuit without transpile
        original_expectation = circuit(param)

        # build circuit with transpile
        transpiled_qnode = transpile(circuit, coupling_map=[(0, 1), (1, 2)])
        transpiled_expectation = transpiled_qnode(param)

        assert qp.math.allclose(
            original_expectation, transpiled_expectation, atol=np.finfo(float).eps
        )

    def test_transpile_state(self):
        """Test that transpile works with state measurement process."""

        tape = qp.tape.QuantumScript([qp.PauliX(0), qp.CNOT((0, 2))], [qp.state()], shots=100)
        batch, fn = qp.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)])

        assert len(batch) == 1
        assert fn(("a",)) == "a"

        assert batch[0][0] == qp.PauliX(0)
        assert batch[0][1] == qp.SWAP((1, 2))
        assert batch[0][2] == qp.CNOT((0, 1))
        assert batch[0][3] == qp.state()
        assert batch[0].shots == tape.shots

    def test_transpile_state_with_device(self):
        """Test that if a device is provided and a state is measured, then the state will be transposed during post processing."""

        dev = qp.device("default.qubit", wires=(0, 1, 2))

        tape = qp.tape.QuantumScript([qp.PauliX(0), qp.CNOT(wires=(0, 2))], [qp.state()])
        batch, fn = qp.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)], device=dev)

        original_mat = np.arange(8)
        new_mat = fn((original_mat,))
        expected_new_mat = np.swapaxes(np.reshape(original_mat, [2, 2, 2]), 1, 2).flatten()
        assert qp.math.allclose(new_mat, expected_new_mat)

        assert batch[0][0] == qp.PauliX(0)
        assert batch[0][1] == qp.SWAP((1, 2))
        assert batch[0][2] == qp.CNOT((0, 1))
        assert batch[0][3] == qp.state()

        pre, post = dev.preprocess_transforms()((tape,))
        original_results = post(dev.execute(pre))
        transformed_results = fn(dev.execute(batch))
        assert qp.math.allclose(original_results, transformed_results)

    def test_transpile_state_with_device_multiple_measurements(self):
        """Test that if a device is provided and a state is measured, then the state will be transposed during post processing."""

        dev = qp.device("default.qubit", wires=(0, 1, 2))

        tape = qp.tape.QuantumScript(
            [qp.PauliX(0), qp.CNOT(wires=(0, 2))], [qp.state(), qp.expval(qp.PauliZ(2))]
        )
        batch, fn = qp.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)], device=dev)

        original_mat = np.arange(8)
        new_mat, _ = fn(((original_mat, 2.0),))
        expected_new_mat = np.swapaxes(np.reshape(original_mat, [2, 2, 2]), 1, 2).flatten()
        assert qp.math.allclose(new_mat, expected_new_mat)

        assert batch[0][0] == qp.PauliX(0)
        assert batch[0][1] == qp.SWAP((1, 2))
        assert batch[0][2] == qp.CNOT((0, 1))
        assert batch[0][3] == qp.state()
        assert batch[0][4] == qp.expval(qp.PauliZ(1))

        pre, post = dev.preprocess_transforms()((tape,))
        original_results = post(dev.execute(pre))
        transformed_results = fn(dev.execute(batch))
        assert qp.math.allclose(original_results[0][0], transformed_results[0])
        assert qp.math.allclose(original_results[0][1], transformed_results[1])

    def test_transpile_with_state_default_mixed(self):
        """Test that if the state is default mixed, state measurements are converted in to density measurements with the device wires."""

        dev = qp.device("default.mixed", wires=(0, 1, 2))

        tape = qp.tape.QuantumScript([qp.PauliX(0), qp.CNOT(wires=(0, 2))], [qp.state()])
        batch, fn = qp.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)], device=dev)

        assert batch[0][-1] == qp.density_matrix(wires=(0, 2, 1))

        pre, post = dev.preprocess_transforms()((tape,))
        original_results = post(dev.execute(pre))
        transformed_results = fn(dev.execute(batch))
        assert qp.math.allclose(original_results, transformed_results)

    def test_transpile_probs_sample_filled_in_wires(self):
        """Test that if probs or sample are requested broadcasted over all wires, transpile fills in the device wires."""
        dev = qp.device("default.qubit", wires=(0, 1, 2))

        tape = qp.tape.QuantumScript(
            [qp.PauliX(0), qp.CNOT(wires=(0, 2))], [qp.probs(), qp.sample()], shots=100
        )
        batch, fn = qp.transforms.transpile(tape, coupling_map=[(0, 1), (1, 2)], device=dev)

        assert batch[0].measurements[0] == qp.probs(wires=(0, 2, 1))
        assert batch[0].measurements[1] == qp.sample(wires=(0, 2, 1))

        pre, post = dev.preprocess_transforms()((tape,))
        original_results = post(dev.execute(pre))[0]
        transformed_results = fn(dev.execute(batch))
        assert qp.math.allclose(original_results[0], transformed_results[0])
        assert qp.math.allclose(original_results[1], transformed_results[1])

    def test_custom_qnode_transform(self):
        """Test that applying the transform to a qnode adds the device to the transform kwargs."""

        dev = qp.device("default.qubit", wires=(0, 1, 2))

        def qfunc():
            return qp.state()

        original_qnode = qp.QNode(qfunc, dev)
        transformed_qnode = transpile(original_qnode, coupling_map=[(0, 1), (1, 2)])

        assert len(transformed_qnode.transform_program) == 1
        assert transformed_qnode.transform_program[0].kwargs["device"] is dev

    def test_qnode_transform_raises_if_device_kwarg(self):
        """Test an error is raised if a device is provided as a keyword argument to a qnode transform."""

        dev = qp.device("default.qubit", wires=[0, 1, 2, 3])

        @qp.qnode(dev)
        def circuit():
            return qp.state()

        with pytest.raises(ValueError, match=r"Cannot provide a "):
            qp.transforms.transpile(
                circuit, coupling_map=[(0, 1), (1, 3), (3, 2), (2, 0)], device=dev
            )
