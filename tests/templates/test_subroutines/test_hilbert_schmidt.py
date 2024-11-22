# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the Hilbert-Schmidt templates.
"""
import pytest
import numpy as np
import pennylane as qml

# pylint: disable=expression-not-assigned


def global_v_circuit(params):
    qml.RZ(params, wires=1)


# pylint: disable=protected-access
@pytest.mark.parametrize("op_type", (qml.HilbertSchmidt, qml.LocalHilbertSchmidt))
def test_flatten_unflatten_standard_checks(op_type):
    """Test the flatten and unflatten methods."""

    u_tape = qml.tape.QuantumScript([qml.Hadamard("a"), qml.Identity("b")])

    v_wires = qml.wires.Wires((0, 1))
    op = op_type([0.1], v_function=global_v_circuit, v_wires=v_wires, u_tape=u_tape)
    qml.ops.functions.assert_valid(op, skip_wire_mapping=True, skip_differentiation=True)

    data, metadata = op._flatten()

    assert data == (0.1,)
    assert metadata == (
        ("v_function", global_v_circuit),
        ("v_wires", v_wires),
        ("u_tape", u_tape),
    )

    assert hash(metadata)

    new_op = type(op)._unflatten(*op._flatten())
    assert qml.math.allclose(op.data, new_op.data)
    assert op.hyperparameters["v_function"] == new_op.hyperparameters["v_function"]
    assert op.hyperparameters["v_wires"] == new_op.hyperparameters["v_wires"]
    for op1, op2 in zip(op.hyperparameters["u_tape"], new_op.hyperparameters["u_tape"]):
        qml.assert_equal(op1, op2)
    assert new_op is not op


class TestHilbertSchmidt:
    """Tests for the Hilbert-Schmidt template."""

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2])
    def test_maximal_cost(self, param):
        """Test that the result is 0 when when the Hilbert-Schmidt inner product is vanishing."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)
        u_tape = qml.tape.QuantumScript.from_queue(q_U)

        def v_function(param):
            qml.Identity(wires=1)
            qml.GlobalPhase(param, wires=1)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        result = hilbert_test(param, v_function, [1], u_tape)[0]
        # This is expected to be 0, since Tr(V†U) = 0
        assert qml.math.allclose(result, 0)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2])
    def test_minimal_cost(self, param):
        """Test that the result is 1 when the Hilbert-Schmidt inner product is maximal."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)
        u_tape = qml.tape.QuantumScript.from_queue(q_U)

        def v_function(param):
            qml.Hadamard(wires=1)
            qml.GlobalPhase(param, wires=1)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        result = hilbert_test(param, v_function, [1], u_tape)[0]
        # This is expected to be 1, since U and V are the same up to a global phase
        assert qml.math.allclose(result, 1)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2, 0.5])
    def test_intermediate_cost_1_qubits(self, param):
        """Test that Hilbert-Schmidt test provides the correct cost for a 1 qubit unitary."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)

        u_tape = qml.tape.QuantumScript.from_queue(q_U)

        def v_function(param):
            qml.RZ(param, wires=1)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        result = hilbert_test(param, v_function, [1], u_tape)[0]

        # We compare the result with 1/d^2 * | Tr(V†U) |^2
        # (see Section 4.1 of https://arxiv.org/pdf/1807.00800 for more details)
        d = 2
        u_matrix = qml.matrix(u_tape)

        with qml.queuing.AnnotatedQueue() as v_queue:
            v_function(param)  # Example parameter value
        v_tape = qml.tape.QuantumScript.from_queue(v_queue)
        v_matrix = qml.matrix(v_tape).reshape(d, d)

        trace = np.trace(np.conj(v_matrix).T @ u_matrix)
        expected = (1 / d**2) * abs(trace) ** 2

        assert qml.math.allclose(result, expected)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2, 0.5])
    def test_intermediate_cost_2_qubits(self, param):
        """Test that Hilbert-Schmidt test provides the correct cost for a 2 qubit unitary."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.SWAP(wires=[0, 1])
            qml.Hadamard(wires=0) @ qml.RY(0.1, wires=1)
            qml.CNOT(wires=[0, 1])

        u_tape = qml.tape.QuantumScript.from_queue(q_U)

        def v_function(param):
            qml.RZ(param, wires=2) @ qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[2, 3])
            qml.RY(param, wires=3) @ qml.Z(3)
            qml.RX(param, wires=2)

        @qml.qnode(qml.device("default.qubit", wires=4))
        def hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        result = hilbert_test(param, v_function, [2, 3], u_tape)[0]

        # We compare the result with 1/d^2 * | Tr(V†U) |^2
        # (see Section 4.1 of https://arxiv.org/pdf/1807.00800 for more details)
        d = 4
        u_matrix = qml.matrix(u_tape, wire_order=[0, 1])

        with qml.queuing.AnnotatedQueue() as v_queue:
            v_function(param)
        v_tape = qml.tape.QuantumScript.from_queue(v_queue)
        v_matrix = qml.matrix(v_tape, wire_order=[2, 3]).reshape(d, d)

        trace = np.trace(np.conj(v_matrix).T @ u_matrix)
        expected = (1 / d**2) * abs(trace) ** 2

        assert qml.math.allclose(result, expected)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2, 0.5])
    def test_intermediate_cost_3_qubits(self, param):
        """Test that Hilbert-Schmidt test provides the correct cost for a 3 qubit unitary."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.RY(0.1, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

        u_tape = qml.tape.QuantumScript.from_queue(q_U)

        def v_function(param):
            qml.RY(param, wires=3)
            qml.CNOT(wires=[3, 4])
            qml.Hadamard(wires=5)

        @qml.qnode(qml.device("default.qubit", wires=6))
        def hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        result = hilbert_test(param, v_function, [3, 4, 5], u_tape)[0]

        # We compare the result with 1/d^2 * | Tr(V†U) |^2
        # (see Section 4.1 of https://arxiv.org/pdf/1807.00800 for more details)
        d = 8
        u_matrix = qml.matrix(u_tape, wire_order=[0, 1, 2])

        with qml.queuing.AnnotatedQueue() as v_queue:
            v_function(param)
        v_tape = qml.tape.QuantumScript.from_queue(v_queue)
        v_matrix = qml.matrix(v_tape, wire_order=[3, 4, 5]).reshape(d, d)

        trace = np.trace(np.conj(v_matrix).T @ u_matrix)
        expected = (1 / d**2) * abs(trace) ** 2

        assert qml.math.allclose(result, expected)

    @pytest.mark.parametrize("op_type", (qml.HilbertSchmidt, qml.LocalHilbertSchmidt))
    def test_map_wires_errors_out(self, op_type):
        """Test that map_wires raises an error."""
        u_tape = qml.tape.QuantumScript([qml.Hadamard("a"), qml.Identity("b")])

        v_wires = qml.wires.Wires((0, 1))
        op = op_type([0.1], v_function=global_v_circuit, v_wires=v_wires, u_tape=u_tape)
        with pytest.raises(NotImplementedError, match="Mapping the wires of HilbertSchmidt"):
            op.map_wires({0: "a", 1: "b"})

    def test_hs_decomposition_1_qubit(self):
        """Test if the HS operation is correctly decomposed for a 1 qubit unitary."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        op = qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=U)

        with qml.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        tape_dec = qml.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qml.H(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.H(wires=[0]),
            qml.QubitUnitary(qml.RZ(0.1, wires=1).matrix().conjugate(), wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.H(0),
        ]
        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.wires == j.wires
            assert qml.math.allclose(i.data, j.data)

    def test_hs_decomposition_2_qubits(self):
        """Test if the HS operation is correctly decomposed for 2 qubits."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.SWAP(wires=[0, 1])

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=2)
            qml.CNOT(wires=[2, 3])

        op = qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2, 3], u_tape=U)

        with qml.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        tape_dec = qml.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
            qml.CNOT(wires=[0, 2]),
            qml.CNOT(wires=[1, 3]),
            qml.SWAP(wires=[0, 1]),
            qml.QubitUnitary(qml.RZ(0.1, wires=[2]).matrix().conjugate(), wires=[2]),
            qml.QubitUnitary(qml.CNOT(wires=[2, 3]).matrix().conjugate(), wires=[2, 3]),
            qml.CNOT(wires=[1, 3]),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.wires == j.wires
            assert qml.math.allclose(i.data, j.data)

    def test_hs_decomposition_2_qubits_custom_wires(self):
        """Test if the HS operation is correctly decomposed for 2 qubits with custom wires."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.SWAP(wires=["a", "b"])

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires="c")
            qml.CNOT(wires=["c", "d"])

        op = qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=["c", "d"], u_tape=U)

        with qml.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        # make sure it works in non-queuing situations too.
        decomp = op.decomposition()

        tape_dec = qml.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qml.Hadamard(wires=["a"]),
            qml.Hadamard(wires=["b"]),
            qml.CNOT(wires=["a", "c"]),
            qml.CNOT(wires=["b", "d"]),
            qml.SWAP(wires=["a", "b"]),
            qml.QubitUnitary(qml.RZ(0.1, wires=["c"]).matrix().conjugate(), wires=["c"]),
            qml.QubitUnitary(qml.CNOT(wires=["b", "d"]).matrix().conjugate(), wires=["c", "d"]),
            qml.CNOT(wires=["b", "d"]),
            qml.CNOT(wires=["a", "c"]),
            qml.Hadamard(wires=["a"]),
            qml.Hadamard(wires=["b"]),
        ]

        for op1, op2 in zip(tape_dec.operations, expected_operations):
            qml.assert_equal(op1, op2)

        for op1, op2 in zip(decomp, expected_operations):
            qml.assert_equal(op1, op2)

    def test_v_not_quantum_function(self):
        """Test that we cannot pass a non quantum function to the HS operation"""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)

        U = qml.tape.QuantumScript.from_queue(q_U)
        with qml.queuing.AnnotatedQueue() as q_v_circuit:
            qml.RZ(0.1, wires=1)

        v_circuit = qml.tape.QuantumScript.from_queue(q_v_circuit)
        with pytest.raises(
            qml.QuantumFunctionError,
            match="The argument v_function must be a callable quantum " "function.",
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=U)

    def test_u_v_same_number_of_wires(self):
        """Test that U and V must have the same number of wires."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.CNOT(wires=[0, 1])

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="U and V must have the same number of wires."
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2], u_tape=U)

    def test_u_quantum_tape(self):
        """Test that U must be a quantum tape."""

        def u_circuit():
            qml.CNOT(wires=[0, 1])

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="The argument u_tape must be a QuantumTape."
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=u_circuit)

    def test_v_wires(self):
        """Test that all wires in V are also in v_wires."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=2)

        with pytest.raises(
            qml.QuantumFunctionError, match="All wires in v_tape must be in v_wires."
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=U)

    def test_distinct_wires(self):
        """Test that U and V have distinct wires."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=0)

        with pytest.raises(
            qml.QuantumFunctionError, match="u_tape and v_tape must act on distinct wires."
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[0], u_tape=U)

    @pytest.mark.jax
    def test_jax_jit(self):
        import jax

        with qml.QueuingManager.stop_recording():
            u_tape = qml.tape.QuantumTape([qml.Hadamard(0)])

        def v_function(params):
            qml.RZ(params[0], wires=1)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(v_params):
            qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=[1], u_tape=u_tape)
            return qml.probs(u_tape.wires + [1])

        jit_circuit = jax.jit(circuit)

        assert qml.math.allclose(circuit(np.array([np.pi / 2])), jit_circuit(np.array([np.pi / 2])))


class TestLocalHilbertSchmidt:
    """Tests for the Local Hilbert-Schmidt template."""

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2])
    def test_maximal_cost(self, param):
        """Test that the result is 0 when when the Hilbert-Schmidt inner product is vanishing."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)
        u_tape = qml.tape.QuantumScript.from_queue(q_U)

        def v_function(param):
            qml.Identity(wires=1)
            qml.GlobalPhase(param, wires=1)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        result = hilbert_test(param, v_function, [1], u_tape)[0]
        # This is expected to be 0, since Tr(V†U) = 0
        assert qml.math.allclose(result, 0)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2])
    def test_minimal_cost(self, param):
        """Test that the result is 1 when the Hilbert-Schmidt inner product is maximal."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)
        u_tape = qml.tape.QuantumScript.from_queue(q_U)

        def v_function(param):
            qml.Hadamard(wires=1)
            qml.GlobalPhase(param, wires=1)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        result = hilbert_test(param, v_function, [1], u_tape)[0]
        # This is expected to be 1, since U and V are the same up to a global phase
        assert qml.math.allclose(result, 1)

    def test_lhs_decomposition_1_qubit(self):
        """Test if the LHS operation is correctly decomposed"""
        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        op = qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=U)

        with qml.queuing.AnnotatedQueue() as q_tape_dec:
            decomp = op.decomposition()

        unqueued_decomp = op.decomposition()

        tape_dec = qml.tape.QuantumScript.from_queue(q_tape_dec)

        for o1, o2 in zip(decomp, tape_dec):
            qml.assert_equal(o1, o2)
        for o1, o2 in zip(decomp, unqueued_decomp):
            qml.assert_equal(o1, o2)

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
            qml.QubitUnitary(qml.RZ(0.1, wires=[1]).matrix().conjugate(), wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.wires == j.wires
            assert qml.math.allclose(i.data, j.data)

    def test_lhs_decomposition_1_qubit_custom_wires(self):
        """Test if the LHS operation is correctly decomposed with custom wires."""
        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires="a")

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires="b")

        op = qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=["b"], u_tape=U)

        with qml.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        tape_dec = qml.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qml.Hadamard(wires=["a"]),
            qml.CNOT(wires=["a", "b"]),
            qml.Hadamard(wires=["a"]),
            qml.QubitUnitary(qml.RZ(0.1, wires=["b"]).matrix().conjugate(), wires=["b"]),
            qml.CNOT(wires=["a", "b"]),
            qml.Hadamard(wires=["a"]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.wires == j.wires
            assert qml.math.allclose(i.data, j.data)

    def test_lhs_decomposition_2_qubits(self):
        """Test if the LHS operation is correctly decomposed for 2 qubits."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.SWAP(wires=[0, 1])

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=2)
            qml.CNOT(wires=[2, 3])

        op = qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2, 3], u_tape=U)

        with qml.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        tape_dec = qml.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
            qml.CNOT(wires=[0, 2]),
            qml.CNOT(wires=[1, 3]),
            qml.SWAP(wires=[0, 1]),
            qml.QubitUnitary(qml.RZ(0.1, wires=[2]).matrix().conjugate(), wires=[2]),
            qml.QubitUnitary(qml.CNOT(wires=[2, 3]).matrix().conjugate(), wires=[2, 3]),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=[0]),
        ]
        assert tape_dec.operations == expected_operations

    def test_qnode_integration(self):
        """Test that the local hilbert schmidt template can be used inside a qnode."""

        u_tape = qml.tape.QuantumTape([qml.CZ(wires=(0, 1))])

        def v_function(params):
            qml.RZ(params[0], wires=2)
            qml.RZ(params[1], wires=3)
            qml.CNOT(wires=[2, 3])
            qml.RZ(params[2], wires=3)
            qml.CNOT(wires=[2, 3])

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def local_hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        # pylint: disable=unsubscriptable-object
        def cost_lhst(parameters, v_function, v_wires, u_tape):
            return (
                1
                - local_hilbert_test(
                    v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape
                )[0]
            )

        res = cost_lhst(
            [3 * qml.numpy.pi / 2, 3 * qml.numpy.pi / 2, qml.numpy.pi / 2],
            v_function=v_function,
            v_wires=[2, 3],
            u_tape=u_tape,
        )

        # The exact analytic expression to be compared against is given by eq. (25) of https://arxiv.org/pdf/1807.00800 with j=1.
        # Unfortunately, we don't have an immediate way to compute such an expression in PennyLane. However, since the
        # local Hilbert-Schmidt test is very similar to the Hilbert-Schmidt test, the compare the latter with the
        # analytic expression and use that as a proxy for correctness.

        assert qml.math.allclose(res, 0.5)
        # the answer is currently 0.5, and I'm going to assume that's correct. This test will let us know
        # if the answer changes.

    def test_v_not_quantum_function(self):
        """Test that we cannot pass a non quantum function to the HS operation"""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)

        U = qml.tape.QuantumScript.from_queue(q_U)
        with qml.queuing.AnnotatedQueue() as q_v_circuit:
            qml.RZ(0.1, wires=1)

        v_circuit = qml.tape.QuantumScript.from_queue(q_v_circuit)
        with pytest.raises(
            qml.QuantumFunctionError,
            match="The argument v_function must be a callable quantum " "function.",
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=U)

    def test_u_v_same_number_of_wires(self):
        """Test that U and V must have the same number of wires."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.CNOT(wires=[0, 1])

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="U and V must have the same number of wires."
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2], u_tape=U)

    def test_u_quantum_tape(self):
        """Test that U must be a quantum tape."""

        def u_circuit():
            qml.CNOT(wires=[0, 1])

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="The argument u_tape must be a QuantumTape."
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=u_circuit)

    def test_v_wires(self):
        """Test that all wires in V are also in v_wires."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=2)

        with pytest.raises(
            qml.QuantumFunctionError, match="All wires in v_tape must be in v_wires."
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_tape=U)

    def test_distinct_wires(self):
        """Test that U and V have distinct wires."""

        with qml.queuing.AnnotatedQueue() as q_U:
            qml.Hadamard(wires=0)

        U = qml.tape.QuantumScript.from_queue(q_U)

        def v_circuit(params):
            qml.RZ(params[0], wires=0)

        with pytest.raises(
            qml.QuantumFunctionError, match="u_tape and v_tape must act on distinct wires."
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[0], u_tape=U)

    @pytest.mark.jax
    def test_jit(self):
        import jax

        with qml.QueuingManager.stop_recording():
            u_tape = qml.tape.QuantumTape([qml.CZ(wires=(0, 1))])

        def v_function(params):
            qml.RZ(params[0], wires=2)
            qml.RZ(params[1], wires=3)
            qml.CNOT(wires=[2, 3])
            qml.RZ(params[2], wires=3)
            qml.CNOT(wires=[2, 3])

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(v_params):
            qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=[2, 3], u_tape=u_tape)
            return qml.probs(u_tape.wires + [2, 3])

        jit_circuit = jax.jit(circuit)

        params = np.array([3 * np.pi / 2, 3 * np.pi / 2, np.pi / 2])
        assert qml.math.allclose(circuit(params), jit_circuit(params))
