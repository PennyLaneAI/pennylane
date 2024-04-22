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

import pennylane as qml


def global_v_circuit(params):
    qml.RZ(params, wires=1)


# pylint: disable=protected-access
@pytest.mark.parametrize("op_type", (qml.HilbertSchmidt, qml.LocalHilbertSchmidt))
def test_flatten_unflatten_standard_checks(op_type):
    """Test the flatten and unflatten methods."""
    u_tape = qml.tape.QuantumScript([qml.Hadamard("a"), qml.Identity("b")])

    v_wires = qml.wires.Wires((0, 1))
    op = op_type([0.1], v_function=global_v_circuit, v_wires=v_wires, u_tape=u_tape)
    qml.ops.functions.assert_valid(op)

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
        assert qml.equal(op1, op2)
    assert new_op is not op


class TestHilbertSchmidt:
    """Tests for the Hilbert-Schmidt template."""

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
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
            qml.RZ(-0.1, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]
        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

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
            qml.RZ(-0.1, wires=[2]),
            qml.CNOT(wires=[2, 3]),
            qml.CNOT(wires=[1, 3]),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

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
            qml.RZ(-0.1, wires=["c"]),
            qml.CNOT(wires=["c", "d"]),
            qml.CNOT(wires=["b", "d"]),
            qml.CNOT(wires=["a", "c"]),
            qml.Hadamard(wires=["a"]),
            qml.Hadamard(wires=["b"]),
        ]

        for op1, op2 in zip(tape_dec.operations, expected_operations):
            assert qml.equal(op1, op2)

        for op1, op2 in zip(decomp, expected_operations):
            assert qml.equal(op1, op2)

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


class TestLocalHilbertSchmidt:
    """Tests for the Local Hilbert-Schmidt template."""

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

        assert all(qml.equal(o1, o2) for o1, o2 in zip(decomp, tape_dec))
        assert all(qml.equal(o1, o2) for o1, o2 in zip(decomp, unqueued_decomp))

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
            qml.RZ(-0.1, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

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
            qml.RZ(-0.1, wires=["b"]),
            qml.CNOT(wires=["a", "b"]),
            qml.Hadamard(wires=["a"]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

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
            qml.RZ(-0.1, wires=[2]),
            qml.CNOT(wires=[2, 3]),
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
