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
import copy

import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule

# pylint: disable=expression-not-assigned


# pylint: disable=protected-access
@pytest.mark.jax
@pytest.mark.parametrize("op_type", (qp.HilbertSchmidt, qp.LocalHilbertSchmidt))
def test_flatten_unflatten_standard_checks(op_type):
    """Test the flatten and unflatten methods."""

    U = (qp.Hadamard("a"), qp.Identity("b"))
    V = (qp.RZ(0.1, wires=0), qp.RZ(0.2, wires=1))

    op = op_type(V, U)
    qp.ops.functions.assert_valid(op, skip_wire_mapping=True, skip_differentiation=True)

    data, metadata = op._flatten()
    assert data == (V, U)
    assert not metadata

    new_op = type(op)._unflatten(*op._flatten())
    assert qp.math.allclose(op.data, new_op.data)

    for op1, op2 in zip(op.hyperparameters["U"], new_op.hyperparameters["U"]):
        qp.assert_equal(op1, op2)
    for op1, op2 in zip(op.hyperparameters["V"], new_op.hyperparameters["V"]):
        qp.assert_equal(op1, op2)

    assert new_op is not op


class TestHilbertSchmidt:
    """Tests for the Hilbert-Schmidt template."""

    @pytest.mark.parametrize(
        ("V", "U"),
        [
            (qp.RZ(0, wires=1), qp.Hadamard(0)),
            ((qp.RZ(0, wires=1),), (qp.Hadamard(0),)),
            (qp.RZ(0, wires=1), (qp.Hadamard(0))),
            ((qp.RZ(0, wires=1),), qp.Hadamard(0)),
        ],
    )
    def test_HilbertSchmidt(self, V, U):
        """Test the HilbertSchmidt template with multiple operations format."""

        @qp.qnode(device=qp.device("default.qubit", wires=2))
        def hilbert_test(V, U):
            qp.HilbertSchmidt(V, U)
            return qp.probs()

        def cost_hst(V, U):
            # pylint:disable=unsubscriptable-object
            return 1 - hilbert_test(V, U)[0]

        res = cost_hst(V, U)
        expected = 1.0
        assert np.isclose(res, expected)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2])
    def test_maximal_cost(self, param):
        """Test that the result is 0 when when the Hilbert-Schmidt inner product is vanishing."""

        U = [qp.Hadamard(wires=0)]
        V = [qp.Identity(wires=1), qp.GlobalPhase(param, wires=1)]

        @qp.qnode(qp.device("default.qubit", wires=2))
        def hilbert_test(V, U):
            qp.HilbertSchmidt(V, U)
            return qp.probs()

        result = hilbert_test(V, U)[0]
        # This is expected to be 0, since Tr(V†U) = 0
        assert qp.math.allclose(result, 0)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2])
    def test_minimal_cost(self, param):
        """Test that the result is 1 when the Hilbert-Schmidt inner product is maximal."""

        U = [qp.Hadamard(0)]
        V = [qp.Hadamard(wires=1), qp.GlobalPhase(param, wires=1)]

        @qp.qnode(qp.device("default.qubit", wires=2))
        def hilbert_test(V, U):
            qp.HilbertSchmidt(V, U)
            return qp.probs()

        result = hilbert_test(V, U)[0]
        # This is expected to be 1, since U and V are the same up to a global phase
        assert qp.math.allclose(result, 1)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2, 0.5])
    def test_intermediate_cost_1_qubits(self, param):
        """Test that Hilbert-Schmidt test provides the correct cost for a 1 qubit unitary."""

        U = [qp.Hadamard(0)]
        V = qp.RZ(param, wires=1)

        @qp.qnode(qp.device("default.qubit", wires=2))
        def hilbert_test(V, U):
            qp.HilbertSchmidt(V, U)
            return qp.probs()

        result = hilbert_test(V, U)[0]

        # We compare the result with 1/d^2 * | Tr(V†U) |^2
        # (see Section 4.1 of https://arxiv.org/pdf/1807.00800 for more details)
        d = 2
        u_matrix = qp.matrix(U[0])
        v_matrix = qp.matrix(V).reshape(d, d)

        trace = np.trace(np.conj(v_matrix).T @ u_matrix)
        expected = (1 / d**2) * abs(trace) ** 2

        assert qp.math.allclose(result, expected)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2, 0.5])
    def test_intermediate_cost_2_qubits(self, param):
        """Test that Hilbert-Schmidt test provides the correct cost for a 2 qubit unitary."""

        U = [
            qp.SWAP(wires=[0, 1]),
            qp.Hadamard(wires=0) @ qp.RY(0.1, wires=1),
            qp.CNOT(wires=[0, 1]),
        ]

        V = [
            qp.RZ(param, wires=2) @ qp.CNOT(wires=[2, 3]),
            qp.CNOT(wires=[2, 3]),
            qp.RY(param, wires=3) @ qp.Z(3),
            qp.RX(param, wires=2),
        ]

        @qp.qnode(qp.device("default.qubit", wires=4))
        def hilbert_test(V, U):
            qp.HilbertSchmidt(V, U)
            return qp.probs()

        result = hilbert_test(V, U)[0]

        u_tape = qp.tape.QuantumScript(U)
        v_tape = qp.tape.QuantumScript(V)

        # We compare the result with 1/d^2 * | Tr(V†U) |^2
        # (see Section 4.1 of https://arxiv.org/pdf/1807.00800 for more details)
        d = 4
        u_matrix = qp.matrix(u_tape, wire_order=[0, 1])
        v_matrix = qp.matrix(v_tape, wire_order=[2, 3]).reshape(d, d)

        trace = np.trace(np.conj(v_matrix).T @ u_matrix)
        expected = (1 / d**2) * abs(trace) ** 2
        assert qp.math.allclose(result, expected)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2, 0.5])
    def test_intermediate_cost_3_qubits(self, param):
        """Test that Hilbert-Schmidt test provides the correct cost for a 3 qubit unitary."""

        U = [qp.RY(0.1, wires=0), qp.CNOT(wires=[0, 1]), qp.CNOT(wires=[1, 2])]
        V = [qp.RY(param, wires=3), qp.CNOT(wires=[3, 4]), qp.Hadamard(wires=5)]

        @qp.qnode(qp.device("default.qubit", wires=6))
        def hilbert_test(V, U):
            qp.HilbertSchmidt(V, U)
            return qp.probs()

        result = hilbert_test(V, U)[0]

        u_tape = qp.tape.QuantumScript(U)
        v_tape = qp.tape.QuantumScript(V)

        # We compare the result with 1/d^2 * | Tr(V†U) |^2
        # (see Section 4.1 of https://arxiv.org/pdf/1807.00800 for more details)
        d = 8
        u_matrix = qp.matrix(u_tape, wire_order=[0, 1, 2])
        v_matrix = qp.matrix(v_tape, wire_order=[3, 4, 5]).reshape(d, d)

        trace = np.trace(np.conj(v_matrix).T @ u_matrix)
        expected = (1 / d**2) * abs(trace) ** 2
        assert qp.math.allclose(result, expected)

    @pytest.mark.parametrize("op_type", (qp.HilbertSchmidt, qp.LocalHilbertSchmidt))
    def test_map_wires_errors_out(self, op_type):
        """Test that map_wires raises an error."""

        U = [qp.Hadamard("a"), qp.Identity("b")]
        V = [qp.RZ(0.1, wires=0), qp.RZ(0.1, wires=1)]

        op = op_type(V=V, U=U)

        with pytest.raises(NotImplementedError, match="Mapping the wires of HilbertSchmidt"):
            op.map_wires({0: "a", 1: "b"})

    def test_hs_decomposition_1_qubit(self):
        """Test if the HS operation is correctly decomposed for a 1 qubit unitary."""

        U = qp.Hadamard(wires=0)
        V = qp.RZ(0.1, wires=1)

        op = qp.HilbertSchmidt(V, U)

        with qp.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        tape_dec = qp.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qp.H(wires=[0]),
            qp.CNOT(wires=[0, 1]),
            qp.H(wires=[0]),
            qp.QubitUnitary(qp.RZ(0.1, wires=1).matrix().conjugate(), wires=[1]),
            qp.CNOT(wires=[0, 1]),
            qp.H(0),
        ]
        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.wires == j.wires
            assert qp.math.allclose(i.data, j.data)

    def test_hs_decomposition_2_qubits(self):
        """Test if the HS operation is correctly decomposed for 2 qubits."""

        U = qp.SWAP(wires=[0, 1])
        V = [qp.RZ(0.1, wires=2), qp.CNOT(wires=[2, 3])]

        op = qp.HilbertSchmidt(V, U)

        with qp.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        tape_dec = qp.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qp.Hadamard(wires=[0]),
            qp.Hadamard(wires=[1]),
            qp.CNOT(wires=[0, 2]),
            qp.CNOT(wires=[1, 3]),
            qp.SWAP(wires=[0, 1]),
            qp.QubitUnitary(qp.RZ(0.1, wires=[2]).matrix().conjugate(), wires=[2]),
            qp.QubitUnitary(qp.CNOT(wires=[2, 3]).matrix().conjugate(), wires=[2, 3]),
            qp.CNOT(wires=[1, 3]),
            qp.CNOT(wires=[0, 2]),
            qp.Hadamard(wires=[0]),
            qp.Hadamard(wires=[1]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.wires == j.wires
            assert qp.math.allclose(i.data, j.data)

    def test_hs_decomposition_2_qubits_custom_wires(self):
        """Test if the HS operation is correctly decomposed for 2 qubits with custom wires."""

        U = qp.SWAP(wires=["a", "b"])
        V = [qp.RZ(0.1, wires="c"), qp.CNOT(wires=["c", "d"])]

        op = qp.HilbertSchmidt(V, U)

        with qp.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        # make sure it works in non-queuing situations too.
        decomp = op.decomposition()

        tape_dec = qp.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qp.Hadamard(wires=["a"]),
            qp.Hadamard(wires=["b"]),
            qp.CNOT(wires=["a", "c"]),
            qp.CNOT(wires=["b", "d"]),
            qp.SWAP(wires=["a", "b"]),
            qp.QubitUnitary(qp.RZ(0.1, wires=["c"]).matrix().conjugate(), wires=["c"]),
            qp.QubitUnitary(qp.CNOT(wires=["b", "d"]).matrix().conjugate(), wires=["c", "d"]),
            qp.CNOT(wires=["b", "d"]),
            qp.CNOT(wires=["a", "c"]),
            qp.Hadamard(wires=["a"]),
            qp.Hadamard(wires=["b"]),
        ]

        for op1, op2 in zip(tape_dec.operations, expected_operations):
            qp.assert_equal(op1, op2)

        for op1, op2 in zip(decomp, expected_operations):
            qp.assert_equal(op1, op2)

    def test_data(self):
        """Test that the data property gets and sets the correct values"""
        op = qp.HilbertSchmidt(
            [qp.RX(1, wires=0), qp.RX(2, wires=1)], [qp.RY(3, wires=2), qp.RZ(4, wires=3)]
        )
        assert op.data == (1, 2, 3, 4)
        op.data = [4, 5, 6, 7]
        assert op.data == (4, 5, 6, 7)

    def test_copy(self):
        """Test that a HilbertSchmidt operator can be copied."""
        orig_op = qp.HilbertSchmidt(
            [qp.RX(1, wires=0), qp.RX(2, wires=1)], [qp.RY(3, wires=2), qp.RZ(4, wires=3)]
        )
        copy_op = copy.copy(orig_op)
        qp.assert_equal(orig_op, copy_op)

        # Ensure the (nested) operations are copied instead of aliased.
        assert orig_op is not copy_op

        orig_U = orig_op.hyperparameters["U"]
        copy_U = copy_op.hyperparameters["U"]
        assert all(u1 is not u2 for u1, u2 in zip(orig_U, copy_U))

        orig_V = orig_op.hyperparameters["V"]
        copy_V = copy_op.hyperparameters["V"]
        assert all(v1 is not v2 for v1, v2 in zip(orig_V, copy_V))

    @pytest.mark.parametrize(
        ("U", "V", "results"),
        [
            (
                qp.Hadamard(wires=0),
                qp.RX(0, wires=1),
                [
                    qp.H(0),
                    qp.CNOT(wires=[0, 1]),
                    qp.H(0),
                    qp.QubitUnitary(
                        [[1.0 - 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 - 0.0j]], wires=[1]
                    ),
                    qp.CNOT(wires=[0, 1]),
                    qp.H(0),
                ],
            ),
        ],
    )
    def test_queuing_ops(self, U, V, results):
        """Test that qp.HilbertSchmidt queues operations in the correct order."""
        with qp.tape.QuantumTape() as tape:
            qp.HilbertSchmidt(V, U)

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.wires == results[idx].wires
            assert qp.math.allclose(val.parameters, results[idx].parameters)

    def test_v_not_operator(self):
        """Test that V must be a an Operator or an iterable of Operators."""

        U = qp.Hadamard(wires=0)
        V = "qp.RZ(0.1, wires=1)"

        with pytest.raises(
            ValueError,
            match="The argument 'V' must be an Operator or an iterable of Operators.",
        ):
            qp.HilbertSchmidt(V, U)

    def test_u_not_operator(self):
        """Test that U must be a an Operator or an iterable of Operators."""

        U = "qp.CNOT(wires=[0, 1])"
        V = qp.RZ(0.1, wires=1)

        with pytest.raises(
            ValueError,
            match="The argument 'U' must be an Operator or an iterable of Operators.",
        ):
            qp.HilbertSchmidt(V, U)

    def test_u_v_same_number_of_wires(self):
        """Test that U and V must have the same number of wires."""

        U = qp.CNOT(wires=[0, 1])
        V = qp.RZ(0.1, wires=1)

        with pytest.raises(
            ValueError,
            match="U and V must have the same number of wires.",
        ):
            qp.HilbertSchmidt(V, U)

    def test_distinct_wires(self):
        """Test that U and V have distinct wires."""

        U = qp.Hadamard(wires=0)
        V = qp.RZ(0.1, wires=0)

        with pytest.raises(
            ValueError,
            match="Operators in U and V must act on distinct wires.",
        ):
            qp.HilbertSchmidt(V, U)

    @pytest.mark.jax
    def test_jax_jit(self):
        import jax

        U = qp.Hadamard(0)

        @qp.qnode(device=qp.device("default.qubit", wires=2))
        def circuit(params):
            qp.HilbertSchmidt(V=qp.RZ(params[0], wires=1), U=U)
            return qp.probs()

        jit_circuit = jax.jit(circuit)
        assert qp.math.allclose(circuit(np.array([np.pi / 2])), jit_circuit(np.array([np.pi / 2])))

    DECOMP_PARAMS = [
        (qp.SWAP(wires=[0, 1]), [qp.RZ(0.1, wires=2), qp.CNOT(wires=[2, 3])]),
        ([qp.RY(0.1, wires=2), qp.SWAP(wires=[2, 3])], qp.CNOT(wires=[0, 1])),
        (
            [qp.RX(0.1, wires=0), qp.SWAP(wires=[1, 2])],
            [qp.RZ(0.1, wires=3), qp.CNOT(wires=[5, 4])],
        ),
    ]

    @pytest.mark.capture
    @pytest.mark.parametrize(("U", "V"), DECOMP_PARAMS)
    def test_decomposition_new(self, U, V):
        op = qp.HilbertSchmidt(V, U)
        for rule in qp.list_decomps(qp.HilbertSchmidt):
            _test_decomposition_rule(op, rule)


class TestLocalHilbertSchmidt:
    """Tests for the Local Hilbert-Schmidt template."""

    @pytest.mark.parametrize(
        "U", [[qp.CZ(wires=(0, 1))], (qp.CZ(wires=(0, 1)),), qp.CZ(wires=(0, 1))]
    )
    def test_LocalHilbertSchmidt(self, U):
        """Test the LocalHilbertSchmidt template with multiple operations format."""

        def V_function(params):
            return [
                qp.RZ(params[0], wires=2),
                qp.RZ(params[1], wires=3),
                qp.CNOT(wires=[2, 3]),
                qp.RZ(params[2], wires=3),
                qp.CNOT(wires=[2, 3]),
            ]

        @qp.qnode(device=qp.device("default.qubit", wires=4))
        def local_hilbert_test(V, U):
            qp.LocalHilbertSchmidt(V, U)
            return qp.probs()

        def cost_lhst(V, U):
            # pylint:disable=unsubscriptable-object
            return 1 - local_hilbert_test(V, U)[0]

        v_params = [3 * np.pi / 2, 3 * np.pi / 2, np.pi / 2]
        V = V_function(v_params)
        res = cost_lhst(V, U)
        expected = 0.5
        assert np.isclose(res, expected)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2])
    def test_maximal_cost(self, param):
        """Test that the result is 0 when when the Hilbert-Schmidt inner product is vanishing."""

        U = [qp.Hadamard(wires=0)]
        V = [qp.Identity(wires=1), qp.GlobalPhase(param, wires=1)]

        @qp.qnode(qp.device("default.qubit", wires=2))
        def hilbert_test(V, U):
            qp.LocalHilbertSchmidt(V, U)
            return qp.probs()

        result = hilbert_test(V, U)[0]
        # This is expected to be 0, since Tr(V†U) = 0
        assert qp.math.allclose(result, 0)

    @pytest.mark.parametrize("param", [0.1, -np.pi / 2])
    def test_minimal_cost(self, param):
        """Test that the result is 1 when the Hilbert-Schmidt inner product is maximal."""

        U = [qp.Hadamard(0)]
        V = [qp.Hadamard(wires=1), qp.GlobalPhase(param, wires=1)]

        @qp.qnode(qp.device("default.qubit", wires=2))
        def hilbert_test(V, U):
            qp.LocalHilbertSchmidt(V, U)
            return qp.probs()

        result = hilbert_test(V, U)[0]
        # This is expected to be 1, since U and V are the same up to a global phase
        assert qp.math.allclose(result, 1)

    def test_lhs_decomposition_1_qubit(self):
        """Test if the LHS operation is correctly decomposed"""

        U = qp.Hadamard(wires=0)
        V = qp.RZ(0.1, wires=1)

        op = qp.LocalHilbertSchmidt(V, U)

        with qp.queuing.AnnotatedQueue() as q_tape_dec:
            decomp = op.decomposition()

        unqueued_decomp = op.decomposition()

        tape_dec = qp.tape.QuantumScript.from_queue(q_tape_dec)

        for o1, o2 in zip(decomp, tape_dec):
            qp.assert_equal(o1, o2)
        for o1, o2 in zip(decomp, unqueued_decomp):
            qp.assert_equal(o1, o2)

        expected_operations = [
            qp.Hadamard(wires=[0]),
            qp.CNOT(wires=[0, 1]),
            qp.Hadamard(wires=[0]),
            qp.QubitUnitary(qp.RZ(0.1, wires=[1]).matrix().conjugate(), wires=[1]),
            qp.CNOT(wires=[0, 1]),
            qp.Hadamard(wires=[0]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.wires == j.wires
            assert qp.math.allclose(i.data, j.data)

    def test_lhs_decomposition_1_qubit_custom_wires(self):
        """Test if the LHS operation is correctly decomposed with custom wires."""

        U = qp.Hadamard(wires="a")
        V = qp.RZ(0.1, wires="b")

        op = qp.LocalHilbertSchmidt(V, U)

        with qp.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        tape_dec = qp.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qp.Hadamard(wires=["a"]),
            qp.CNOT(wires=["a", "b"]),
            qp.Hadamard(wires=["a"]),
            qp.QubitUnitary(qp.RZ(0.1, wires=["b"]).matrix().conjugate(), wires=["b"]),
            qp.CNOT(wires=["a", "b"]),
            qp.Hadamard(wires=["a"]),
        ]

        for i, j in zip(tape_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.wires == j.wires
            assert qp.math.allclose(i.data, j.data)

    def test_lhs_decomposition_2_qubits(self):
        """Test if the LHS operation is correctly decomposed for 2 qubits."""

        U = qp.SWAP(wires=[0, 1])
        V = [qp.RZ(0.1, wires=2), qp.CNOT(wires=[2, 3])]

        op = qp.LocalHilbertSchmidt(V, U)

        with qp.queuing.AnnotatedQueue() as q_tape_dec:
            op.decomposition()

        tape_dec = qp.tape.QuantumScript.from_queue(q_tape_dec)
        expected_operations = [
            qp.Hadamard(wires=[0]),
            qp.Hadamard(wires=[1]),
            qp.CNOT(wires=[0, 2]),
            qp.CNOT(wires=[1, 3]),
            qp.SWAP(wires=[0, 1]),
            qp.QubitUnitary(qp.RZ(0.1, wires=[2]).matrix().conjugate(), wires=[2]),
            qp.QubitUnitary(qp.CNOT(wires=[2, 3]).matrix().conjugate(), wires=[2, 3]),
            qp.CNOT(wires=[0, 2]),
            qp.Hadamard(wires=[0]),
        ]
        assert tape_dec.operations == expected_operations

    DECOMP_PARAMS = [
        (qp.SWAP(wires=[0, 1]), [qp.RZ(0.1, wires=2), qp.CNOT(wires=[2, 3])]),
        ([qp.RY(0.1, wires=2), qp.SWAP(wires=[2, 3])], qp.CNOT(wires=[0, 1])),
        (
            [qp.RX(0.1, wires=0), qp.SWAP(wires=[1, 2])],
            [qp.RZ(0.1, wires=3), qp.CNOT(wires=[5, 4])],
        ),
    ]

    @pytest.mark.capture
    @pytest.mark.parametrize(("U", "V"), DECOMP_PARAMS)
    def test_local_decomposition_new(self, U, V):
        op = qp.LocalHilbertSchmidt(V, U)
        for rule in qp.list_decomps(qp.LocalHilbertSchmidt):
            _test_decomposition_rule(op, rule)

    def test_qnode_integration(self):
        """Test that the local hilbert schmidt template can be used inside a qnode."""

        U = [qp.CZ(wires=(0, 1))]

        def V_function(params):
            return [
                qp.RZ(params[0], wires=2),
                qp.RZ(params[1], wires=3),
                qp.CNOT(wires=[2, 3]),
                qp.RZ(params[2], wires=3),
                qp.CNOT(wires=[2, 3]),
            ]

        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev)
        def local_hilbert_test(V, U):
            qp.LocalHilbertSchmidt(V, U)
            return qp.probs()

        # pylint: disable=unsubscriptable-object
        def cost_lhst(V, U):
            return 1 - local_hilbert_test(V, U)[0]

        v_params = [3 * qp.numpy.pi / 2, 3 * qp.numpy.pi / 2, qp.numpy.pi / 2]
        V = V_function(v_params)
        res = cost_lhst(V, U)

        # The exact analytic expression to be compared against is given by eq. (25) of https://arxiv.org/pdf/1807.00800 with j=1.
        # Unfortunately, we don't have an immediate way to compute such an expression in PennyLane. However, since the
        # local Hilbert-Schmidt test is very similar to the Hilbert-Schmidt test, we compare the latter with the
        # analytic expression and use that as a proxy for correctness.

        assert qp.math.allclose(res, 0.5)
        # the answer is currently 0.5, and I'm going to assume that's correct. This test will let us know
        # if the answer changes.

    def test_v_not_operator(self):
        """Test that V must be a an Operator or an iterable of Operators."""

        U = qp.Hadamard(wires=0)
        V = "qp.RZ(0.1, wires=1)"

        with pytest.raises(
            ValueError,
            match="The argument 'V' must be an Operator or an iterable of Operators.",
        ):
            qp.LocalHilbertSchmidt(V, U)

    def test_u_not_operator(self):
        """Test that U must be a an Operator or an iterable of Operators."""

        U = "qp.CNOT(wires=[0, 1])"
        V = qp.RZ(0.1, wires=1)

        with pytest.raises(
            ValueError,
            match="The argument 'U' must be an Operator or an iterable of Operators.",
        ):
            qp.LocalHilbertSchmidt(V, U)

    def test_u_v_same_number_of_wires(self):
        """Test that U and V must have the same number of wires."""

        U = qp.CNOT(wires=[0, 1])
        V = qp.RZ(0.1, wires=1)

        with pytest.raises(
            ValueError,
            match="U and V must have the same number of wires.",
        ):
            qp.LocalHilbertSchmidt(V, U)

    def test_distinct_wires(self):
        """Test that U and V have distinct wires."""

        U = qp.Hadamard(wires=0)
        V = qp.RZ(0.1, wires=0)

        with pytest.raises(
            ValueError,
            match="Operators in U and V must act on distinct wires.",
        ):
            qp.LocalHilbertSchmidt(V, U)

    @pytest.mark.jax
    def test_jit(self):
        import jax

        U = qp.CZ(wires=(0, 1))

        def V_function(params):
            return [
                qp.RZ(params[0], wires=2),
                qp.RZ(params[1], wires=3),
                qp.CNOT(wires=[2, 3]),
                qp.RZ(params[2], wires=3),
                qp.CNOT(wires=[2, 3]),
            ]

        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev)
        def circuit(v_params):
            qp.LocalHilbertSchmidt(V=V_function(v_params), U=U)
            return qp.probs()

        jit_circuit = jax.jit(circuit)

        params = np.array([3 * np.pi / 2, 3 * np.pi / 2, np.pi / 2])
        assert qp.math.allclose(circuit(params), jit_circuit(params))
