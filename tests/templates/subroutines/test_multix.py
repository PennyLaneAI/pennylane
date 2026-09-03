# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the MultiX template."""

import numpy as np
import pytest
from scipy import sparse

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.typing import AbstractArray, AbstractWires, Bool, Float, Int, Wire
from pennylane.wires import Wires


class TestInitialization:
    """Tests for MultiX initialization and input canonicalization."""

    @pytest.mark.parametrize(
        ("bitstring_input", "wires_input"),
        [
            ([0, 1, 1, 0], ("a", "b", "c", "d")),
            ([False, True, True, False], ("a", "b", "c", "d")),
            ((False, True, False), ("a", "b", "c")),
            ([False, True], (0, 1)),
            (np.array([0, 0]), ["Alice", "Bob"]),
            (np.array([False, True]), ["Alice", "Bob"]),
            (Int[3], Wire[3]),
            (Bool[2], Wire[2]),
        ],
    )
    def test_input_arguments_parsed_correctly(self, bitstring_input, wires_input):
        """Tests that MultiX handles and sanitizes its input arguments correctly."""
        op = qp.MultiX(bitstring_input, wires=wires_input)

        assert isinstance(op.bitstring, (np.ndarray, AbstractArray))
        assert isinstance(op.wires, (Wires, AbstractWires))
        assert op.is_fully_abstract == (
            isinstance(bitstring_input, AbstractArray) or isinstance(wires_input, AbstractWires)
        )
        assert op.bitstring.dtype == bool
        if not isinstance(bitstring_input, AbstractArray):
            assert np.all(op.bitstring == np.array(bitstring_input, dtype=bool))
        if not isinstance(wires_input, AbstractWires):
            assert op.wires == Wires(wires_input)

        assert op.dynamic_args == {"bitstring": op.bitstring}
        assert op.wire_args == {"wires": op.wires}
        assert op.num_wires == len(wires_input)
        assert op.num_params == 1
        assert op.grad_method is None

    def test_canonicalize_inputs_does_not_validate_or_cast(self):
        """Tests that canonicalization changes input containers but does not validate or cast."""
        bitstring = np.array([0.0, 1.0])
        wires = ("a", "b")

        # pylint: disable-next=protected-access
        canonical_bitstring, canonical_wires = qp.MultiX._canonicalize_inputs(bitstring, wires)

        assert canonical_bitstring is bitstring
        assert canonical_bitstring.dtype == float
        assert canonical_wires == Wires(wires)

    def test_custom_repr_override(self):
        """Tests that the Boolean bitstring is represented with integer values."""
        op = qp.MultiX([True, False, True], wires=["a", "b", "c"])

        assert repr(op) == "MultiX([1 0 1], wires=['a', 'b', 'c'])"


class TestValidation:
    """Tests for MultiX input and operator validation."""

    @pytest.mark.capture
    def test_standard_checks(self):
        """Runs the standard Operator2 validity checks for MultiX."""
        op = qp.MultiX([1, 0, 1], wires=[0, 1, 2])
        qp.ops.functions.assert_valid(op)

    @pytest.mark.parametrize(
        ("bitstring", "wires", "error_match"),
        [
            ([0, 1, 1, 0], ["a", "b", "c"], "length"),
            (Int[2], Wire[3], "length"),
            ([0, 1, 2], ["a", "b", "c"], "binary"),
            ([[0, 1, 0]], ["a", "b", "c"], "dimension"),
            (Int[1, 3], Wire[1], "dimension"),
            (np.array([0.0, 1.0]), ["a", "b"], "boolean"),
            (Float[3], Wire[3], "boolean"),
            ([0], [], "wire"),
        ],
    )
    def test_invalid_arguments(self, bitstring, wires, error_match):
        """Tests that MultiX raises clear errors when input arguments are invalid."""
        with pytest.raises(ValueError, match=error_match):
            qp.MultiX(bitstring, wires=wires)


class TestMatrixGeneration:
    """Tests for dense and sparse matrix representations."""

    @pytest.mark.parametrize(
        ("bitstring", "expected_matrix"),
        [
            ([0], np.eye(2)),
            ([1], qp.X.compute_matrix()),
            ([1, 0], np.kron(qp.X.compute_matrix(), np.eye(2))),
            ([0, 1, 1], np.kron(np.eye(2), np.kron(qp.X.compute_matrix(), qp.X.compute_matrix()))),
        ],
    )
    def test_matrix(self, bitstring, expected_matrix):
        """Tests that MultiX computes the tensor product selected by the bitstring."""
        op = qp.MultiX(bitstring, wires=range(len(bitstring)))

        assert np.allclose(op.matrix(), expected_matrix)

    @pytest.mark.parametrize("sparse_format", ["csr", "csc", "lil", "coo"])
    @pytest.mark.parametrize(
        ("bitstring", "expected_matrix"),
        [
            ([0], sparse.csr_matrix(np.eye(2))),
            ([1], sparse.csr_matrix(qp.X.compute_matrix())),
            ([1, 0], sparse.csr_matrix(np.kron(qp.X.compute_matrix(), np.eye(2)))),
            (
                [0, 1, 1],
                sparse.csr_matrix(
                    np.kron(np.eye(2), np.kron(qp.X.compute_matrix(), qp.X.compute_matrix()))
                ),
            ),
        ],
    )
    def test_sparse_matrix(self, bitstring, expected_matrix, sparse_format):
        """Tests that MultiX computes its sparse matrix in the requested format."""
        wires = range(len(bitstring))
        op = qp.MultiX(bitstring, wires=wires)

        expected_matrix_correct_format = expected_matrix.asformat(sparse_format)
        sparse_matrix = op.sparse_matrix(format=sparse_format)
        dense_matrix = op.matrix()

        assert qp.MultiX.has_sparse_matrix
        assert sparse_matrix.format == sparse_format
        assert (sparse_matrix - expected_matrix_correct_format).nnz == 0
        assert np.allclose(sparse_matrix.toarray(), dense_matrix)

    @pytest.mark.jax
    def test_jit_matrix(self):
        """Tests that MultiX matrix computations work with JAX bitstrings."""

        import jax  # pylint: disable=import-outside-toplevel

        bitstring = jax.numpy.array([1, 0])
        matrix_fn = jax.jit(lambda bits: qp.MultiX.compute_matrix(bits, wires=[0, 1]))
        expected = np.kron(qp.X.compute_matrix(), np.eye(2))
        assert qp.math.allclose(matrix_fn(bitstring), expected)


class TestEigvalDiagonalization:
    """Tests for eigenvalues and diagonalizing gates."""

    @pytest.mark.parametrize(
        ("bitstring", "expected_eigvals"),
        [
            ([0], [1, 1]),
            ([1], [1, -1]),
            ([1, 0], [1, 1, -1, -1]),
            ([0, 1], [1, -1, 1, -1]),
            ([0, 1, 1], [1, -1, -1, 1, 1, -1, -1, 1]),
        ],
    )
    def test_eigvals(self, bitstring, expected_eigvals):
        """Tests that MultiX computes the eigenvalues correctly."""
        wires = range(len(bitstring))
        op = qp.MultiX(bitstring, wires=wires)

        computed_eigvals = qp.MultiX.compute_eigvals(bitstring, wires)

        assert qp.math.allclose(computed_eigvals, expected_eigvals)
        assert qp.math.allclose(op.eigvals(), expected_eigvals)

    @pytest.mark.jax
    def test_jit_eigvals(self):
        """Tests that MultiX eigenvalue computations work with JAX bitstrings."""

        import jax  # pylint: disable=import-outside-toplevel

        bitstring = jax.numpy.array([1, 0])
        eigvals_fn = jax.jit(lambda bits: qp.MultiX.compute_eigvals(bits, wires=[0, 1]))
        assert qp.math.allclose(eigvals_fn(bitstring), [1, 1, -1, -1])

    @pytest.mark.parametrize(
        ("bitstring", "wires", "gates_expected"),
        [
            ([0], [0], []),
            ([1], [1], [qp.H(1)]),
            ([1, 0], [0, 1], [qp.H(0)]),
            ([0, 1, 1], [0, 1, 2], [qp.H(1), qp.H(2)]),
        ],
    )
    def test_diagonalizing_gates(self, bitstring, wires, gates_expected):
        """Tests that MultiX is diagonalized by its diagonalizing gates."""
        op = qp.MultiX(bitstring, wires=wires)

        assert op.has_diagonalizing_gates

        diag_gates = op.diagonalizing_gates()

        assert len(diag_gates) == len(gates_expected)

        for i, gate in enumerate(diag_gates):
            qp.assert_equal(gate, gates_expected[i])

    @pytest.mark.parametrize("bitstring", ([0], [1], [1, 0], [0, 1, 1]))
    def test_diagonalizing_gates_match_eigvals(self, bitstring):
        """Tests that the diagonalizing gates produce the diagonal eigenvalue matrix."""
        wires = range(len(bitstring))
        op = qp.MultiX(bitstring, wires=wires)

        unitary = np.eye(2 ** len(op.wires))
        for gate in op.diagonalizing_gates():
            gate_matrix = qp.matrix(gate, wire_order=op.wires)
            unitary = gate_matrix @ unitary

        diagonalized_matrix = unitary @ op.matrix() @ unitary.conj().T
        expected_matrix = np.diag(op.eigvals())

        assert np.allclose(diagonalized_matrix, expected_matrix)


class TestOperatorArithmetic:
    """Tests for adjoint and power operations."""

    def test_adjoint(self):
        """Tests that the adjoint is a distinct, equivalent MultiX instance."""
        op = qp.MultiX([1, 0, 1], wires=["a", "b", "c"])

        adjoint_op = op.adjoint()

        assert op.has_adjoint
        assert adjoint_op is not op
        qp.assert_equal(adjoint_op, op)

    @pytest.mark.parametrize("exponent", [-5, -3, -1, 1, 3, 5])
    def test_pow_odd_integer(self, exponent):
        """Tests that MultiX raised to an odd integer is equivalent to itself."""
        op = qp.MultiX([1, 0, 1], wires=["a", "b", "c"])
        pow_ops = op.pow(exponent)

        assert len(pow_ops) == 1
        qp.assert_equal(pow_ops[0], op)

    @pytest.mark.parametrize("exponent", [-6, -4, -2, 0, 2, 4, 6])
    def test_pow_even_integer(self, exponent):
        """Tests that MultiX raised to an even integer is the identity."""
        op = qp.MultiX([1, 0, 1], wires=["a", "b", "c"])
        pow_ops = op.pow(exponent)

        assert len(pow_ops) == 0


class TestDecomposition:
    """Tests for registered and operator-level decompositions."""

    @pytest.mark.parametrize(
        ("bitstring", "wires"),
        [
            ([0], [0]),
            ([1], [0]),
            ([1, 0], [0, 1]),
            ([0, 1, 1], [0, 1, 2]),
            ([False, True, False], [0, 1, 2]),
            ([1, 0, 1, 1], [0, 1, 2, 3]),
        ],
    )
    def test_decomposition(self, bitstring, wires):
        """Tests that MultiX decomposition contains X gates at the locations in the bitstring marked by 1."""

        op = qp.MultiX(bitstring, wires=wires)
        assert op.has_decomposition
        decomp = op.decomposition()
        assert len(decomp) == sum(bitstring)  # each bit contributes one X gate

        # checking that the decomposed PauliX gates have the correct wire indices
        decomp_idx = 0
        for i, bit in enumerate(bitstring):
            if bit == 1:
                qp.assert_equal(decomp[decomp_idx], qp.X(wires[i]))
                decomp_idx += 1

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_decomposition_capture_compatibility(self):
        """Tests that the MultiX decomposition rule is capture compatible."""
        op = qp.MultiX([1, 0, 1], wires=[0, 1, 2])

        for rule in qp.list_decomps(qp.MultiX):
            _test_decomposition_rule(op, rule)

    @pytest.mark.capture
    def test_decomposition_capture_with_tuple_wires(self):
        """Tests graph capture with tuple wires and a dynamically traced loop index."""

        import jax  # pylint: disable=import-outside-toplevel

        from pennylane.capture.primitives import (  # pylint: disable=import-outside-toplevel
            for_loop_prim,
        )

        bitstring = jax.numpy.array([1, 0, 1])
        wires = (0, 1, 2)
        decomposition = qp.list_decomps(qp.MultiX)[0]

        # The loop index is a JAX tracer. This call fails if the python tuple wires is
        # not converted to a JAX array before the decomposition for MultiX calls wires[i].
        jaxpr = jax.make_jaxpr(lambda bits: decomposition(bits, wires))(bitstring)

        # Ensures exactly one for loop primitive appears in the jaxpr
        loop_eqns = [eqn for eqn in jaxpr.eqns if eqn.primitive == for_loop_prim]
        assert len(loop_eqns) == 1

        # Check that the only constants are the wires
        assert len(jaxpr.consts) == 1
        assert np.all(jaxpr.consts[0] == wires)

        # Validate the produced tape
        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, bitstring)
        expected = [qp.X(0), qp.X(2)]

        assert len(tape.operations) == len(expected)
        assert tape.wires == Wires([0, 2])

        # Ensure the tape matches what is expected
        for i, expected_op in enumerate(expected):
            qp.assert_equal(tape.operations[i], expected_op)

    def test_adjoint_decomposition(self):
        """Tests that Adjoint(MultiX) decomposes to MultiX."""
        op = qp.MultiX([1, 0, 1], wires=[0, 1, 2])

        decomposition = qp.adjoint(op).decomposition()

        assert len(decomposition) == 1
        qp.assert_equal(decomposition[0], op)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_adjoint_decomposition_capture_compatibility(self):
        """Tests that the MultiX decomposition rule is capture compatible."""
        op = qp.MultiX([1, 0, 1], wires=[0, 1, 2])
        adjoint_op = qp.adjoint(op)

        if qp.capture.enabled():
            pytest.xfail(
                "When capture is enabled, ends up passing identical `ArgInfo` placeholder leaves into MultiX's`__init__` as wires since `ArgInfo` are not recognized as abstract. Since they are not unique, we get an error comparing them."
            )

        for rule in qp.list_decomps("Adjoint(MultiX)"):
            _test_decomposition_rule(adjoint_op, rule)

    @pytest.mark.parametrize("exponent", [0, 1, 2, 3, 4, 5])
    def test_pow_decomposition(self, exponent):
        """Tests non-negative integer powers for Pow(MultiX)."""
        op = qp.MultiX([1, 0, 1], wires=[0, 1, 2])

        decomposition = qp.pow(op, exponent).decomposition()
        if exponent % 2 == 0:  # even
            assert len(decomposition) == 0
        else:  # odd
            assert len(decomposition) == 1
            qp.assert_equal(decomposition[0], op)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_pow_decomposition_capture_compatibility(self):
        """Tests that the MultiX decomposition rule is capture compatible."""
        op = qp.MultiX([1, 0, 1], wires=[0, 1, 2])
        pow_op = qp.pow(op, 3)

        if qp.capture.enabled():
            pytest.xfail(
                "When capture is enabled, ends up passing identical `ArgInfo` placeholder leaves into MultiX's`__init__` as wires since `ArgInfo` are not recognized as abstract. Since they are not unique, we get an error comparing them."
            )

        for rule in qp.list_decomps("Pow(MultiX)"):
            _test_decomposition_rule(pow_op, rule)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_controlled_decomposition_capture_compatibility(self):
        """Tests that the C(MultiX) decomposition rule is a valid, capture-compatible rule."""
        op = qp.ctrl(qp.MultiX([1, 0, 1], wires=[1, 2, 3]), control=0)

        for rule in qp.list_decomps("C(MultiX)"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(
        ("control_values", "gate_set", "expected"),
        [
            # single control on |1>: one CNOT per set bit (PauliX covers the abstract control-value
            # branch of the general rule but is not emitted for a concrete control on |1>)
            ([1], {"CNOT", "PauliX"}, {"CNOT": 2}),
            # single control on |0>: one CNOT per set bit, plus a PauliX flip of the control wire
            # before and after.
            ([0], {"CNOT", "PauliX"}, {"CNOT": 2, "PauliX": 2}),
        ],
    )
    def test_controlled_decomposition_gate_set(self, control_values, gate_set, expected):
        """A single-control C(MultiX) fans out to one controlled-X per set bit (a CNOT for a
        control on |1>), lowering to the requested gate set."""

        @qp.transforms.decompose(gate_set=gate_set)
        @qp.qnode(qp.device("null.qubit", wires=4))
        def circuit():
            qp.ctrl(
                qp.MultiX([1, 0, 1], wires=[1, 2, 3]),
                control=0,
                control_values=control_values,
            )
            return qp.state()

        specs = qp.specs(circuit)()["resources"].quantum_operations
        assert dict(specs) == expected

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("control_values", [[1], [0]])
    def test_controlled_decomposition_matrix(self, control_values):
        """The controlled MultiX decomposition matches the op matrix for controls on |0> and |1>."""
        op = qp.ctrl(
            qp.MultiX([1, 0, 1], wires=[1, 2, 3]), control=0, control_values=control_values
        )

        for rule in qp.list_decomps("C(MultiX)"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("control_values", [[1, 1, 1], [0, 1, 0]])
    def test_controlled_multi_control_ladder_decomposition_capture_compatibility(
        self, control_values
    ):
        """With more than one control wire, the TemporaryAND-ladder rules (which load the fanout
        through a single work wire instead of repeating the multi-control structure per target)
        become applicable, and should also be valid, capture-compatible decomposition rules."""
        op = qp.ctrl(
            qp.MultiX([1, 0, 1], wires=[3, 4, 5]),
            control=[0, 1, 2],
            control_values=control_values,
        )

        for rule in qp.list_decomps("C(MultiX)"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("control_values", [[1, 1, 1], [0, 1, 0]])
    def test_controlled_multi_control_ladder_decomposition_matrix(self, control_values):
        """The TemporaryAND-ladder decomposition rules match the op matrix for a multi-control
        C(MultiX), for both zeroed work wires supplied explicitly and dynamically allocated."""
        op = qp.ctrl(
            qp.MultiX([1, 0, 1], wires=[3, 4, 5]),
            control=[0, 1, 2],
            control_values=control_values,
        )

        for rule in qp.list_decomps("C(MultiX)"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_controlled_multi_control_ladder_uses_fewer_non_clifford_gates(self):
        """With a work wire available, a multi-control C(MultiX) loads the multi-control
        structure once (as a TemporaryAND ladder) and fans it out to the targets with CNOTs,
        instead of repeating a MultiControlledX once per target bit."""
        bitstring = [1, 0, 1, 1]
        control = [0, 1, 2, 3]
        targets = [4, 5, 6, 7]

        @qp.transforms.decompose(
            # PauliX is required for solvability (flip_zero_control's declared resources), even
            # though this concrete, all-1-control circuit never emits one.
            gate_set={"TemporaryAND", "Adjoint(TemporaryAND)", "CNOT", "PauliX"},
            num_work_wires=len(control) - 1,
        )
        @qp.qnode(qp.device("null.qubit", wires=11))
        def circuit():
            qp.ctrl(qp.MultiX(bitstring, wires=targets), control=control)
            return qp.state()

        specs = qp.specs(circuit)()["resources"].quantum_operations
        num_controls = len(control)
        num_set_bits = sum(bitstring)
        # One CNOT per set target bit fanned out from the ladder's single work wire, instead of
        # one MultiControlledX (which itself needs O(num_controls) gates) per set target bit.
        assert dict(specs) == {
            "Allocate": 1,
            "Deallocate": 1,
            "TemporaryAND": num_controls - 1,
            "Adjoint(TemporaryAND)": num_controls - 1,
            "CNOT": num_set_bits,
        }


class TestExecution:
    """Tests execution on PennyLane devices and compilation interfaces."""

    @pytest.mark.parametrize(
        ("bitstring", "wires", "expected_index"),
        [
            ([0], [0], 0),
            ([1], [0], 1),
            ([1, 0], [0, 1], 2),
            ([0, 1, 1], [0, 1, 2], 3),
            ([1, 0, 1, 1], [0, 1, 2, 3], 11),
        ],
    )
    def test_evalutation(self, bitstring, wires, expected_index):
        """Tests that MultiX works correctly on 'default.qubit' device and |0...0> input state."""

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            qp.MultiX(bitstring, wires=wires)
            return qp.probs(wires=wires)

        # qp.MultiX( bitstring, ... ) should set |0...0> to |bitstring>
        expected_result = np.zeros(2 ** len(wires))
        expected_result[expected_index] = 1

        obtained_result = circuit()

        assert np.allclose(obtained_result, expected_result)

    @pytest.mark.catalyst
    def test_qjit_circuit(self):
        """Tests that a circuit containing MultiX compiles and runs correctly under qjit."""
        pytest.importorskip("catalyst")

        wires = (0, 1, 2)
        bitstring = np.array([True, False, True])

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=len(wires)))
        def circuit(bits):
            qp.MultiX(bits, wires=wires)
            return qp.state()

        expected_state = np.zeros(2 ** len(wires), dtype=complex)
        expected_state[5] = 1

        assert np.allclose(circuit(bitstring), expected_state)
