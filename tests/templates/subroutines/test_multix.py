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

import pennylane as qp
from pennylane.typing import Bool, Float, Int, Wire
from pennylane.wires import Wires


@pytest.mark.parametrize(
    ("bitstring_input", "wires_input"),
    [
        ([0, 1, 1, 0], ("a", "b", "c", "d")),
        ([False, True, True, False], ("a", "b", "c", "d")),
        ((False, True, False), ("a", "b", "c")),
        ([False, True], (0, 1)),
        (np.array([0, 0]), ["Alice", "Bob"]),
        (np.array([False, True]), ["Alice", "Bob"]),
    ],
)
def test_input_arguments_parsed_correctly(bitstring_input, wires_input):
    """Tests that MultiX handles and sanitizes its input arguments correctly."""
    op = qp.MultiX(bitstring_input, wires=wires_input)

    assert isinstance(op.bitstring, np.ndarray)
    assert op.bitstring.dtype == bool
    assert np.all(op.bitstring == np.array(bitstring_input, dtype=bool))
    assert op.wires == Wires(wires_input)
    assert op.dynamic_args == {"bitstring": op.bitstring}
    assert op.wire_args == {"wires": Wires(wires_input)}
    assert op.num_wires == len(wires_input)
    assert op.num_params == 0
    assert op.grad_method is None


def test_boolean_bitstring_accepted():
    """Tests that Boolean input is accepted and remains Boolean after initialization."""
    bitstring = np.array([True, False, True])

    op = qp.MultiX(bitstring, wires=[0, 1, 2])

    assert op.bitstring.dtype == bool
    assert np.array_equal(op.bitstring, bitstring)


def test_canonicalize_inputs_does_not_validate_or_cast():
    """Tests that canonicalization changes input containers but does not validate or cast."""
    bitstring = np.array([0.0, 1.0])
    wires = ("a", "b")

    # pylint: disable-next=protected-access
    canonical_bitstring, canonical_wires = qp.MultiX._canonicalize_inputs(bitstring, wires)

    assert canonical_bitstring is bitstring
    assert canonical_bitstring.dtype == float
    assert canonical_wires == Wires(wires)


@pytest.mark.jax
def test_standard_checks():
    """Runs the standard Operator2 validity checks for MultiX."""
    op = qp.MultiX([1, 0, 1], wires=[0, 1, 2])
    qp.ops.functions.assert_valid(op)


def test_abstract_init():
    """Tests that MultiX can be initialized from abstract argument metadata."""
    bitstring = Int[3]
    wires = Wire[3]

    op = qp.MultiX(bitstring, wires)

    assert op.is_abstract
    assert op.bitstring == Bool[3]
    assert op.wires == wires


def test_custom_repr_override():
    """Tests that the Boolean bitstring is represented with integer values."""
    op = qp.MultiX([True, False, True], wires=["a", "b", "c"])

    assert repr(op) == "MultiX([1 0 1], wires=['a', 'b', 'c'])"


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
def test_invalid_arguments(bitstring, wires, error_match):
    """Tests that MultiX raises clear errors when input arguments are invalid."""
    with pytest.raises(ValueError, match=error_match):
        qp.MultiX(bitstring, wires=wires)


@pytest.mark.parametrize(
    ("bitstring", "expected_matrix"),
    [
        ([0], np.eye(2)),
        ([1], qp.X.compute_matrix()),
        ([1, 0], np.kron(qp.X.compute_matrix(), np.eye(2))),
        ([0, 1, 1], np.kron(np.eye(2), np.kron(qp.X.compute_matrix(), qp.X.compute_matrix()))),
    ],
)
def test_matrix(bitstring, expected_matrix):
    """Tests that MultiX computes the tensor product selected by the bitstring."""
    op = qp.MultiX(bitstring, wires=range(len(bitstring)))

    assert np.allclose(op.matrix(), expected_matrix)


@pytest.mark.parametrize("sparse_format", ["csr", "csc", "lil", "coo"])
@pytest.mark.parametrize(
    "bitstring", [[0], [1], [1, 0], [0, 1, 1], [1, 0, 1, 1], [False, False, True, True]]
)
def test_sparse_matrix(bitstring, sparse_format):
    """Tests that MultiX computes its sparse matrix in the requested format."""
    wires = range(len(bitstring))
    op = qp.MultiX(bitstring, wires=wires)

    static_matrix = qp.MultiX.compute_sparse_matrix(bitstring, wires, format=sparse_format)
    instance_matrix = op.sparse_matrix(format=sparse_format)

    assert qp.MultiX.has_sparse_matrix
    assert static_matrix.format == sparse_format
    assert instance_matrix.format == sparse_format
    assert np.allclose(static_matrix.toarray(), op.matrix())
    assert np.allclose(instance_matrix.toarray(), op.matrix())


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
def test_eigvals(bitstring, expected_eigvals):
    """Tests that MultiX computes the eigenvalues correctly."""
    wires = range(len(bitstring))
    op = qp.MultiX(bitstring, wires=wires)

    computed_eigvals = qp.MultiX.compute_eigvals(bitstring, wires)

    assert qp.math.allclose(computed_eigvals, expected_eigvals)
    assert qp.math.allclose(op.eigvals(), expected_eigvals)


@pytest.mark.jax
def test_jit_eigvals():
    """Tests that MultiX eigenvalue computations work with JAX bitstrings."""

    import jax  # pylint: disable=import-outside-toplevel

    bitstring = jax.numpy.array([1, 0])
    eigvals_fn = jax.jit(lambda bits: qp.MultiX.compute_eigvals(bits, wires=[0, 1]))
    assert qp.math.allclose(eigvals_fn(bitstring), [1, 1, -1, -1])


@pytest.mark.parametrize(
    ("bitstring", "wires"),
    [
        ([0], [0]),
        ([1], [1]),
        ([1, 0], [0, 1]),
        ([0, 1, 1], [0, 1, 2]),
    ],
)
def test_diagonalizing_gates(bitstring, wires):
    """Tests that MultiX is diagonalized by its diagonalizing gates."""
    op = qp.MultiX(bitstring, wires=wires)

    assert op.has_diagonalizing_gates

    diag_gates = qp.MultiX.compute_diagonalizing_gates(bitstring, wires)
    diag_gates_from_instance = op.diagonalizing_gates()
    diag_gates_expected = [qp.H(wire) for wire in wires]

    assert len(diag_gates) == len(diag_gates_from_instance)
    assert len(diag_gates) == len(diag_gates_expected)

    for i, gate in enumerate(diag_gates):
        qp.assert_equal(gate, diag_gates_from_instance[i])
        qp.assert_equal(gate, diag_gates_expected[i])


@pytest.mark.jax
def test_jit_matrix():
    """Tests that MultiX matrix computations work with JAX bitstrings."""

    import jax  # pylint: disable=import-outside-toplevel

    bitstring = jax.numpy.array([1, 0])
    matrix_fn = jax.jit(lambda bits: qp.MultiX.compute_matrix(bits, wires=[0, 1]))
    expected = np.kron(qp.X.compute_matrix(), np.eye(2))
    assert qp.math.allclose(matrix_fn(bitstring), expected)


@pytest.mark.capture
def test_decomposition_capture_with_tuple_wires():
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

    assert any(eqn.primitive == for_loop_prim for eqn in jaxpr.eqns)


@pytest.mark.capture
def test_decomposition_capture_with_tuple_bitstring():
    """Tests that a tuple bitstring can be indexed by a dynamically traced loop index."""

    import jax  # pylint: disable=import-outside-toplevel

    from pennylane.capture.primitives import (  # pylint: disable=import-outside-toplevel
        for_loop_prim,
    )

    bitstring = (True, False, True)
    wires = jax.numpy.array([0, 1, 2])
    decomposition = qp.list_decomps(qp.MultiX)[0]

    jaxpr = jax.make_jaxpr(lambda: decomposition(bitstring, wires))()

    assert any(eqn.primitive == for_loop_prim for eqn in jaxpr.eqns)


def test_adjoint():
    """Tests that the adjoint is a distinct, equivalent MultiX instance."""
    op = qp.MultiX([1, 0, 1], wires=["a", "b", "c"])

    adjoint_op = op.adjoint()

    assert op.has_adjoint
    assert adjoint_op is not op
    qp.assert_equal(adjoint_op, op)


@pytest.mark.parametrize("exponent", [-5, -3, -1, 1, 3, 5])
def test_pow_odd_integer(exponent):
    """Tests that MultiX raised to an odd integer is equivalent to itself."""
    op = qp.MultiX([1, 0, 1], wires=["a", "b", "c"])
    pow_ops = op.pow(exponent)

    assert len(pow_ops) == 1
    qp.assert_equal(pow_ops[0], op)


@pytest.mark.parametrize("exponent", [-6, -4, -2, 0, 2, 4, 6])
def test_pow_even_integer(exponent):
    """Tests that MultiX raised to an even integer is the identity."""
    op = qp.MultiX([1, 0, 1], wires=["a", "b", "c"])
    pow_ops = op.pow(exponent)

    assert len(pow_ops) == 0


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
def test_evalutation(bitstring, wires, expected_index):
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
def test_decomposition(bitstring, wires):
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


@pytest.mark.catalyst
def test_qjit_circuit():
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
