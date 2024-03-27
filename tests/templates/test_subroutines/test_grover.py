# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the Grover Diffusion Operator template
"""
import functools
import itertools
import pytest
import numpy as np
import pennylane as qml
from pennylane.ops import Hadamard, PauliZ, MultiControlledX


def test_repr():
    """Tests the repr method for GroverOperator."""
    op = qml.GroverOperator(wires=(0, 1, 2), work_wires=(3, 4))
    expected = "GroverOperator(wires=[0, 1, 2], work_wires=[3, 4])"
    assert repr(op) == expected


# pylint: disable=protected-access
def test_flatten_unflatten():
    """Tests the flatten and unflatten methods for GroverOperator."""
    work_wires = qml.wires.Wires((3, 4))
    op = qml.GroverOperator(wires=(0, 1, 2), work_wires=work_wires)
    data, metadata = op._flatten()
    assert data == tuple()
    assert len(metadata) == 2
    assert metadata[0] == op.wires
    assert metadata[1] == (("work_wires", work_wires),)

    # make sure metadata hashable
    assert hash(metadata)

    new_op = type(op)._unflatten(*op._flatten())
    assert qml.equal(op, new_op)
    assert new_op is not op


def test_work_wires():
    """Assert work wires get passed to MultiControlledX"""
    wires = ("a", "b")
    work_wire = ("aux",)

    op = qml.GroverOperator(wires=wires, work_wires=work_wire)

    assert op.hyperparameters["work_wires"] == work_wire

    ops = op.expand().operations

    assert ops[2].hyperparameters["work_wires"] == work_wire


def test_work_wires_None():
    """Test that work wires of None are not inpreted as work wires."""
    op = qml.GroverOperator(wires=(0, 1, 2, 3), work_wires=None)
    assert op.hyperparameters["work_wires"] == qml.wires.Wires([])


@pytest.mark.parametrize("bad_wires", [0, (0,), tuple()])
def test_single_wire_error(bad_wires):
    """Assert error raised when called with only a single wire"""

    with pytest.raises(ValueError, match="GroverOperator must have at least"):
        qml.GroverOperator(wires=bad_wires)


def test_id():
    """Assert id keyword works"""

    op = qml.GroverOperator(wires=(0, 1), id="hello")

    assert op.id == "hello"


decomp_3wires = [
    qml.Hadamard,
    qml.Hadamard,
    qml.PauliZ,
    qml.MultiControlledX,
    qml.PauliZ,
    qml.Hadamard,
    qml.Hadamard,
    qml.GlobalPhase,
]


def decomposition_wires(wires):
    wire_order = [
        wires[0],
        wires[1],
        wires[2],
        wires,
        wires[2],
        wires[0],
        wires[1],
        wires,
    ]
    return wire_order


@pytest.mark.parametrize("n_wires", [2, 4, 7])
def test_grover_diffusion_matrix(n_wires):
    """Test that the Grover diffusion matrix is the same as when constructed in a different way"""
    wires = list(range(n_wires))

    # Test-oracle
    oracle = np.identity(2**n_wires)
    oracle[0, 0] = -1

    # s1 = H|0>, Hadamard on a single qubit in the ground state
    s1 = np.array([1, 1]) / np.sqrt(2)

    # uniform superposition state
    s = functools.reduce(np.kron, list(itertools.repeat(s1, n_wires)))
    # Grover matrix
    G_matrix = qml.GroverOperator(wires=wires).matrix()

    amplitudes = G_matrix @ oracle @ s
    probs = amplitudes**2

    # Create Grover diffusion matrix G in alternative way
    oplist = list(itertools.repeat(Hadamard.compute_matrix(), n_wires - 1))
    oplist.append(PauliZ.compute_matrix())

    ctrl_str = [0] * (n_wires - 1)
    CX = MultiControlledX(
        control_values=ctrl_str,
        wires=wires,
        work_wires=None,
    ).matrix()

    M = functools.reduce(np.kron, oplist)
    G = M @ CX @ M

    amplitudes2 = G @ oracle @ s
    probs2 = amplitudes2**2

    assert np.allclose(probs, probs2)


def test_grover_diffusion_matrix_results():
    """Test that the matrix gives the same result as when running the example in the documentation
    `here <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.templates.subroutines.GroverOperator.html>`_
    """
    n_wires = 3
    wires = list(range(n_wires))

    def oracle():
        qml.Hadamard(wires[-1])
        qml.Toffoli(wires=wires)
        qml.Hadamard(wires[-1])

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def GroverSearch(num_iterations=1):
        for wire in wires:
            qml.Hadamard(wire)

        for _ in range(num_iterations):
            oracle()
            qml.GroverOperator(wires=wires)
        return qml.probs(wires)

    # Get probabilities from example
    probs_example = GroverSearch(num_iterations=1)

    # Grover diffusion matrix
    G_matrix = qml.GroverOperator(wires=wires).matrix()

    oracle_matrix = np.identity(2**n_wires)
    oracle_matrix[-1, -1] = -1

    # s1 = H|0>, Hadamard on a single qubit in the ground state
    s1 = np.array([1, 1]) / np.sqrt(2)

    # uniform superposition state
    s = functools.reduce(np.kron, list(itertools.repeat(s1, n_wires)))

    amplitudes = G_matrix @ oracle_matrix @ s
    # Check that the probabilities are the same
    probs_matrix = amplitudes**2

    assert np.allclose(probs_example, probs_matrix)


@pytest.mark.parametrize("wires", ((0, 1, 2), ("a", "c", "b")))
def test_expand(wires):
    """Asserts decomposition uses expected operations and wires"""
    op = qml.GroverOperator(wires=wires)

    decomp = op.expand().operations

    expected_wires = decomposition_wires(wires)

    assert len(decomp) == len(decomp_3wires) == len(expected_wires)

    for actual_op, expected_class, expected_wire in zip(decomp, decomp_3wires, expected_wires):
        assert isinstance(actual_op, expected_class)
        assert actual_op.wires == qml.wires.Wires(expected_wire)


@pytest.mark.parametrize("n_wires", [6, 13])
def test_findstate(n_wires):
    """Asserts can find state marked by oracle, with operation full matrix and decomposition."""
    wires = list(range(n_wires))

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circ():
        for wire in wires:
            qml.Hadamard(wire)

        for _ in range(2):
            qml.Hadamard(wires[0])
            qml.MultiControlledX(wires=wires[1:] + wires[0:1])
            qml.Hadamard(wires[0])
            qml.GroverOperator(wires=wires)

        return qml.probs(wires=wires)

    probs = circ()

    assert np.argmax(probs) == len(probs) - 1


def test_matrix(tol):
    """Test that the matrix representation is correct."""

    res_static = qml.GroverOperator.compute_matrix(2, work_wires=None)
    res_dynamic = qml.GroverOperator(wires=[0, 1]).matrix()
    res_reordered = qml.GroverOperator(wires=[0, 1]).matrix([1, 0])

    expected = np.array(
        [[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5], [0.5, 0.5, -0.5, 0.5], [0.5, 0.5, 0.5, -0.5]]
    )

    assert np.allclose(res_static, expected, atol=tol, rtol=0)
    assert np.allclose(res_dynamic, expected, atol=tol, rtol=0)
    # reordering should not affect this particular matrix
    assert np.allclose(res_reordered, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("n_wires", [2, 3, 5])
def test_decomposition_matrix(n_wires):
    """Test that the decomposition and the matrix match."""
    wires = list(range(n_wires))
    op = qml.GroverOperator(wires)
    mat1 = op.matrix()
    mat2 = qml.matrix(qml.tape.QuantumScript(op.decomposition()), wire_order=wires)
    assert np.allclose(mat1, mat2)
