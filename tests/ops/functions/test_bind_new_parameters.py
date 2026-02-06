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
This module contains unit tests for ``qp.bind_parameters``.
"""

import numpy as np
import pytest
from gate_data import GELL_MANN, I, X, Y, Z

import pennylane as qp
from pennylane.ops.functions import bind_new_parameters


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (
            qp.sum(qp.s_prod(1.1, qp.PauliX(0)), qp.PauliZ(1)),
            [0.1],
            qp.sum(qp.s_prod(0.1, qp.PauliX(0)), qp.PauliZ(1)),
        ),
        (
            qp.sum(qp.Hermitian(X, 0), qp.Hermitian(Y, 0)),
            [I, Z],
            qp.sum(qp.Hermitian(I, 0), qp.Hermitian(Z, 0)),
        ),
        (
            qp.sum(qp.s_prod(0.5, qp.Hermitian(X, 0)), qp.PauliZ(1)),
            [-0.5, Z],
            qp.sum(qp.s_prod(-0.5, qp.Hermitian(Z, 0)), qp.PauliZ(1)),
        ),
        (
            qp.sum(
                qp.s_prod(0.5, qp.sum(qp.THermitian(GELL_MANN[0], 0), qp.GellMann(0, 2))),
                qp.prod(qp.GellMann(1, 3), qp.THermitian(GELL_MANN[5], 2)),
            ),
            [-0.5, GELL_MANN[1], GELL_MANN[6]],
            qp.sum(
                qp.s_prod(-0.5, qp.sum(qp.THermitian(GELL_MANN[1], 0), qp.GellMann(0, 2))),
                qp.prod(qp.GellMann(1, 3), qp.THermitian(GELL_MANN[6], 2)),
            ),
        ),
        (
            qp.prod(qp.s_prod(1.1, qp.PauliX(0)), qp.PauliZ(1)),
            [0.1],
            qp.prod(qp.s_prod(0.1, qp.PauliX(0)), qp.PauliZ(1)),
        ),
        (
            qp.prod(qp.Hermitian(X, 0), qp.Hermitian(Y, 1)),
            [I, Z],
            qp.prod(qp.Hermitian(I, 0), qp.Hermitian(Z, 1)),
        ),
        (
            qp.prod(qp.s_prod(0.5, qp.Hermitian(X, 0)), qp.PauliZ(1)),
            [-0.5, Z],
            qp.prod(qp.s_prod(-0.5, qp.Hermitian(Z, 0)), qp.PauliZ(1)),
        ),
        (
            qp.prod(
                qp.s_prod(0.5, qp.sum(qp.THermitian(GELL_MANN[0], 0), qp.GellMann(0, 2))),
                qp.sum(qp.GellMann(1, 3), qp.THermitian(GELL_MANN[5], 1)),
            ),
            [-0.5, GELL_MANN[1], GELL_MANN[6]],
            qp.prod(
                qp.s_prod(-0.5, qp.sum(qp.THermitian(GELL_MANN[1], 0), qp.GellMann(0, 2))),
                qp.sum(qp.GellMann(1, 3), qp.THermitian(GELL_MANN[6], 1)),
            ),
        ),
    ],
)
def test_composite_ops(op, new_params, expected_op):
    """Test that `bind_new_parameters` with `CompositeOp` returns a new
    operator with the new parameters without mutating the original
    operator."""
    new_op = bind_new_parameters(op, new_params)

    qp.assert_equal(new_op, expected_op)
    assert new_op is not op
    assert all(no is not o for no, o in zip(new_op.operands, op.operands))


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qp.evolve(qp.PauliX(0), 0.5), [-0.5], qp.evolve(qp.PauliX(0), -0.5)),
        (qp.exp(qp.PauliX(0), 0.5), [-0.5], qp.exp(qp.PauliX(0), -0.5)),
        (qp.pow(qp.RX(0.123, 0), 2), [0.456], qp.pow(qp.RX(0.456, 0), 2)),
        (qp.s_prod(0.5, qp.RX(0.123, 0)), [-0.5, 0.456], qp.s_prod(-0.5, qp.RX(0.456, 0))),
        (
            qp.s_prod(0.5, qp.sum(qp.Hermitian(X, 0), qp.PauliZ(1))),
            [-0.5, Y],
            qp.s_prod(-0.5, qp.sum(qp.Hermitian(Y, 0), qp.PauliZ(1))),
        ),
    ],
)
def test_scalar_symbolic_ops(op, new_params, expected_op):
    """Test that `bind_new_parameters` with `ScalarSymbolicOp` returns a new
    operator with the new parameters without mutating the original
    operator."""
    new_op = bind_new_parameters(op, new_params)

    qp.assert_equal(new_op, expected_op)
    assert new_op is not op
    assert new_op.base is not op.base


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qp.adjoint(qp.RX(0.123, wires=0)), [0.456], qp.adjoint(qp.RX(0.456, wires=0))),
        (qp.adjoint(qp.T(0)), [], qp.adjoint(qp.T(0))),
        (
            qp.adjoint(qp.s_prod(0.5, qp.RX(0.123, 0))),
            [-0.5, 0.456],
            qp.adjoint(qp.s_prod(-0.5, qp.RX(0.456, 0))),
        ),
        (
            qp.ctrl(
                qp.IsingZZ(0.123, wires=[0, 1]),
                [2, 3],
                control_values=[True, False],
                work_wires=[4],
            ),
            [0.456],
            qp.ctrl(
                qp.IsingZZ(0.456, wires=[0, 1]),
                [2, 3],
                control_values=[True, False],
                work_wires=[4],
            ),
        ),
        (
            qp.ctrl(qp.s_prod(0.5, qp.RX(0.123, 0)), [1]),
            [-0.5, 0.456],
            qp.ctrl(qp.s_prod(-0.5, qp.RX(0.456, 0)), [1]),
        ),
        (
            qp.ControlledQubitUnitary(X, wires=[1, 0]),
            [Y],
            qp.ControlledQubitUnitary(Y, wires=[1, 0]),
        ),
        (
            qp.ControlledQubitUnitary(qp.QubitUnitary(X, 0).matrix(), wires=[1, 0]),
            [Y],
            qp.ControlledQubitUnitary(qp.QubitUnitary(Y, 0).matrix(), wires=[1, 0]),
        ),
    ],
)
def test_symbolic_ops(op, new_params, expected_op):
    """Test that `bind_new_parameters` with `SymbolicOp` returns a new
    operator with the new parameters without mutating the original
    operator."""
    new_op = bind_new_parameters(op, new_params)

    qp.assert_equal(new_op, expected_op)
    assert new_op is not op
    assert new_op.base is not op.base


def test_controlled_sequence():
    """Test integration of controlled sequence with bind_new_parameters."""
    op = qp.ControlledSequence(qp.RX(0.25, wires=3), control=[0, 1, 2])
    new_op = bind_new_parameters(op, (0.5,))
    assert qp.math.allclose(new_op.data[0], 0.5)
    qp.assert_equal(new_op.base, qp.RX(0.5, wires=3))


TEST_BIND_LINEARCOMBINATION = [
    (  # LinearCombination with only data being the coeffs
        qp.ops.LinearCombination(
            [1.1, 2.1, 3.1],
            [qp.prod(qp.PauliZ(0), qp.X(1)), qp.Hadamard(1), qp.Y(0)],
        ),
        [1.2, 2.2, 3.2],
        qp.ops.LinearCombination(
            [1.2, 2.2, 3.2],
            [qp.prod(qp.PauliZ(0), qp.X(1)), qp.Hadamard(1), qp.Y(0)],
        ),
    ),
    (  # LinearCombination with Hermitian that carries extra data
        qp.ops.LinearCombination(
            [1.6, -1], [qp.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=1), qp.X(1)]
        ),
        [-1, np.array([[1.0, 1.0], [1.0, 1.0]]), 1.6],
        qp.ops.LinearCombination(
            [-1, 1.6], [qp.Hermitian(np.array([[1.0, 1.0], [1.0, 1.0]]), wires=1), qp.X(1)]
        ),
    ),
    (  # LinearCombination with prod that contains Hermitian that carries extra data
        qp.ops.LinearCombination(
            [1.6, -1],
            [
                qp.prod(qp.X(0), qp.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=1)),
                qp.X(1),
            ],
        ),
        [-1, np.array([[1.0, 1.0], [1.0, 1.0]]), 1.6],
        qp.ops.LinearCombination(
            [-1, 1.6],
            [
                qp.prod(qp.X(0), qp.Hermitian(np.array([[1.0, 1.0], [1.0, 1.0]]), wires=1)),
                qp.X(1),
            ],
        ),
    ),
    (  # LinearCombination with prod that contains Hermitian that carries extra data
        qp.ops.LinearCombination(
            [1.6, -1],
            [
                qp.prod(qp.X(0), qp.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=1)),
                qp.X(1),
            ],
        ),
        [-1, np.array([[1.0, 1.0], [1.0, 1.0]]), 1.6],
        qp.ops.LinearCombination(
            [-1, 1.6],
            [
                qp.prod(qp.X(0), qp.Hermitian(np.array([[1.0, 1.0], [1.0, 1.0]]), wires=1)),
                qp.X(1),
            ],
        ),
    ),
    (  # LinearCombination with Projector that carries extra data and prod that contains Hermitian that carries extra data
        qp.ops.LinearCombination(
            [1.0, 1.6, -1],
            [
                qp.Projector(np.array([1.0, 0.0]), 0),
                qp.prod(qp.X(0), qp.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=1)),
                qp.X(1),
            ],
        ),
        [-1.0, np.array([0.0, 1.0]), -1, np.array([[1.0, 1.0], [1.0, 1.0]]), 1.6],
        qp.ops.LinearCombination(
            [-1.0, -1, 1.6],
            [
                qp.Projector(np.array([0.0, 1.0]), 0),
                qp.prod(qp.X(0), qp.Hermitian(np.array([[1.0, 1.0], [1.0, 1.0]]), wires=1)),
                qp.X(1),
            ],
        ),
    ),
]


@pytest.mark.parametrize(
    "H, new_coeffs, expected_H",
    TEST_BIND_LINEARCOMBINATION,
)
def test_linear_combination(H, new_coeffs, expected_H):
    """Test that `bind_new_parameters` with `LinearCombination` returns a new
    operator with the new parameters without mutating the original
    operator."""
    new_H = bind_new_parameters(H, new_coeffs)

    qp.assert_equal(new_H, expected_H)
    assert new_H is not H


def test_hamiltonian_grouping_indices():
    """Test that bind_new_parameters with a Hamiltonian preserves the grouping indices."""
    H = qp.Hamiltonian([1.0, 2.0], [qp.PauliX(0), qp.PauliX(1)])
    H.compute_grouping()
    new_H = bind_new_parameters(H, [2.3, 3.4])
    assert H.grouping_indices == new_H.grouping_indices
    assert new_H.data == (2.3, 3.4)


old_hamiltonian = qp.Hamiltonian(
    [0.1, 0.2, 0.3], [qp.PauliX(0), qp.PauliZ(0) @ qp.PauliX(1), qp.PauliZ(2)]
)
new_hamiltonian = qp.Hamiltonian(
    [0.4, 0.5, 0.6], [qp.PauliX(0), qp.PauliZ(0) @ qp.PauliX(1), qp.PauliZ(2)]
)


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (
            qp.ApproxTimeEvolution(old_hamiltonian, 5, 10),
            [0.4, 0.5, 0.6, 10],
            qp.ApproxTimeEvolution(new_hamiltonian, 10, 10),
        ),
        (
            qp.CommutingEvolution(old_hamiltonian, 5),
            [10, 0.4, 0.5, 0.6],
            qp.CommutingEvolution(new_hamiltonian, 10),
        ),
        (
            qp.QDrift(old_hamiltonian, 5, n=4, seed=251),
            [0.4, 0.5, 0.6, 10],
            qp.QDrift(new_hamiltonian, 10, n=4, seed=251),
        ),
    ],
)
def test_evolution_template_ops(op, new_params, expected_op):
    """Test that `bind_new_parameters` with template operators returns a new
    template operator with the new parameters without mutating the original
    operator."""
    new_op = bind_new_parameters(op, new_params)

    assert new_op is not op
    # Need the following assertions since there is a `Hamiltonian` in the hyperparameters so
    # `qp.equal` fails.
    assert isinstance(new_op, type(op))
    assert new_op.arithmetic_depth == expected_op.arithmetic_depth
    assert all(qp.math.allclose(d1, d2) for d1, d2 in zip(new_op.data, expected_op.data))
    assert new_op.wires == op.wires
    for val1, val2 in zip(new_op.hyperparameters.values(), expected_op.hyperparameters.values()):
        if isinstance(val1, qp.Hamiltonian):
            qp.assert_equal(val1, val2)
        else:
            assert val1 == val2


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (
            qp.FermionicDoubleExcitation(0.123, wires1=[0, 1, 2], wires2=[3, 4]),
            [0.456],
            qp.FermionicDoubleExcitation(0.456, wires1=[0, 1, 2], wires2=[3, 4]),
        ),
        (
            qp.FermionicSingleExcitation(0.123, wires=[0, 1]),
            [0.456],
            qp.FermionicSingleExcitation(0.456, wires=[0, 1]),
        ),
    ],
)
def test_fermionic_template_ops(op, new_params, expected_op):
    """Test that `bind_new_parameters` with fermionic template operators returns a new operator
    with the new parameters."""
    new_op = bind_new_parameters(op, new_params)

    qp.assert_equal(new_op, expected_op)
    assert new_op is not op


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qp.RX(0.123, 0), [0.456], qp.RX(0.456, 0)),
        (
            qp.Rot(0.123, 0.456, 0.789, wires=1),
            [1.23, 4.56, 7.89],
            qp.Rot(1.23, 4.56, 7.89, wires=1),
        ),
        (
            qp.QubitUnitary(X, wires=0),
            [I],
            qp.QubitUnitary(I, wires=0),
        ),
        (
            qp.BasisState([0, 1], wires=[0, 1]),
            [[1, 1]],
            qp.BasisState([1, 1], wires=[0, 1]),
        ),
        (qp.GellMann(wires=0, index=4), [], qp.GellMann(wires=0, index=4)),
        (qp.Identity(wires=[1, 2]), [], qp.Identity(wires=[1, 2])),
        (qp.pow(qp.RX(0.123, 0), 2, lazy=False), [0.456], qp.RX(0.456, 0)),
        (
            qp.adjoint(qp.RX(0.123, wires=0), lazy=False),
            [0.456],
            qp.RX(0.456, wires=0),
        ),
        (qp.PCPhase(0.123, 2, wires=[0, 1]), [0.456], qp.PCPhase(0.456, 2, wires=[0, 1])),
    ],
)
def test_vanilla_operators(op, new_params, expected_op):
    """Test that `bind_new_parameters` with vanilla operators returns a new
    operator with the new parameters without mutating the original
    operator."""
    new_op = bind_new_parameters(op, new_params)

    qp.assert_equal(new_op, expected_op)
    assert new_op is not op


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qp.Projector([0, 0], [0, 1]), [[0, 1]], qp.Projector([0, 1], [0, 1])),
        (qp.Projector([1, 0, 0, 0], [0, 1]), [[0, 0]], qp.Projector([0, 0], [0, 1])),
        (qp.Projector([1, 1], [0, 1]), [[0.5] * 4], qp.Projector([0.5] * 4, [0, 1])),
        (qp.Projector([1, 0, 0, 0], [0, 1]), [[0.5] * 4], qp.Projector([0.5] * 4, [0, 1])),
    ],
)
def test_projector(op, new_params, expected_op):
    """Test that `bind_new_parameters` with projectors returns a new projector with the new
    parameters without mutating the original operator."""
    new_op = bind_new_parameters(op, new_params)

    qp.assert_equal(new_op, expected_op)
    assert new_op is not op


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qp.RX(0.123, 0), (0.456,), qp.RX(0.456, 0)),
        (
            qp.QubitUnitary([[1, 0], [0, -1]], 0),
            ([[0, 1], [1, 0]],),
            qp.QubitUnitary([[0, 1], [1, 0]], 0),
        ),
        (qp.Rot(1.23, 4.56, 7.89, 0), (9.87, 6.54, 3.21), qp.Rot(9.87, 6.54, 3.21, 0)),
    ],
)
def test_conditional_ops(op, new_params, expected_op):
    """Test that Conditional ops are bound correctly."""
    mp0 = qp.ops.MidMeasure(qp.wires.Wires(0), reset=True, id="foo")
    mv0 = qp.ops.MeasurementValue([mp0], lambda v: v)
    cond_op = qp.ops.Conditional(mv0, op)
    new_op = bind_new_parameters(cond_op, new_params)

    assert isinstance(new_op, qp.ops.Conditional)
    assert new_op.base == expected_op
    assert new_op.meas_val.measurements == [mp0]


def test_unsupported_op_copy_and_set():
    """Test that trying to use `bind_new_parameters` on an operator without
    a supported dispatcher will fall back to copying the operator and setting
    `new_op.data` to the new parameters."""
    op = qp.PCPhase(0.123, 2, wires=[1, 2])
    new_op = bind_new_parameters(op, [0.456])

    expected_op = qp.PCPhase(0.456, 2, wires=[1, 2])

    qp.assert_equal(new_op, expected_op)
    assert new_op is not op
