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
This module contains unit tests for ``qml.bind_parameters``.
"""

import numpy as np
import pytest
from gate_data import GELL_MANN, I, X, Y, Z

import pennylane as qml
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.ops.functions import bind_new_parameters


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (
            qml.sum(qml.s_prod(1.1, qml.PauliX(0)), qml.PauliZ(1)),
            [0.1],
            qml.sum(qml.s_prod(0.1, qml.PauliX(0)), qml.PauliZ(1)),
        ),
        (
            qml.sum(qml.Hermitian(X, 0), qml.Hermitian(Y, 0)),
            [I, Z],
            qml.sum(qml.Hermitian(I, 0), qml.Hermitian(Z, 0)),
        ),
        (
            qml.sum(qml.s_prod(0.5, qml.Hermitian(X, 0)), qml.PauliZ(1)),
            [-0.5, Z],
            qml.sum(qml.s_prod(-0.5, qml.Hermitian(Z, 0)), qml.PauliZ(1)),
        ),
        (
            qml.sum(
                qml.s_prod(0.5, qml.sum(qml.THermitian(GELL_MANN[0], 0), qml.GellMann(0, 2))),
                qml.prod(qml.GellMann(1, 3), qml.THermitian(GELL_MANN[5], 2)),
            ),
            [-0.5, GELL_MANN[1], GELL_MANN[6]],
            qml.sum(
                qml.s_prod(-0.5, qml.sum(qml.THermitian(GELL_MANN[1], 0), qml.GellMann(0, 2))),
                qml.prod(qml.GellMann(1, 3), qml.THermitian(GELL_MANN[6], 2)),
            ),
        ),
        (
            qml.prod(qml.s_prod(1.1, qml.PauliX(0)), qml.PauliZ(1)),
            [0.1],
            qml.prod(qml.s_prod(0.1, qml.PauliX(0)), qml.PauliZ(1)),
        ),
        (
            qml.prod(qml.Hermitian(X, 0), qml.Hermitian(Y, 1)),
            [I, Z],
            qml.prod(qml.Hermitian(I, 0), qml.Hermitian(Z, 1)),
        ),
        (
            qml.prod(qml.s_prod(0.5, qml.Hermitian(X, 0)), qml.PauliZ(1)),
            [-0.5, Z],
            qml.prod(qml.s_prod(-0.5, qml.Hermitian(Z, 0)), qml.PauliZ(1)),
        ),
        (
            qml.prod(
                qml.s_prod(0.5, qml.sum(qml.THermitian(GELL_MANN[0], 0), qml.GellMann(0, 2))),
                qml.sum(qml.GellMann(1, 3), qml.THermitian(GELL_MANN[5], 1)),
            ),
            [-0.5, GELL_MANN[1], GELL_MANN[6]],
            qml.prod(
                qml.s_prod(-0.5, qml.sum(qml.THermitian(GELL_MANN[1], 0), qml.GellMann(0, 2))),
                qml.sum(qml.GellMann(1, 3), qml.THermitian(GELL_MANN[6], 1)),
            ),
        ),
    ],
)
def test_composite_ops(op, new_params, expected_op):
    """Test that `bind_new_parameters` with `CompositeOp` returns a new
    operator with the new parameters without mutating the original
    operator."""
    new_op = bind_new_parameters(op, new_params)

    qml.assert_equal(new_op, expected_op)
    assert new_op is not op
    assert all(no is not o for no, o in zip(new_op.operands, op.operands))


def test_num_steps_is_deprecated():
    """Test that providing `num_steps` to `qml.evolve` and `qml.exp` raises a deprecation warning."""
    with pytest.warns(
        PennyLaneDeprecationWarning,
        match="Providing 'num_steps' to 'qml.evolve' and 'qml.exp' is deprecated",
    ):
        qml.evolve(qml.PauliX(0), 0.5, num_steps=15)

    with pytest.warns(
        PennyLaneDeprecationWarning,
        match="Providing 'num_steps' to 'qml.evolve' and 'qml.exp' is deprecated",
    ):
        qml.exp(qml.PauliX(0), 0.5, num_steps=15)


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qml.evolve(qml.PauliX(0), 0.5), [-0.5], qml.evolve(qml.PauliX(0), -0.5)),
        (qml.exp(qml.PauliX(0), 0.5), [-0.5], qml.exp(qml.PauliX(0), -0.5)),
        (qml.pow(qml.RX(0.123, 0), 2), [0.456], qml.pow(qml.RX(0.456, 0), 2)),
        (qml.s_prod(0.5, qml.RX(0.123, 0)), [-0.5, 0.456], qml.s_prod(-0.5, qml.RX(0.456, 0))),
        (
            qml.s_prod(0.5, qml.sum(qml.Hermitian(X, 0), qml.PauliZ(1))),
            [-0.5, Y],
            qml.s_prod(-0.5, qml.sum(qml.Hermitian(Y, 0), qml.PauliZ(1))),
        ),
    ],
)
def test_scalar_symbolic_ops(op, new_params, expected_op):
    """Test that `bind_new_parameters` with `ScalarSymbolicOp` returns a new
    operator with the new parameters without mutating the original
    operator."""
    new_op = bind_new_parameters(op, new_params)

    qml.assert_equal(new_op, expected_op)
    assert new_op is not op
    assert new_op.base is not op.base


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qml.adjoint(qml.RX(0.123, wires=0)), [0.456], qml.adjoint(qml.RX(0.456, wires=0))),
        (qml.adjoint(qml.T(0)), [], qml.adjoint(qml.T(0))),
        (
            qml.adjoint(qml.s_prod(0.5, qml.RX(0.123, 0))),
            [-0.5, 0.456],
            qml.adjoint(qml.s_prod(-0.5, qml.RX(0.456, 0))),
        ),
        (
            qml.ctrl(
                qml.IsingZZ(0.123, wires=[0, 1]),
                [2, 3],
                control_values=[True, False],
                work_wires=[4],
            ),
            [0.456],
            qml.ctrl(
                qml.IsingZZ(0.456, wires=[0, 1]),
                [2, 3],
                control_values=[True, False],
                work_wires=[4],
            ),
        ),
        (
            qml.ctrl(qml.s_prod(0.5, qml.RX(0.123, 0)), [1]),
            [-0.5, 0.456],
            qml.ctrl(qml.s_prod(-0.5, qml.RX(0.456, 0)), [1]),
        ),
        (
            qml.ControlledQubitUnitary(X, wires=[1, 0]),
            [Y],
            qml.ControlledQubitUnitary(Y, wires=[1, 0]),
        ),
        (
            qml.ControlledQubitUnitary(qml.QubitUnitary(X, 0).matrix(), wires=[1, 0]),
            [Y],
            qml.ControlledQubitUnitary(qml.QubitUnitary(Y, 0).matrix(), wires=[1, 0]),
        ),
    ],
)
def test_symbolic_ops(op, new_params, expected_op):
    """Test that `bind_new_parameters` with `SymbolicOp` returns a new
    operator with the new parameters without mutating the original
    operator."""
    new_op = bind_new_parameters(op, new_params)

    qml.assert_equal(new_op, expected_op)
    assert new_op is not op
    assert new_op.base is not op.base


def test_controlled_sequence():
    """Test integration of controlled sequence with bind_new_parameters."""
    op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=[0, 1, 2])
    new_op = bind_new_parameters(op, (0.5,))
    assert qml.math.allclose(new_op.data[0], 0.5)
    qml.assert_equal(new_op.base, qml.RX(0.5, wires=3))


TEST_BIND_LINEARCOMBINATION = [
    (  # LinearCombination with only data being the coeffs
        qml.ops.LinearCombination(
            [1.1, 2.1, 3.1],
            [qml.prod(qml.PauliZ(0), qml.X(1)), qml.Hadamard(1), qml.Y(0)],
        ),
        [1.2, 2.2, 3.2],
        qml.ops.LinearCombination(
            [1.2, 2.2, 3.2],
            [qml.prod(qml.PauliZ(0), qml.X(1)), qml.Hadamard(1), qml.Y(0)],
        ),
    ),
    (  # LinearCombination with Hermitian that carries extra data
        qml.ops.LinearCombination(
            [1.6, -1], [qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=1), qml.X(1)]
        ),
        [-1, np.array([[1.0, 1.0], [1.0, 1.0]]), 1.6],
        qml.ops.LinearCombination(
            [-1, 1.6], [qml.Hermitian(np.array([[1.0, 1.0], [1.0, 1.0]]), wires=1), qml.X(1)]
        ),
    ),
    (  # LinearCombination with prod that contains Hermitian that carries extra data
        qml.ops.LinearCombination(
            [1.6, -1],
            [
                qml.prod(qml.X(0), qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=1)),
                qml.X(1),
            ],
        ),
        [-1, np.array([[1.0, 1.0], [1.0, 1.0]]), 1.6],
        qml.ops.LinearCombination(
            [-1, 1.6],
            [
                qml.prod(qml.X(0), qml.Hermitian(np.array([[1.0, 1.0], [1.0, 1.0]]), wires=1)),
                qml.X(1),
            ],
        ),
    ),
    (  # LinearCombination with prod that contains Hermitian that carries extra data
        qml.ops.LinearCombination(
            [1.6, -1],
            [
                qml.prod(qml.X(0), qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=1)),
                qml.X(1),
            ],
        ),
        [-1, np.array([[1.0, 1.0], [1.0, 1.0]]), 1.6],
        qml.ops.LinearCombination(
            [-1, 1.6],
            [
                qml.prod(qml.X(0), qml.Hermitian(np.array([[1.0, 1.0], [1.0, 1.0]]), wires=1)),
                qml.X(1),
            ],
        ),
    ),
    (  # LinearCombination with Projector that carries extra data and prod that contains Hermitian that carries extra data
        qml.ops.LinearCombination(
            [1.0, 1.6, -1],
            [
                qml.Projector(np.array([1.0, 0.0]), 0),
                qml.prod(qml.X(0), qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=1)),
                qml.X(1),
            ],
        ),
        [-1.0, np.array([0.0, 1.0]), -1, np.array([[1.0, 1.0], [1.0, 1.0]]), 1.6],
        qml.ops.LinearCombination(
            [-1.0, -1, 1.6],
            [
                qml.Projector(np.array([0.0, 1.0]), 0),
                qml.prod(qml.X(0), qml.Hermitian(np.array([[1.0, 1.0], [1.0, 1.0]]), wires=1)),
                qml.X(1),
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

    qml.assert_equal(new_H, expected_H)
    assert new_H is not H


def test_hamiltonian_grouping_indices():
    """Test that bind_new_parameters with a Hamiltonian preserves the grouping indices."""
    H = qml.Hamiltonian([1.0, 2.0], [qml.PauliX(0), qml.PauliX(1)])
    H.compute_grouping()
    new_H = bind_new_parameters(H, [2.3, 3.4])
    assert H.grouping_indices == new_H.grouping_indices
    assert new_H.data == (2.3, 3.4)


old_hamiltonian = qml.Hamiltonian(
    [0.1, 0.2, 0.3], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliZ(2)]
)
new_hamiltonian = qml.Hamiltonian(
    [0.4, 0.5, 0.6], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliZ(2)]
)


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (
            qml.ApproxTimeEvolution(old_hamiltonian, 5, 10),
            [0.4, 0.5, 0.6, 10],
            qml.ApproxTimeEvolution(new_hamiltonian, 10, 10),
        ),
        (
            qml.CommutingEvolution(old_hamiltonian, 5),
            [10, 0.4, 0.5, 0.6],
            qml.CommutingEvolution(new_hamiltonian, 10),
        ),
        (
            qml.QDrift(old_hamiltonian, 5, n=4, seed=251),
            [0.4, 0.5, 0.6, 10],
            qml.QDrift(new_hamiltonian, 10, n=4, seed=251),
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
    # `qml.equal` fails.
    assert isinstance(new_op, type(op))
    assert new_op.arithmetic_depth == expected_op.arithmetic_depth
    assert all(qml.math.allclose(d1, d2) for d1, d2 in zip(new_op.data, expected_op.data))
    assert new_op.wires == op.wires
    for val1, val2 in zip(new_op.hyperparameters.values(), expected_op.hyperparameters.values()):
        if isinstance(val1, qml.Hamiltonian):
            qml.assert_equal(val1, val2)
        else:
            assert val1 == val2


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (
            qml.FermionicDoubleExcitation(0.123, wires1=[0, 1, 2], wires2=[3, 4]),
            [0.456],
            qml.FermionicDoubleExcitation(0.456, wires1=[0, 1, 2], wires2=[3, 4]),
        ),
        (
            qml.FermionicSingleExcitation(0.123, wires=[0, 1]),
            [0.456],
            qml.FermionicSingleExcitation(0.456, wires=[0, 1]),
        ),
    ],
)
def test_fermionic_template_ops(op, new_params, expected_op):
    """Test that `bind_new_parameters` with fermionic template operators returns a new operator
    with the new parameters."""
    new_op = bind_new_parameters(op, new_params)

    qml.assert_equal(new_op, expected_op)
    assert new_op is not op


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qml.RX(0.123, 0), [0.456], qml.RX(0.456, 0)),
        (
            qml.Rot(0.123, 0.456, 0.789, wires=1),
            [1.23, 4.56, 7.89],
            qml.Rot(1.23, 4.56, 7.89, wires=1),
        ),
        (
            qml.QubitUnitary(X, wires=0),
            [I],
            qml.QubitUnitary(I, wires=0),
        ),
        (
            qml.BasisState([0, 1], wires=[0, 1]),
            [[1, 1]],
            qml.BasisState([1, 1], wires=[0, 1]),
        ),
        (qml.GellMann(wires=0, index=4), [], qml.GellMann(wires=0, index=4)),
        (qml.Identity(wires=[1, 2]), [], qml.Identity(wires=[1, 2])),
        (qml.pow(qml.RX(0.123, 0), 2, lazy=False), [0.456], qml.RX(0.456, 0)),
        (
            qml.adjoint(qml.RX(0.123, wires=0), lazy=False),
            [0.456],
            qml.RX(0.456, wires=0),
        ),
        (qml.PCPhase(0.123, 2, wires=[0, 1]), [0.456], qml.PCPhase(0.456, 2, wires=[0, 1])),
    ],
)
def test_vanilla_operators(op, new_params, expected_op):
    """Test that `bind_new_parameters` with vanilla operators returns a new
    operator with the new parameters without mutating the original
    operator."""
    new_op = bind_new_parameters(op, new_params)

    qml.assert_equal(new_op, expected_op)
    assert new_op is not op


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qml.Projector([0, 0], [0, 1]), [[0, 1]], qml.Projector([0, 1], [0, 1])),
        (qml.Projector([1, 0, 0, 0], [0, 1]), [[0, 0]], qml.Projector([0, 0], [0, 1])),
        (qml.Projector([1, 1], [0, 1]), [[0.5] * 4], qml.Projector([0.5] * 4, [0, 1])),
        (qml.Projector([1, 0, 0, 0], [0, 1]), [[0.5] * 4], qml.Projector([0.5] * 4, [0, 1])),
    ],
)
def test_projector(op, new_params, expected_op):
    """Test that `bind_new_parameters` with projectors returns a new projector with the new
    parameters without mutating the original operator."""
    new_op = bind_new_parameters(op, new_params)

    qml.assert_equal(new_op, expected_op)
    assert new_op is not op


@pytest.mark.parametrize(
    "op, new_params, expected_op",
    [
        (qml.RX(0.123, 0), (0.456,), qml.RX(0.456, 0)),
        (
            qml.QubitUnitary([[1, 0], [0, -1]], 0),
            ([[0, 1], [1, 0]],),
            qml.QubitUnitary([[0, 1], [1, 0]], 0),
        ),
        (qml.Rot(1.23, 4.56, 7.89, 0), (9.87, 6.54, 3.21), qml.Rot(9.87, 6.54, 3.21, 0)),
    ],
)
def test_conditional_ops(op, new_params, expected_op):
    """Test that Conditional ops are bound correctly."""
    mp0 = qml.measurements.MidMeasureMP(qml.wires.Wires(0), reset=True, id="foo")
    mv0 = qml.measurements.MeasurementValue([mp0], lambda v: v)
    cond_op = qml.ops.Conditional(mv0, op)
    new_op = bind_new_parameters(cond_op, new_params)

    assert isinstance(new_op, qml.ops.Conditional)
    assert new_op.base == expected_op
    assert new_op.meas_val.measurements == [mp0]


def test_unsupported_op_copy_and_set():
    """Test that trying to use `bind_new_parameters` on an operator without
    a supported dispatcher will fall back to copying the operator and setting
    `new_op.data` to the new parameters."""
    op = qml.PCPhase(0.123, 2, wires=[1, 2])
    new_op = bind_new_parameters(op, [0.456])

    expected_op = qml.PCPhase(0.456, 2, wires=[1, 2])

    qml.assert_equal(new_op, expected_op)
    assert new_op is not op
