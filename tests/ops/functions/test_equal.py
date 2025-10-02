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
Unit tests for the equal function.
Tests are divided by number of parameters and wires different operators take.
"""
import itertools
import re

# pylint: disable=too-many-arguments, too-many-public-methods
from copy import deepcopy

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as npp
from pennylane.measurements import ExpectationMP
from pennylane.measurements.probs import ProbabilityMP
from pennylane.operation import Operator
from pennylane.ops import Conditional
from pennylane.ops.functions.equal import (
    BASE_OPERATION_MISMATCH_ERROR_MESSAGE,
    OPERANDS_MISMATCH_ERROR_MESSAGE,
    _equal_dispatch,
    assert_equal,
)
from pennylane.ops.op_math import Controlled, SymbolicOp
from pennylane.templates.subroutines import ControlledSequence

PARAMETRIZED_OPERATIONS_1P_1W = [
    qml.RX,
    qml.RY,
    qml.RZ,
    qml.PhaseShift,
    qml.U1,
]

PARAMETRIZED_OPERATIONS_1P_2W = [
    qml.IsingXX,
    qml.IsingYY,
    qml.IsingZZ,
    qml.IsingXY,
    qml.ControlledPhaseShift,
    qml.CRX,
    qml.CRY,
    qml.CRZ,
    qml.SingleExcitation,
    qml.SingleExcitationPlus,
    qml.SingleExcitationMinus,
]


PARAMETRIZED_OPERATIONS_3P_1W = [
    qml.Rot,
    qml.U3,
]


PARAMETRIZED_OPERATIONS_1P_4W = [
    qml.DoubleExcitation,
    qml.DoubleExcitationPlus,
    qml.DoubleExcitationMinus,
]

PARAMETRIZED_OPERATIONS_Remaining = [
    qml.PauliRot,
    qml.QubitUnitary,
    qml.DiagonalQubitUnitary,
    qml.ControlledQubitUnitary,
]
PARAMETRIZED_OPERATIONS_2P_1W = [qml.U2]
PARAMETRIZED_OPERATIONS_1P_3W = [qml.MultiRZ]
PARAMETRIZED_OPERATIONS_3P_2W = [qml.CRot]


PARAMETRIZED_OPERATIONS = [
    qml.RX(0.123, wires=0),
    qml.RY(1.434, wires=0),
    qml.RZ(2.774, wires=0),
    qml.PauliRot(0.123, "Y", wires=0),
    qml.IsingXX(0.123, wires=[0, 1]),
    qml.IsingYY(0.123, wires=[0, 1]),
    qml.IsingZZ(0.123, wires=[0, 1]),
    qml.IsingXY(0.123, wires=[0, 1]),
    qml.Rot(0.123, 0.456, 0.789, wires=0),
    qml.PhaseShift(2.133, wires=0),
    qml.ControlledPhaseShift(1.777, wires=[0, 2]),
    qml.MultiRZ(0.112, wires=[1, 2, 3]),
    qml.CRX(0.836, wires=[2, 3]),
    qml.CRY(0.721, wires=[2, 3]),
    qml.CRZ(0.554, wires=[2, 3]),
    qml.U1(0.123, wires=0),
    qml.U2(3.556, 2.134, wires=0),
    qml.U3(2.009, 1.894, 0.7789, wires=0),
    qml.CRot(0.123, 0.456, 0.789, wires=[0, 1]),
    qml.QubitUnitary(np.eye(2) * 1j, wires=0),
    qml.DiagonalQubitUnitary(np.array([1.0, 1.0j]), wires=1),
    qml.ControlledQubitUnitary(np.eye(2) * 1j, wires=[2, 0]),
    qml.SingleExcitation(0.123, wires=[0, 3]),
    qml.SingleExcitationPlus(0.123, wires=[0, 3]),
    qml.SingleExcitationMinus(0.123, wires=[0, 3]),
    qml.DoubleExcitation(0.123, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationPlus(0.123, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationMinus(0.123, wires=[0, 1, 2, 3]),
]
PARAMETRIZED_OPERATIONS_COMBINATIONS = list(
    itertools.combinations(
        PARAMETRIZED_OPERATIONS,
        2,
    )
)

PARAMETRIZED_MEASUREMENTS = [
    qml.sample(qml.PauliY(0)),
    qml.sample(wires=0),
    qml.sample(),
    qml.counts(qml.PauliZ(0)),
    qml.counts(wires=[0, 1]),
    qml.counts(wires=[1, 0]),
    qml.counts(),
    qml.density_matrix(wires=0),
    qml.density_matrix(wires=1),
    qml.var(qml.PauliY(1)),
    qml.var(qml.PauliY(0)),
    qml.expval(qml.PauliX(0)),
    qml.expval(qml.PauliX(1)),
    qml.probs(wires=1),
    qml.probs(wires=0),
    qml.probs(op=qml.PauliZ(0)),
    qml.probs(op=qml.PauliZ(1)),
    qml.state(),
    qml.vn_entropy(wires=0),
    qml.vn_entropy(wires=0, log_base=np.e),
    qml.mutual_info(wires0=[0], wires1=[1]),
    qml.mutual_info(wires0=[1], wires1=[0]),
    qml.mutual_info(wires0=[1], wires1=[0], log_base=2),
    qml.classical_shadow(wires=[0, 1]),
    qml.classical_shadow(wires=[1, 0]),
    qml.shadow_expval(
        H=qml.Hamiltonian(
            [1.0, 1.0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
        ),
        k=2,
    ),
    qml.shadow_expval(
        H=qml.Hamiltonian(
            [1.0, 1.0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
        )
    ),
    qml.shadow_expval(
        H=qml.Hamiltonian(
            [1.0, 1.0], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        ),
        k=3,
    ),
    qml.shadow_expval(
        H=[
            qml.Hamiltonian(
                [1.0, 1.0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
            )
        ],
        k=2,
    ),
    qml.shadow_expval(
        H=[
            qml.Hamiltonian(
                [1.0, 1.0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
            )
        ]
    ),
    qml.shadow_expval(
        H=[
            qml.Hamiltonian(
                [1.0, 1.0], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)]
            )
        ],
        k=3,
    ),
    ExpectationMP(eigvals=[1, -1]),
    ExpectationMP(eigvals=[1, 2]),
]
PARAMETRIZED_MEASUREMENTS_COMBINATIONS = list(
    itertools.combinations(
        PARAMETRIZED_MEASUREMENTS,
        2,
    )
)

equal_hamiltonians = [
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliZ(0)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(0)]),
        True,
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliY(2) @ qml.PauliZ(0)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliY(2) @ qml.Identity(1)]),
        True,
    ),
    (
        qml.Hamiltonian(
            [1, 1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliZ(0), qml.Identity(1)]
        ),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(0)]),
        False,
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(1), qml.PauliZ(0)]),
        False,
    ),
    (
        qml.Hamiltonian([1, 2], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        False,
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX("a"), qml.PauliZ("b")]),
        qml.Hamiltonian([1, 1], [qml.PauliX("a"), qml.PauliZ("b")]),
        True,
    ),
    (
        qml.Hamiltonian([1, 2], [qml.PauliX("a"), qml.PauliZ("b")]),
        qml.Hamiltonian([1, 1], [qml.PauliX("b"), qml.PauliZ("a")]),
        False,
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliZ(3) @ qml.Identity(1.2), qml.PauliZ(3)]),
        qml.Hamiltonian([2], [qml.PauliZ(3)]),
        True,
    ),
]

equal_pauli_operators = [
    (qml.PauliX(0), qml.PauliX(0), True),
    (qml.PauliY("a"), qml.PauliY("a"), True),
    (qml.PauliY(0.3), qml.PauliY(0.3), True),
    (qml.PauliX(0), qml.PauliX(1), False),
    (qml.PauliY("a"), qml.PauliY("b"), False),
    (qml.PauliY(0.3), qml.PauliY(0.7), False),
    (qml.PauliY(0), qml.PauliX(0), False),
    (qml.PauliY("a"), qml.PauliX("a"), False),
    (qml.PauliZ(0.3), qml.PauliY(0.3), False),
    (qml.PauliZ(0), qml.RX(1.23, 0), False),
]


def test_assert_equal_types():
    """Test that assert equal raises if the operator types are different."""

    op1 = qml.S(0)
    op2 = qml.T(0)
    with pytest.raises(AssertionError, match="op1 and op2 are of different types"):
        assert_equal(op1, op2)


def test_assert_equal_unspecified():

    # pylint: disable=too-few-public-methods
    class RandomType:
        """dummy type"""

        def __init__(self):
            pass

    # pylint: disable=unused-argument
    @_equal_dispatch.register
    def _(op1: RandomType, op2, **_):
        """always returns false"""
        return False

    with pytest.raises(AssertionError, match=r"for an unspecified reason"):
        assert_equal(RandomType(), RandomType())


class TestEqual:

    def test_identity_equal(self):
        """Test that comparing two Identities always returns True regardless of wires"""
        I1 = qml.Identity()
        I2 = qml.Identity(wires=[-1])
        I3 = qml.Identity(wires=[0, 1, 2, 3])
        I4 = qml.Identity(wires=["a", "b"])

        assert qml.equal(I1, I2) is True
        assert qml.equal(I1, I3) is True
        assert qml.equal(I1, I4) is True
        assert qml.equal(I2, I3) is True
        assert qml.equal(I2, I4) is True
        assert qml.equal(I3, I4) is True

    @pytest.mark.parametrize(("op1", "op2", "res"), equal_pauli_operators)
    def test_pauli_operator_equals(self, op1, op2, res):
        """Tests that equality can be checked between PauliX/Y/Z operators, and between Pauli operators
        and Hamiltonians"""

        assert qml.equal(op1, op2) == qml.equal(op2, op1)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize("ops", PARAMETRIZED_OPERATIONS_COMBINATIONS)
    def test_equal_simple_diff_op(self, ops):
        """Test different operators return False"""
        assert qml.equal(ops[0], ops[1], check_trainability=False, check_interface=False) is False
        with pytest.raises(AssertionError, match="op1 and op2 are of different types"):
            assert_equal(ops[0], ops[1], check_trainability=False, check_interface=False)

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS)
    def test_equal_simple_same_op(self, op1):
        """Test same operators return True"""
        assert qml.equal(op1, op1, check_trainability=False, check_interface=False) is True
        assert_equal(op1, op1, check_trainability=False, check_interface=False)

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_1W)
    def test_equal_simple_op_1p1w(self, op1):
        """Test changing parameter or wire returns False"""
        wire = 0
        param = 0.123
        test_operator = op1(param, wires=wire)
        assert (
            qml.equal(
                test_operator,
                test_operator,
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert_equal(
            test_operator,
            test_operator,
            check_trainability=False,
            check_interface=False,
        )

        test_operator_diff_parameter = op1(param * 2, wires=wire)
        assert (
            qml.equal(
                test_operator,
                test_operator_diff_parameter,
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        with pytest.raises(AssertionError, match="op1 and op2 have different data."):
            assert_equal(
                test_operator,
                test_operator_diff_parameter,
                check_trainability=False,
                check_interface=False,
            )
        test_operator_diff_wire = op1(param, wires=wire + 1)
        assert (
            qml.equal(
                test_operator,
                test_operator_diff_wire,
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        with pytest.raises(AssertionError, match="op1 and op2 have different wires."):
            assert_equal(
                test_operator,
                test_operator_diff_wire,
                check_trainability=False,
                check_interface=False,
            )

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_1W)
    def test_equal_op_1p1w(self, op1):
        """Test optional arguments are working"""
        wire = 0

        import jax
        import torch

        param_torch = torch.tensor(0.123)
        param_jax = jax.numpy.array(0.123)
        param_qml = npp.array(0.123)
        param_np = np.array(0.123)

        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

        with pytest.raises(AssertionError, match="Parameters have different trainability"):
            assert_equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_2W)
    def test_equal_op_1p2w(self, op1):
        """Test optional arguments are working"""
        wire = [0, 1]

        import jax
        import torch

        param_torch = torch.tensor(0.123)
        param_jax = jax.numpy.array(0.123)
        param_qml = npp.array(0.123)
        param_np = np.array(0.123)

        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_2W)
    def test_equal_simple_op_1p2w(self, op1):
        """Test changing parameter or wire returns False"""
        wire = [0, 1]
        param = 0.123
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param * 2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=[w + 1 for w in wire]),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_4W)
    def test_equal_op_1p4w(self, op1):
        """Test optional arguments are working"""
        wire = [0, 1, 2, 3]

        import jax
        import torch

        param_torch = torch.tensor(0.123)
        param_jax = jax.numpy.array(0.123)
        param_qml = npp.array(0.123)
        param_np = np.array(0.123)

        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_4W)
    def test_equal_simple_op_1p4w(self, op1):
        """Test changing parameter or wire returns False"""
        wire = [0, 1, 2, 3]
        param = 0.123
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param * 2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=[w + 1 for w in wire]),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_3P_1W)
    def test_equal_op_3p1w(self, op1):
        """Test optional arguments are working"""
        wire = 0

        import jax
        import torch

        param_torch = torch.tensor([1, 2, 3])
        param_jax = jax.numpy.array([1, 2, 3])
        param_qml = npp.array([1, 2, 3])
        param_np = np.array([1, 2, 3])

        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1[0], p1[1], p1[2], wires=wire),
                    op1(p2[0], p2[1], p2[2], wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1[0], p1[1], p1[2], wires=wire),
                    op1(p2[0], p2[1], p2[2], wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(*param_qml, wires=wire),
                op1(*param_qml_1, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(*param_qml, wires=wire),
                op1(*param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_3P_1W)
    def test_equal_simple_op_3p1w(self, op1):
        """Test changing parameter or wire returns False"""
        wire = 0
        param = [0.123] * 3
        assert (
            qml.equal(
                op1(*param, wires=wire),
                op1(*param, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(*param, wires=wire),
                op1(*param, wires=wire + 1),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(*param, wires=wire),
                op1(param[0] * 2, param[1], param[2], wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

    @pytest.mark.all_interfaces
    def test_equal_op_remaining(self):  # pylint: disable=too-many-statements
        """Test optional arguments are working"""
        # pylint: disable=too-many-statements
        wire = 0

        import jax
        import torch

        param_torch = torch.tensor([1, 2])
        param_jax = jax.numpy.array([1, 2])
        param_qml = npp.array([1, 2])
        param_np = np.array([1, 2])

        op1 = PARAMETRIZED_OPERATIONS_2P_1W[0]
        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1[0], p1[1], wires=wire),
                    op1(p2[0], p2[1], wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1[0], p1[1], wires=wire),
                    op1(p2[0], p2[1], wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(*param_qml, wires=wire),
                op1(*param_qml_1, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(*param_qml, wires=wire),
                op1(*param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

        wire = [1, 2, 3]
        param_torch = torch.tensor(1)
        param_jax = jax.numpy.array(1)
        param_qml = npp.array(1)
        param_np = np.array(1)

        op1 = PARAMETRIZED_OPERATIONS_1P_3W[0]
        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

        wire = [1, 2]
        param_torch = torch.tensor([1, 2, 3])
        param_jax = jax.numpy.array([1, 2, 3])
        param_qml = npp.array([1, 2, 3])
        param_np = np.array([1, 2, 3])

        op1 = PARAMETRIZED_OPERATIONS_3P_2W[0]
        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1[0], p1[1], p1[2], wires=wire),
                    op1(p2[0], p2[1], p2[2], wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1[0], p1[1], p1[2], wires=wire),
                    op1(p2[0], p2[1], p2[2], wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(*param_qml, wires=wire),
                op1(*param_qml_1, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(*param_qml, wires=wire),
                op1(*param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

        wire = 0
        param_torch = torch.tensor(1)
        param_jax = jax.numpy.array(1)
        param_qml = npp.array(1)
        param_np = np.array(1)

        op1 = PARAMETRIZED_OPERATIONS_Remaining[0]
        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1, "Y", wires=wire),
                    op1(p2, "Y", wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1, "Y", wires=wire),
                    op1(p2, "Y", wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(param_qml, "Y", wires=wire),
                op1(param_qml_1, "Y", wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param_qml, "Y", wires=wire),
                op1(param_qml_1, "Y", wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

        wire = 0
        param_torch = torch.tensor([[1, 0], [0, 1]]) * 1j
        param_jax = jax.numpy.eye(2) * 1j
        param_qml = npp.eye(2) * 1j
        param_np = np.eye(2) * 1j

        op1 = PARAMETRIZED_OPERATIONS_Remaining[1]
        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

        wire = 0
        param_torch = torch.tensor([1.0, 1.0j])
        param_jax = jax.numpy.array([1.0, 1.0j])
        param_qml = npp.array([1.0, 1.0j])
        param_np = np.array([1.0, 1.0j])

        op1 = PARAMETRIZED_OPERATIONS_Remaining[2]
        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1, wires=wire),
                    op1(p2, wires=wire),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param_qml, wires=wire),
                op1(param_qml_1, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

        wire = 0
        param_torch = torch.tensor([[1, 0], [0, 1]]) * 1j
        param_jax = jax.numpy.eye(2) * 1j
        param_qml = npp.eye(2) * 1j
        param_np = np.eye(2) * 1j

        # ControlledQubitUnitary
        op1 = PARAMETRIZED_OPERATIONS_Remaining[3]
        wires = [wire + 1, wire]
        param_list = [param_qml, param_torch, param_jax, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert (
                qml.equal(
                    op1(p1, wires=wires),
                    op1(p2, wires=wires),
                    check_trainability=False,
                    check_interface=False,
                )
                is True
            )
            assert (
                qml.equal(
                    op1(p1, wires=wires),
                    op1(p2, wires=wires),
                    check_trainability=False,
                    check_interface=True,
                )
                is False
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert (
            qml.equal(
                op1(param_qml, wires=wires),
                op1(param_qml_1, wires=wires),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param_qml, wires=wires),
                op1(param_qml_1, wires=wires),
                check_trainability=True,
                check_interface=False,
            )
            is False
        )

    def test_equal_simple_op_remaining(self):
        """Test changing parameter or wire returns False"""
        wire = 0
        param = [0.123] * 2
        op1 = PARAMETRIZED_OPERATIONS_2P_1W[0]
        assert (
            qml.equal(
                op1(*param, wires=wire),
                op1(*param, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(*param, wires=wire),
                op1(*param, wires=wire + 1),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(*param, wires=wire),
                op1(param[0] * 2, param[1], wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

        wire = [1, 2, 3]
        param = 0.123
        op1 = PARAMETRIZED_OPERATIONS_1P_3W[0]
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param * 2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=[w + 1 for w in wire]),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

        wire = [1, 2]
        param = [0.123] * 3
        op1 = PARAMETRIZED_OPERATIONS_3P_2W[0]
        assert qml.equal(
            op1(*param, wires=wire),
            op1(*param, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert (
            qml.equal(
                op1(*param, wires=wire),
                op1(*param, wires=[w + 1 for w in wire]),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(*param, wires=wire),
                op1(param[0] * 2, param[1], param[2], wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

        wire = 0
        param = 0.123
        op1 = PARAMETRIZED_OPERATIONS_Remaining[0]
        assert (
            qml.equal(
                op1(param, "Y", wires=wire),
                op1(param, "Y", wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param, "Y", wires=wire),
                op1(param * 2, "Y", wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(param, "Y", wires=wire),
                op1(param, "Y", wires=wire + 1),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(param, "Y", wires=wire),
                op1(param, "Z", wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        with pytest.raises(
            AssertionError, match="The hyperparameters are not equal for op1 and op2."
        ):
            assert_equal(
                op1(param, "Y", wires=wire),
                op1(param, "Z", wires=wire),
                check_trainability=False,
                check_interface=False,
            )

        wire = 0
        param = np.eye(2) * 1j
        op1 = PARAMETRIZED_OPERATIONS_Remaining[1]
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param * 2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=wire + 1),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

        wire = 0
        param = np.array([1.0, 1.0j])
        op1 = PARAMETRIZED_OPERATIONS_Remaining[2]
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param * 2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(param, wires=wire),
                op1(param, wires=wire + 1),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

        wire = 0
        param = np.eye(2) * 1j
        op1 = PARAMETRIZED_OPERATIONS_Remaining[3]
        wires = [wire + 1, wire]
        assert (
            qml.equal(
                op1(param, wires=wires),
                op1(param, wires=wires),
                check_trainability=False,
                check_interface=False,
            )
            is True
        )
        assert (
            qml.equal(
                op1(param, wires=wires),
                op1(param * 2, wires=wires),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )
        assert (
            qml.equal(
                op1(param, wires=wires),
                op1(param, wires=[wire + 1, wire + 2]),
                check_trainability=False,
                check_interface=False,
            )
            is False
        )

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_1W)
    def test_equal_trainable_different_interface(self, op1):
        """Test equal method with two operators with trainable inputs and
        different interfaces.

        This test case tests the 4 interface with each other pairwise,
        totalling a 4*3/2=6 total assertions in the following order (assuming
        symmetry doesn't affect the behaviour):

        -JAX and Autograd
        -JAX and Torch
        -Autograd and Torch
        """
        import jax
        import torch

        wire = 0

        pl_tensor = qml.numpy.array(0.3, requires_grad=True)
        torch_tensor = torch.tensor(0.3, requires_grad=True)

        non_jax_tensors = [pl_tensor, torch_tensor]

        # JAX and the others
        # ------------------
        # qml.math.requires_grad returns True for a Tracer with JAX, the
        # assertion involves using a JAX function that transforms a JAX NumPy
        # array into a Tracer
        def jax_assertion_func(x, other_tensor):
            operation1 = op1(jax.numpy.array(x), wires=1)
            operation2 = op1(other_tensor, wires=1)
            assert qml.equal(operation1, operation2, check_interface=False, check_trainability=True)
            return x

        par = 0.3
        for tensor in non_jax_tensors:
            jax.grad(jax_assertion_func, argnums=0)(par, tensor)

        # Autograd and Torch
        # ------------------
        assert (
            qml.equal(
                op1(pl_tensor, wires=wire),
                op1(torch_tensor, wires=wire),
                check_trainability=True,
                check_interface=False,
            )
            is True
        )

        with pytest.raises(AssertionError, match="Parameters have different interfaces"):
            assert_equal(
                op1(pl_tensor, wires=wire),
                op1(torch_tensor, wires=wire),
                check_trainability=True,
                check_interface=True,
            )

    def test_equal_with_different_arithmetic_depth(self):
        """Test equal method with two operators with different arithmetic depth."""
        op1 = Operator(wires=0)
        op2 = DepthIncreaseOperator(op1)

        assert qml.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="op1 and op2 have different arithmetic depths"):
            assert_equal(op1, op2)

    def test_equal_with_unsupported_nested_operators_returns_false(self):
        """Test that the equal method with two operators with the same arithmetic depth (>0) returns
        `False` unless there is a singledispatch function specifically comparing that operator type.
        """

        op1 = SymbolicOp(qml.PauliY(0))
        op2 = SymbolicOp(qml.PauliY(0))

        assert op1.arithmetic_depth == op2.arithmetic_depth
        assert op1.arithmetic_depth > 0

        assert qml.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="op1 and op2 have arithmetic depth > 0"):
            assert_equal(op1, op2)

    # Measurements test cases
    @pytest.mark.parametrize("ops", PARAMETRIZED_MEASUREMENTS_COMBINATIONS)
    def test_not_equal_diff_measurement(self, ops):
        """Test different measurements return False"""
        assert qml.equal(ops[0], ops[1]) is False

    @pytest.mark.parametrize("op1", PARAMETRIZED_MEASUREMENTS)
    def test_equal_same_measurement(self, op1):
        """Test same measurements return True"""
        assert qml.equal(op1, op1) is True

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS)
    @pytest.mark.parametrize("op2", PARAMETRIZED_MEASUREMENTS)
    def test_not_equal_operator_measurement(self, op1, op2):
        """Test operator not equal to measurement"""
        assert qml.equal(op1, op2) is False


equal_pauli_words = [
    ({0: "X", 1: "Y"}, {1: "Y", 0: "X"}, True, None),
    ({0: "X", 1: "Y"}, {0: "X"}, False, "Different wires in Pauli words."),
    ({0: "X", 1: "Z"}, {1: "Y", 0: "X"}, False, "agree on wires but differ in Paulis."),
    ({0: "X", 1: "Y"}, {"X": "Y", 0: "X"}, False, "Different wires in Pauli words."),
]


class TestPauliErrorEqual:
    """Tests for qml.equal with PauliErrors."""

    ARGS_ONE = [
        ["XY", 0.1, (0, 1)],
        ["XY", 0.1, (0, 1), "one"],
        ["XY", 0.1, (0, 1), "one"],
        ["XY", 0.1, (0, 1), "one"],
        ["XY", 0.1, (0, 1), "one"],
    ]
    ARGS_TWO = [
        ["XY", 0.1, (0, 1)],
        ["XY", 0.1, (0, 1), "two"],  # id is not in op.data
        ["XYZ", 0.1, (0, 1, 2), "two"],  # different Pauli strs, number of wires
        ["XZ", 0.1, (0, 1), "two"],  # different Pauli strs
        ["XY", 0.1, (0, 2), "two"],  # different wire numbers
    ]
    EQS = [True, True, False, False, False]

    @pytest.mark.parametrize("args1, args2, eqs", list(zip(ARGS_ONE, ARGS_TWO, EQS)))
    def test_equality(self, args1, args2, eqs):
        e1 = qml.PauliError(*args1)
        e2 = qml.PauliError(*args2)

        eq = qml.equal(e1, e2)
        if eqs:
            assert eq
        else:
            assert not eq

    def test_equal_with_different_arithmetic_depth(self):
        """Test equal method with two operators with different arithmetic depth."""
        op1 = qml.PauliError("XY", 0.1, (0, 1))
        op2 = DepthIncreaseOperator(op1)

        assert qml.equal(op1, op2) is False

    @pytest.mark.torch
    def test_trainability_and_interface(self):
        """Test that trainability and interface are compared correctly."""

        x1 = qml.numpy.array(0.5, requires_grad=True)
        pes = [qml.PauliError("XY", x1, (0, 1)), qml.PauliError("XY", 0.5, (0, 1))]

        assert qml.equal(pes[0], pes[1]) is False
        with pytest.raises(AssertionError, match="Parameters have different trainability"):
            assert_equal(pes[0], pes[1])

        with pytest.raises(AssertionError, match="Parameters have different interfaces"):
            assert_equal(pes[0], pes[1], check_trainability=False)


# pylint: disable=too-few-public-methods
class TestPauliWordsEqual:
    """Tests for qml.equal with PauliSentences."""

    @pytest.mark.parametrize("pw1, pw2, res, error_match", equal_pauli_words)
    def test_equality(self, pw1, pw2, res, error_match):
        """Test basic equalities/inequalities."""
        pw1 = qml.pauli.PauliWord(pw1)
        pw2 = qml.pauli.PauliWord(pw2)
        assert qml.equal(pw1, pw2) is res
        assert qml.equal(pw2, pw1) is res

        if res:
            assert_equal(pw1, pw2)
            assert_equal(pw2, pw1)
        else:
            with pytest.raises(AssertionError, match=error_match):
                assert_equal(pw1, pw2)
            with pytest.raises(AssertionError, match=error_match):
                assert_equal(pw2, pw1)


equal_pauli_sentences = [
    (qml.X(0) @ qml.Y(2), 1.0 * qml.Y(2) @ qml.X(0), True, None),
    (
        qml.X(0) @ qml.Y(2),
        1.0 * qml.X(2) @ qml.Y(0),
        False,
        "Different Pauli words in PauliSentences",
    ),
    (qml.X(0) - qml.Y(2), -1.0 * (qml.Y(2) - qml.X(0)), True, None),
    (qml.X(0) @ qml.Y(2), qml.Y(2) + qml.X(0), False, "Different Pauli words in PauliSentences"),
    (qml.SISWAP([0, "a"]) @ qml.Z("b"), qml.Z("b") @ qml.SISWAP((0, "a")), True, None),
    (qml.SWAP([0, "a"]) @ qml.S("b"), qml.S("b") @ qml.SWAP(("a", 0)), True, None),
]


class TestPauliSentencesEqual:
    """Tests for qml.equal with PauliSentences."""

    @pytest.mark.parametrize("ps1, ps2, res, error_match", equal_pauli_sentences)
    def test_equality(self, ps1, ps2, res, error_match):
        """Test basic equalities/inequalities."""
        ps1 = qml.simplify(ps1).pauli_rep
        ps2 = qml.simplify(ps2).pauli_rep

        assert qml.equal(ps1, ps2) is res
        assert qml.equal(ps1 * 0.6, ps2 * 0.6) is res
        assert qml.equal(ps2, ps1) is res

        if res:
            assert_equal(ps1, ps2)
            assert_equal(ps2, ps1)
        else:
            with pytest.raises(AssertionError, match=error_match):
                assert_equal(ps1, ps2)
            with pytest.raises(AssertionError, match=error_match):
                assert_equal(ps2, ps1)

    @pytest.mark.torch
    def test_trainability_and_interface(self):
        """Test that trainability and interface are compared correctly."""
        import torch

        x1 = qml.numpy.array(0.5, requires_grad=True)
        x2 = qml.numpy.array(0.5, requires_grad=False)
        x3 = torch.tensor(0.5, requires_grad=True)
        x4 = torch.tensor(0.5, requires_grad=False)
        pws = [qml.pauli.PauliWord({1: "X", 39: "Y"}), qml.pauli.PauliWord({0: "Z", 1: "Y"})]
        ps1 = pws[0] * x1 - 0.7 * pws[1]
        ps2 = pws[0] * x2 - 0.7 * pws[1]
        ps3 = pws[0] * x3 - 0.7 * pws[1]
        ps4 = pws[0] * x4 - 0.7 * pws[1]

        assert qml.equal(ps1, ps2) is False
        with pytest.raises(AssertionError, match="Parameters have different trainability"):
            assert_equal(ps1, ps2)
        assert qml.equal(ps1, ps3) is False
        assert qml.equal(ps1, ps4) is False
        assert qml.equal(ps2, ps3) is False
        assert qml.equal(ps2, ps4) is False
        assert qml.equal(ps3, ps4) is False

        assert qml.equal(ps1, ps2, check_trainability=False) is True
        assert_equal(ps1, ps2, check_trainability=False)
        assert qml.equal(ps1, ps3, check_trainability=False) is False
        with pytest.raises(AssertionError, match="Parameters have different interfaces"):
            assert_equal(ps1, ps3, check_trainability=False)
        assert qml.equal(ps1, ps4, check_trainability=False) is False
        assert qml.equal(ps2, ps3, check_trainability=False) is False
        assert qml.equal(ps2, ps4, check_trainability=False) is False
        assert qml.equal(ps3, ps4, check_trainability=False) is True

        assert qml.equal(ps1, ps2, check_interface=False) is False
        with pytest.raises(AssertionError, match="Parameters have different trainability"):
            assert_equal(ps1, ps2, check_interface=False)
        assert qml.equal(ps1, ps3, check_interface=False) is True
        assert_equal(ps1, ps3, check_interface=False)
        assert qml.equal(ps1, ps4, check_interface=False) is False
        assert qml.equal(ps2, ps3, check_interface=False) is False
        assert qml.equal(ps2, ps4, check_interface=False) is True
        assert qml.equal(ps3, ps4, check_interface=False) is False

        assert qml.equal(ps1, ps2, check_trainability=False, check_interface=False) is True
        assert_equal(ps1, ps2, check_trainability=False, check_interface=False)
        assert qml.equal(ps1, ps3, check_trainability=False, check_interface=False) is True
        assert qml.equal(ps1, ps4, check_trainability=False, check_interface=False) is True
        assert qml.equal(ps2, ps3, check_trainability=False, check_interface=False) is True
        assert qml.equal(ps2, ps4, check_trainability=False, check_interface=False) is True
        assert qml.equal(ps3, ps4, check_trainability=False, check_interface=False) is True

    @pytest.mark.parametrize(
        "atol, rtol, res", [(1e-9, 0.0, False), (1e-7, 0.0, True), (0.0, 1e-9, True)]
    )
    def test_tolerance(self, atol, rtol, res):
        """Test that tolerances are taken into account correctly."""
        x1 = 100
        x2 = 100 + 1e-8
        pws = [qml.pauli.PauliWord({1: "X", 39: "Y"}), qml.pauli.PauliWord({0: "Z", 1: "Y"})]
        ps1 = pws[0] * x1 - 0.7 * pws[1]
        ps2 = pws[0] * x2 - 0.7 * pws[1]
        assert qml.equal(ps1, ps2, atol=atol, rtol=rtol) is res


class TestMeasurementsEqual:
    @pytest.mark.jax
    def test_observables_different_interfaces(self):
        """Check that the check_interface keyword is used when comparing observables."""

        import jax

        M1 = np.eye(2)
        M2 = jax.numpy.eye(2)
        ob1 = qml.Hermitian(M1, 0)
        ob2 = qml.Hermitian(M2, 0)

        assert qml.equal(qml.expval(ob1), qml.expval(ob2), check_interface=True) is False
        assert qml.equal(qml.expval(ob1), qml.expval(ob2), check_interface=False) is True

    def test_observables_different_trainability(self):
        """Check the check_trainability keyword argument affects comparisons of measurements."""
        M1 = qml.numpy.eye(2, requires_grad=True)
        M2 = qml.numpy.eye(2, requires_grad=False)

        ob1 = qml.Hermitian(M1, 0)
        ob2 = qml.Hermitian(M2, 0)

        assert qml.equal(qml.expval(ob1), qml.expval(ob2), check_trainability=True) is False
        assert qml.equal(qml.expval(ob1), qml.expval(ob2), check_trainability=False) is True

    def test_observables_atol(self):
        """Check that the atol keyword argument affects comparisons of measurements."""
        M1 = np.eye(2)
        M2 = M1 + 1e-3

        ob1 = qml.Hermitian(M1, 0)
        ob2 = qml.Hermitian(M2, 0)

        assert qml.equal(qml.expval(ob1), qml.expval(ob2)) is False
        assert qml.equal(qml.expval(ob1), qml.expval(ob2), atol=1e-1) is True

    def test_observables_rtol(self):
        """Check rtol affects comparison of measurement observables."""
        M1 = np.eye(2)
        M2 = np.diag([1 + 1e-3, 1 - 1e-3])

        ob1 = qml.Hermitian(M1, 0)
        ob2 = qml.Hermitian(M2, 0)

        assert qml.equal(qml.expval(ob1), qml.expval(ob2)) is False
        assert qml.equal(qml.expval(ob1), qml.expval(ob2), rtol=1e-2) is True

    def test_eigvals_atol(self):
        """Check atol affects comparisons of eigenvalues."""
        m1 = ProbabilityMP(eigvals=(1, 1e-3))
        m2 = ProbabilityMP(eigvals=(1, 0))

        assert qml.equal(m1, m2) is False
        assert qml.equal(m1, m2, atol=1e-2) is True

    def test_eigvals_rtol(self):
        """Check that rtol affects comparisons of eigenvalues."""
        m1 = ProbabilityMP(eigvals=(1 + 1e-3, 0))
        m2 = ProbabilityMP(eigvals=(1, 0))

        assert qml.equal(m1, m2) is False
        assert qml.equal(m1, m2, rtol=1e-2) is True

    def test_observables_equal_but_wire_order_not(self):
        """Test that when the wire orderings are not equal but the observables are, that
        we still get True."""

        x1 = qml.PauliX(1)
        z0 = qml.PauliZ(0)

        o1 = qml.prod(x1, z0)
        o2 = qml.prod(z0, x1)
        assert qml.equal(qml.expval(o1), qml.expval(o2)) is True

    def test_mid_measure(self):
        """Test that `MidMeasureMP`s are equal only if their wires
        an id are equal and their `reset` attribute match."""
        mp = qml.measurements.MidMeasureMP(wires=qml.wires.Wires([0]), reset=True, id="test_id")

        mp1 = qml.measurements.MidMeasureMP(wires=qml.wires.Wires([1]), reset=True, id="test_id")
        mp2 = qml.measurements.MidMeasureMP(wires=qml.wires.Wires([0]), reset=False, id="test_id")
        mp3 = qml.measurements.MidMeasureMP(wires=qml.wires.Wires([0]), reset=True, id="foo")

        assert qml.equal(mp, mp1) is False
        assert qml.equal(mp, mp2) is False
        assert qml.equal(mp, mp3) is False

        assert (
            qml.equal(
                mp,
                qml.measurements.MidMeasureMP(wires=qml.wires.Wires([0]), reset=True, id="test_id"),
            )
            is True
        )

    def test_equal_measurement_value(self):
        """Test that MeasurementValue's are equal when their measurements are the same."""
        mv1 = qml.measure(0)
        mv2 = qml.measure(0)
        # qml.equal of MidMeasureMP checks the id
        mv2.measurements[0]._id = mv1.measurements[0].id  # pylint: disable=protected-access

        assert qml.equal(mv1, mv1) is True
        assert qml.equal(mv1, mv2) is True

    def test_different_measurement_value(self):
        """Test that MeasurementValue's are different when their measurements are not the same."""
        mv1 = qml.measure(0)
        mv2 = qml.measure(1)
        assert qml.equal(mv1, mv2) is False

    def test_composed_measurement_value(self):
        """Test that composition of MeasurementValue's are checked correctly."""
        mv1 = qml.measure(0)
        mv2 = qml.measure(1)
        mv3 = qml.measure(0)
        # qml.equal of MidMeasureMP checks the id
        mv3.measurements[0]._id = mv1.measurements[0].id  # pylint: disable=protected-access

        assert qml.equal(mv1 * mv2, mv2 * mv1) is True
        assert qml.equal(mv1 + mv2, mv3 + mv2) is True
        # NOTE: we are deliberatily just checking for measurements and not for processing_fn, such that two MeasurementValue objects composed from the same operators will be qml.equal
        assert qml.equal(3 * mv1 + 1, 4 * mv3 + 2) is True

    @pytest.mark.parametrize("mp_fn", [qml.probs, qml.sample, qml.counts])
    def test_mv_list_as_op(self, mp_fn):
        """Test that MeasurementProcesses that measure a list of MeasurementValues check for equality
        correctly."""
        mv1 = qml.measure(0)
        mv2 = qml.measure(1)
        mv3 = qml.measure(1)
        mv4 = qml.measure(0)
        mv4.measurements[0]._id = mv1.measurements[0].id  # pylint: disable=protected-access

        mp1 = mp_fn(op=[mv1, mv2])
        mp2 = mp_fn(op=[mv4, mv2])
        mp3 = mp_fn(op=[mv1, mv3])
        mp4 = mp_fn(op=[mv2, mv1])

        assert qml.equal(mp1, mp1) is True
        assert qml.equal(mp1, mp2) is True
        assert qml.equal(mp1, mp3) is False
        assert qml.equal(mp1, mp4) is False

    def test_mv_list_and_arithmetic_as_op(self):
        """Test that comparing measurements using composite measurement values and
        a list of measurement values fails."""
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        mp1 = qml.sample(op=m0 * m1)
        mp2 = qml.sample(op=[m0, m1])

        assert qml.equal(mp1, mp2) is False

    @pytest.mark.parametrize("mp_fn", [qml.expval, qml.var, qml.sample, qml.counts])
    def test_mv_arithmetic_as_op(self, mp_fn):
        """Test that MeasurementProcesses that measure a list of MeasurementValues check for equality
        correctly."""
        mv1 = qml.measure(0)
        mv2 = qml.measure(1)
        mv3 = qml.measure(1)
        mv4 = qml.measure(0)
        mv4.measurements[0]._id = mv1.measurements[0].id  # pylint: disable=protected-access

        mp1 = mp_fn(op=mv1 * mv2)
        mp2 = mp_fn(op=mv4 * mv2)
        mp3 = mp_fn(op=mv2 * mv1)
        mp4 = mp_fn(op=mv1 * mv3)

        assert qml.equal(mp1, mp1) is True
        assert qml.equal(mp1, mp2) is True
        assert qml.equal(mp1, mp3) is True
        assert qml.equal(mp1, mp4) is False

    @pytest.mark.jax
    @pytest.mark.parametrize("mp_fn", [qml.expval, qml.var, qml.sample, qml.counts, qml.probs])
    def test_abstract_mv_equality(self, mp_fn):
        """Test that equality is verified correctly for measurements collecting statistics for
        abstract mid-circuit measurement values"""
        import jax  # pylint: disable=import-outside-toplevel

        m1 = True
        m2 = False

        @jax.jit
        def eq_traced(a, b):
            assert qml.math.is_abstract(a)
            assert qml.math.is_abstract(b)

            mp1 = mp_fn(op=a)
            mp2 = mp_fn(op=a)
            mp3 = mp_fn(op=b)

            return qml.equal(mp1, mp2), qml.equal(mp1, mp3)

        res = eq_traced(m1, m2)
        assert res[0]
        assert not res[1]

    def test_shadow_expval_list_versus_operator(self):
        """Check that if one shadow expval has an operator and the other has a list, they are not equal."""

        op = qml.X(0)
        m1 = qml.shadow_expval(H=op)
        m2 = qml.shadow_expval(H=[op])
        assert qml.equal(m1, m2) is False


def test_unsupported_object_type_not_implemented():
    dev = qml.device("default.qubit", wires=1)

    with pytest.raises(NotImplementedError, match="Comparison of"):
        qml.equal(dev, dev)


class TestSymbolicOpComparison:
    """Test comparison for subclasses of SymbolicOp"""

    WIRES = [(5, 5, True), (6, 7, False)]
    CONTROL_WIRES_SEQUENCE = [
        ([1, 2], [1, 2], True),
        ([1, 2], [1, 3], False),
        ([1, 2, 3], [3, 2, 1], False),
    ]

    BASES = [
        (qml.PauliX(0), qml.PauliX(0), True),
        (qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1), True),
        (qml.CRX(1.23, [0, 1]), qml.CRX(1.23, [0, 1]), True),
        (qml.CRX(1.23, [1, 0]), qml.CRX(1.23, [0, 1]), False),
        (qml.PauliY(1), qml.PauliY(0), False),
        (qml.PauliX(1), qml.PauliY(1), False),
        (qml.PauliX(0) @ qml.PauliY(1), qml.PauliZ(1) @ qml.PauliY(0), False),
    ]

    PARAMS = [(1.23, 1.23, True), (5, 5, True), (2, -2, False), (1.2, 1, False)]

    def test_mismatched_arithmetic_depth(self):
        """Test that comparing SymoblicOp operators of mismatched arithmetic depth returns False"""
        base1 = qml.PauliX(0)
        base2 = qml.prod(qml.PauliX(0), qml.PauliY(1))

        op1 = Controlled(base1, control_wires=2)
        op2 = Controlled(base2, control_wires=2)

        assert op1.arithmetic_depth == 1
        assert op2.arithmetic_depth == 2
        assert qml.equal(op1, op2) is False

        op1 = ControlledSequence(base1, control=2)
        op2 = ControlledSequence(base2, control=2)

        assert op1.arithmetic_depth == 1
        assert op2.arithmetic_depth == 2
        assert qml.equal(op1, op2) is False

    def test_comparison_of_base_not_implemented_returns_false(self):
        """Test that comparing SymbolicOps of base operators whose comparison is not yet implemented returns False"""
        base = SymbolicOp(qml.RX(1.2, 0))
        op1 = Controlled(base, control_wires=2)
        op2 = Controlled(base, control_wires=2)

        assert qml.equal(op1, op2) is False

        op1 = ControlledSequence(base, control=2)
        op2 = ControlledSequence(base, control=2)

        assert qml.equal(op1, op2) is False

    @pytest.mark.torch
    @pytest.mark.jax
    def test_kwargs_for_base_operator_comparison(self):
        """Test that setting kwargs check_interface and check_trainability are applied when comparing the bases"""
        import jax
        import torch

        base1 = qml.RX(torch.tensor(1.2), wires=0)
        base2 = qml.RX(jax.numpy.array(1.2), wires=0)

        op1 = Controlled(base1, control_wires=1)
        op2 = Controlled(base2, control_wires=1)

        assert qml.equal(op1, op2) is False
        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True

        op1 = ControlledSequence(base1, control=1)
        op2 = ControlledSequence(base2, control=1)

        assert qml.equal(op1, op2) is False
        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True

    @pytest.mark.parametrize("base", PARAMETRIZED_OPERATIONS)
    def test_controlled_comparison(self, base):
        """Test that Controlled operators can be compared"""
        op1 = Controlled(base, control_wires=7, control_values=0)
        op2 = Controlled(base, control_wires=7, control_values=0)
        assert qml.equal(op1, op2) is True

    @pytest.mark.parametrize("base", PARAMETRIZED_OPERATIONS)
    def test_controlled_sequence_comparison(self, base):
        """Test that ControlledSequence operators can be compared"""
        op1 = ControlledSequence(base, control=7)
        op2 = ControlledSequence(base, control=7)
        assert qml.equal(op1, op2) is True

    @pytest.mark.parametrize("base", PARAMETRIZED_OPERATIONS)
    def test_controlled_sequence_deepcopy_comparison(self, base):
        """Test that equal is compatible with deepcopy"""
        op1 = ControlledSequence(base, control=7)
        assert qml.equal(op1, deepcopy(op1)) is True

    @pytest.mark.parametrize(("wire1", "wire2", "res"), WIRES)
    def test_controlled_base_operator_wire_comparison(self, wire1, wire2, res):
        """Test that equal compares operator wires for Controlled operators"""
        base1 = qml.PauliX(wire1)
        base2 = qml.PauliX(wire2)
        op1 = Controlled(base1, control_wires=1)
        op2 = Controlled(base2, control_wires=1)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("wire1", "wire2", "res"), WIRES)
    def test_controlled_sequence_base_operator_wire_comparison(self, wire1, wire2, res):
        """Test that equal compares operator wires for ControlledSequence operators"""
        base1 = qml.PauliX(wire1)
        base2 = qml.PauliX(wire2)
        op1 = ControlledSequence(base1, control=1)
        op2 = ControlledSequence(base2, control=1)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("base1", "base2", "res"), BASES)
    def test_controlled_base_operator_comparison(self, base1, base2, res):
        """Test that equal compares base operators for Controlled operators"""
        op1 = Controlled(base1, control_wires=2)
        op2 = Controlled(base2, control_wires=2)
        if res:
            assert qml.equal(op1, op2) is True
        else:
            assert qml.equal(op1, op2) is False
            with pytest.raises(AssertionError, match=BASE_OPERATION_MISMATCH_ERROR_MESSAGE):
                assert_equal(op1, op2)

    @pytest.mark.parametrize(("base1", "base2", "res"), BASES)
    def test_controlled_sequence_base_operator_comparison(self, base1, base2, res):
        """Test that equal compares base operators for ControlledSequence operators"""
        op1 = ControlledSequence(base1, control=2)
        op2 = ControlledSequence(base2, control=2)
        assert qml.equal(op1, op2) == res

    def test_controlled_sequence_with_different_base_operator(self):
        """Test controlled sequence operator with different base operators"""
        op1 = ControlledSequence(qml.PauliX(0), control=2)
        op2 = ControlledSequence(qml.PauliY(0), control=2)
        with pytest.raises(AssertionError, match=BASE_OPERATION_MISMATCH_ERROR_MESSAGE):
            assert_equal(op1, op2)

    def test_controlled_sequence_with_different_arithmetic_depth(self):
        """The depths of controlled sequence operators are different due to nesting"""
        base = qml.MultiRZ(1.23, [0, 1])
        depth_increased_base = DepthIncreaseOperator(base)
        op1 = ControlledSequence(base, control=5)
        op2 = ControlledSequence(depth_increased_base, control=5)

        with pytest.raises(AssertionError, match="op1 and op2 have different arithmetic depths."):
            assert_equal(op1, op2)

    @pytest.mark.parametrize(("wire1", "wire2", "res"), WIRES)
    def test_control_wires_comparison(self, wire1, wire2, res):
        """Test that equal compares control_wires for Controlled operators"""
        base1 = qml.Hadamard(0)
        base2 = qml.Hadamard(0)
        op1 = Controlled(base1, control_wires=wire1)
        op2 = Controlled(base2, control_wires=wire2)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("controls1", "controls2"), [([0, 1], [0, 1]), ([1, 1], [1, 0])])
    def test_control_values_comparison(self, controls1, controls2):
        """Test that equal compares control values for Controlled operators"""
        base1 = qml.PauliX(wires=0)
        base2 = qml.PauliX(wires=0)

        op1 = qml.ops.op_math.Controlled(base1, control_wires=[1, 2], control_values=controls1)
        op2 = qml.ops.op_math.Controlled(base2, control_wires=[1, 2], control_values=controls2)

        if np.allclose(controls1, controls2):
            assert qml.equal(op1, op2)
            assert_equal(op1, op2)
        else:
            assert qml.equal(op1, op2) is False
            with pytest.raises(
                AssertionError, match="op1 and op2 have different control dictionaries."
            ):
                assert_equal(op1, op2)

    # pylint: disable=too-many-positional-arguments
    @pytest.mark.parametrize(
        ("wires1", "controls1", "wires2", "controls2", "res"),
        [
            ([1, 2], [0, 1], [1, 2], [0, 1], True),
            ([1, 2], [0, 1], [2, 1], [0, 1], False),
            ([1, 2], [0, 1], [2, 1], [1, 0], True),
        ],
    )
    def test_differing_control_wire_order(self, wires1, controls1, wires2, controls2, res):
        """Test that equal compares control wires and their respective control values
        without regard for wire order"""
        base1 = qml.PauliX(wires=0)
        base2 = qml.PauliX(wires=0)

        op1 = qml.ops.op_math.Controlled(base1, control_wires=wires1, control_values=controls1)
        op2 = qml.ops.op_math.Controlled(base2, control_wires=wires2, control_values=controls2)

        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("wires1", "wires2", "res"), CONTROL_WIRES_SEQUENCE)
    def test_control_sequence_wires_comparison(self, wires1, wires2, res):
        """Test that equal compares control for ControlledSequence operators"""
        base1 = qml.Hadamard(0)
        base2 = qml.Hadamard(0)
        op1 = ControlledSequence(base1, control=wires1)
        op2 = ControlledSequence(base2, control=wires2)
        assert qml.equal(op1, op2) == res
        if not res:
            with pytest.raises(AssertionError, match="op1 and op2 have different wires."):
                assert_equal(op1, op2)

    @pytest.mark.parametrize(("wire1", "wire2", "res"), WIRES)
    def test_controlled_work_wires_comparison(self, wire1, wire2, res):
        """Test that equal compares work_wires for Controlled operators"""
        base1 = qml.MultiRZ(1.23, [0, 1])
        base2 = qml.MultiRZ(1.23, [0, 1])
        op1 = Controlled(base1, control_wires=2, work_wires=wire1)
        op2 = Controlled(base2, control_wires=2, work_wires=wire2)
        if res:
            assert qml.equal(op1, op2) == res
            assert_equal(op1, op2)
        else:
            assert qml.equal(op1, op2) is False
            with pytest.raises(AssertionError, match="op1 and op2 have different work wires."):
                assert_equal(op1, op2)

    def test_controlled_arithmetic_depth(self):
        """The depths of controlled operators are different due to nesting"""
        base = qml.MultiRZ(1.23, [0, 1])
        op1 = Controlled(base, control_wires=5)
        op2 = Controlled(op1, control_wires=6)

        with pytest.raises(AssertionError, match="op1 and op2 have different arithmetic depths."):
            assert_equal(op1, op2)

    @pytest.mark.parametrize("base", PARAMETRIZED_OPERATIONS)
    def test_adjoint_comparison(self, base):
        """Test that equal compares two objects of the Adjoint class"""
        op1 = qml.adjoint(base)
        op2 = qml.adjoint(base)
        op3 = qml.adjoint(qml.PauliX(15))

        assert qml.equal(op1, op2) is True
        assert qml.equal(op1, op3) is False
        with pytest.raises(AssertionError, match=BASE_OPERATION_MISMATCH_ERROR_MESSAGE):
            assert_equal(op1, op3)

    def test_adjoint_comparison_with_tolerance(self):
        """Test that equal compares the parameters within a provided tolerance of the Adjoint class."""
        op1 = qml.adjoint(qml.RX(1.2, wires=0))
        op2 = qml.adjoint(qml.RX(1.2 + 1e-4, wires=0))

        assert qml.equal(op1, op2, atol=1e-3, rtol=0) is True
        assert qml.equal(op1, op2, atol=1e-5, rtol=0) is False
        assert qml.equal(op1, op2, atol=0, rtol=1e-3) is True
        assert qml.equal(op1, op2, atol=0, rtol=1e-5) is False

    def test_adjoint_base_op_comparison_with_interface(self):
        """Test that equal compares the parameters within a provided interface of the base operator of Adjoint class."""
        op1 = qml.adjoint(qml.RX(1.2, wires=0))
        op2 = qml.adjoint(qml.RX(npp.array(1.2), wires=0))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=True, check_trainability=False) is False

    def test_adjoint_base_op_comparison_with_trainability(self):
        """Test that equal compares the parameters within a provided trainability of the base operator of Adjoint class."""
        op1 = qml.adjoint(qml.RX(npp.array(1.2, requires_grad=False), wires=0))
        op2 = qml.adjoint(qml.RX(npp.array(1.2, requires_grad=True), wires=0))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=False, check_trainability=True) is False

    @pytest.mark.parametrize(("wire1", "wire2", "res"), WIRES)
    def test_conditional_base_operator_wire_comparison(self, wire1, wire2, res):
        """Test that equal compares operator wires for Conditional operators"""
        m = qml.measure(0)
        base1 = qml.PauliX(wire1)
        base2 = qml.PauliX(wire2)
        op1 = Conditional(m, base1)
        op2 = Conditional(m, base2)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("wire1", "wire2", "res"), WIRES)
    def test_conditional_measurement_value_wire_comparison(self, wire1, wire2, res):
        """Test that equal compares operator wires for Conditional operators"""
        m1 = qml.measure(wire1)
        m2 = qml.measure(wire2)
        if wire1 == wire2:
            # qml.equal checks id for MidMeasureMP, but here we only care about them acting on the same wire
            m2.measurements[0]._id = m1.measurements[0].id  # pylint: disable=protected-access
        base = qml.PauliX(wire2)
        op1 = Conditional(m1, base)
        op2 = Conditional(m2, base)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("base1", "base2", "res"), BASES)
    def test_conditional_base_operator_comparison(self, base1, base2, res):
        """Test that equal compares base operators for Conditional operators"""
        m = qml.measure(0)
        op1 = Conditional(m, base1)
        op2 = Conditional(m, base2)
        assert qml.equal(op1, op2) == res

    def test_conditional_comparison_with_tolerance(self):
        """Test that equal compares the parameters within a provided tolerance of the Conditional class."""
        m = qml.measure(0)
        base1 = qml.RX(1.2, wires=0)
        base2 = qml.RX(1.2 + 1e-4, wires=0)
        op1 = Conditional(m, base1)
        op2 = Conditional(m, base2)

        assert qml.equal(op1, op2, atol=1e-3, rtol=0) is True
        assert qml.equal(op1, op2, atol=1e-5, rtol=0) is False
        assert qml.equal(op1, op2, atol=0, rtol=1e-3) is True
        assert qml.equal(op1, op2, atol=0, rtol=1e-5) is False

    def test_conditional_base_op_comparison_with_interface(self):
        """Test that equal compares the parameters within a provided interface of the base operator of Conditional class."""
        m = qml.measure(0)
        base1 = qml.RX(1.2, wires=0)
        base2 = qml.RX(npp.array(1.2), wires=0)
        op1 = Conditional(m, base1)
        op2 = Conditional(m, base2)

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=True, check_trainability=False) is False

    def test_conditional_base_op_comparison_with_trainability(self):
        """Test that equal compares the parameters within a provided trainability of the base operator of Conditional class."""

        m = qml.measure(0)
        base1 = qml.RX(npp.array(1.2, requires_grad=False), wires=0)
        base2 = qml.RX(npp.array(1.2, requires_grad=True), wires=0)
        op1 = Conditional(m, base1)
        op2 = Conditional(m, base2)

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=False, check_trainability=True) is False

    @pytest.mark.parametrize("bases_bases_match", BASES)
    @pytest.mark.parametrize("params_params_match", PARAMS)
    def test_pow_comparison(self, bases_bases_match, params_params_match):
        """Test that equal compares two objects of the Pow class"""
        base1, base2, bases_match = bases_bases_match
        param1, param2, params_match = params_params_match
        op1 = qml.pow(base1, param1)
        op2 = qml.pow(base2, param2)
        assert qml.equal(op1, op2) == (bases_match and params_match)

    def test_diff_pow_comparison(self):
        """Test different exponents"""
        base = qml.PauliX(0)
        op1 = qml.pow(base, 0.2)
        op2 = qml.pow(base, 0.3)
        with pytest.raises(AssertionError, match="Exponent are different."):
            assert_equal(op1, op2)

    def test_pow_comparison_with_tolerance(self):
        """Test that equal compares the parameters within a provided tolerance of the Pow class."""
        op1 = qml.pow(qml.RX(1.2, wires=0), 2)
        op2 = qml.pow(qml.RX(1.2 + 1e-4, wires=0), 2)

        assert qml.equal(op1, op2, atol=1e-3, rtol=0) is True
        assert qml.equal(op1, op2, atol=1e-5, rtol=0) is False
        assert qml.equal(op1, op2, atol=0, rtol=1e-3) is True
        assert qml.equal(op1, op2, atol=0, rtol=1e-5) is False

    def test_pow_comparison_with_interface(self):
        """Test that equal compares the parameters within a provided interface of the Pow class."""
        op1 = qml.pow(qml.RX(1.2, wires=0), 2)
        op2 = qml.pow(qml.RX(1.2, wires=0), npp.array(2))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=True, check_trainability=False) is False
        with pytest.raises(AssertionError, match="Exponent have different interfaces.\n"):
            assert_equal(op1, op2, check_interface=True, check_trainability=False)

    def test_pow_comparison_with_trainability(self):
        """Test that equal compares the parameters within a provided trainability of the Pow class."""
        op1 = qml.pow(qml.RX(1.2, wires=0), npp.array(2, requires_grad=False))
        op2 = qml.pow(qml.RX(1.2, wires=0), npp.array(2, requires_grad=True))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=False, check_trainability=True) is False
        with pytest.raises(AssertionError, match="Exponent have different trainability.\n"):
            assert_equal(op1, op2, check_interface=True, check_trainability=True)

    def test_pow_base_op_comparison_with_interface(self):
        """Test that equal compares the parameters within a provided interface of the base operator of Pow class."""
        op1 = qml.pow(qml.RX(1.2, wires=0), 2)
        op2 = qml.pow(qml.RX(npp.array(1.2), wires=0), 2)

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=True, check_trainability=False) is False

    def test_pow_base_op_comparison_with_trainability(self):
        """Test that equal compares the parameters within a provided trainability of the base operator of Pow class."""
        op1 = qml.pow(qml.RX(npp.array(1.2, requires_grad=False), wires=0), 2)
        op2 = qml.pow(qml.RX(npp.array(1.2, requires_grad=True), wires=0), 2)

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=False, check_trainability=True) is False

    @pytest.mark.parametrize("bases_bases_match", BASES)
    @pytest.mark.parametrize("params_params_match", PARAMS)
    def test_exp_comparison(self, bases_bases_match, params_params_match):
        """Test that equal compares two objects of the Exp class"""
        base1, base2, bases_match = bases_bases_match
        param1, param2, params_match = params_params_match
        op1 = qml.exp(base1, param1)
        op2 = qml.exp(base2, param2)

        assert qml.equal(op1, op2) == (bases_match and params_match)

    def test_exp_with_different_coeffs(self):
        """Test that assert_equal fails when coeffs are different"""
        op1 = qml.exp(qml.X(0), 0.5j)
        op2 = qml.exp(qml.X(0), 1.0j)

        with pytest.raises(AssertionError, match="op1 and op2 have different coefficients."):
            assert_equal(op1, op2)

    def test_exp_with_different_base_operator(self):
        """Test that assert_equal fails when base operators are different"""
        op1 = qml.exp(qml.X(0), 0.5j)
        op2 = qml.exp(qml.Y(0), 0.5j)

        with pytest.raises(AssertionError, match=BASE_OPERATION_MISMATCH_ERROR_MESSAGE):
            assert_equal(op1, op2)

    def test_exp_comparison_with_tolerance(self):
        """Test that equal compares the parameters within a provided tolerance of the Exp class."""
        op1 = qml.exp(qml.PauliX(0), 0.12)
        op2 = qml.exp(qml.PauliX(0), 0.12 + 1e-4)

        assert qml.equal(op1, op2, atol=1e-3, rtol=0) is True
        assert qml.equal(op1, op2, atol=1e-5, rtol=0) is False
        assert qml.equal(op1, op2, atol=0, rtol=1e-2) is True
        assert qml.equal(op1, op2, atol=0, rtol=1e-5) is False

    def test_exp_comparison_with_interface(self):
        """Test that equal compares the parameters within a provided interface of the Exp class."""
        op1 = qml.exp(qml.PauliX(0), 1.2)
        op2 = qml.exp(qml.PauliX(0), npp.array(1.2))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=True, check_trainability=False) is False

        assert_equal(op1, op2, check_interface=False, check_trainability=False)
        with pytest.raises(AssertionError, match="Parameters have different interface"):
            assert_equal(op1, op2, check_interface=True, check_trainability=False)

    def test_exp_comparison_with_trainability(self):
        """Test that equal compares the parameters within a provided trainability of the Exp class."""
        op1 = qml.exp(qml.PauliX(0), npp.array(1.2, requires_grad=False))
        op2 = qml.exp(qml.PauliX(0), npp.array(1.2, requires_grad=True))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=False, check_trainability=True) is False

        assert_equal(op1, op2, check_interface=False, check_trainability=False)
        with pytest.raises(AssertionError, match="Parameters have different trainability"):
            assert_equal(op1, op2, check_interface=False, check_trainability=True)

    def test_exp_base_op_comparison_with_interface(self):
        """Test that equal compares the parameters within a provided interface of the base operator of Exp class."""
        op1 = qml.exp(qml.RX(0.5, wires=0), 1.2)
        op2 = qml.exp(qml.RX(npp.array(0.5), wires=0), 1.2)

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=True, check_trainability=False) is False

        assert_equal(op1, op2, check_interface=False, check_trainability=False)
        with pytest.raises(AssertionError, match="Parameters have different interface"):
            assert_equal(op1, op2, check_interface=True, check_trainability=False)

    def test_exp_base_op_comparison_with_trainability(self):
        """Test that equal compares the parameters within a provided trainability of the base operator of Exp class."""
        op1 = qml.exp(qml.RX(npp.array(0.5, requires_grad=False), wires=0), 1.2)
        op2 = qml.exp(qml.RX(npp.array(0.5, requires_grad=True), wires=0), 1.2)

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=False, check_trainability=True) is False

    additional_cases = [
        (qml.sum(qml.PauliX(0), qml.PauliY(0)), qml.sum(qml.PauliY(0), qml.PauliX(0)), True),
        (qml.sum(qml.PauliX(0), qml.PauliY(1)), qml.sum(qml.PauliX(1), qml.PauliY(0)), False),
        (qml.prod(qml.PauliX(0), qml.PauliY(1)), qml.prod(qml.PauliY(1), qml.PauliX(0)), True),
        (qml.prod(qml.PauliX(0), qml.PauliY(1)), qml.prod(qml.PauliX(1), qml.PauliY(0)), False),
    ]

    @pytest.mark.parametrize("bases_bases_match", BASES + additional_cases)
    @pytest.mark.parametrize("params_params_match", PARAMS)
    def test_s_prod_comparison(self, bases_bases_match, params_params_match):
        """Test that equal compares two objects of the SProd class"""
        base1, base2, bases_match = bases_bases_match
        param1, param2, params_match = params_params_match
        op1 = qml.s_prod(param1, base1)
        op2 = qml.s_prod(param2, base2)
        assert qml.equal(op1, op2) == (bases_match and params_match)

    def test_s_prod_comparison_different_scalar(self):
        """Test that equal compares two objects of the SProd class with different scalars"""
        base = qml.PauliX(0) @ qml.PauliY(1)
        op1 = qml.s_prod(0.2, base)
        op2 = qml.s_prod(0.3, base)

        with pytest.raises(
            AssertionError, match="op1 and op2 have different scalars. Got 0.2 and 0.3"
        ):
            assert_equal(op1, op2)

    def test_s_prod_comparison_different_operands(self):
        """Test that equal compares two objects of the SProd class with different operands"""
        base1 = qml.PauliX(0) @ qml.PauliY(1)
        base2 = qml.PauliX(0) @ qml.PauliY(2)
        op1 = qml.s_prod(0.2, base1)
        op2 = qml.s_prod(0.2, base2)

        with pytest.raises(AssertionError, match=OPERANDS_MISMATCH_ERROR_MESSAGE):
            assert_equal(op1, op2)

    def test_s_prod_comparison_with_tolerance(self):
        """Test that equal compares the parameters within a provided tolerance of the SProd class."""
        op1 = qml.s_prod(0.12, qml.PauliX(0))
        op2 = qml.s_prod(0.12 + 1e-4, qml.PauliX(0))

        assert qml.equal(op1, op2, atol=1e-3, rtol=0) is True
        assert qml.equal(op1, op2, atol=1e-5, rtol=0) is False
        assert qml.equal(op1, op2, atol=0, rtol=1e-3) is True
        assert qml.equal(op1, op2, atol=0, rtol=1e-5) is False

    def test_s_prod_comparison_with_interface(self):
        """Test that equal compares the parameters within a provided interface of the SProd class."""
        op1 = qml.s_prod(0.12, qml.PauliX(0))
        op2 = qml.s_prod(npp.array(0.12), qml.PauliX(0))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=True, check_trainability=False) is False
        with pytest.raises(AssertionError, match="Parameters have different interfaces."):
            assert_equal(op1, op2)

    def test_s_prod_comparison_with_trainability(self):
        """Test that equal compares the parameters within a provided trainability of the SProd class."""
        op1 = qml.s_prod(npp.array(0.12, requires_grad=False), qml.PauliX(0))
        op2 = qml.s_prod(npp.array(0.12, requires_grad=True), qml.PauliX(0))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=False, check_trainability=True) is False
        with pytest.raises(AssertionError, match="Parameters have different trainability."):
            assert_equal(op1, op2)

    def test_s_prod_base_op_comparison_with_interface(self):
        """Test that equal compares the parameters within a provided interface of the base operator of SProd class."""
        op1 = qml.s_prod(0.12, qml.RX(0.5, wires=0))
        op2 = qml.s_prod(0.12, qml.RX(npp.array(0.5), wires=0))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=True, check_trainability=False) is False

    def test_s_prod_base_op_comparison_with_trainability(self):
        """Test that equal compares the parameters within a provided trainability of the base operator of SProd class."""
        op1 = qml.s_prod(0.12, qml.RX(npp.array(0.5, requires_grad=False), wires=0))
        op2 = qml.s_prod(0.12, qml.RX(npp.array(0.5, requires_grad=True), wires=0))

        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True
        assert qml.equal(op1, op2, check_interface=False, check_trainability=True) is False


class TestProdComparisons:
    """Tests comparisons between Prod operators"""

    SINGLE_WIRE_BASES = [
        ([qml.PauliX(0), qml.PauliY(1)], [qml.PauliX(0), qml.PauliY(1)], True),
        ([qml.PauliX(0), qml.PauliY(1)], [qml.PauliY(1), qml.PauliX(0)], True),
        (
            [qml.RX(1.23, 0), qml.adjoint(qml.RY(1.23, 1))],
            [qml.RX(1.23, 0), qml.adjoint(qml.RY(1.23, 1))],
            True,
        ),
        ([qml.PauliX(1), qml.PauliY(0)], [qml.PauliY(1), qml.PauliX(0)], False),
        (
            [qml.PauliX(0), qml.PauliY(1), qml.PauliX(0), qml.PauliY(1)],
            [qml.PauliX(0), qml.PauliX(0), qml.PauliY(1), qml.PauliY(1)],
            True,
        ),
        (
            [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(0), qml.PauliY(1)],
            [qml.PauliZ(0), qml.PauliX(0), qml.PauliY(1), qml.PauliY(1)],
            False,
        ),
        ([qml.PauliZ(0), qml.PauliZ(1)], [qml.PauliZ(0), qml.PauliZ(1), qml.PauliY(2)], False),
        ([qml.RX(1.23, 0), qml.RX(1.23, 1)], [qml.RX(2.34, 0), qml.RX(1.23, 1)], False),
    ]

    MULTI_WIRE_BASES = [
        ([qml.CRX(1.23, [0, 1]), qml.PauliX(0)], [qml.CRX(1.23, [0, 1]), qml.PauliX(0)], True),
        ([qml.CRX(1.23, [0, 1]), qml.PauliX(0)], [qml.PauliX(0), qml.CRX(1.23, [0, 1])], False),
        ([qml.CRX(1.23, [0, 1]), qml.PauliX(2)], [qml.PauliX(2), qml.CRX(1.23, [0, 1])], True),
        (
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [1, 0])],
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [1, 0])],
            True,
        ),
        (
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [1, 0])],
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [0, 1])],
            False,
        ),
        (
            [qml.CRX(1.34, [1, 0]), qml.CRY(2.34, [1, 0])],
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [1, 0])],
            False,
        ),
    ]

    @pytest.mark.parametrize(
        ("T1", "T2", "res"),
        [
            (qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(1) @ qml.PauliX(0), True),
            (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliZ(2), qml.PauliX(0) @ qml.PauliZ(2), True),
            (qml.PauliX(0) @ qml.Identity(2) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliZ(2), False),
            (qml.PauliX(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliZ(2), False),
            (qml.PauliX("a") @ qml.PauliZ("b"), qml.PauliX("a") @ qml.PauliZ("b"), True),
            (qml.PauliX("a") @ qml.PauliZ("b"), qml.PauliX("c") @ qml.PauliZ("d"), False),
            (qml.PauliX("a") @ qml.PauliZ("b"), qml.PauliX("b") @ qml.PauliZ("a"), False),
            (qml.PauliX(1.1) @ qml.PauliZ(1.2), qml.PauliX(1.1) @ qml.PauliZ(1.2), True),
            (qml.PauliX(1.1) @ qml.PauliZ(1.2), qml.PauliX(1.2) @ qml.PauliZ(0.9), False),
        ],
    )
    def test_prods_equal(self, T1, T2, res):
        """Tests that equality can be checked between Prods"""
        assert qml.equal(T1, T2) == qml.equal(T2, T1)
        assert qml.equal(T1, T2) == res

    def test_non_commuting_order_swap_not_equal(self):
        """Test that changing the order of non-commuting operators is not equal"""
        op1 = qml.prod(qml.PauliX(0), qml.PauliY(0))
        op2 = qml.prod(qml.PauliY(0), qml.PauliX(0))
        assert qml.equal(op1, op2) is False

    def test_commuting_order_swap_equal(self):
        """Test that changing the order of commuting operators is equal"""
        op1 = qml.prod(qml.PauliX(0), qml.PauliY(1))
        op2 = qml.prod(qml.PauliY(1), qml.PauliX(0))
        assert qml.equal(op1, op2) is True

    @pytest.mark.all_interfaces
    def test_prod_kwargs_used_for_base_operator_comparison(self):
        """Test that setting kwargs check_interface and check_trainability are applied when comparing the bases"""
        import jax
        import torch

        base_list1 = [qml.RX(torch.tensor(1.2), wires=0), qml.RX(torch.tensor(2.3), wires=1)]
        base_list2 = [qml.RX(jax.numpy.array(1.2), wires=0), qml.RX(jax.numpy.array(2.3), wires=1)]

        op1 = qml.prod(*base_list1)
        op2 = qml.prod(*base_list2)

        assert qml.equal(op1, op2) is False
        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True

    @pytest.mark.parametrize(("base_list1", "base_list2", "res"), SINGLE_WIRE_BASES)
    def test_prod_comparisons_single_wire_bases(self, base_list1, base_list2, res):
        """Test comparison of products of operators where all operators have a single wire"""
        op1 = qml.prod(*base_list1)
        op2 = qml.prod(*base_list2)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("base_list1", "base_list2", "res"), MULTI_WIRE_BASES)
    def test_prod_with_multi_wire_bases(self, base_list1, base_list2, res):
        """Test comparison of products of operators where some operators work on multiple wires"""
        op1 = qml.prod(*base_list1)
        op2 = qml.prod(*base_list2)
        assert qml.equal(op1, op2) == res

    def test_prod_of_prods(self):
        """Test that prod of prods and just an equivalent Prod get compared correctly"""
        X = qml.PauliX

        op1 = (0.5 * X(0)) @ (0.5 * X(1)) @ (0.5 * X(2)) @ (0.5 * X(3)) @ (0.5 * X(4))
        op2 = qml.prod(*[0.5 * X(i) for i in range(5)])
        assert qml.equal(op1, op2) is True

    def test_prod_global_phase(self):
        """Test that a prod with a global phase can be used with qml.equal."""

        p1 = qml.GlobalPhase(np.pi) @ qml.X(0)
        p2 = qml.X(0) @ qml.GlobalPhase(np.pi)

        assert qml.equal(p1, p2) is True


class TestSumComparisons:
    """Tests comparisons between Sum operators"""

    SINGLE_WIRE_BASES = [
        ([qml.PauliX(0), qml.PauliY(1)], [qml.PauliX(0), qml.PauliY(1)], True),
        ([qml.PauliX(0), qml.PauliY(1)], [qml.PauliY(1), qml.PauliX(0)], True),
        (
            [qml.RX(1.23, 0), qml.adjoint(qml.RY(1.23, 1))],
            [qml.RX(1.23, 0), qml.adjoint(qml.RY(1.23, 1))],
            True,
        ),
        ([qml.PauliX(1), qml.PauliY(0)], [qml.PauliY(1), qml.PauliX(0)], False),
        (
            [qml.PauliX(0), qml.PauliY(1), qml.PauliX(0), qml.PauliY(1)],
            [qml.PauliX(0), qml.PauliX(0), qml.PauliY(1), qml.PauliY(1)],
            True,
        ),
        ([qml.PauliZ(0), qml.PauliZ(1)], [qml.PauliZ(0), qml.PauliZ(1), qml.PauliY(2)], False),
        ([qml.RX(1.23, 0), qml.RX(1.23, 1)], [qml.RX(2.34, 0), qml.RX(1.23, 1)], False),
    ]

    MULTI_WIRE_BASES = [
        ([qml.CRX(1.23, [0, 1]), qml.PauliX(0)], [qml.CRX(1.23, [0, 1]), qml.PauliX(0)], True),
        ([qml.CRX(1.23, [0, 1]), qml.PauliX(0)], [qml.PauliX(0), qml.CRX(1.23, [0, 1])], True),
        (
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [1, 0])],
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [1, 0])],
            True,
        ),
        (
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [1, 0])],
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [0, 1])],
            False,
        ),
        (
            [qml.CRX(1.34, [1, 0]), qml.CRY(2.34, [1, 0])],
            [qml.CRX(1.23, [1, 0]), qml.CRY(2.34, [1, 0])],
            False,
        ),
    ]

    def test_sum_different_order_still_equal(self):
        """Test that changing the order of the terms doesn't affect comparison of sums"""
        op1 = qml.sum(qml.PauliX(0), qml.PauliY(1))
        op2 = qml.sum(qml.PauliY(1), qml.PauliX(0))
        assert qml.equal(op1, op2)

    @pytest.mark.all_interfaces
    def test_sum_kwargs_used_for_base_operator_comparison(self):
        """Test that setting kwargs check_interface and check_trainability are applied when comparing the bases"""
        import jax
        import torch

        base_list1 = [qml.RX(torch.tensor(1.2), wires=0), qml.RX(torch.tensor(2.3), wires=1)]
        base_list2 = [qml.RX(jax.numpy.array(1.2), wires=0), qml.RX(jax.numpy.array(2.3), wires=1)]

        op1 = qml.sum(*base_list1)
        op2 = qml.sum(*base_list2)

        assert qml.equal(op1, op2) is False
        assert qml.equal(op1, op2, check_interface=False, check_trainability=False) is True

    @pytest.mark.parametrize(("base_list1", "base_list2", "res"), SINGLE_WIRE_BASES)
    def test_sum_comparisons_single_wire_bases(self, base_list1, base_list2, res):
        """Test comparison of sums of operators where all operators have a single wire"""
        op1 = qml.sum(*base_list1)
        op2 = qml.sum(*base_list2)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("base_list1", "base_list2", "res"), MULTI_WIRE_BASES)
    def test_sum_with_multi_wire_operations(self, base_list1, base_list2, res):
        """Test comparison of sums of operators where some operators act on multiple wires"""
        op1 = qml.sum(*base_list1)
        op2 = qml.sum(*base_list2)
        assert qml.equal(op1, op2) == res

    def test_sum_with_different_operands(self):
        """Test sum equals with different operands"""
        operands1 = [qml.PauliX(0), qml.PauliY(1)]
        operands2 = [qml.PauliY(0), qml.PauliY(1)]
        op1 = qml.sum(*operands1)
        op2 = qml.sum(*operands2)

        with pytest.raises(AssertionError, match=OPERANDS_MISMATCH_ERROR_MESSAGE):
            assert_equal(op1, op2)

    def test_sum_with_different_number_of_operands(self):
        """Test sum equals with different number of operands"""
        operands1 = [qml.PauliX(0), qml.PauliY(1)]
        operands2 = [qml.PauliY(1)]
        op1 = qml.sum(*operands1)
        op2 = qml.sum(*operands2)

        with pytest.raises(
            AssertionError, match="op1 and op2 have different number of operands. Got 2 and 1"
        ):
            assert_equal(op1, op2)

    def test_sum_equal_order_invarient(self):
        """Test that the order of operations doesn't affect equality"""
        H1 = qml.prod(qml.PauliX(0), qml.PauliX(1))
        H2 = qml.s_prod(1.0, qml.sum(qml.PauliY(0), qml.PauliY(1)))

        true_res = qml.sum(
            qml.s_prod(2j, qml.prod(qml.PauliZ(0), qml.PauliX(1))),
            qml.s_prod(2j, qml.prod(qml.PauliX(0), qml.PauliZ(1))),
        )
        true_res = true_res.simplify()

        res = qml.prod(H1, H2) - qml.prod(H2, H1)
        res = res.simplify()

        assert true_res == res

    def test_sum_of_sums(self):
        """Test that sum of sums and just an equivalent sum get compared correctly"""
        X = qml.PauliX
        op1 = (
            0.5 * X(0)
            + 0.5 * X(1)
            + 0.5 * X(2)
            + 0.5 * X(3)
            + 0.5 * X(4)
            + 0.5 * X(5)
            + 0.5 * X(6)
            + 0.5 * X(7)
            + 0.5 * X(8)
            + 0.5 * X(9)
        )
        op2 = qml.sum(*[0.5 * X(i) for i in range(10)])
        assert qml.equal(op1, op2)

    def test_sum_global_phase(self):
        """Test that a sum containing a no-wires op can still be compared."""
        op1 = qml.sum(qml.X(0), qml.GlobalPhase(np.pi))
        op2 = qml.sum(qml.GlobalPhase(np.pi), qml.X(0))
        assert qml.equal(op1, op2) is True

    @pytest.mark.parametrize(("H1", "H2", "res"), equal_hamiltonians)
    def test_hamiltonian_equal(self, H1, H2, res):
        """Tests that equality can be checked between LinearCombinations"""

        assert qml.equal(H1, H2) == qml.equal(H2, H1)
        assert qml.equal(H1, H2) == res
        if not res:
            if len(H1) != len(H2):
                error_message = "op1 and op2 have different number of operands"
            else:
                error_message = re.compile(r"op1 and op2 have different operands")
            with pytest.raises(AssertionError, match=error_message):
                assert_equal(H1, H2)


def f1(p, t):
    return np.polyval(p, t)


def f2(p, t):
    return p[0] * t**2 + p[1]


@pytest.mark.jax
class TestParametrizedEvolutionComparisons:
    """Tests comparisons between ParametrizedEvolution operators"""

    def test_params_comparison(self):
        """Test that params are compared for two ParametrizedEvolution ops"""
        coeffs = [3, f1, f2]
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]

        h1 = qml.dot(coeffs, ops)

        ev1 = qml.evolve(h1)
        ev2 = qml.evolve(h1)

        params1 = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0]]
        params2 = [[1.0, 2.0, 3.0, 4.0, 5.0], [8.0, 9.0]]

        t = 3

        assert qml.equal(ev1(params1, t), ev2(params1, t)) is True
        assert qml.equal(ev1(params1, t), ev2(params2, t)) is False

    def test_wires_comparison(self):
        """Test that wires are compared for two ParametrizedEvolution ops"""
        coeffs = [3, f1, f2]
        ops1 = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        ops2 = [qml.PauliX(1), qml.PauliY(2), qml.PauliZ(3)]

        h1 = qml.dot(coeffs, ops1)
        h2 = qml.dot(coeffs, ops2)

        ev1 = qml.evolve(h1)
        ev2 = qml.evolve(h1)
        ev3 = qml.evolve(h2)

        params = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0]]
        t = 3

        assert qml.equal(ev1(params, t), ev2(params, t)) is True
        assert qml.equal(ev1(params, t), ev3(params, t)) is False

    def test_times_comparison(self):
        """Test that times are compared for two ParametrizedEvolution ops"""
        coeffs = [3, f1, f2]
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]

        h1 = qml.dot(coeffs, ops)

        ev1 = qml.evolve(h1)

        params = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0]]

        t1 = 3
        t2 = 4

        assert qml.equal(ev1(params, t1), ev1(params, t1)) is True
        assert qml.equal(ev1(params, t1), ev1(params, t2)) is False

    def test_operator_comparison(self):
        """Test that operators are compared for two ParametrizedEvolution ops"""
        coeffs = [3, f1, f2]
        ops1 = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        ops2 = [qml.PauliZ(0), qml.PauliX(1), qml.PauliY(2)]

        h1 = qml.dot(coeffs, ops1)
        h2 = qml.dot(coeffs, ops2)

        ev1 = qml.evolve(h1)
        ev2 = qml.evolve(h1)
        ev3 = qml.evolve(h2)

        params = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0]]
        t = 3

        assert qml.equal(ev1(params, t), ev2(params, t)) is True
        assert qml.equal(ev1(params, t), ev3(params, t)) is False

    def test_coefficients_comparison(self):
        """Test that coefficients are compared for two ParametrizedEvolution ops"""
        coeffs1 = [3, f1, f2]
        coeffs2 = [3, 4, f2]
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]

        h1 = qml.dot(coeffs1, ops)
        h2 = qml.dot(coeffs2, ops)

        ev1 = qml.evolve(h1)
        ev2 = qml.evolve(h1)
        ev3 = qml.evolve(h2)

        params1 = [6.0, 7.0]
        params2 = [6.0, 7.0]
        params3 = [[6.0, 7.0]]
        t = 3

        assert qml.equal(ev1(params1, t), ev2(params2, t)) is True
        assert qml.equal(ev1(params1, t), ev3(params3, t)) is False

    def test_different_times(self):
        """Test that times are compared for two ParametrizedEvolution ops"""
        coeffs1 = [3, f1, f2]
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]

        h1 = qml.dot(coeffs1, ops)

        params1 = [6.0, 7.0]
        t = 3

        ev1 = qml.pulse.ParametrizedEvolution(h1, params1)
        ev11 = qml.pulse.ParametrizedEvolution(h1, params1)
        ev2 = qml.pulse.ParametrizedEvolution(h1, params1, t)
        ev3 = qml.pulse.ParametrizedEvolution(h1, params1, 0.5)

        assert qml.equal(ev1, ev11) is True
        assert qml.equal(ev1, ev2) is False
        assert qml.equal(ev1, ev3) is False
        assert qml.equal(ev2, ev3) is False


class TestQuantumScriptComparisons:
    tape1 = qml.tape.QuantumScript(
        [qml.PauliX(0), qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0))], shots=10
    )
    tape2 = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))], shots=10)
    tape3 = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))], shots=None)
    tape4 = qml.tape.QuantumScript(
        [qml.PauliX(0), qml.RX(1.2 + 1e-6, wires=0)], [qml.expval(qml.PauliZ(0))], shots=10
    )
    tape5 = qml.tape.QuantumScript(
        [qml.PauliX(0)],
        [qml.expval(qml.PauliZ(0))],
        shots=10,
        trainable_params=[2],
    )
    tape6 = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliX(0))], shots=10)
    tape7 = qml.tape.QuantumScript(
        [qml.PauliX(0)],
        [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))],
        shots=10,
    )

    @pytest.mark.parametrize("tape, other_tape", [(tape2, tape7), (tape2, tape6)])
    def test_non_equal_measurement_comparison(self, tape, other_tape):
        assert qml.equal(tape, other_tape) is False

    @pytest.mark.parametrize("tape, other_tape", [(tape2, tape3)])
    def test_non_equal_shot_comparison(self, tape, other_tape):
        assert qml.equal(tape, other_tape) is False

    @pytest.mark.parametrize("tape, other_tape", [(tape2, tape2)])
    def test_equal_comparison(self, tape, other_tape):
        assert qml.equal(tape, other_tape) is True

    @pytest.mark.parametrize("tape, other_tape", [(tape1, tape2)])
    def test_non_equal_operators_comparison(self, tape, other_tape):
        assert qml.equal(tape, other_tape) is False

    @pytest.mark.parametrize("tape, other_tape", [(tape1, tape4)])
    def test_different_tolerances_comparison(self, tape, other_tape):
        assert qml.equal(tape, other_tape, atol=1e-5)
        assert qml.equal(tape, other_tape, rtol=0, atol=1e-7) is False

    @pytest.mark.parametrize("tape, other_tape", [(tape2, tape5)])
    def test_non_equal_training_params_comparison(self, tape, other_tape):
        assert qml.equal(tape, other_tape) is False


class TestBasisRotation:
    """Test that qml.equal works with qml.BasisRotation."""

    rotation_mat = np.array(
        [
            [-0.618452, -0.68369054 - 0.38740723j],
            [-0.78582258, 0.53807284 + 0.30489424j],
        ]
    )
    op1 = qml.BasisRotation(wires=range(2), unitary_matrix=rotation_mat)
    op2 = qml.BasisRotation(wires=range(2), unitary_matrix=np.array(rotation_mat))
    op3 = qml.BasisRotation(wires=range(2), unitary_matrix=rotation_mat + 1e-7)
    op4 = qml.BasisRotation(wires=range(2, 4), unitary_matrix=rotation_mat)

    @pytest.mark.parametrize("op, other_op", [(op1, op3)])
    def test_different_tolerances_comparison(self, op, other_op):
        assert qml.equal(op, other_op, atol=1e-5) is True
        assert_equal(op, other_op, atol=1e-5)
        assert qml.equal(op, other_op, rtol=0, atol=1e-9) is False

        with pytest.raises(AssertionError, match="op1 and op2 have different data"):
            assert_equal(op, other_op, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("op, other_op", [(op1, op2)])
    def test_non_equal_training_params_comparison(self, op, other_op):
        assert qml.equal(op, other_op) is True
        assert_equal(op, other_op)

    @pytest.mark.parametrize("op, other_op", [(op1, op4)])
    def test_non_equal_training_wires(self, op, other_op):
        assert qml.equal(op, other_op) is False

        with pytest.raises(AssertionError, match="op1 and op2 have different wires."):
            assert_equal(op, other_op)

    @pytest.mark.jax
    @pytest.mark.parametrize("op", [op1])
    def test_non_equal_interfaces(self, op):
        import jax

        rotation_mat_jax = jax.numpy.array(
            [
                [-0.618452, -0.68369054 - 0.38740723j],
                [-0.78582258, 0.53807284 + 0.30489424j],
            ]
        )
        other_op = qml.BasisRotation(wires=range(2), unitary_matrix=rotation_mat_jax)
        assert qml.equal(op, other_op, check_interface=False) is True
        assert_equal(op, other_op, check_interface=False)
        assert qml.equal(op, other_op) is False

        with pytest.raises(AssertionError, match=r"have different interfaces"):
            assert_equal(op, other_op)


class TestHilbertSchmidt:
    """Test that qml.equal works with qml.HilbertSchmidt."""

    # pylint: disable=no-self-argument

    def v_function1(params):
        """Returns a v_function that is used in the HilbertSchmidt operator."""
        return qml.RZ(params[0], wires=1)

    def v_function2(params):
        """Differs from v_function1 by operation type and used parameter."""
        return qml.RX(params[1], wires=1)

    def v_function3(params):
        """Differs from v_function1 by the used wire."""
        return qml.RZ(params[0], wires=2)

    def v_function4(params):
        """Differs from v_function1 by the functional parameter dependence, but
        produces the same tape at params[0]=0.2."""
        return qml.RZ(params[0] * 2 - 0.2, wires=1)

    u_tape1 = qml.tape.QuantumScript([qml.RX(0.2, 0)])
    u_tape1_eps = qml.tape.QuantumScript([qml.RX(0.2 + 1e-7, 0)])
    u_tape1_trainable = qml.tape.QuantumScript([qml.RX(npp.array(0.2, requires_grad=True), 0)])
    u_tape1_untrainable = qml.tape.QuantumScript([qml.RX(npp.array(0.2, requires_grad=False), 0)])
    u_tape2 = qml.tape.QuantumScript([qml.Hadamard(2)])

    v_params1 = [0.2, 0.3]
    v_params2 = [0.1, 0.5]
    v_params1_eps = [0.2 + 1e-7, 0.3]
    v_params1_trainable = npp.array(v_params1, requires_grad=True)
    v_params1_untrainable = npp.array(v_params1, requires_grad=False)

    op1 = qml.HilbertSchmidt(V=v_function1(v_params1), U=u_tape1.operations)

    op1_trainable = qml.HilbertSchmidt(V=v_function1(v_params1), U=u_tape1_trainable.operations)

    op1_untrainable = qml.HilbertSchmidt(V=v_function1(v_params1), U=u_tape1_untrainable.operations)

    op1_eps = qml.HilbertSchmidt(V=v_function1(v_params1_eps), U=u_tape1.operations)

    op1_eps_tape = qml.HilbertSchmidt(V=v_function1(v_params1), U=u_tape1_eps.operations)

    op2 = qml.HilbertSchmidt(V=v_function1(v_params2), U=u_tape1.operations)

    op3_tapediff = qml.HilbertSchmidt(V=v_function2(v_params1), U=u_tape1.operations)

    op3_fundiff = qml.HilbertSchmidt(V=v_function4(v_params1), U=u_tape1.operations)

    op4 = qml.HilbertSchmidt(V=v_function1(v_params1), U=u_tape2.operations)

    op5 = qml.HilbertSchmidt(V=v_function1(v_params1_trainable), U=u_tape1.operations)

    op6 = qml.HilbertSchmidt(V=v_function1(v_params1_untrainable), U=u_tape1.operations)

    op7 = qml.HilbertSchmidt(V=v_function3(v_params1), U=u_tape1.operations)

    @pytest.mark.parametrize("op, other_op", [(op1, op1), (op2, op2), (op4, op4)])
    def test_equality(self, op, other_op):
        """Test that two identical HilbertSchmidt operators are found equal."""
        assert qml.equal(op, other_op) is True

    # The second test case (incl op1_eps_tape) ensures that the kwargs of equal are
    # passed to the tape comparisons correctly
    @pytest.mark.parametrize("op, other_op", [(op1, op1_eps), (op1, op1_eps_tape)])
    def test_different_tolerances_comparison(self, op, other_op):
        """Test that the tolerance parameters are used correctly."""
        assert qml.equal(op, other_op) is True
        assert qml.equal(op, other_op, rtol=0) is False
        assert qml.equal(op, other_op, rtol=0, atol=1e-5) is True

    @pytest.mark.parametrize("op, other_op", [(op1, op2)])
    def test_non_equal_data(self, op, other_op):
        """Test that differing data is found."""
        assert qml.equal(op, other_op) is False
        other_op.data = op.data

        v_ops = op.hyperparameters["V"]
        op_params = qml.tape.QuantumScript(v_ops).get_parameters()

        new_ops = qml.tape.QuantumScript(other_op.hyperparameters["V"]).bind_new_parameters(
            op_params, [0]
        )

        new_other_op = deepcopy(other_op)
        new_other_op.hyperparameters["V"] = new_ops
        assert qml.equal(op, new_other_op)

    @pytest.mark.parametrize("op, other_op", [(op1, op4)])
    def test_non_equal_u_ops(self, op, other_op):
        """Test that differing u operations are found."""
        assert qml.equal(op, other_op) is False

        new_other_op = deepcopy(other_op)
        new_other_op.hyperparameters["U"] = op.hyperparameters["U"]
        assert qml.equal(op, new_other_op) is True

    @pytest.mark.parametrize("op, other_op", [(op1, op7)])
    def test_non_equal_v_wires(self, op, other_op):
        """Test that differing v_wires are found."""
        assert qml.equal(op, other_op) is False
        new_other_op = deepcopy(other_op)
        new_other_op.hyperparameters["V"] = op.hyperparameters["V"]
        assert qml.equal(op, new_other_op) is True

    @pytest.mark.parametrize("op, other_op", [(op5, op6), (op1_trainable, op1_untrainable)])
    def test_trainability(self, op, other_op):
        """Test that differing trainabilities are found."""
        assert qml.equal(op, other_op) is False
        assert qml.equal(op, other_op, check_trainability=False) is True

    @pytest.mark.parametrize("op, other_op", [(op1, op6), (op1, op1_untrainable)])
    def test_interface(self, op, other_op):
        """Test that differing interfaces are found."""
        assert qml.equal(op, other_op) is False
        assert qml.equal(op, other_op, check_interface=False) is True

    @pytest.mark.parametrize("op, other_op", [(op1, op5), (op1, op1_trainable)])
    def test_interface_and_trainability(self, op, other_op):
        """Test that simultaneously differing interfaces and trainabilities are found."""
        assert qml.equal(op, other_op) is False
        assert qml.equal(op, other_op, check_interface=False) is False
        assert qml.equal(op, other_op, check_trainability=False) is False
        assert qml.equal(op, other_op, check_interface=False, check_trainability=False) is True


# pylint: disable=too-few-public-methods
class DepthIncreaseOperator(Operator):
    """Dummy class which increases depth by one"""

    # pylint: disable=super-init-not-called
    def __init__(self, op: Operator):
        self._op = op

    @property
    def arithmetic_depth(self) -> int:
        """Arithmetic depth of the operator."""
        return 1 + self._op.arithmetic_depth

    @property
    def wires(self):
        return self._op.wires


@pytest.mark.jax
def test_ops_with_abstract_parameters_not_equal():
    """Test that ops are not equal if any data is tracers."""

    import jax

    assert not jax.jit(qml.equal)(qml.RX(0.1, 0), qml.RX(0.1, 0))
    with pytest.raises(AssertionError, match="Data contains a tracer"):
        jax.jit(assert_equal)(qml.RX(0.1, 0), qml.RX(0.1, 0))


@pytest.mark.parametrize(
    "op, other_op",
    [
        (
            qml.PrepSelPrep(qml.dot([1.0, 2.0], [qml.Z(0), qml.X(0)]), control=1),
            qml.PrepSelPrep(qml.dot([1.0, 2.0], [qml.Z(0), qml.X(0)]), control=2),
        ),
        (
            qml.PrepSelPrep(qml.dot([1.0, 2.0], [qml.Z(2), qml.X(2)]), control=1),
            qml.PrepSelPrep(qml.dot([1.0, 2.0], [qml.Z(0), qml.X(0)]), control=1),
        ),
        (
            qml.PrepSelPrep(qml.dot([1.0, -2.0], [qml.Z(0), qml.X(0)]), control=1),
            qml.PrepSelPrep(qml.dot([1.0, 2.0], [qml.Z(0), qml.X(0)]), control=1),
        ),
        (
            qml.PrepSelPrep(qml.dot([1.0, 2.0], [qml.Z(0), qml.X(0)]), control=1),
            qml.PrepSelPrep(qml.dot([1.0, 2.0], [qml.Y(0), qml.X(0)]), control=1),
        ),
    ],
)
def test_not_equal_prep_sel_prep(op, other_op):
    """Test that two PrepSelPrep operators with different Hamiltonian are not equal."""
    assert qml.equal(op, other_op) is False


def test_qsvt():
    """Test that QSVT operators can be compared."""

    projectors = [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)]
    op1 = qml.QSVT(qml.X(0), projectors)
    op2 = qml.QSVT(qml.Y(0), projectors)
    op3 = qml.QSVT(qml.X(0), projectors[:1])
    op4 = qml.QSVT(qml.X(0), projectors[::-1])

    for op in [op1, op2, op3, op4]:
        qml.assert_equal(op, op)

    with pytest.raises(AssertionError, match=r"different block encodings"):
        qml.assert_equal(op1, op2)

    with pytest.raises(AssertionError, match=r"different number of projectors"):
        qml.assert_equal(op1, op3)

    with pytest.raises(AssertionError, match=r"different projectors at position 0"):
        qml.assert_equal(op1, op4)


def test_select():
    """Test that Select operators can be compared."""

    op1 = qml.Select((qml.X(0),), control=2)
    op2 = qml.Select((qml.X(0),), control=3)
    with pytest.raises(AssertionError, match=r"different control wires"):
        qml.assert_equal(op1, op2)
    assert qml.equal(op1, op2) is False

    op1 = qml.Select((qml.X(0), qml.Y(0)), control=2)
    op2 = qml.Select((qml.X(0),), control=2)
    with pytest.raises(AssertionError, match=r"different number of target operators"):
        qml.assert_equal(op1, op2)
    assert qml.equal(op1, op2) is False

    op1 = qml.Select((qml.X(0), qml.Y(0)), control=2)
    op2 = qml.Select((qml.X(0), qml.X(0)), control=2)
    with pytest.raises(AssertionError, match=r"different operations at index 1"):
        qml.assert_equal(op1, op2)
    assert qml.equal(op1, op2) is False

    op1 = qml.Select((qml.X(0),), control=2)
    op2 = qml.Select((qml.X(0),), control=2)
    qml.assert_equal(op1, op2)
    assert qml.equal(op1, op2) is True
