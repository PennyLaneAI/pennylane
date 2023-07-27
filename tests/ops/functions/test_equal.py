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
# pylint: disable=too-many-arguments
import itertools

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as npp
from pennylane.measurements import ExpectationMP
from pennylane.measurements.probs import ProbabilityMP
from pennylane.ops.op_math import SymbolicOp, Controlled

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
    qml.ControlledQubitUnitary(np.eye(2) * 1j, wires=[0], control_wires=[2]),
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
    (qml.Hamiltonian([1], [qml.PauliZ(0) @ qml.PauliX(1)]), qml.PauliZ(0) @ qml.PauliX(1), True),
    (qml.Hamiltonian([1], [qml.PauliZ(0)]), qml.PauliZ(0), True),
    (
        qml.Hamiltonian(
            [1, 1, 1],
            [
                qml.Hermitian(np.array([[1, 0], [0, -1]]), "b") @ qml.Identity(7),
                qml.PauliZ(3),
                qml.Identity(1.2),
            ],
        ),
        qml.Hamiltonian(
            [1, 1, 1],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), qml.PauliZ(3), qml.Identity(1.2)],
        ),
        True,
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliZ(3) @ qml.Identity(1.2), qml.PauliZ(3)]),
        qml.Hamiltonian([2], [qml.PauliZ(3)]),
        True,
    ),
]

equal_tensors = [
    (qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(1) @ qml.PauliX(0), True),
    (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliZ(2), qml.PauliX(0) @ qml.PauliZ(2), True),
    (qml.PauliX(0) @ qml.Identity(2) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliZ(2), False),
    (qml.PauliX(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliZ(2), False),
    (qml.PauliX("a") @ qml.PauliZ("b"), qml.PauliX("a") @ qml.PauliZ("b"), True),
    (qml.PauliX("a") @ qml.PauliZ("b"), qml.PauliX("c") @ qml.PauliZ("d"), False),
    (qml.PauliX("a") @ qml.PauliZ("b"), qml.PauliX("b") @ qml.PauliZ("a"), False),
    (qml.PauliX(1.1) @ qml.PauliZ(1.2), qml.PauliX(1.1) @ qml.PauliZ(1.2), True),
    (qml.PauliX(1.1) @ qml.PauliZ(1.2), qml.PauliX(1.2) @ qml.PauliZ(0.9), False),
]

equal_hamiltonians_and_tensors = [
    (qml.Hamiltonian([1], [qml.PauliX(0) @ qml.PauliY(1)]), qml.PauliY(1) @ qml.PauliX(0), True),
    (
        qml.Hamiltonian(
            [0.5, 0.5],
            [qml.PauliZ(0) @ qml.PauliY(1), qml.PauliY(1) @ qml.PauliZ(0) @ qml.Identity("a")],
        ),
        qml.PauliZ(0) @ qml.PauliY(1),
        True,
    ),
    (qml.Hamiltonian([1], [qml.PauliX(0) @ qml.PauliY(1)]), qml.PauliX(0) @ qml.PauliY(1), True),
    (qml.Hamiltonian([2], [qml.PauliX(0) @ qml.PauliY(1)]), qml.PauliX(0) @ qml.PauliY(1), False),
    (qml.Hamiltonian([1], [qml.PauliX(0) @ qml.PauliY(1)]), qml.PauliX(4) @ qml.PauliY(1), False),
    (
        qml.Hamiltonian([1], [qml.PauliX("a") @ qml.PauliZ("b")]),
        qml.PauliX("a") @ qml.PauliZ("b"),
        True,
    ),
    (
        qml.Hamiltonian([1], [qml.PauliX("a") @ qml.PauliZ("b")]),
        qml.PauliX("b") @ qml.PauliZ("a"),
        False,
    ),
    (
        qml.Hamiltonian([1], [qml.PauliX(1.2) @ qml.PauliZ(0.2)]),
        qml.PauliX(1.2) @ qml.PauliZ(0.2),
        True,
    ),
    (
        qml.Hamiltonian([1], [qml.PauliX(1.2) @ qml.PauliZ(0.2)]),
        qml.PauliX(1.3) @ qml.PauliZ(2),
        False,
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
    (qml.Hamiltonian([1], [qml.PauliX("a")]), qml.PauliX("a"), True),
    (qml.Hamiltonian([1], [qml.PauliX("a")]), qml.PauliX("b"), False),
    (qml.Hamiltonian([1], [qml.PauliX(1.2)]), qml.PauliX(1.2), True),
    (qml.Hamiltonian([1], [qml.PauliX(1.2)]), qml.PauliX(1.3), False),
]


class TestEqual:
    @pytest.mark.parametrize("ops", PARAMETRIZED_OPERATIONS_COMBINATIONS)
    def test_equal_simple_diff_op(self, ops):
        """Test different operators return False"""
        assert not qml.equal(ops[0], ops[1], check_trainability=False, check_interface=False)

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS)
    def test_equal_simple_same_op(self, op1):
        """Test same operators return True"""
        assert qml.equal(op1, op1, check_trainability=False, check_interface=False)

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_1W)
    def test_equal_simple_op_1p1w(self, op1):
        """Test changing parameter or wire returns False"""
        wire = 0
        param = 0.123
        assert qml.equal(
            op1(param, wires=wire),
            op1(param, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param * 2, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param, wires=wire + 1),
            check_trainability=False,
            check_interface=False,
        )

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_1W)
    def test_equal_op_1p1w(self, op1):
        """Test optional arguments are working"""
        wire = 0

        import jax
        import tensorflow as tf
        import torch

        param_torch = torch.tensor(0.123)
        param_tf = tf.Variable(0.123)
        param_jax = jax.numpy.array(0.123)
        param_qml = npp.array(0.123)
        param_np = np.array(0.123)

        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
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
        import tensorflow as tf
        import torch

        param_torch = torch.tensor(0.123)
        param_tf = tf.Variable(0.123)
        param_jax = jax.numpy.array(0.123)
        param_qml = npp.array(0.123)
        param_np = np.array(0.123)

        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=True,
            check_interface=False,
        )

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_2W)
    def test_equal_simple_op_1p2w(self, op1):
        """Test changing parameter or wire returns False"""
        wire = [0, 1]
        param = 0.123
        assert qml.equal(
            op1(param, wires=wire),
            op1(param, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param * 2, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param, wires=[w + 1 for w in wire]),
            check_trainability=False,
            check_interface=False,
        )

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_4W)
    def test_equal_op_1p4w(self, op1):
        """Test optional arguments are working"""
        wire = [0, 1, 2, 3]

        import jax
        import tensorflow as tf
        import torch

        param_torch = torch.tensor(0.123)
        param_tf = tf.Variable(0.123)
        param_jax = jax.numpy.array(0.123)
        param_qml = npp.array(0.123)
        param_np = np.array(0.123)

        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=True,
            check_interface=False,
        )

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_1P_4W)
    def test_equal_simple_op_1p4w(self, op1):
        """Test changing parameter or wire returns False"""
        wire = [0, 1, 2, 3]
        param = 0.123
        assert qml.equal(
            op1(param, wires=wire),
            op1(param, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param * 2, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param, wires=[w + 1 for w in wire]),
            check_trainability=False,
            check_interface=False,
        )

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_3P_1W)
    def test_equal_op_3p1w(self, op1):
        """Test optional arguments are working"""
        wire = 0

        import jax
        import tensorflow as tf
        import torch

        param_torch = torch.tensor([1, 2, 3])
        param_tf = tf.Variable([1, 2, 3])
        param_jax = jax.numpy.array([1, 2, 3])
        param_qml = npp.array([1, 2, 3])
        param_np = np.array([1, 2, 3])

        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1[0], p1[1], p1[2], wires=wire),
                op1(p2[0], p2[1], p2[2], wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1[0], p1[1], p1[2], wires=wire),
                op1(p2[0], p2[1], p2[2], wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(*param_qml, wires=wire),
            op1(*param_qml_1, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(*param_qml, wires=wire),
            op1(*param_qml_1, wires=wire),
            check_trainability=True,
            check_interface=False,
        )

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS_3P_1W)
    def test_equal_simple_op_3p1w(self, op1):
        """Test changing parameter or wire returns False"""
        wire = 0
        param = [0.123] * 3
        assert qml.equal(
            op1(*param, wires=wire),
            op1(*param, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(*param, wires=wire),
            op1(*param, wires=wire + 1),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(*param, wires=wire),
            op1(param[0] * 2, param[1], param[2], wires=wire),
            check_trainability=False,
            check_interface=False,
        )

    @pytest.mark.all_interfaces
    def test_equal_op_remaining(self):  # pylint: disable=too-many-statements
        """Test optional arguments are working"""
        # pylint: disable=too-many-statements
        wire = 0

        import jax
        import tensorflow as tf
        import torch

        param_torch = torch.tensor([1, 2])
        param_tf = tf.Variable([1, 2])
        param_jax = jax.numpy.array([1, 2])
        param_qml = npp.array([1, 2])
        param_np = np.array([1, 2])

        op1 = PARAMETRIZED_OPERATIONS_2P_1W[0]
        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1[0], p1[1], wires=wire),
                op1(p2[0], p2[1], wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1[0], p1[1], wires=wire),
                op1(p2[0], p2[1], wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(*param_qml, wires=wire),
            op1(*param_qml_1, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(*param_qml, wires=wire),
            op1(*param_qml_1, wires=wire),
            check_trainability=True,
            check_interface=False,
        )

        wire = [1, 2, 3]
        param_torch = torch.tensor(1)
        param_tf = tf.Variable(1)
        param_jax = jax.numpy.array(1)
        param_qml = npp.array(1)
        param_np = np.array(1)

        op1 = PARAMETRIZED_OPERATIONS_1P_3W[0]
        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=True,
            check_interface=False,
        )

        wire = [1, 2]
        param_torch = torch.tensor([1, 2, 3])
        param_tf = tf.Variable([1, 2, 3])
        param_jax = jax.numpy.array([1, 2, 3])
        param_qml = npp.array([1, 2, 3])
        param_np = np.array([1, 2, 3])

        op1 = PARAMETRIZED_OPERATIONS_3P_2W[0]
        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1[0], p1[1], p1[2], wires=wire),
                op1(p2[0], p2[1], p2[2], wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1[0], p1[1], p1[2], wires=wire),
                op1(p2[0], p2[1], p2[2], wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(*param_qml, wires=wire),
            op1(*param_qml_1, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(*param_qml, wires=wire),
            op1(*param_qml_1, wires=wire),
            check_trainability=True,
            check_interface=False,
        )

        wire = 0
        param_torch = torch.tensor(1)
        param_tf = tf.Variable(1)
        param_jax = jax.numpy.array(1)
        param_qml = npp.array(1)
        param_np = np.array(1)

        op1 = PARAMETRIZED_OPERATIONS_Remaining[0]
        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1, "Y", wires=wire),
                op1(p2, "Y", wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1, "Y", wires=wire),
                op1(p2, "Y", wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(param_qml, "Y", wires=wire),
            op1(param_qml_1, "Y", wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param_qml, "Y", wires=wire),
            op1(param_qml_1, "Y", wires=wire),
            check_trainability=True,
            check_interface=False,
        )

        wire = 0
        param_torch = torch.tensor([[1, 0], [0, 1]]) * 1j
        param_tf = tf.Variable([[1, 0], [0, 1]], dtype=tf.complex64) * 1j
        param_jax = jax.numpy.eye(2) * 1j
        param_qml = npp.eye(2) * 1j
        param_np = np.eye(2) * 1j

        op1 = PARAMETRIZED_OPERATIONS_Remaining[1]
        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=True,
            check_interface=False,
        )

        wire = 0
        param_torch = torch.tensor([1.0, 1.0j])
        param_tf = tf.Variable([1.0 + 0j, 1.0j])
        param_jax = jax.numpy.array([1.0, 1.0j])
        param_qml = npp.array([1.0, 1.0j])
        param_np = np.array([1.0, 1.0j])

        op1 = PARAMETRIZED_OPERATIONS_Remaining[2]
        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1, wires=wire),
                op1(p2, wires=wire),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param_qml, wires=wire),
            op1(param_qml_1, wires=wire),
            check_trainability=True,
            check_interface=False,
        )

        wire = 0
        param_torch = torch.tensor([[1, 0], [0, 1]]) * 1j
        param_tf = tf.Variable([[1, 0], [0, 1]], dtype=tf.complex64) * 1j
        param_jax = jax.numpy.eye(2) * 1j
        param_qml = npp.eye(2) * 1j
        param_np = np.eye(2) * 1j

        op1 = PARAMETRIZED_OPERATIONS_Remaining[3]
        param_list = [param_qml, param_torch, param_jax, param_tf, param_np]
        for p1, p2 in itertools.combinations(param_list, 2):
            assert qml.equal(
                op1(p1, wires=wire, control_wires=wire + 1),
                op1(p2, wires=wire, control_wires=wire + 1),
                check_trainability=False,
                check_interface=False,
            )
            assert not qml.equal(
                op1(p1, wires=wire, control_wires=wire + 1),
                op1(p2, wires=wire, control_wires=wire + 1),
                check_trainability=False,
                check_interface=True,
            )

        param_qml_1 = param_qml.copy()
        param_qml_1.requires_grad = False
        assert qml.equal(
            op1(param_qml, wires=wire, control_wires=wire + 1),
            op1(param_qml_1, wires=wire, control_wires=wire + 1),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param_qml, wires=wire, control_wires=wire + 1),
            op1(param_qml_1, wires=wire, control_wires=wire + 1),
            check_trainability=True,
            check_interface=False,
        )

    def test_equal_simple_op_remaining(self):
        """Test changing parameter or wire returns False"""
        wire = 0
        param = [0.123] * 2
        op1 = PARAMETRIZED_OPERATIONS_2P_1W[0]
        assert qml.equal(
            op1(*param, wires=wire),
            op1(*param, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(*param, wires=wire),
            op1(*param, wires=wire + 1),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(*param, wires=wire),
            op1(param[0] * 2, param[1], wires=wire),
            check_trainability=False,
            check_interface=False,
        )

        wire = [1, 2, 3]
        param = 0.123
        op1 = PARAMETRIZED_OPERATIONS_1P_3W[0]
        assert qml.equal(
            op1(param, wires=wire),
            op1(param, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param * 2, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param, wires=[w + 1 for w in wire]),
            check_trainability=False,
            check_interface=False,
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
        assert not qml.equal(
            op1(*param, wires=wire),
            op1(*param, wires=[w + 1 for w in wire]),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(*param, wires=wire),
            op1(param[0] * 2, param[1], param[2], wires=wire),
            check_trainability=False,
            check_interface=False,
        )

        wire = 0
        param = 0.123
        op1 = PARAMETRIZED_OPERATIONS_Remaining[0]
        assert qml.equal(
            op1(param, "Y", wires=wire),
            op1(param, "Y", wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, "Y", wires=wire),
            op1(param * 2, "Y", wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, "Y", wires=wire),
            op1(param, "Y", wires=wire + 1),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, "Y", wires=wire),
            op1(param, "Z", wires=wire),
            check_trainability=False,
            check_interface=False,
        )

        wire = 0
        param = np.eye(2) * 1j
        op1 = PARAMETRIZED_OPERATIONS_Remaining[1]
        assert qml.equal(
            op1(param, wires=wire),
            op1(param, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param * 2, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param, wires=wire + 1),
            check_trainability=False,
            check_interface=False,
        )

        wire = 0
        param = np.array([1.0, 1.0j])
        op1 = PARAMETRIZED_OPERATIONS_Remaining[2]
        assert qml.equal(
            op1(param, wires=wire),
            op1(param, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param * 2, wires=wire),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire),
            op1(param, wires=wire + 1),
            check_trainability=False,
            check_interface=False,
        )

        wire = 0
        param = np.eye(2) * 1j
        op1 = PARAMETRIZED_OPERATIONS_Remaining[3]
        assert qml.equal(
            op1(param, wires=wire, control_wires=wire + 1),
            op1(param, wires=wire, control_wires=wire + 1),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire, control_wires=wire + 1),
            op1(param * 2, wires=wire, control_wires=wire + 1),
            check_trainability=False,
            check_interface=False,
        )
        assert not qml.equal(
            op1(param, wires=wire, control_wires=wire + 1),
            op1(param, wires=wire + 2, control_wires=wire + 1),
            check_trainability=False,
            check_interface=False,
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
        -JAX and TF
        -JAX and Torch
        -TF and Autograd
        -TF and Torch
        -Autograd and Torch
        """
        import jax
        import tensorflow as tf
        import torch

        wire = 0

        pl_tensor = qml.numpy.array(0.3, requires_grad=True)
        tf_tensor = tf.Variable(0.3)
        torch_tensor = torch.tensor(0.3, requires_grad=True)

        non_jax_tensors = [pl_tensor, tf_tensor, torch_tensor]

        # JAX and the others
        # ------------------
        # qml.math.requires_grad returns True for a Tracer with JAX, the
        # assertion involves using a JAX function that transforms a JAX NumPy
        # array into a Tracer
        def jax_assertion_func(x, other_tensor):
            operation1 = op1(jax.numpy.array(x), wires=1)
            operation2 = op1(other_tensor, wires=1)
            if isinstance(other_tensor, tf.Variable):
                with tf.GradientTape():
                    assert qml.equal(
                        operation1, operation2, check_interface=False, check_trainability=True
                    )
            else:
                assert qml.equal(
                    operation1, operation2, check_interface=False, check_trainability=True
                )
            return x

        par = 0.3
        for tensor in non_jax_tensors:
            jax.grad(jax_assertion_func, argnums=0)(par, tensor)

        # TF and Autograd
        # ------------------
        with tf.GradientTape():
            assert qml.equal(
                op1(tf_tensor, wires=wire),
                op1(pl_tensor, wires=wire),
                check_trainability=True,
                check_interface=False,
            )

        # TF and Torch
        # ------------------
        with tf.GradientTape():
            assert qml.equal(
                op1(tf_tensor, wires=wire),
                op1(torch_tensor, wires=wire),
                check_trainability=True,
                check_interface=False,
            )

        # Autograd and Torch
        # ------------------
        assert qml.equal(
            op1(pl_tensor, wires=wire),
            op1(torch_tensor, wires=wire),
            check_trainability=True,
            check_interface=False,
        )

    def test_equal_with_different_arithmetic_depth(self):
        """Test equal method with two operators with different arithmetic depth."""
        op1 = qml.RX(0.3, wires=0)
        op2 = qml.prod(op1, qml.RY(0.25, wires=1))
        assert not qml.equal(op1, op2)

    def test_equal_with_unsupported_nested_operators_returns_false(self):
        """Test that the equal method with two operators with the same arithmetic depth (>0) returns
        `False` unless there is a singledispatch function specifically comparing that operator type.
        """

        op1 = SymbolicOp(qml.PauliY(0))
        op2 = SymbolicOp(qml.PauliY(0))

        assert op1.arithmetic_depth == op2.arithmetic_depth
        assert op1.arithmetic_depth > 0

        assert not qml.equal(op1, op2)

    # Measurements test cases
    @pytest.mark.parametrize("ops", PARAMETRIZED_MEASUREMENTS_COMBINATIONS)
    def test_not_equal_diff_measurement(self, ops):
        """Test different measurements return False"""
        assert not qml.equal(ops[0], ops[1])

    @pytest.mark.parametrize("op1", PARAMETRIZED_MEASUREMENTS)
    def test_equal_same_measurement(self, op1):
        """Test same measurements return True"""
        assert qml.equal(op1, op1)

    @pytest.mark.parametrize("op1", PARAMETRIZED_OPERATIONS)
    @pytest.mark.parametrize("op2", PARAMETRIZED_MEASUREMENTS)
    def test_not_equal_operator_measurement(self, op1, op2):
        """Test operator not equal to measurement"""
        assert not qml.equal(op1, op2)


class TestMeasurementsEqual:
    @pytest.mark.jax
    def test_observables_different_interfaces(self):
        """Check that the check_interface keyword is used when comparing observables."""

        import jax

        M1 = np.eye(2)
        M2 = jax.numpy.eye(2)
        ob1 = qml.Hermitian(M1, 0)
        ob2 = qml.Hermitian(M2, 0)

        assert not qml.equal(qml.expval(ob1), qml.expval(ob2), check_interface=True)
        assert qml.equal(qml.expval(ob1), qml.expval(ob2), check_interface=False)

    def test_observables_different_trainability(self):
        """Check the check_trainability keyword argument affects comparisons of measurements."""
        M1 = qml.numpy.eye(2, requires_grad=True)
        M2 = qml.numpy.eye(2, requires_grad=False)

        ob1 = qml.Hermitian(M1, 0)
        ob2 = qml.Hermitian(M2, 0)

        assert not qml.equal(qml.expval(ob1), qml.expval(ob2), check_trainability=True)
        assert qml.equal(qml.expval(ob1), qml.expval(ob2), check_trainability=False)

    def test_observables_atol(self):
        """Check that the atol keyword argument affects comparisons of measurements."""
        M1 = np.eye(2)
        M2 = M1 + 1e-3

        ob1 = qml.Hermitian(M1, 0)
        ob2 = qml.Hermitian(M2, 0)

        assert not qml.equal(qml.expval(ob1), qml.expval(ob2))
        assert qml.equal(qml.expval(ob1), qml.expval(ob2), atol=1e-1)

    def test_observables_rtol(self):
        """Check rtol affects comparison of measurement observables."""
        M1 = np.eye(2)
        M2 = np.diag([1 + 1e-3, 1 - 1e-3])

        ob1 = qml.Hermitian(M1, 0)
        ob2 = qml.Hermitian(M2, 0)

        assert not qml.equal(qml.expval(ob1), qml.expval(ob2))
        assert qml.equal(qml.expval(ob1), qml.expval(ob2), rtol=1e-2)

    def test_eigvals_atol(self):
        """Check atol affects comparisons of eigenvalues."""
        m1 = ProbabilityMP(eigvals=(1, 1e-3))
        m2 = ProbabilityMP(eigvals=(1, 0))

        assert not qml.equal(m1, m2)
        assert qml.equal(m1, m2, atol=1e-2)

    def test_eigvals_rtol(self):
        """Check that rtol affects comparisons of eigenvalues."""
        m1 = ProbabilityMP(eigvals=(1 + 1e-3, 0))
        m2 = ProbabilityMP(eigvals=(1, 0))

        assert not qml.equal(m1, m2)
        assert qml.equal(m1, m2, rtol=1e-2)

    def test_observables_equal_but_wire_order_not(self):
        """Test that when the wire orderings are not equal but the observables are, that
        we still get True."""

        x1 = qml.PauliX(1)
        z0 = qml.PauliZ(0)

        o1 = qml.prod(x1, z0)
        o2 = qml.prod(z0, x1)
        assert qml.equal(qml.expval(o1), qml.expval(o2))


class TestObservablesComparisons:
    """Tests comparisons between Hamiltonians, Tensors and PauliX/Y/Z operators"""

    @pytest.mark.parametrize(("H1", "H2", "res"), equal_hamiltonians)
    def test_hamiltonian_equal(self, H1, H2, res):
        """Tests that equality can be checked between Hamiltonians"""
        assert qml.equal(H1, H2) == qml.equal(H2, H1)
        assert qml.equal(H1, H2) == res

    @pytest.mark.parametrize(("T1", "T2", "res"), equal_tensors)
    def test_tensors_equal(self, T1, T2, res):
        """Tests that equality can be checked between Tensors"""
        assert qml.equal(T1, T2) == qml.equal(T2, T1)
        assert qml.equal(T1, T2) == res

    @pytest.mark.parametrize(("H", "T", "res"), equal_hamiltonians_and_tensors)
    def test_hamiltonians_and_tensors_equal(self, H, T, res):
        """Tests that equality can be checked between a Hamiltonian and a Tensor"""
        assert qml.equal(H, T) == qml.equal(T, H)
        assert qml.equal(H, T) == res

    @pytest.mark.parametrize(("op1", "op2", "res"), equal_pauli_operators)
    def test_pauli_operator_equals(self, op1, op2, res):
        """Tests that equality can be checked between PauliX/Y/Z operators, and between Pauli operators and Hamiltonians"""
        assert qml.equal(op1, op2) == qml.equal(op2, op1)
        assert qml.equal(op1, op2) == res

    def test_hamiltonian_and_operation_not_equal(self):
        """Tests that comparing a Hamiltonian with an Operator that is not an Observable returns False"""
        op1 = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliY(0)])
        op2 = qml.RX(1.2, 0)
        assert qml.equal(op1, op2) is False
        assert qml.equal(op2, op1) is False

    def test_tensor_and_operation_not_equal(self):
        """Tests that comparing a Tensor with an Operator that is not an Observable returns False"""
        op1 = qml.PauliX(0) @ qml.PauliY(1)
        op2 = qml.RX(1.2, 0)
        assert qml.equal(op1, op2) is False
        assert qml.equal(op2, op1) is False

    def test_tensor_and_unsupported_observable_returns_false(self):
        """Tests that trying to compare a Tensor to something other than another Tensor or a Hamiltonian returns False"""
        op1 = qml.PauliX(0) @ qml.PauliY(1)
        op2 = qml.Hermitian([[0, 1], [1, 0]], 0)

        assert not qml.equal(op1, op2)

    def test_unsupported_object_type_not_implemented(self):
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(NotImplementedError, match="Comparison of"):
            qml.equal(dev, dev)


class TestSymbolicOpComparison:
    """Test comparison for subclasses of SymbolicOp"""

    WIRES = [(5, 5, True), (6, 7, False)]

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

    def test_comparison_of_base_not_implemented_returns_false(self):
        """Test that comparing SymbolicOps of base operators whose comparison is not yet implemented returns False"""
        base = SymbolicOp(qml.RX(1.2, 0))
        op1 = Controlled(base, control_wires=2)
        op2 = Controlled(base, control_wires=2)

        assert not qml.equal(op1, op2)

    @pytest.mark.torch
    @pytest.mark.jax
    def test_kwargs_for_base_operator_comparison(self):
        """Test that setting kwargs check_interface and check_trainability are applied when comparing the bases"""
        import torch
        import jax

        base1 = qml.RX(torch.tensor(1.2), wires=0)
        base2 = qml.RX(jax.numpy.array(1.2), wires=0)

        op1 = Controlled(base1, control_wires=1)
        op2 = Controlled(base2, control_wires=1)

        assert not qml.equal(op1, op2)
        assert qml.equal(op1, op2, check_interface=False, check_trainability=False)

    @pytest.mark.parametrize("base", PARAMETRIZED_OPERATIONS)
    def test_controlled_comparison(self, base):
        """Test that Controlled operators can be compared"""
        op1 = Controlled(base, control_wires=7, control_values=0)
        op2 = Controlled(base, control_wires=7, control_values=0)
        assert qml.equal(op1, op2)

    @pytest.mark.parametrize(("wire1", "wire2", "res"), WIRES)
    def test_controlled_base_operator_wire_comparison(self, wire1, wire2, res):
        """Test that equal compares operator wires for Controlled operators"""
        base1 = qml.PauliX(wire1)
        base2 = qml.PauliX(wire2)
        op1 = Controlled(base1, control_wires=1)
        op2 = Controlled(base2, control_wires=1)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(
        ("base1", "base2", "res"),
        [(qml.PauliX(0), qml.PauliX(0), True), (qml.PauliX(0), qml.PauliY(0), False)],
    )
    def test_controlled_base_operator_comparison(self, base1, base2, res):
        """Test that equal compares base operators for Controlled operators"""
        op1 = Controlled(base1, control_wires=1)
        op2 = Controlled(base2, control_wires=1)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("wire1", "wire2", "res"), WIRES)
    def test_control_wires_comparison(self, wire1, wire2, res):
        """Test that equal compares control_wires for Controlled operators"""
        base1 = qml.Hadamard(0)
        base2 = qml.Hadamard(0)
        op1 = Controlled(base1, control_wires=wire1)
        op2 = Controlled(base2, control_wires=wire2)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize(("wire1", "wire2", "res"), WIRES)
    def test_controlled_work_wires_comparison(self, wire1, wire2, res):
        """Test that equal compares work_wires for Controlled operators"""
        base1 = qml.MultiRZ(1.23, [0, 1])
        base2 = qml.MultiRZ(1.23, [0, 1])
        op1 = Controlled(base1, control_wires=2, work_wires=wire1)
        op2 = Controlled(base2, control_wires=2, work_wires=wire2)
        assert qml.equal(op1, op2) == res

    @pytest.mark.parametrize("base", PARAMETRIZED_OPERATIONS)
    def test_adjoint_comparison(self, base):
        """Test that equal compares two objects of the Adjoint class"""
        op1 = qml.adjoint(base)
        op2 = qml.adjoint(base)
        op3 = qml.adjoint(qml.PauliX(15))

        assert qml.equal(op1, op2)
        assert not qml.equal(op1, op3)

    @pytest.mark.parametrize("bases_bases_match", BASES)
    @pytest.mark.parametrize("params_params_match", PARAMS)
    def test_pow_comparison(self, bases_bases_match, params_params_match):
        """Test that equal compares two objects of the Pow class"""
        base1, base2, bases_match = bases_bases_match
        param1, param2, params_match = params_params_match
        op1 = qml.pow(base1, param1)
        op2 = qml.pow(base2, param2)
        assert qml.equal(op1, op2) == (bases_match and params_match)

    @pytest.mark.parametrize("bases_bases_match", BASES)
    @pytest.mark.parametrize("params_params_match", PARAMS)
    def test_exp_comparison(self, bases_bases_match, params_params_match):
        """Test that equal compares two objects of the Exp class"""
        base1, base2, bases_match = bases_bases_match
        param1, param2, params_match = params_params_match
        op1 = qml.exp(base1, param1)
        op2 = qml.exp(base2, param2)
        assert qml.equal(op1, op2) == (bases_match and params_match)

    @pytest.mark.parametrize("bases_bases_match", BASES)
    @pytest.mark.parametrize("params_params_match", PARAMS)
    def test_s_prod_comparison(self, bases_bases_match, params_params_match):
        """Test that equal compares two objects of the SProd class"""
        base1, base2, bases_match = bases_bases_match
        param1, param2, params_match = params_params_match
        op1 = qml.s_prod(param1, base1)
        op2 = qml.s_prod(param2, base2)
        assert qml.equal(op1, op2) == (bases_match and params_match)


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

    def test_non_commuting_order_swap_not_equal(self):
        """Test that changing the order of non-commuting operators is not equal"""
        op1 = qml.prod(qml.PauliX(0), qml.PauliY(0))
        op2 = qml.prod(qml.PauliY(0), qml.PauliX(0))
        assert not qml.equal(op1, op2)

    def test_commuting_order_swap_equal(self):
        """Test that changing the order of commuting operators is equal"""
        op1 = qml.prod(qml.PauliX(0), qml.PauliY(1))
        op2 = qml.prod(qml.PauliY(1), qml.PauliX(0))
        assert qml.equal(op1, op2)

    @pytest.mark.all_interfaces
    def test_prod_kwargs_used_for_base_operator_comparison(self):
        """Test that setting kwargs check_interface and check_trainability are applied when comparing the bases"""
        import torch
        import jax

        base_list1 = [qml.RX(torch.tensor(1.2), wires=0), qml.RX(torch.tensor(2.3), wires=1)]
        base_list2 = [qml.RX(jax.numpy.array(1.2), wires=0), qml.RX(jax.numpy.array(2.3), wires=1)]

        op1 = qml.prod(*base_list1)
        op2 = qml.prod(*base_list2)

        assert not qml.equal(op1, op2)
        assert qml.equal(op1, op2, check_interface=False, check_trainability=False)

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
        import torch
        import jax

        base_list1 = [qml.RX(torch.tensor(1.2), wires=0), qml.RX(torch.tensor(2.3), wires=1)]
        base_list2 = [qml.RX(jax.numpy.array(1.2), wires=0), qml.RX(jax.numpy.array(2.3), wires=1)]

        op1 = qml.sum(*base_list1)
        op2 = qml.sum(*base_list2)

        assert not qml.equal(op1, op2)
        assert qml.equal(op1, op2, check_interface=False, check_trainability=False)

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

        assert qml.equal(ev1(params1, t), ev2(params1, t))
        assert not qml.equal(ev1(params1, t), ev2(params2, t))

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

        assert qml.equal(ev1(params, t), ev2(params, t))
        assert not qml.equal(ev1(params, t), ev3(params, t))

    def test_times_comparison(self):
        """Test that times are compared for two ParametrizedEvolution ops"""
        coeffs = [3, f1, f2]
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]

        h1 = qml.dot(coeffs, ops)

        ev1 = qml.evolve(h1)

        params = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0]]

        t1 = 3
        t2 = 4

        assert qml.equal(ev1(params, t1), ev1(params, t1))
        assert not qml.equal(ev1(params, t1), ev1(params, t2))

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

        assert qml.equal(ev1(params, t), ev2(params, t))
        assert not qml.equal(ev1(params, t), ev3(params, t))

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

        assert qml.equal(ev1(params1, t), ev2(params2, t))
        assert not qml.equal(ev1(params1, t), ev3(params3, t))
