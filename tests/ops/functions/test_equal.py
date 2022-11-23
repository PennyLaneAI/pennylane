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

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as npp
from pennylane.measurements import MeasurementProcess, ObservableReturnTypes

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
    qml.probs(qml.PauliZ(0)),
    qml.probs(qml.PauliZ(1)),
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
        )
    ),
    MeasurementProcess(ObservableReturnTypes.Expectation, eigvals=[1, -1]),
    MeasurementProcess(ObservableReturnTypes.Expectation, eigvals=[1, 2]),
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
    def test_equal_op_remaining(self):
        """Test optional arguments are working"""
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
                with tf.GradientTape() as tape:
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
        with tf.GradientTape() as tape:
            assert qml.equal(
                op1(tf_tensor, wires=wire),
                op1(pl_tensor, wires=wire),
                check_trainability=True,
                check_interface=False,
            )

        # TF and Torch
        # ------------------
        with tf.GradientTape() as tape:
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

    def test_equal_with_nested_operators_raises_error(self):
        """Test that the equal method with two operators with the same arithmetic depth (>0) raises
        an error."""
        with pytest.raises(
            NotImplementedError,
            match="Comparison of operators with an arithmetic"
            + " depth larger than 0 is not yet implemented.",
        ):
            qml.equal(qml.adjoint(qml.RX(1.2, 0)), qml.adjoint(qml.RX(1.2, 0)))

    def test_equal_same_inversion(self):
        """Test operations are equal if they are both inverted."""
        op1 = qml.RX(1.2, wires=0).inv()
        op2 = qml.RX(1.2, wires=0).inv()
        assert qml.equal(op1, op2)

    def test_not_equal_different_inversion(self):
        """Test operations are not equal if one is inverted and the other is not."""
        op1 = qml.PauliX(0)
        op2 = qml.PauliX(0).inv()
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
        assert qml.equal(op1, op2) == False
        assert qml.equal(op2, op1) == False

    def test_tensor_and_operation_not_equal(self):
        """Tests that comparing a Tensor with an Operator that is not an Observable returns False"""
        op1 = qml.PauliX(0) @ qml.PauliY(1)
        op2 = qml.RX(1.2, 0)
        assert qml.equal(op1, op2) == False
        assert qml.equal(op2, op1) == False

    def test_tensor_and_unsupported_observable_not_implemented(self):
        """Tests that trying to compare a Tensor to something other than another Tensor or a Hamiltonian raises a NotImplmenetedError"""
        op1 = qml.PauliX(0) @ qml.PauliY(1)
        op2 = qml.Hermitian([[0, 1], [1, 0]], 0)

        with pytest.raises(NotImplementedError, match="Comparison of"):
            qml.equal(op1, op2)

    def test_unsupported_object_type_not_implemented(self):
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(NotImplementedError, match="Comparison of"):
            qml.equal(dev, dev)
