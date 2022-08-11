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
        jax_array = jax.core.Tracer(0.3)
        tf_tensor = tf.Variable(0.3, trainable=True)
        torch_tensor = torch.tensor(0.3, requires_grad=True)

        non_jax_tensors = [pl_tensor, tf_tensor, torch_tensor]

        # JAX and the others
        # ------------------
        # qml.math.requires_grad returns True for a Tracer with JAX, the
        # assertion involves using a JAX function that transforms a JAX NumPy
        # array into a Tracer
        def jax_assertion_func(x, y):
            op1 = qml.RY(jax.numpy.array(x), wires=1)
            op2 = qml.RY(torch.tensor(y, requires_grad=True), wires=1)
            assert qml.equal(op1, op2, check_interface=False, check_trainability=True)
            return x

        par = 0.3
        for tensor in non_jax_tensors:
            jax.grad(jax_assertion_func, argnums=0)(par, par)

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
        with tf.GradientTape() as tape:
            assert qml.equal(
                op1(pl_tensor, wires=wire),
                op1(torch_tensor, wires=wire),
                check_trainability=True,
                check_interface=False,
            )

    def test_equal_with_nested_operators_raises_error(self):
        """Test that the equal method with two operators with the same arithmetic depth (>0) raises
        an error."""
        with pytest.raises(
            NotImplementedError,
            match="Comparison of operators with an arithmetic"
            + " depth larger than 0 is not yet implemented.",
        ):
            qml.equal(qml.adjoint(qml.PauliX(0)), qml.adjoint(qml.PauliX(0)))
