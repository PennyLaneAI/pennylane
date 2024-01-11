# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for create_initial_state in devices/qutrit_mixed/apply_operation."""

import os
import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
from pennylane import math
from pennylane.operation import Channel
from pennylane.devices.qutrit_mixed.apply_operation import (
    apply_operation_einsum,
    apply_operation_tensordot,
    apply_operation,
)



ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


methods = [apply_operation_einsum, apply_operation_tensordot, apply_operation]
broadcasting_methods = [apply_operation_einsum, apply_operation]
subspaces = [(0, 1), (0, 2), (1, 2)]

@pytest.fixture
def one_qutrit_state():
    path = os.path.join(os.getcwd(), "test_ref_files", "one_qutrit_state")
    return np.load(path).item()
@pytest.fixture
def two_qutrit_state():
    path = os.path.join(os.getcwd(), "test_ref_files", "two_qutrit_state")
    return np.load(path).item()[0]

@pytest.fixture
def two_qutrit_states():
    path = os.path.join(os.getcwd(), "test_ref_files", "two_qutrit_state")
    return np.load(path).item()

@pytest.fixture
def three_qutrit_state():
    path = os.path.join(os.getcwd(), "test_ref_files", "three_qutrit_state")
    return np.load(path).item()[0]

@pytest.fixture
def three_qutrit_states():
    path = os.path.join(os.getcwd(), "test_ref_files", "three_qutrit_state")
    return np.load(path).item()


def test_custom_operator_with_matrix(one_qutrit_state):
    """Test that apply_operation works with any operation that defines a matrix."""
    mat = np.array(
        [
            [-0.35546532 - 0.03636115j, -0.19051888 - 0.38049108j, 0.07943913 - 0.8276115j],
            [-0.2766807 - 0.71617593j, -0.1227771 + 0.61271557j, -0.0872488 - 0.11150285j],
            [-0.2312502 - 0.47894201j, -0.04564929 - 0.65295532j, -0.3629075 + 0.3962342j],
        ]
    )

    # pylint: disable=too-few-public-methods
    class CustomOp(qml.operation.Operation):
        num_wires = 1

        def matrix(self):
            return mat

    new_state = apply_operation(CustomOp(0), one_qutrit_state)
    assert qml.math.allclose(new_state, mat @ one_qutrit_state @ np.conj(mat).T)


@pytest.mark.usefixtures("two_qutrit_state")
@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("wire", (0, 1))
class TestTwoQubitStateSpecialCases:
    """Test the special cases on a two qutrit state.  Also tests the special cases for einsum and tensor application methods
    for additional testing of these generic matrix application methods."""

    def test_TAdd(self, method, wire, ml_framework):
        """Test the application of a TAdd gate on a two qutrit state."""
        initial_state = math.asarray(two_qutrit_state, like=ml_framework)

        control = wire
        target = int(not control)
        new_state = method(qml.TAdd((control, target)), initial_state)

        def check_TAdd_second_roll(initial_input, new_input):
            initial_input0 = math.take(initial_input, 0, axis=control + 1)
            new_input0 = math.take(new_input, 0, axis=control + 1)
            assert math.allclose(initial_input0, new_input0)

            initial_input1 = math.take(initial_input, 1, axis=control + 1)
            initial_input1_rolled = math.roll(initial_input1, 1, 1)
            new_input1 = math.take(new_input, 1, axis=control + 1)
            assert math.allclose(initial_input1_rolled, new_input1)

            initial_input2 = math.take(initial_input, 2, axis=control + 1)
            initial_input2_rolled = math.roll(initial_input2, -1, 1)
            new_input2 = math.take(new_input, 2, axis=control + 1)
            assert math.allclose(initial_input2_rolled, new_input2)

        initial0 = math.take(two_qutrit_state, 0, axis=control)
        new0 = math.take(new_state, 0, axis=control)
        check_TAdd_second_roll(initial0, new0)

        initial1 = math.take(two_qutrit_state, 1, axis=control)
        initial1_rolled = np.roll(initial1, 1, 0)
        new1 = math.take(new_state, 1, axis=control)
        check_TAdd_second_roll(initial1_rolled, new1)

        initial2 = math.take(two_qutrit_state, 2, axis=control)
        initial2_rolled = math.roll(initial2, -1, 0)
        new2 = math.take(new_state, 2, axis=control)
        check_TAdd_second_roll(initial2_rolled, new2)

    def test_TShift(self, method, wire, ml_framework):
        """Test the application of a TShift gate on a two qutrit state."""
        initial_state = math.asarray(two_qutrit_state, like=ml_framework)
        new_state = method(qml.TShift(wire), initial_state)

        def check_second_roll(initial_input, new_input):
            initial_input0 = math.take(initial_input, 0, axis=wire + 1)
            new_input1 = math.take(new_input, 1, axis=wire + 1)
            assert math.allclose(initial_input0, new_input1)

            initial_input1 = math.take(initial_input, 1, axis=wire + 1)
            new_input2 = math.take(new_input, 2, axis=wire + 1)
            assert math.allclose(initial_input1, new_input2)

            initial_input2 = math.take(initial_input, 2, axis=wire + 1)
            new_input0 = math.take(new_input, 0, axis=wire + 1)
            assert math.allclose(initial_input2, new_input0)

        initial0 = math.take(two_qutrit_state, 0, axis=wire)
        new1 = math.take(new_state, 1, axis=wire)
        check_second_roll(initial0, new1)

        initial1 = math.take(two_qutrit_state, 1, axis=wire)
        new2 = math.take(new_state, 2, axis=wire)
        check_second_roll(initial1, new2)

        initial2 = math.take(two_qutrit_state, 2, axis=wire)
        new0 = math.take(new_state, 0, axis=wire)
        check_second_roll(initial2, new0)

    def test_TClock(self, method, wire, ml_framework):
        """Test the application of a TClock gate on a two qutrit state."""
        initial_state = math.asarray(two_qutrit_state, like=ml_framework)
        new_state = method(qml.TClock(wire), initial_state)
        w = math.exp(2j * np.pi / 3)
        w2 = math.exp(4j * np.pi / 3)

        def check_second_roll(initial_input, new_input):
            initial_input0 = math.take(initial_input, 0, axis=wire + 1)
            new_input0 = math.take(new_input, 0, axis=wire + 1)
            assert math.allclose(initial_input0, new_input0)

            initial_input1 = math.take(initial_input, 1, axis=wire + 1)
            new_input1 = math.take(new_input, 1, axis=wire + 1)
            print(initial_input1)
            print(new_input1)
            assert math.allclose(initial_input1 / w, new_input1)

            initial_input2 = math.take(initial_input, 2, axis=wire + 1)
            new_input2 = math.take(new_input, 2, axis=wire + 1)
            assert math.allclose(initial_input2 / w2, new_input2)

        initial0 = math.take(two_qutrit_state, 0, axis=wire)
        new0 = math.take(new_state, 0, axis=wire)
        check_second_roll(initial0, new0)

        initial1 = math.take(two_qutrit_state, 1, axis=wire)
        new1 = math.take(new_state, 1, axis=wire)
        check_second_roll(w * initial1, new1)

        initial2 = math.take(two_qutrit_state, 2, axis=wire)
        new2 = math.take(new_state, 2, axis=wire)
        check_second_roll(w2 * initial2, new2)

    @pytest.mark.parametrize("subspace", subspaces)
    def test_THadamard(self, method, wire, ml_framework, subspace):
        initial_state = math.asarray(two_qutrit_state, like=ml_framework)
        op = qml.THadamard(wire, subspace=subspace)
        new_state = method(op, initial_state)

        flattened_state = two_qutrit_state.reshape(9, 9)
        sizes = [3, 3]
        sizes[wire] = 1
        expanded_mat = np.kron(np.kron(np.eye(sizes[0]), op.matrix()), np.eye(sizes[1]))
        adjoint_mat = np.conj(expanded_mat).T
        expected = (expanded_mat @ flattened_state @ adjoint_mat).reshape([3] * 4)

        assert math.allclose(expected, new_state)

    # TODO: Add more tests as Special cases are added


states_and_shapes = [
    (two_qutrit_state(), (9, 9)),
    (two_qutrit_states(), (3, 9, 9)),
]


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("state,shape", states_and_shapes)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""

    class Debugger:  # pylint: disable=too-few-public-methods
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    def test_no_debugger(self, ml_framework, state, shape):  # pylint: disable=unused-argument
        """Test nothing happens when there is no debugger"""
        initial_state = math.asarray(state, like=ml_framework)
        new_state = apply_operation(qml.Snapshot(), initial_state, is_state_batched=len(shape) != 2)

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

    def test_empty_tag(self, ml_framework, state, shape):
        """Test a snapshot is recorded properly when there is no tag"""
        initial_state = math.asarray(state, like=ml_framework)

        debugger = self.Debugger()
        new_state = apply_operation(
            qml.Snapshot(), initial_state, debugger=debugger, is_state_batched=len(shape) != 2
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [0]
        assert debugger.snapshots[0].shape == shape
        assert math.allclose(debugger.snapshots[0], math.reshape(initial_state, shape))

    def test_provided_tag(self, ml_framework, state, shape):
        """Test a snapshot is recorded property when provided a tag"""
        initial_state = math.asarray(state, like=ml_framework)

        debugger = self.Debugger()
        tag = "dense"
        new_state = apply_operation(
            qml.Snapshot(tag), initial_state, debugger=debugger, is_state_batched=len(shape) != 2
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]
        assert debugger.snapshots[tag].shape == shape
        assert math.allclose(debugger.snapshots[tag], math.reshape(initial_state, shape))


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", broadcasting_methods)
class TestBroadcasting:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations (not channels) are applied correctly."""

    broadcasted_ops = [
        qml.TRX(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2, subspace=(0, 1)),
        qml.TRY(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2, subspace=(0, 1)),
        qml.TRZ(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2, subspace=(1, 2)),
        qml.QutritUnitary(
            np.array([unitary_group.rvs(27), unitary_group.rvs(27), unitary_group.rvs(27)]),
            wires=[0, 1, 2],
        ),
    ]
    unbroadcasted_ops = [
        qml.THadamard(wires=2),
        qml.TClock(wires=2),
        qml.TShift(wires=2),
        qml.TAdd(wires=[1, 2]),
        qml.TRX(np.pi / 3, wires=2, subspace=(0, 2)),
        qml.TRY(2 * np.pi / 3, wires=2, subspace=(1, 2)),
        qml.TRZ(np.pi / 6, wires=2, subspace=(0, 1)),
        qml.QutritUnitary(unitary_group.rvs(27), wires=[0, 1, 2]),
    ]
    num_qutrits = 3
    num_batched = 3
    dims = (3**num_qutrits, 3**num_qutrits)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op(self, op, method, ml_framework, three_qutrits_state):
        """Tests that batched operations are applied correctly to an unbatched state."""

        state = three_qutrits_state
        flattened_state = state.reshape(self.dims)

        res = method(op, qml.math.asarray(state, like=ml_framework))
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mats = [
            np.kron(np.eye(3**missing_wires), mat[i]) if missing_wires else mat[i]
            for i in range(self.num_batched)
        ]
        expected = []

        for i in range(self.num_batched):
            expanded_mat = expanded_mats[i]
            adjoint_mat = np.conj(expanded_mat).T
            expected.append(
                (expanded_mat @ flattened_state @ adjoint_mat).reshape([3] * (self.num_qutrits * 2))
            )

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_broadcasted_state(self, op, method, ml_framework, three_qutrits_states):
        """Tests that unbatched operations are applied correctly to a batched state."""
        state = three_qutrits_states

        res = method(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        missing_wires = self.num_qutrits - len(op.wires)
        mat = op.matrix()
        expanded_mat = np.kron(np.eye(3**missing_wires), mat) if missing_wires else mat
        adjoint_mat = np.conj(expanded_mat).T
        expected = []

        for i in range(self.num_batched):
            flattened_state = state[i].reshape(self.dims)
            expected.append(
                (expanded_mat @ flattened_state @ adjoint_mat).reshape([3] * (self.num_qutrits * 2))
            )

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op_broadcasted_state(self, op, method, ml_framework, three_qutrits_states):
        """Tests that batched operations are applied correctly to a batched state."""
        state = three_qutrits_states

        res = method(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        missing_wires = self.num_qutrits - len(op.wires)
        mat = op.matrix()
        expanded_mats = [
            np.kron(np.eye(3**missing_wires), mat[i]) if missing_wires else mat[i]
            for i in range(self.num_batched)
        ]
        expected = []

        for i in range(self.num_batched):
            expanded_mat = expanded_mats[i]
            adjoint_mat = np.conj(expanded_mat).T
            flattened_state = state[i].reshape(self.dims)
            expected.append(
                (expanded_mat @ flattened_state @ adjoint_mat).reshape([3] * (self.num_qutrits * 2))
            )
        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_batch_size_set_if_missing(self, method, ml_framework, one_qutrit_state):
        """Tests that the batch_size is set on an operator if it was missing before."""
        param = qml.math.asarray([0.1, 0.2, 0.3], like=ml_framework)
        state = one_qutrit_state
        op = qml.TRX(param, 0)
        op._batch_size = None  # pylint:disable=protected-access
        state = method(op, state)
        assert state.shape == (3, 3, 3)
        assert op.batch_size == self.num_batched


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
class TestChannels:  # pylint: disable=too-few-public-methods
    """Tests that Channel operations are applied correctly."""

    class CustomChannel(Channel):
        num_params = 1
        num_wires = 1

        def __init__(self, p, wires, id=None):
            super().__init__(p, wires=wires, id=id)

        @staticmethod
        def compute_kraus_matrices(p):
            K0 = (np.sqrt(1 - p) * math.cast_like(np.eye(3), p)).astype(complex)
            K1 = (
                np.sqrt(p) * math.cast_like(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), p)
            ).astype(complex)
            return [K0, K1]

    def test_non_broadcasted_state(self, method, ml_framework, two_qutrits_state):
        """Tests that Channel operations are applied correctly to a state."""
        state = two_qutrits_state
        test_channel = self.CustomChannel(0.3, wires=1)
        res = method(test_channel, math.asarray(state, like=ml_framework))
        flattened_state = state.reshape(9, 9)

        mat = test_channel.kraus_matrices()

        expanded_mats = [np.kron(np.eye(3), mat[i]) for i in range(len(mat))]
        expected = np.zeros((9, 9)).astype(complex)
        for i in range(len(mat)):
            expanded_mat = expanded_mats[i]
            adjoint_mat = np.conj(expanded_mat).T
            expected += expanded_mat @ flattened_state @ adjoint_mat
        expected = expected.reshape([3] * 4)

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_broadcasted_state(self, method, ml_framework, two_qutrits_states):
        """Tests that Channel operations are applied correctly to a batched state."""
        if method is apply_operation_tensordot:
            pytest.skip("Tensordot doesn't support batched operations.")
        state = two_qutrits_states
        test_channel = self.CustomChannel(0.3, wires=1)
        res = method(test_channel, math.asarray(state, like=ml_framework))

        mat = test_channel.kraus_matrices()
        expanded_mats = [np.kron(np.eye(3), mat[i]) for i in range(len(mat))]
        expected = [np.zeros((9, 9)).astype(complex) for _ in range(3)]
        for i in range(3):
            flattened_state = state[i].reshape(9, 9)
            for j in range(len(mat)):
                expanded_mat = expanded_mats[j]
                adjoint_mat = np.conj(expanded_mat).T
                expected[i] += expanded_mat @ flattened_state @ adjoint_mat
            expected[i] = expected[i].reshape([3] * 4)

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)
