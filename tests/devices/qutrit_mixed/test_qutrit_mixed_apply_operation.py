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
"""Unit tests for apply_operation in devices/qutrit_mixed/apply_operation."""

from functools import reduce

import numpy as np
import pytest
from dummy_debugger import Debugger
from scipy.stats import unitary_group

import pennylane as qp
from pennylane import math
from pennylane.core.operator import Channel
from pennylane.devices.qutrit_mixed import apply_operation, measure
from pennylane.devices.qutrit_mixed.utils import QUDIT_DIM
from pennylane.operation import Channel

# Small additive constant to prevent negative sqrt arguments from floating-point errors
_SQRT_STABILITY_EPS = 1e-14

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]

subspaces = [(0, 1), (0, 2), (1, 2)]
kraus_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)


class CustomChannel(Channel):  # pylint: disable=too-few-public-methods
    num_params = 1
    num_wires = 1

    def __init__(self, p, wires):
        super().__init__(p, wires=wires)

    @staticmethod
    def compute_kraus_matrices(p):
        if math.get_interface(p) == "tensorflow":
            p = math.cast_like(p, 1j)

        K0 = math.sqrt(1 - p + _SQRT_STABILITY_EPS) * math.convert_like(
            math.eye(3, dtype=complex), p
        )
        K1 = math.sqrt(p + _SQRT_STABILITY_EPS) * math.convert_like(kraus_matrix, p)
        return [K0, K1]


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
    class CustomOp(qp.operation.Operation):
        num_wires = 1

        def matrix(self):
            return mat

    new_state = apply_operation(CustomOp(0), one_qutrit_state)
    assert qp.math.allclose(new_state, mat @ one_qutrit_state @ np.conj(mat).T)


# TODO: add tests for special cases [sc-79348]


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize(
    "state,shape", [("two_qutrit_state", (9, 9)), ("two_qutrit_batched_state", (2, 9, 9))]
)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""

    @pytest.mark.usefixtures("two_qutrit_state")
    def test_no_debugger(
        self, ml_framework, state, shape, request
    ):  # pylint: disable=unused-argument
        """Test nothing happens when there is no debugger"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        new_state = apply_operation(qp.Snapshot(), initial_state, is_state_batched=len(shape) != 2)
        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

    def test_empty_tag(self, ml_framework, state, shape, request):
        """Test a snapshot is recorded properly when there is no tag"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        debugger = Debugger()
        new_state = apply_operation(
            qp.Snapshot(), initial_state, debugger=debugger, is_state_batched=len(shape) != 2
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [0]
        assert debugger.snapshots[0].shape == shape
        assert math.allclose(debugger.snapshots[0], math.reshape(initial_state, shape))

    def test_provided_tag(self, ml_framework, state, shape, request):
        """Test a snapshot is recorded properly when provided a tag"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        debugger = Debugger()
        tag = "dense"
        new_state = apply_operation(
            qp.Snapshot(tag), initial_state, debugger=debugger, is_state_batched=len(shape) != 2
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]
        assert debugger.snapshots[tag].shape == shape
        assert math.allclose(debugger.snapshots[tag], math.reshape(initial_state, shape))

    def test_snapshot_with_measurement(self, ml_framework, state, shape, request):
        """Test a snapshot with measurement throws NotImplementedError"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)
        tag = "expected_value"

        debugger = Debugger()

        new_state = apply_operation(
            qp.Snapshot(tag, measurement=qp.expval(qp.GellMann(0, 1))),
            initial_state,
            debugger=debugger,
            is_state_batched=len(shape) != 2,
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]

        if len(shape) == 2:
            assert debugger.snapshots[tag].shape == ()
            assert math.allclose(debugger.snapshots[tag], 0.018699118213231336)
        else:
            assert debugger.snapshots[tag].shape == (2,)
            assert math.allclose(
                debugger.snapshots[tag], [0.018699118213231336, 0.018699118213231336]
            )


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestOperation:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations (not channels) are applied correctly."""

    broadcasted_ops = [
        qp.TRX(np.array([np.pi, np.pi / 2]), wires=0, subspace=(0, 1)),
        qp.TRY(np.array([np.pi, np.pi / 2]), wires=1, subspace=(0, 1)),
        qp.TRZ(np.array([np.pi, np.pi / 2]), wires=2, subspace=(1, 2)),
        qp.QutritUnitary(
            np.array([unitary_group.rvs(27), unitary_group.rvs(27)]),
            wires=[0, 1, 2],
        ),
    ]
    unbroadcasted_ops = [
        qp.THadamard(wires=0),
        qp.TClock(wires=1),
        qp.TShift(wires=2),
        qp.TAdd(wires=[1, 2]),
        qp.TRX(np.pi / 3, wires=0, subspace=(0, 2)),
        qp.TRY(2 * np.pi / 3, wires=1, subspace=(1, 2)),
        qp.TRZ(np.pi / 6, wires=2, subspace=(0, 1)),
        qp.QutritUnitary(unitary_group.rvs(27), wires=[0, 1, 2]),
    ]
    num_qutrits = 3
    num_batched = 2

    @classmethod
    def expand_matrices(cls, op, batch_size=0):
        """Find expanded operator matrices, since qp.matrix isn't working for qutrits #4367"""
        pre_wires_identity = np.eye(3 ** op.wires[0])
        post_wires_identity = np.eye(3 ** ((cls.num_qutrits - 1) - op.wires[-1]))
        mat = op.matrix()

        def expand_matrix(matrix):
            return reduce(np.kron, (pre_wires_identity, matrix, post_wires_identity))

        if batch_size:
            return [expand_matrix(mat[i]) for i in range(batch_size)]
        return expand_matrix(mat)

    @classmethod
    def get_expected_state(cls, expanded_operator, state):
        """Finds expected state after applying operator"""
        flattened_state = state.reshape((3**cls.num_qutrits,) * 2)
        adjoint_matrix = np.conj(expanded_operator).T
        new_state = expanded_operator @ flattened_state @ adjoint_matrix
        return new_state.reshape([3] * (cls.num_qutrits * 2))

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_no_broadcasting(self, op, ml_framework, three_qutrit_state):
        """Tests that unbatched operations are applied correctly to an unbatched state."""
        state = three_qutrit_state
        res = apply_operation(op, qp.math.asarray(state, like=ml_framework))

        expanded_operator = self.expand_matrices(op)
        expected = self.get_expected_state(expanded_operator, state)

        assert qp.math.get_interface(res) == ml_framework
        assert qp.math.allclose(res, expected)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op(self, op, ml_framework, three_qutrit_state):
        """Tests that batched operations are applied correctly to an unbatched state."""
        state = three_qutrit_state
        res = apply_operation(op, qp.math.asarray(state, like=ml_framework))
        expanded_operators = self.expand_matrices(op, self.num_batched)

        def get_expected(m):
            return self.get_expected_state(m, state)

        expected = [get_expected(expanded_operators[i]) for i in range(self.num_batched)]

        assert qp.math.get_interface(res) == ml_framework
        assert qp.math.allclose(res, expected)

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_broadcasted_state(self, op, ml_framework, three_qutrit_batched_state):
        """Tests that unbatched operations are applied correctly to a batched state."""
        state = three_qutrit_batched_state
        res = apply_operation(op, qp.math.asarray(state, like=ml_framework), is_state_batched=True)
        expanded_operator = self.expand_matrices(op)

        def get_expected(s):
            return self.get_expected_state(expanded_operator, s)

        expected = [get_expected(state[i]) for i in range(self.num_batched)]

        assert qp.math.get_interface(res) == ml_framework
        assert qp.math.allclose(res, expected)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op_broadcasted_state(self, op, ml_framework, three_qutrit_batched_state):
        """Tests that batched operations are applied correctly to a batched state."""
        state = three_qutrit_batched_state
        res = apply_operation(op, qp.math.asarray(state, like=ml_framework), is_state_batched=True)
        expanded_operators = self.expand_matrices(op, self.num_batched)

        expected = [
            self.get_expected_state(expanded_operators[i], state[i])
            for i in range(self.num_batched)
        ]

        assert qp.math.get_interface(res) == ml_framework
        assert qp.math.allclose(res, expected)

    def test_batch_size_set_if_missing(self, ml_framework, one_qutrit_state):
        """Tests that the batch_size is set on an operator if it was missing before."""
        param = qp.math.asarray([0.1, 0.2], like=ml_framework)
        state = one_qutrit_state
        op = qp.TRX(param, 0)
        op._batch_size = None  # pylint:disable=protected-access
        state = apply_operation(op, state)
        assert state.shape == (self.num_batched, 3, 3)
        assert op.batch_size == self.num_batched


@pytest.mark.parametrize("wire", [0, 1])
@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestChannels:  # pylint: disable=too-few-public-methods
    """Tests that Channel operations are applied correctly."""

    num_qutrits = 2
    num_batched = 2

    @classmethod
    def expand_krons(cls, op):
        """Find expanded operator kraus matrices"""
        pre_wires_identity = np.eye(3 ** op.wires[0])
        post_wires_identity = np.eye(3 ** ((cls.num_qutrits - 1) - op.wires[-1]))
        krons = op.kraus_matrices()
        return [reduce(np.kron, (pre_wires_identity, kron, post_wires_identity)) for kron in krons]

    @classmethod
    def get_expected_state(cls, expanded_krons, state):
        """Finds expected state after applying channel"""
        flattened_state = state.reshape((3**cls.num_qutrits,) * 2)
        adjoint_krons = np.conj(np.transpose(expanded_krons, (0, 2, 1)))
        new_state = np.sum(
            [
                expanded_kron @ flattened_state @ adjoint_krons[i]
                for i, expanded_kron in enumerate(expanded_krons)
            ],
            axis=0,
        )
        return new_state.reshape([3] * (cls.num_qutrits * 2))

    class CustomChannel(Channel):
        num_params = 1
        num_wires = 1

        def __init__(self, p, wires):
            super().__init__(p, wires=wires)

        @staticmethod
        def compute_kraus_matrices(p):
            K0 = (np.sqrt(1 - p) * math.cast_like(np.eye(3), p)).astype(complex)
            K1 = (
                np.sqrt(p) * math.cast_like(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), p)
            ).astype(complex)
            return [K0, K1]

    def test_non_broadcasted_state(self, ml_framework, wire, two_qutrit_state):
        """Tests that Channel operations are applied correctly to a state."""
        state = two_qutrit_state
        test_channel = self.CustomChannel(0.3, wires=wire)
        res = apply_operation(test_channel, math.asarray(state, like=ml_framework))

        expanded_krons = self.expand_krons(test_channel)
        expected = self.get_expected_state(expanded_krons, state)

        assert qp.math.get_interface(res) == ml_framework
        assert qp.math.allclose(res, expected)

    def test_broadcasted_state(self, ml_framework, wire, two_qutrit_batched_state):
        """Tests that Channel operations are applied correctly to a batched state."""
        state = two_qutrit_batched_state

        test_channel = self.CustomChannel(0.3, wires=wire)
        res = apply_operation(test_channel, math.asarray(state, like=ml_framework))

        expanded_krons = self.expand_krons(test_channel)
        expected = [
            self.get_expected_state(expanded_krons, state[i]) for i in range(self.num_batched)
        ]

        assert qp.math.get_interface(res) == ml_framework
        assert qp.math.allclose(res, expected)


@pytest.mark.parametrize("subspace", [(0, 1), (0, 2), (1, 2)])
class TestTRXCalcGrad:
    """Tests the application and differentiation of a TRX gate in the different interfaces."""

    phi = 0.325

    @staticmethod
    def compare_expected_result(phi, state, probs, subspace, jacobian):
        """Compare the expected result for this circuit and gradient with observed values"""
        trx = qp.TRX.compute_matrix(phi, subspace)
        trx_adj = qp.TRX.compute_matrix(-phi, subspace)
        state = math.reshape(state, (9, 9))

        expected_probs = math.real(
            math.diagonal(np.kron(trx, np.eye(3)) @ state @ np.kron(trx_adj, np.eye(3)))
        )
        assert qp.math.allclose(probs, expected_probs)

        if subspace[0] == 0:
            gell_mann_index = 1 if subspace[1] == 1 else 4
        else:
            gell_mann_index = 6
        gell_mann_matrix = qp.GellMann.compute_matrix(gell_mann_index)
        trx_derivative = -0.5j * gell_mann_matrix @ trx
        trx_adj_derivative = 0.5j * gell_mann_matrix @ trx_adj

        expected_derivative_state = (
            np.kron(trx_derivative, np.eye(3)) @ state @ np.kron(trx_adj, np.eye(3))
        ) + (np.kron(trx, np.eye(3)) @ state @ np.kron(trx_adj_derivative, np.eye(3)))
        expected_derivative = np.real(np.diagonal(expected_derivative_state))
        assert qp.math.allclose(jacobian, expected_derivative)

    @pytest.mark.autograd
    def test_trx_grad_autograd(self, two_qutrit_state, subspace):
        """Test that the application of a trx gate is differentiable with autograd."""

        state = qp.numpy.array(two_qutrit_state)

        def f(phi):
            op = qp.TRX(phi, wires=0, subspace=subspace)
            new_state = apply_operation(op, state)
            return measure(qp.probs(), new_state)

        phi = qp.numpy.array(self.phi, requires_grad=True)

        probs = f(phi)
        jacobian = qp.jacobian(lambda x: qp.math.real(f(x)))(phi)
        self.compare_expected_result(phi, state, probs, subspace, jacobian)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_trx_grad_jax(self, use_jit, two_qutrit_state, subspace):
        """Test that the application of a trx gate is differentiable with jax."""

        import jax

        state = jax.numpy.array(two_qutrit_state)

        def f(phi):
            op = qp.TRX(phi, wires=0, subspace=subspace)
            new_state = apply_operation(op, state)
            return measure(qp.probs(), new_state)

        if use_jit:
            f = jax.jit(f)

        probs = f(self.phi)
        jacobian = jax.jacobian(f)(self.phi)
        self.compare_expected_result(self.phi, state, probs, subspace, jacobian)

    @pytest.mark.torch
    def test_trx_grad_torch(self, two_qutrit_state, subspace):
        """Tests the application and differentiation of a trx gate with torch."""

        import torch

        state = torch.tensor(two_qutrit_state)

        def f(phi):
            op = qp.TRX(phi, wires=0, subspace=subspace)
            new_state = apply_operation(op, state)
            return measure(qp.probs(), new_state)

        phi = torch.tensor(self.phi, requires_grad=True)

        probs = f(phi)
        jacobian = torch.autograd.functional.jacobian(f, phi)

        self.compare_expected_result(
            phi.detach().numpy(),
            state.detach().numpy(),
            probs.detach().numpy(),
            subspace,
            jacobian.detach().numpy(),
        )

    @pytest.mark.tf
    def test_trx_grad_tf(self, two_qutrit_state, subspace):
        """Tests the application and differentiation of a trx gate with tensorflow"""
        import tensorflow as tf

        state = tf.Variable(two_qutrit_state, dtype="complex128")
        phi = tf.Variable(0.8589, trainable=True, dtype="float64")

        with tf.GradientTape() as grad_tape:
            op = qp.TRX(phi, wires=0, subspace=subspace)
            new_state = apply_operation(op, state)
            probs = measure(qp.probs(), new_state)

        jacobians = grad_tape.jacobian(probs, [phi])
        phi_jacobian = jacobians[0]

        self.compare_expected_result(phi, state, probs, subspace, phi_jacobian)


class TestChannelCalcGrad:
    """Tests the application and differentiation of a Channel in the different interfaces."""

    p = 0.325

    @staticmethod
    def compare_expected_result(p, state, new_state, jacobian):
        """Compare the expected result for this channel and gradient with observed values"""
        kraus_matrix_two_qutrits = np.kron(np.eye(3), kraus_matrix)
        kraus_matrix_two_qutrits_adj = kraus_matrix_two_qutrits.transpose()
        state = math.reshape(state, (9, 9))

        state_kraus_applied = kraus_matrix_two_qutrits @ state @ kraus_matrix_two_qutrits_adj

        expected_state = (1 - p) * state + (p * state_kraus_applied)
        expected_probs = np.diagonal(expected_state)
        assert qp.math.allclose(new_state, expected_probs)

        expected_derivative_state = state_kraus_applied - state
        expected_derivative = np.diagonal(expected_derivative_state)
        assert qp.math.allclose(jacobian, expected_derivative)

    @pytest.mark.autograd
    def test_channel_grad_autograd(self, two_qutrit_state):
        """Test that the application of a channel is differentiable with autograd."""

        state = qp.numpy.array(two_qutrit_state)

        def f(p):
            channel = CustomChannel(p, wires=1)
            new_state = apply_operation(channel, state)
            return measure(qp.probs(), new_state)

        p = qp.numpy.array(self.p, requires_grad=True)

        probs = f(p)
        jacobian = qp.jacobian(lambda x: qp.math.real(f(x)))(p)
        self.compare_expected_result(p, state, probs, jacobian)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_channel_grad_jax(self, use_jit, two_qutrit_state):
        """Test that the application of a channel is differentiable with jax."""

        import jax

        state = jax.numpy.array(two_qutrit_state)

        def f(p):
            op = CustomChannel(p, wires=1)
            new_state = apply_operation(op, state)
            return measure(qp.probs(), new_state)

        if use_jit:
            f = jax.jit(f)

        probs = f(self.p)
        jacobian = jax.jacobian(f)(self.p)
        self.compare_expected_result(self.p, state, probs, jacobian)

    @pytest.mark.torch
    def test_channel_grad_torch(self, two_qutrit_state):
        """Tests the application and differentiation of a channel with torch."""

        import torch

        state = torch.tensor(two_qutrit_state)

        def f(p):
            op = CustomChannel(p, wires=1)
            new_state = apply_operation(op, state)
            return measure(qp.probs(), new_state)

        p = torch.tensor(self.p, requires_grad=True)

        probs = f(p)
        jacobian = torch.autograd.functional.jacobian(f, p)

        self.compare_expected_result(
            p.detach().numpy(),
            state.detach().numpy(),
            probs.detach().numpy(),
            jacobian.detach().numpy(),
        )

    @pytest.mark.tf
    def test_channel_grad_tf(self, two_qutrit_state):
        """Tests the application and differentiation of a channel with tensorflow"""
        import tensorflow as tf

        state = tf.Variable(two_qutrit_state, dtype="complex128")
        p = tf.Variable(0.8589, trainable=True, dtype="complex128")

        with tf.GradientTape() as grad_tape:
            op = CustomChannel(p, wires=1)
            new_state = apply_operation(op, state)
            probs = measure(qp.probs(), new_state)

        jacobians = grad_tape.jacobian(probs, [p])
        # tf takes gradient with respect to conj(z), so we need to conj the gradient
        phi_jacobian = tf.math.conj(jacobians[0])

        self.compare_expected_result(p, state, probs, phi_jacobian)


def get_valid_density_matrix(num_wires):
    """Helper function to create a valid density matrix"""
    # Create a pure state first
    state = np.zeros(QUDIT_DIM**num_wires, dtype=np.complex128)
    state[0] = 1 / np.sqrt(2)
    state[-1] = 1 / np.sqrt(2)
    # Convert to density matrix
    return np.outer(state, state.conjugate())


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestDensityMatrix:
    """Test that apply_operation works for QutritDensityMatrix"""

    num_qutrits = [1, 2, 3]

    @pytest.mark.parametrize("num_q", num_qutrits)
    def test_valid_density_matrix(self, num_q, ml_framework):
        """Test applying a valid density matrix to the state"""
        density_matrix = get_valid_density_matrix(num_q)
        # Convert density matrix to the given ML framework and ensure complex dtype
        density_matrix = math.asarray(density_matrix, like=ml_framework)
        density_matrix = math.cast(density_matrix, dtype=complex)  # ensure complex

        op = qp.QutritDensityMatrix(density_matrix, wires=range(num_q))

        # Create the initial state as zeros in the same framework and ensure complex dtype
        shape = (QUDIT_DIM,) * (2 * num_q)
        state = np.zeros(shape, dtype=np.complex128)
        state = math.asarray(state, like=ml_framework)
        state = math.cast(state, dtype=complex)

        # Apply operation
        result = qp.devices.qutrit_mixed.apply_operation(op, state)

        # Reshape and cast expected result
        expected = math.reshape(density_matrix, shape)
        expected = math.cast(expected, dtype=complex)

        assert math.allclose(result, expected)

    @pytest.mark.parametrize("num_q", num_qutrits)
    def test_batched_state(self, num_q, ml_framework):
        """Test applying density matrix to batched states"""
        batch_size = 3
        density_matrix = get_valid_density_matrix(num_q)
        density_matrix = math.asarray(density_matrix, like=ml_framework)
        density_matrix = math.cast(density_matrix, dtype=complex)

        op = qp.QutritDensityMatrix(density_matrix, wires=range(num_q))

        shape = (batch_size,) + (QUDIT_DIM,) * (2 * num_q)
        state = math.zeros(shape, like=ml_framework)
        state = math.cast(state, dtype=complex)

        result = qp.devices.qutrit_mixed.apply_operation(op, state, is_state_batched=True)

        expected_single = math.reshape(density_matrix, (QUDIT_DIM,) * (2 * num_q))
        expected_single = math.cast(expected_single, dtype=complex)
        # Tile along batch dimension
        expected = math.stack([expected_single] * batch_size, axis=0)

        assert math.allclose(result, expected)

    def test_partial_trace_single_qutrit_update(self, ml_framework):
        """Minimal test for partial tracing when applying QutritDensityMatrix to a subset of wires."""

        # Initial 2-qutrit state as a (9,9) density matrix representing |00><00|
        # |00> in vector form = [1,0,0,0,0,0,0,0,0]
        # |00><00| as a 9x9 matrix = diag([1,0,0,0,0,0,0,0,0])
        initial_state = np.zeros((9, 9), dtype=complex)
        initial_state[0, 0] = 1.0
        initial_state = math.asarray(initial_state, like=ml_framework)
        initial_state = math.reshape(initial_state, (QUDIT_DIM,) * 4)

        # Define the single-qutrit density matrix |+><+| = 1/3 * [[1,1,1],[1,1,1],[1,1,1]]
        plus_state = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=complex) / 3
        plus_state = math.asarray(plus_state, like=ml_framework)

        # Apply QutritDensityMatrix on the first wire (wire=0)
        op = qp.QutritDensityMatrix(plus_state, wires=[0])

        # The expected final state should be |+><+| ⊗ |0><0|
        # |0><0| = [[1,0,0],[0,0,0],[0,0,0]]
        zero_dm = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
        expected = np.kron(plus_state, zero_dm)  # shape (9,9)
        expected = math.reshape(expected, [QUDIT_DIM] * 4)
        # Apply the operation
        result = qp.devices.qutrit_mixed.apply_operation(op, initial_state)

        assert math.allclose(result, expected, atol=1e-8)

    def test_partial_trace_batched_update(self, ml_framework):  # TODO
        """Minimal test for partial tracing when applying QutritDensityMatrix to a subset of wires, batched."""

        batch_size = 3

        # Initial 2-qutrit state as a (9,9) density matrix representing |00><00| batched
        initial_state = np.zeros((batch_size, 9, 9), dtype=complex)
        for b in range(batch_size):
            initial_state[b, 0, 0] = 1.0
        initial_state = math.asarray(initial_state, like=ml_framework)
        initial_state = math.reshape(initial_state, [batch_size] + [QUDIT_DIM] * 4)

        # Define the single-qutrit density matrix |+><+| = 0.5 * [[1,1],[1,1]]
        plus_state = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=complex) / 3
        plus_state = math.asarray(plus_state, like=ml_framework)

        # Apply QutritDensityMatrix on the first wire (wire=0)
        op = qp.QutritDensityMatrix(plus_state, wires=[0])

        # The expected final state should be |+><+| ⊗ |0><0| for each batch
        zero_dm = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
        expected_single = np.kron(plus_state, zero_dm)  # shape (9,9)
        expected = np.stack([expected_single] * batch_size, axis=0)
        expected = math.reshape(expected, [batch_size] + [QUDIT_DIM] * 4)

        # Apply the operation
        result = qp.devices.qutrit_mixed.apply_operation(op, initial_state, is_state_batched=True)

        assert math.allclose(result, expected, atol=1e-8)

    @pytest.mark.parametrize("wires", ([0, 1], [0, 2]))
    def test_partial_trace_tensor_format_state(self, wires, ml_framework):
        """Test partial tracing with state in tensor format (as used by the actual mixed device).

        This test reproduces the bug from GitHub issue #8932 where QutritDensityMatrix
        fails when applied to a subset of wires because the state is in tensor format
        (3, 3, ..., 3) rather than 2D matrix format (dim, dim).
        """
        # Initial 4-qutrit state in tensor format (3,3,3,3,3,3,3,3) representing |0000><0000|
        num_wires = 4
        initial_state = np.zeros([QUDIT_DIM] * (2 * num_wires), dtype=complex)
        initial_state[0, 0, 0, 0, 0, 0, 0, 0] = 1.0  # |0000><0000|
        initial_state = math.asarray(initial_state, like=ml_framework)

        # Define the 2-qutrit density matrix for GHZ state: (|00> + |11> + |22>)/sqrt(3)
        ghz = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=complex) / np.sqrt(3)
        ghz_dm = np.outer(ghz, np.conj(ghz))  # shape (9, 9)
        ghz_dm = math.asarray(ghz_dm, like=ml_framework)

        # Apply QutritDensityMatrix on the first 2 wires (wires=[0, 1])
        op = qp.QutritDensityMatrix(ghz_dm, wires=wires)

        # Apply the operation - this should not raise ValueError
        result = qp.devices.qutrit_mixed.apply_operation(op, initial_state)

        # Verify result shape matches input shape
        assert result.shape == initial_state.shape

        # Verify the result is a valid density matrix
        result_2d = math.reshape(result, (81, 81))
        # Trace should be 1
        trace_val = math.trace(result_2d)
        assert math.allclose(trace_val, 1.0, atol=1e-8)


def test_qutrit_density_matrix_qnode_integration():
    """Integration test for QutritDensityMatrix on subset of wires using QNode.

    This reproduces the exact bug scenario from GitHub issue #8932.
    """
    n = 2
    dev = qp.device("default.qutrit.mixed", wires=2 * n)

    @qp.qnode(dev)
    def test_circuit(rho):
        # Only initialize n of the 2n qutrits using with rho
        qp.QutritDensityMatrix(rho, wires=range(0, n))

        # Apply THadamard gate to ancilla qutrits
        for a in range(n, 2 * n):
            qp.THadamard(a)

        return qp.probs(wires=range(n))

    # Define the 2-qutrit density matrix for GHZ state: (|00> + |11> + |22>)/sqrt(3)
    ghz = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=complex) / np.sqrt(3)
    ghz_dm = np.outer(ghz, np.conj(ghz))  # shape (9, 9)

    # This should not raise ValueError
    result = test_circuit(ghz_dm)

    # Expected: probabilities for GHZ state are [1/3, 0, 0,0, 1/3,0,0,0,1/3]
    expected = np.array([1 / 3, 0, 0, 0, 1 / 3, 0, 0, 0, 1 / 3])
    assert np.allclose(result, expected, atol=1e-8)
