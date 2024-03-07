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
import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
from pennylane import math
from pennylane.operation import Channel
from pennylane.devices.qutrit_mixed import apply_operation, measure, create_initial_state

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]

subspaces = [(0, 1), (0, 2), (1, 2)]
krause_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)


class CustomChannel(Channel):  # pylint: disable=too-few-public-methods
    num_params = 1
    num_wires = 1

    def __init__(self, p, wires, id=None):
        super().__init__(p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p):
        if math.get_interface(p) == "tensorflow":
            p = math.cast_like(p, 1j)

        K0 = math.sqrt(1 - p + math.eps) * math.convert_like(math.eye(3, dtype=complex), p)
        K1 = math.sqrt(p + math.eps) * math.convert_like(krause_matrix, p)
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
    class CustomOp(qml.operation.Operation):
        num_wires = 1

        def matrix(self):
            return mat

    new_state = apply_operation(CustomOp(0), one_qutrit_state)
    assert qml.math.allclose(new_state, mat @ one_qutrit_state @ np.conj(mat).T)


# TODO add tests for special cases as they are added


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize(
    "state,shape", [("two_qutrit_state", (9, 9)), ("two_qutrit_batched_state", (2, 9, 9))]
)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""

    class Debugger:  # pylint: disable=too-few-public-methods
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    @pytest.mark.usefixtures("two_qutrit_state")
    def test_no_debugger(
        self, ml_framework, state, shape, request
    ):  # pylint: disable=unused-argument
        """Test nothing happens when there is no debugger"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        new_state = apply_operation(qml.Snapshot(), initial_state, is_state_batched=len(shape) != 2)
        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

    def test_empty_tag(self, ml_framework, state, shape, request):
        """Test a snapshot is recorded properly when there is no tag"""
        state = request.getfixturevalue(state)
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

    def test_provided_tag(self, ml_framework, state, shape, request):
        """Test a snapshot is recorded properly when provided a tag"""
        state = request.getfixturevalue(state)
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

    def test_snapshot_with_measurement(self, ml_framework, state, shape, request):
        """Test a snapshot with measurement throws NotImplementedError"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        debugger = self.Debugger()
        with pytest.raises(NotImplementedError):
            _ = apply_operation(
                qml.Snapshot(measurement=qml.expval(qml.GellMann(0, 1))),
                initial_state,
                debugger=debugger,
                is_state_batched=len(shape) != 2,
            )


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestOperation:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations (not channels) are applied correctly."""

    broadcasted_ops = [
        qml.TRX(np.array([np.pi, np.pi / 2]), wires=0, subspace=(0, 1)),
        qml.TRY(np.array([np.pi, np.pi / 2]), wires=1, subspace=(0, 1)),
        qml.TRZ(np.array([np.pi, np.pi / 2]), wires=2, subspace=(1, 2)),
        qml.QutritUnitary(
            np.array([unitary_group.rvs(27), unitary_group.rvs(27)]),
            wires=[0, 1, 2],
        ),
    ]
    unbroadcasted_ops = [
        qml.THadamard(wires=0),
        qml.TClock(wires=1),
        qml.TShift(wires=2),
        qml.TAdd(wires=[1, 2]),
        qml.TRX(np.pi / 3, wires=0, subspace=(0, 2)),
        qml.TRY(2 * np.pi / 3, wires=1, subspace=(1, 2)),
        qml.TRZ(np.pi / 6, wires=2, subspace=(0, 1)),
        qml.QutritUnitary(unitary_group.rvs(27), wires=[0, 1, 2]),
    ]
    num_qutrits = 3
    num_batched = 2

    @classmethod
    def expand_matrices(cls, op, batch_size=0):
        """Find expanded operator matrices, since qml.matrix isn't working for qutrits #4367"""
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
        res = apply_operation(op, qml.math.asarray(state, like=ml_framework))

        expanded_operator = self.expand_matrices(op)
        expected = self.get_expected_state(expanded_operator, state)

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op(self, op, ml_framework, three_qutrit_state):
        """Tests that batched operations are applied correctly to an unbatched state."""
        state = three_qutrit_state
        res = apply_operation(op, qml.math.asarray(state, like=ml_framework))
        expanded_operators = self.expand_matrices(op, self.num_batched)

        def get_expected(m):
            return self.get_expected_state(m, state)

        expected = [get_expected(expanded_operators[i]) for i in range(self.num_batched)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_broadcasted_state(self, op, ml_framework, three_qutrit_batched_state):
        """Tests that unbatched operations are applied correctly to a batched state."""
        state = three_qutrit_batched_state
        res = apply_operation(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        expanded_operator = self.expand_matrices(op)

        def get_expected(s):
            return self.get_expected_state(expanded_operator, s)

        expected = [get_expected(state[i]) for i in range(self.num_batched)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op_broadcasted_state(self, op, ml_framework, three_qutrit_batched_state):
        """Tests that batched operations are applied correctly to a batched state."""
        state = three_qutrit_batched_state
        res = apply_operation(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        expanded_operators = self.expand_matrices(op, self.num_batched)

        expected = [
            self.get_expected_state(expanded_operators[i], state[i])
            for i in range(self.num_batched)
        ]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_batch_size_set_if_missing(self, ml_framework, one_qutrit_state):
        """Tests that the batch_size is set on an operator if it was missing before."""
        param = qml.math.asarray([0.1, 0.2], like=ml_framework)
        state = one_qutrit_state
        op = qml.TRX(param, 0)
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

        def __init__(self, p, wires, id=None):
            super().__init__(p, wires=wires, id=id)

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

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_broadcasted_state(self, ml_framework, wire, two_qutrit_batched_state):
        """Tests that Channel operations are applied correctly to a batched state."""
        state = two_qutrit_batched_state

        test_channel = self.CustomChannel(0.3, wires=wire)
        res = apply_operation(test_channel, math.asarray(state, like=ml_framework))

        expanded_krons = self.expand_krons(test_channel)
        expected = [
            self.get_expected_state(expanded_krons, state[i]) for i in range(self.num_batched)
        ]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)


@pytest.mark.parametrize("subspace", [(0, 1), (0, 2), (1, 2)])
class TestTRXCalcGrad:
    """Tests the application and differentiation of a TRX gate in the different interfaces."""

    phi = 0.325

    @staticmethod
    def compare_expected_result(phi, state, probs, subspace, g):
        trx = qml.TRX.compute_matrix(phi, subspace)
        trx_adj = qml.TRX.compute_matrix(-phi, subspace)
        state = math.reshape(state, (9, 9))

        expected_probs = math.diagonal(
            np.kron(trx, np.eye(3)) @ state @ np.kron(trx_adj, np.eye(3))
        )
        assert qml.math.allclose(probs, expected_probs)

        if subspace[0] == 0:
            gell_mann_index = 1 if subspace[1] == 1 else 4
        else:
            gell_mann_index = 6
        gell_mann_matrix = qml.GellMann.compute_matrix(gell_mann_index)
        trx_derivative = -0.5j * gell_mann_matrix @ trx
        trx_adj_derivative = 0.5j * gell_mann_matrix @ trx_adj

        expected_derivative_state = (
            np.kron(trx_derivative, np.eye(3)) @ state @ np.kron(trx_adj, np.eye(3))
        ) + (np.kron(trx, np.eye(3)) @ state @ np.kron(trx_adj_derivative, np.eye(3)))
        expected_derivative = np.diagonal(expected_derivative_state)
        print(g, "\n")
        print(expected_derivative)
        assert qml.math.allclose(g, expected_derivative)

    @pytest.mark.autograd
    def test_trx_grad_autograd(self, two_qutrit_state, subspace):
        """Test that the application of a trx gate is differentiable with autograd."""

        state = qml.numpy.array(two_qutrit_state)

        def f(phi):
            op = qml.TRX(phi, wires=0, subspace=subspace)
            new_state = apply_operation(op, state)
            return measure(qml.probs(), new_state)

        phi = qml.numpy.array(self.phi, requires_grad=True)

        probs = f(phi)
        g = qml.jacobian(lambda x: qml.math.real(f(x)))(phi)
        self.compare_expected_result(phi, state, probs, subspace, g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_trx_grad_jax(self, use_jit, two_qutrit_state, subspace):
        """Test that the application of a trx gate is differentiable with jax."""

        import jax

        state = jax.numpy.array(two_qutrit_state)

        def f(phi):
            op = qml.TRX(phi, wires=0, subspace=subspace)
            new_state = apply_operation(op, state)
            return measure(qml.probs(), new_state)

        if use_jit:
            f = jax.jit(f)

        phi = self.phi

        probs = f(phi)
        g = jax.jacobian(f)(phi)
        self.compare_expected_result(phi, state, probs, subspace, g)

    @pytest.mark.torch
    def test_trx_grad_torch(self, two_qutrit_state, subspace):
        """Tests the application and differentiation of a trx gate with torch."""

        import torch

        state = torch.tensor(two_qutrit_state)

        def f(phi):
            op = qml.TRX(phi, wires=0, subspace=subspace)
            new_state = apply_operation(op, state)
            return measure(qml.probs(), new_state)

        phi = torch.tensor(self.phi, requires_grad=True)

        probs = f(phi)
        g = torch.autograd.functional.jacobian(f, phi)

        self.compare_expected_result(
            phi.detach().numpy(),
            state.detach().numpy(),
            probs.detach().numpy(),
            subspace,
            g.detach().numpy(),
        )

    @pytest.mark.tf
    def test_trx_grad_tf(self, two_qutrit_state, subspace):
        """Tests the application and differentiation of a trx gate with tensorflow"""
        import tensorflow as tf

        state = tf.Variable(two_qutrit_state)
        phi = tf.Variable(0.8589, trainable=True)

        with tf.GradientTape() as grad_tape:
            op = qml.TRX(phi, wires=0, subspace=subspace)
            new_state = apply_operation(op, state)
            probs = measure(qml.probs(), new_state)

        grads = grad_tape.jacobian(probs, [phi])
        # tf takes gradient with respect to conj(z), so we need to conj the gradient
        phi_grad = grads[0]  # tf.math.conj(grads[0])

        self.compare_expected_result(phi, state, probs, subspace, phi_grad)


class TestChannelCalcGrad:
    """Tests the application and differentiation of a Channel in the different interfaces."""

    p = 0.325

    @staticmethod
    def compare_expected_result(p, state, new_state, g):
        krause_matrix_two_qutrits = np.kron(np.eye(3), krause_matrix)
        krause_matrix_two_qutrits_adj = krause_matrix_two_qutrits.transpose()
        state = math.reshape(state, (9, 9))

        state_krause_applied = krause_matrix_two_qutrits @ state @ krause_matrix_two_qutrits_adj

        expected_state = (1 - p) * state + (p * state_krause_applied)
        expected_probs = np.diagonal(expected_state)
        assert qml.math.allclose(new_state, expected_probs)

        expected_derivative_state = state_krause_applied - state
        expected_derivative = np.diagonal(expected_derivative_state)
        assert qml.math.allclose(g, expected_derivative)

    @pytest.mark.autograd
    def test_channel_grad_autograd(self, two_qutrit_state):
        """Test that the application of a trx gate is differentiable with autograd."""

        state = qml.numpy.array(two_qutrit_state)

        def f(p):
            channel = CustomChannel(p, wires=1)
            new_state = apply_operation(channel, state)
            return measure(qml.probs(), new_state)

        p = qml.numpy.array(self.p, requires_grad=True)

        probs = f(p)
        g = qml.jacobian(lambda x: qml.math.real(f(x)))(p)
        self.compare_expected_result(p, state, probs, g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_channel_grad_jax(self, use_jit, two_qutrit_state):
        """Test that the application of a trx gate is differentiable with jax."""

        import jax

        state = jax.numpy.array(two_qutrit_state)

        def f(p):
            op = CustomChannel(p, wires=1)
            new_state = apply_operation(op, state)
            return measure(qml.probs(), new_state)

        if use_jit:
            f = jax.jit(f)

        p = self.p

        probs = f(p)
        g = jax.jacobian(f)(p)
        self.compare_expected_result(p, state, probs, g)

    @pytest.mark.torch
    def test_channel_grad_torch(self, two_qutrit_state):
        """Tests the application and differentiation of a trx gate with torch."""

        import torch

        state = torch.tensor(two_qutrit_state)

        def f(p):
            op = CustomChannel(p, wires=1)
            new_state = apply_operation(op, state)
            return measure(qml.probs(), new_state)

        p = torch.tensor(self.p, requires_grad=True)

        probs = f(p)
        g = torch.autograd.functional.jacobian(f, p)

        self.compare_expected_result(
            p.detach().numpy(),
            state.detach().numpy(),
            probs.detach().numpy(),
            g.detach().numpy(),
        )

    @pytest.mark.tf
    def test_channel_grad_tf(self, two_qutrit_state):
        """Tests the application and differentiation of a trx gate with tensorflow"""
        import tensorflow as tf

        state = tf.Variable(two_qutrit_state)
        p = tf.Variable(0.8589 + 0j, trainable=True)

        with tf.GradientTape() as grad_tape:
            op = CustomChannel(p, wires=1)
            new_state = apply_operation(op, state)
            probs = measure(qml.probs(), new_state)

        grads = grad_tape.jacobian(probs, [p])
        # tf takes gradient with respect to conj(z), so we need to conj the gradient
        phi_grad = tf.math.conj(grads[0])

        self.compare_expected_result(p, state, probs, phi_grad)


# TODO: move to other file when measurements PR is done
class TestSumOfTermsDifferentiability:
    x = 0.52

    @staticmethod
    def f(scale, coeffs, n_wires=5, offset=0.1, convert_to_hamiltonian=False):
        ops = [qml.TRX(offset + scale * i, wires=i, subspace=(0, 2)) for i in range(n_wires)]

        if convert_to_hamiltonian:
            H = qml.Hamiltonian(
                coeffs,
                [
                    qml.operation.Tensor(*(qml.GellMann(i, 3) for i in range(n_wires))),
                    qml.operation.Tensor(*(qml.GellMann(i, 5) for i in range(n_wires))),
                ],
            )
        else:
            t1 = qml.s_prod(coeffs[0], qml.prod(*(qml.GellMann(i, 3) for i in range(n_wires))))
            t2 = qml.s_prod(coeffs[1], qml.prod(*(qml.GellMann(i, 5) for i in range(n_wires))))
            H = t1 + t2

        state = create_initial_state(range(n_wires), like=math.get_interface(scale))
        for op in ops:
            state = apply_operation(op, state)
        return measure(qml.expval(H), state)

    @staticmethod
    def expected(scale, coeffs, n_wires=5, offset=0.1, like="numpy"):
        phase = offset + scale * qml.math.asarray(range(n_wires), like=like)
        cosines = math.cos(phase / 2) ** 2
        sines = -math.sin(phase)
        return coeffs[0] * qml.math.prod(cosines) + coeffs[1] * qml.math.prod(sines)

    @pytest.mark.autograd
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    @pytest.mark.parametrize(
        "coeffs",
        [
            (qml.numpy.array(2.5), qml.numpy.array(6.2)),
            (qml.numpy.array(2.5, requires_grad=False), qml.numpy.array(6.2, requires_grad=False)),
        ],
    )
    def test_autograd_backprop(self, convert_to_hamiltonian, coeffs):
        """Test that backpropagation derivatives work in autograd with hamiltonians and large sums."""
        x = qml.numpy.array(self.x)
        out = self.f(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x, coeffs)
        assert qml.math.allclose(out, expected_out)

        g = qml.grad(self.f)(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_g = qml.grad(self.expected)(x, coeffs)
        assert qml.math.allclose(g, expected_g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_jax_backprop(self, convert_to_hamiltonian, use_jit):
        """Test that backpropagation derivatives work with jax with hamiltonians and large sums."""
        import jax

        jax.config.update("jax_enable_x64", True)

        x = jax.numpy.array(self.x, dtype=jax.numpy.float64)
        coeffs = (5.2, 6.7)
        f = jax.jit(self.f, static_argnums=(1, 2, 3, 4)) if use_jit else self.f

        out = f(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x, coeffs)
        assert qml.math.allclose(out, expected_out)

        g = jax.grad(f)(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_g = jax.grad(self.expected)(x, coeffs)
        assert qml.math.allclose(g, expected_g)

    @pytest.mark.torch
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_torch_backprop(self, convert_to_hamiltonian):
        """Test that backpropagation derivatives work with torch with hamiltonians and large sums."""
        import torch

        coeffs = [
            torch.tensor(9.2, requires_grad=False, dtype=torch.float64),
            torch.tensor(6.2, requires_grad=False, dtype=torch.float64),
        ]

        x = torch.tensor(
            -0.289, requires_grad=True, dtype=torch.float64
        )  # TODO what should I make these values
        x2 = torch.tensor(-0.289, requires_grad=True, dtype=torch.float64)
        out = self.f(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x2, coeffs, like="torch")
        assert qml.math.allclose(out, expected_out)

        out.backward()
        expected_out.backward()
        assert qml.math.allclose(x.grad, x2.grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_tf_backprop(self, convert_to_hamiltonian):
        """Test that backpropagation derivatives work with tensorflow with hamiltonians and large sums."""
        import tensorflow as tf

        x = tf.Variable(self.x)
        coeffs = [8.3, 5.7]

        with tf.GradientTape() as tape1:
            out = self.f(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)

        with tf.GradientTape() as tape2:
            expected_out = self.expected(x, coeffs)

        assert qml.math.allclose(out, expected_out)
        g1 = tape1.gradient(out, x)
        g2 = tape2.gradient(expected_out, x)
        assert qml.math.allclose(g1, g2)
