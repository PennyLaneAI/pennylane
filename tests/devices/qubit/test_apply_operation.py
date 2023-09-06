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
Tests the apply_operation functions from devices/qubit
"""
import pytest

import numpy as np
from scipy.stats import unitary_group
import pennylane as qml


from pennylane.devices.qubit.apply_operation import (
    apply_operation,
    apply_operation_einsum,
    apply_operation_tensordot,
)

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


methods = [apply_operation_einsum, apply_operation_tensordot, apply_operation]


def test_custom_operator_with_matrix():
    """Test that apply_operation works with any operation that defines a matrix."""

    mat = np.array(
        [
            [0.39918205 + 0.3024376j, -0.86421077 + 0.04821758j],
            [0.73240679 + 0.46126509j, 0.49576832 - 0.07091251j],
        ]
    )

    # pylint: disable=too-few-public-methods
    class CustomOp(qml.operation.Operation):
        num_wires = 1

        def matrix(self):
            return mat

    state = np.array([-0.30688912 - 0.4768824j, 0.8100052 - 0.14931113j])

    new_state = apply_operation(CustomOp(0), state)
    assert qml.math.allclose(new_state, mat @ state)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("wire", (0, 1))
class TestTwoQubitStateSpecialCases:
    """Test the special cases on a two qubit state.  Also tests the special cases for einsum and tensor application methods
    for additional testing of these generic matrix application methods."""

    def test_paulix(self, method, wire, ml_framework):
        """Test the application of a paulix gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.PauliX(wire), initial_state)

        initial0dim = qml.math.take(initial_state, 0, axis=wire)
        new1dim = qml.math.take(new_state, 1, axis=wire)

        assert qml.math.allclose(initial0dim, new1dim)

        initial1dim = qml.math.take(initial_state, 1, axis=wire)
        new0dim = qml.math.take(new_state, 0, axis=wire)
        assert qml.math.allclose(initial1dim, new0dim)

    def test_pauliz(self, method, wire, ml_framework):
        """Test the application of a pauliz gate on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.PauliZ(wire), initial_state)

        initial0 = qml.math.take(initial_state, 0, axis=wire)
        new0 = qml.math.take(new_state, 0, axis=wire)
        assert qml.math.allclose(initial0, new0)

        initial1 = qml.math.take(initial_state, 1, axis=wire)
        new1 = qml.math.take(new_state, 1, axis=wire)
        assert qml.math.allclose(initial1, -new1)

    def test_pauliy(self, method, wire, ml_framework):
        """Test the application of a pauliy gate on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.PauliY(wire), initial_state)

        initial0 = qml.math.take(initial_state, 0, axis=wire)
        new1 = qml.math.take(new_state, 1, axis=wire)
        assert qml.math.allclose(1j * initial0, new1)

        initial1 = qml.math.take(initial_state, 1, axis=wire)
        new0 = qml.math.take(new_state, 0, axis=wire)
        assert qml.math.allclose(-1j * initial1, new0)

    def test_hadamard(self, method, wire, ml_framework):
        """Test the application of a hadamard on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.Hadamard(wire), initial_state)

        inv_sqrt2 = 1 / np.sqrt(2)

        initial0 = qml.math.take(initial_state, 0, axis=wire)
        initial1 = qml.math.take(initial_state, 1, axis=wire)

        expected0 = inv_sqrt2 * (initial0 + initial1)
        new0 = qml.math.take(new_state, 0, axis=wire)
        assert qml.math.allclose(new0, expected0)

        expected1 = inv_sqrt2 * (initial0 - initial1)
        new1 = qml.math.take(new_state, 1, axis=wire)
        assert qml.math.allclose(new1, expected1)

    def test_phaseshift(self, method, wire, ml_framework):
        """test the application of a phaseshift gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        phase = qml.math.asarray(-2.3, like=ml_framework)
        shift = qml.math.exp(1j * qml.math.cast(phase, np.complex128))

        new_state = method(qml.PhaseShift(phase, wire), initial_state)

        new0 = qml.math.take(new_state, 0, axis=wire)
        initial0 = qml.math.take(initial_state, 0, axis=wire)
        assert qml.math.allclose(new0, initial0)

        initial1 = qml.math.take(initial_state, 1, axis=wire)
        new1 = qml.math.take(new_state, 1, axis=wire)
        assert qml.math.allclose(shift * initial1, new1)

    def test_cnot(self, method, wire, ml_framework):
        """Test the application of a cnot gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        control = wire
        target = int(not control)

        new_state = method(qml.CNOT((control, target)), initial_state)

        initial0 = qml.math.take(initial_state, 0, axis=control)
        new0 = qml.math.take(new_state, 0, axis=control)
        assert qml.math.allclose(initial0, new0)

        initial1 = qml.math.take(initial_state, 1, axis=control)
        new1 = qml.math.take(new_state, 1, axis=control)
        assert qml.math.allclose(initial1[1], new1[0])
        assert qml.math.allclose(initial1[0], new1[1])

    def test_identity(self, method, wire, ml_framework):
        """Test the application of a GlobalPhase gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.Identity(wire), initial_state)

        assert qml.math.allclose(initial_state, new_state)

    def test_globalphase(self, method, wire, ml_framework):
        """Test the application of a GlobalPhase gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        phase = qml.math.asarray(-2.3, like=ml_framework)
        shift = qml.math.exp(-1j * qml.math.cast(phase, np.complex128))

        new_state_with_wire = method(qml.GlobalPhase(phase, wire), initial_state)
        new_state_no_wire = method(qml.GlobalPhase(phase), initial_state)

        assert qml.math.allclose(shift * initial_state, new_state_with_wire)
        assert qml.math.allclose(shift * initial_state, new_state_no_wire)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""

    class Debugger:  # pylint: disable=too-few-public-methods
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    def test_no_debugger(self, ml_framework):
        """Test nothing happens when there is no debugger"""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)
        new_state = apply_operation(qml.Snapshot(), initial_state)

        assert new_state.shape == initial_state.shape
        assert qml.math.allclose(new_state, initial_state)

    def test_empty_tag(self, ml_framework):
        """Test a snapshot is recorded properly when there is no tag"""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        debugger = self.Debugger()
        new_state = apply_operation(qml.Snapshot(), initial_state, debugger=debugger)

        assert new_state.shape == initial_state.shape
        assert qml.math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [0]
        assert debugger.snapshots[0].shape == (4,)
        assert qml.math.allclose(debugger.snapshots[0], qml.math.flatten(initial_state))

    def test_provided_tag(self, ml_framework):
        """Test a snapshot is recorded property when provided a tag"""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        debugger = self.Debugger()
        tag = "abcd"
        new_state = apply_operation(qml.Snapshot(tag), initial_state, debugger=debugger)

        assert new_state.shape == initial_state.shape
        assert qml.math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]
        assert debugger.snapshots[tag].shape == (4,)
        assert qml.math.allclose(debugger.snapshots[tag], qml.math.flatten(initial_state))


@pytest.mark.parametrize("method", methods)
class TestRXCalcGrad:
    """Tests the application and differentiation of an RX gate in the different interfaces."""

    state = np.array(
        [
            [
                [-0.22209168 + 0.21687383j, -0.1302055 - 0.06014422j],
                [-0.24033117 + 0.28282153j, -0.14025702 - 0.13125938j],
            ],
            [
                [-0.42373896 + 0.51912421j, -0.01934135 + 0.07422255j],
                [0.22311677 + 0.2245953j, 0.33154166 + 0.20820744j],
            ],
        ]
    )

    def compare_expected_result(self, phi, state, new_state, g):
        expected0 = np.cos(phi / 2) * state[0, :, :] + -1j * np.sin(phi / 2) * state[1, :, :]
        expected1 = -1j * np.sin(phi / 2) * state[0, :, :] + np.cos(phi / 2) * state[1, :, :]

        assert qml.math.allclose(new_state[0, :, :], expected0)
        assert qml.math.allclose(new_state[1, :, :], expected1)

        g_expected0 = (
            -0.5 * np.sin(phi / 2) * state[0, :, :] - 0.5j * np.cos(phi / 2) * state[1, :, :]
        )
        g_expected1 = (
            -0.5j * np.cos(phi / 2) * state[0, :, :] - 0.5 * np.sin(phi / 2) * state[1, :, :]
        )

        assert qml.math.allclose(g[0], g_expected0)
        assert qml.math.allclose(g[1], g_expected1)

    @pytest.mark.autograd
    def test_rx_grad_autograd(self, method):
        """Test that the application of an rx gate is differentiable with autograd."""

        state = qml.numpy.array(self.state)

        def f(phi):
            op = qml.RX(phi, wires=0)
            return method(op, state)

        phi = qml.numpy.array(0.325 + 0j, requires_grad=True)

        new_state = f(phi)
        g = qml.jacobian(lambda x: qml.math.real(f(x)))(phi)
        self.compare_expected_result(phi, state, new_state, g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_rx_grad_jax(self, method, use_jit):
        """Test that the application of an rx gate is differentiable with jax."""

        import jax

        state = jax.numpy.array(self.state)

        def f(phi):
            op = qml.RX(phi, wires=0)
            return method(op, state)

        if use_jit:
            f = jax.jit(f)

        phi = 0.325

        new_state = f(phi)
        g = jax.jacobian(f, holomorphic=True)(phi + 0j)
        self.compare_expected_result(phi, state, new_state, g)

    @pytest.mark.torch
    def test_rx_grad_torch(self, method):
        """Tests the application and differentiation of an rx gate with torch."""

        import torch

        state = torch.tensor(self.state)

        def f(phi):
            op = qml.RX(phi, wires=0)
            return method(op, state)

        phi = torch.tensor(0.325, requires_grad=True)

        new_state = f(phi)
        g = torch.autograd.functional.jacobian(f, phi + 0j)

        # torch takes gradient with respect to conj(z), so we need to conj the gradient
        g = torch.conj(g).resolve_conj()

        self.compare_expected_result(
            phi.detach().numpy(),
            state.detach().numpy(),
            new_state.detach().numpy(),
            g.detach().numpy(),
        )

    @pytest.mark.tf
    def test_rx_grad_tf(self, method):
        """Tests the application and differentiation of an rx gate with tensorflow"""
        import tensorflow as tf

        state = tf.Variable(self.state)
        phi = tf.Variable(0.8589 + 0j)

        with tf.GradientTape() as grad_tape:
            op = qml.RX(phi, wires=0)
            new_state = method(op, state)

        grads = grad_tape.jacobian(new_state, [phi])
        # tf takes gradient with respect to conj(z), so we need to conj the gradient
        phi_grad = tf.math.conj(grads[0])

        self.compare_expected_result(phi, state, new_state, phi_grad)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
class TestBroadcasting:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations are applied correctly."""

    broadcasted_ops = [
        qml.RX(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2),
        qml.PhaseShift(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2),
        qml.IsingXX(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=[1, 2]),
        qml.QubitUnitary(
            np.array([unitary_group.rvs(8), unitary_group.rvs(8), unitary_group.rvs(8)]),
            wires=[0, 1, 2],
        ),
    ]

    unbroadcasted_ops = [
        qml.PauliX(2),
        qml.PauliZ(2),
        qml.CNOT([1, 2]),
        qml.RX(np.pi, wires=2),
        qml.PhaseShift(np.pi / 2, wires=2),
        qml.IsingXX(np.pi / 2, wires=[1, 2]),
        qml.QubitUnitary(unitary_group.rvs(8), wires=[0, 1, 2]),
    ]

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op(self, op, method, ml_framework):
        """Tests that batched operations are applied correctly to an unbatched state."""
        state = np.ones((2, 2, 2)) / np.sqrt(8)

        res = method(op, qml.math.asarray(state, like=ml_framework))
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mat = [
            np.kron(np.eye(2 ** missing_wires), mat[i]) if missing_wires else mat[i]
            for i in range(3)
        ]
        expected = [(expanded_mat[i] @ state.flatten()).reshape((2, 2, 2)) for i in range(3)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_broadcasted_state(self, op, method, ml_framework):
        """Tests that unbatched operations are applied correctly to a batched state."""
        state = np.ones((3, 2, 2, 2)) / np.sqrt(8)

        res = method(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mat = np.kron(np.eye(2 ** missing_wires), mat) if missing_wires else mat
        expected = [(expanded_mat @ state[i].flatten()).reshape((2, 2, 2)) for i in range(3)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op_broadcasted_state(self, op, method, ml_framework):
        """Tests that batched operations are applied correctly to a batched state."""
        if method is apply_operation_tensordot:
            pytest.skip("Tensordot doesn't support batched operator and batched state.")

        state = np.ones((3, 2, 2, 2)) / np.sqrt(8)

        res = method(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mat = [
            np.kron(np.eye(2 ** missing_wires), mat[i]) if missing_wires else mat[i]
            for i in range(3)
        ]
        expected = [(expanded_mat[i] @ state[i].flatten()).reshape((2, 2, 2)) for i in range(3)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)


@pytest.mark.parametrize("method", methods)
class TestLargerOperations:
    """Tests matrix applications on states and operations with larger numbers of wires."""

    state = np.array(
        [
            [
                [
                    [-0.21733955 - 0.01990267j, 0.22960893 - 0.0312392j],
                    [0.21406652 - 0.07552019j, 0.09527143 + 0.01870987j],
                ],
                [
                    [0.05603182 - 0.26879067j, -0.02755183 - 0.03097822j],
                    [-0.43962358 - 0.17435254j, 0.12820737 + 0.06794554j],
                ],
            ],
            [
                [
                    [-0.09270161 - 0.3132961j, -0.03276799 + 0.07557535j],
                    [-0.15712707 - 0.32666969j, -0.00898954 + 0.1324474j],
                ],
                [
                    [-0.17760532 + 0.08415488j, -0.26872752 - 0.05767781j],
                    [0.23142582 - 0.1970496j, 0.15483611 - 0.15100495j],
                ],
            ],
        ]
    )

    def test_multicontrolledx(self, method):
        """Tests a four qubit multi-controlled x gate."""

        new_state = method(qml.MultiControlledX(wires=(0, 1, 2, 3)), self.state)

        expected_state = np.copy(self.state)
        expected_state[1, 1, 1, 1] = self.state[1, 1, 1, 0]
        expected_state[1, 1, 1, 0] = self.state[1, 1, 1, 1]

        assert qml.math.allclose(new_state, expected_state)

    def test_double_excitation(self, method):
        """Tests a double excitation operation compared to its decomposition."""

        op = qml.DoubleExcitation(np.array(2.14), wires=(3, 1, 2, 0))

        state_v1 = method(op, self.state)

        state_v2 = self.state
        for d_op in op.decomposition():
            state_v2 = method(d_op, state_v2)

        assert qml.math.allclose(state_v1, state_v2)


@pytest.mark.tf
@pytest.mark.parametrize("op", (qml.PauliZ(8), qml.CNOT((5, 6))))
def test_tf_large_state(op):
    """ "Tests that custom kernels that use slicing fall back to a different method when
    the state has a large number of wires."""
    import tensorflow as tf

    state = np.zeros([2] * 10)
    state = tf.Variable(state)
    new_state = apply_operation(op, state)

    # still all zeros.  Mostly just making sure error not raised
    assert qml.math.allclose(state, new_state)
