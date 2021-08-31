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
Tests for the QubitUnitary decomposition transforms.
"""

import pytest

from itertools import product

import pennylane as qml
from pennylane import numpy as np

from pennylane.wires import Wires
from pennylane.transforms import unitary_to_rot

from gate_data import I, Z, S, T, H, X

single_qubit_decomps = [
    # First set of gates are diagonal and converted to RZ
    (I, qml.RZ, [0.0]),
    (Z, qml.RZ, [np.pi]),
    (S, qml.RZ, [np.pi / 2]),
    (T, qml.RZ, [np.pi / 4]),
    (qml.RZ(0.3, wires=0).matrix, qml.RZ, [0.3]),
    (qml.RZ(-0.5, wires=0).matrix, qml.RZ, [-0.5]),
    # Next set of gates are non-diagonal and decomposed as Rots
    (H, qml.Rot, [np.pi, np.pi / 2, 0.0]),
    (X, qml.Rot, [0.0, np.pi, np.pi]),
    (qml.Rot(0.2, 0.5, -0.3, wires=0).matrix, qml.Rot, [0.2, 0.5, -0.3]),
    (np.exp(1j * 0.02) * qml.Rot(-1.0, 2.0, -3.0, wires=0).matrix, qml.Rot, [-1.0, 2.0, -3.0]),
]

# A simple quantum function for testing
def qfunc(U):
    qml.Hadamard(wires="a")
    qml.QubitUnitary(U, wires="a")
    qml.CNOT(wires=["b", "a"])


class TestDecomposeSingleQubitUnitaryTransform:
    """Tests to ensure the transform itself works in all interfaces."""

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_unitary_to_rot(self, U, expected_gate, expected_params):
        """Test that the transform works in the autograd interface."""
        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)(U).operations

        assert len(ops) == 3

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        assert isinstance(ops[1], expected_gate)
        assert ops[1].wires == Wires("a")
        assert qml.math.allclose(ops[1].parameters, expected_params)

        assert isinstance(ops[2], qml.CNOT)
        assert ops[2].wires == Wires(["b", "a"])

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_unitary_to_rot_torch(self, U, expected_gate, expected_params):
        """Test that the transform works in the torch interface."""
        torch = pytest.importorskip("torch")

        U = torch.tensor(U, dtype=torch.complex64)

        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)(U).operations

        assert len(ops) == 3

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        assert isinstance(ops[1], expected_gate)
        assert ops[1].wires == Wires("a")
        assert qml.math.allclose([x.detach() for x in ops[1].parameters], expected_params)

        assert isinstance(ops[2], qml.CNOT)
        assert ops[2].wires == Wires(["b", "a"])

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_unitary_to_rot_tf(self, U, expected_gate, expected_params):
        """Test that the transform works in the Tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        U = tf.Variable(U, dtype=tf.complex64)

        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)(U).operations

        assert len(ops) == 3

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        assert isinstance(ops[1], expected_gate)
        assert ops[1].wires == Wires("a")
        assert qml.math.allclose([x.numpy() for x in ops[1].parameters], expected_params)

        assert isinstance(ops[2], qml.CNOT)
        assert ops[2].wires == Wires(["b", "a"])

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_unitary_to_rot_jax(self, U, expected_gate, expected_params):
        """Test that the transform works in the JAX interface."""
        jax = pytest.importorskip("jax")

        U = jax.numpy.array(U, dtype=jax.numpy.complex64)

        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)(U).operations

        assert len(ops) == 3

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        assert isinstance(ops[1], expected_gate)
        assert ops[1].wires == Wires("a")
        assert qml.math.allclose([jax.numpy.asarray(x) for x in ops[1].parameters], expected_params)

        assert isinstance(ops[2], qml.CNOT)
        assert ops[2].wires == Wires(["b", "a"])


# A simple circuit; we will test QubitUnitary on matrices constructed using trainable
# parameters, and RZ/RX are easy to write the matrices for.
def original_qfunc_for_grad(angles):
    qml.Hadamard(wires="a")
    qml.RZ(angles[0], wires="a")
    qml.RX(angles[1], wires="b")
    qml.CNOT(wires=["b", "a"])
    return qml.expval(qml.PauliX(wires="a"))


dev = qml.device("default.qubit", wires=["a", "b"])

angle_pairs = [[0.3, 0.3], [np.pi, -0.65], [0.0, np.pi / 2], [np.pi / 3, 0.0]]
diff_methods = ["parameter-shift", "backprop"]
angle_diff_pairs = list(product(angle_pairs, diff_methods))


class TestQubitUnitaryDifferentiability:
    """Tests to ensure the transform is fully differentiable in all interfaces."""

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot(self, rot_angles, diff_method):
        """Tests differentiability in autograd interface."""

        def qfunc_with_qubit_unitary(angles):
            z = angles[0]
            x = angles[1]

            Z_mat = np.array([[np.exp(-1j * z / 2), 0.0], [0.0, np.exp(1j * z / 2)]])

            c = np.cos(x / 2)
            s = np.sin(x / 2) * 1j
            X_mat = np.array([[c, -s], [-s, c]])

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        original_qnode = qml.QNode(original_qfunc_for_grad, dev, diff_method=diff_method)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(transformed_qfunc, dev, diff_method=diff_method)

        input = np.array(rot_angles, requires_grad=True)
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        original_grad = qml.grad(original_qnode)(input)
        transformed_grad = qml.grad(transformed_qnode)(input)

        assert qml.math.allclose(original_grad, transformed_grad)

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_torch(self, rot_angles, diff_method):
        """Tests differentiability in torch interface."""
        torch = pytest.importorskip("torch", minversion="1.8")

        def qfunc_with_qubit_unitary(angles):
            z = angles[0]
            x = angles[1]

            # Had to do this in order to make a torch tensor of torch tensors
            Z_mat = torch.stack(
                [
                    torch.exp(-1j * z / 2),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    torch.exp(1j * z / 2),
                ]
            ).reshape(2, 2)

            # Variables need to be complex
            c = torch.cos(x / 2).type(torch.complex64)
            s = torch.sin(x / 2) * 1j

            X_mat = torch.stack([c, -s, -s, c]).reshape(2, 2)

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        original_qnode = qml.QNode(
            original_qfunc_for_grad, dev, interface="torch", diff_method=diff_method
        )
        original_input = torch.tensor(rot_angles, dtype=torch.float64, requires_grad=True)
        original_result = original_qnode(original_input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="torch", diff_method=diff_method
        )
        transformed_input = torch.tensor(rot_angles, dtype=torch.float64, requires_grad=True)
        transformed_result = transformed_qnode(transformed_input)

        assert qml.math.allclose(original_result, transformed_result)

        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(original_input.grad, transformed_input.grad)

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_tf(self, rot_angles, diff_method):
        """Tests differentiability in tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        def qfunc_with_qubit_unitary(angles):
            z = tf.cast(angles[0], tf.complex128)
            x = tf.cast(angles[1], tf.complex128)

            c = tf.cos(x / 2)
            s = tf.sin(x / 2) * 1j

            Z_mat = tf.convert_to_tensor([[tf.exp(-1j * z / 2), 0.0], [0.0, tf.exp(1j * z / 2)]])
            X_mat = tf.convert_to_tensor([[c, -s], [-s, c]])

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        original_qnode = qml.QNode(
            original_qfunc_for_grad, dev, interface="tf", diff_method=diff_method
        )
        original_input = tf.Variable(rot_angles, dtype=tf.float64)
        original_result = original_qnode(original_input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="tf", diff_method=diff_method
        )
        transformed_input = tf.Variable(rot_angles, dtype=tf.float64)
        transformed_result = transformed_qnode(transformed_input)

        assert qml.math.allclose(original_result, transformed_result)

        with tf.GradientTape() as tape:
            loss = original_qnode(original_input)
        original_grad = tape.gradient(loss, original_input)

        with tf.GradientTape() as tape:
            loss = transformed_qnode(transformed_input)

        transformed_grad = tape.gradient(loss, transformed_input)

        # For 64bit values, need to slightly increase the tolerance threshold
        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-7)

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_jax(self, rot_angles, diff_method):
        """Tests differentiability in jax interface."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        def qfunc_with_qubit_unitary(angles):
            z = angles[0]
            x = angles[1]

            Z_mat = jnp.array([[jnp.exp(-1j * z / 2), 0.0], [0.0, jnp.exp(1j * z / 2)]])

            c = jnp.cos(x / 2)
            s = jnp.sin(x / 2) * 1j
            X_mat = jnp.array([[c, -s], [-s, c]])

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        # Setting the dtype to complex64 causes the gradients to be complex...
        input = jnp.array(rot_angles, dtype=jnp.float64)

        original_qnode = qml.QNode(
            original_qfunc_for_grad, dev, interface="jax", diff_method=diff_method
        )
        original_result = original_qnode(input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="jax", diff_method=diff_method
        )
        transformed_result = transformed_qnode(input)
        assert qml.math.allclose(original_result, transformed_result)

        original_grad = jax.grad(original_qnode)(input)
        transformed_grad = jax.grad(transformed_qnode)(input)
        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-7)


# Randomly generated set (scipy.unitary_group) of five U(4) operations.
test_u4_unitaries = [
    [
        [
            -0.07016275 - 0.11813399j,
            -0.46476569 - 0.36887134j,
            -0.18641714 + 0.66322739j,
            -0.36131479 - 0.15452521j,
        ],
        [
            -0.21347395 - 0.47461873j,
            0.45781338 - 0.21930172j,
            0.16991759 - 0.12283128j,
            -0.59626019 + 0.26831674j,
        ],
        [
            -0.50100034 + 0.47448914j,
            0.14346598 + 0.41837463j,
            -0.39898589 + 0.01284601j,
            -0.40027516 - 0.09308024j,
        ],
        [
            0.47310527 + 0.10157557j,
            -0.4411336 - 0.00466688j,
            -0.2328747 - 0.51752603j,
            -0.46274728 + 0.18717456j,
        ],
    ],
    [
        [
            -0.41189319 + 0.06007113j,
            0.15316396 + 0.1458654j,
            -0.17064243 + 0.33405919j,
            0.60457809 + 0.52513855j,
        ],
        [
            0.08694733 + 0.64367692j,
            -0.17413963 + 0.61038522j,
            -0.27698195 - 0.31363701j,
            -0.01169698 - 0.0012099j,
        ],
        [
            -0.39488729 - 0.41241129j,
            0.2202738 + 0.19315024j,
            -0.45149927 - 0.30746997j,
            0.15876369 - 0.51435213j,
        ],
        [
            -0.27620095 - 0.05049386j,
            0.47854591 + 0.48737626j,
            0.6201202 - 0.03549654j,
            -0.25966725 + 0.03722163j,
        ],
    ],
    [
        [
            -0.34812515 + 0.37427723j,
            -0.11092236 + 0.47565307j,
            0.13724183 - 0.29504039j,
            -0.56249794 - 0.27908375j,
        ],
        [
            -0.14408107 + 0.1693212j,
            -0.20483797 - 0.10707915j,
            -0.85376825 + 0.3175112j,
            -0.1054503 - 0.23726165j,
        ],
        [
            0.03106625 - 0.04236712j,
            0.78292822 + 0.03053768j,
            0.01814738 + 0.16830002j,
            0.03513342 - 0.59451003j,
        ],
        [
            -0.00087219 - 0.82857442j,
            0.04840206 + 0.30294214j,
            -0.1884474 + 0.01468393j,
            -0.41353861 + 0.11227088j,
        ],
    ],
    [
        [
            -0.05780187 - 0.06284269j,
            0.13559069 + 0.19399748j,
            0.12381697 + 0.01612151j,
            0.71416466 - 0.64114599j,
        ],
        [
            -0.31103029 - 0.06658675j,
            -0.50183231 + 0.49812898j,
            -0.58061141 - 0.20451914j,
            -0.07379796 - 0.12030957j,
        ],
        [
            0.47241806 - 0.79298028j,
            0.14041019 + 0.06342211j,
            -0.27789855 + 0.19625469j,
            -0.07716877 - 0.05067088j,
        ],
        [
            0.11114093 - 0.16488557j,
            -0.12688073 + 0.63574829j,
            0.68327072 - 0.15122624j,
            -0.21697355 + 0.05813823j,
        ],
    ],
    [
        [
            0.32457875 - 0.36309659j,
            -0.21084067 + 0.48248995j,
            -0.21588245 - 0.42368088j,
            -0.0474262 + 0.50714809j,
        ],
        [
            -0.24688996 - 0.11890225j,
            0.16113004 + 0.1518989j,
            -0.40132738 - 0.28678782j,
            -0.63810805 - 0.4747406j,
        ],
        [
            -0.14474527 - 0.46561401j,
            0.47151308 - 0.32560877j,
            0.51600239 - 0.28298318j,
            -0.18491473 + 0.23103107j,
        ],
        [
            0.42624962 - 0.51795827j,
            -0.17138618 - 0.56213399j,
            -0.36133453 + 0.23168462j,
            -0.0167845 - 0.14191731j,
        ],
    ],
]

dev = qml.device('default.qubit', wires=["a", "b"])

def qfunc_with_qubit_unitary_two_qubit(U):
    qml.Hadamard(wires="a")
    qml.QubitUnitary(U, wires=["a", "b"])
    qml.CNOT(wires=["b", "a"])
    return qml.expval(qml.PauliX(wires="a"))


class TestTwoQubitUnitaryDifferentiability:
    """Tests to ensure the transform is fully differentiable in all interfaces
    when the circuit contains two-qubit QubitUnitary operations."""

    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("U", test_u4_unitaries)
    def test_gradient_unitary_to_rot_two_qubit(self, U, diff_method):
        """Tests differentiability in autograd interface."""

        U = np.array(U, requires_grad=True)

        original_qnode = qml.QNode(qfunc_with_qubit_unitary_two_qubit, device=dev, diff_method=diff_method)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary_two_qubit)
        transformed_qnode = qml.QNode(transformed_qfunc, dev, diff_method=diff_method)

        assert qml.math.allclose(original_qnode(U), transformed_qnode(U))

        original_grad = qml.grad(original_qnode)(U)
        transformed_grad = qml.grad(transformed_qnode)(U)

        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-6)

    @pytest.mark.parametrize("U", test_u4_unitaries)
    def test_gradient_unitary_to_rot_torch_two_qubit(self, U):
        """Tests differentiability in autograd interface."""
        torch = pytest.importorskip("torch")
        
        U = torch.tensor(U, dtype=torch.complex128, requires_grad=True)
        transformed_U = torch.tensor(U, dtype=torch.complex128, requires_grad=True)
        
        original_qnode = qml.QNode(
            qfunc_with_qubit_unitary_two_qubit, dev, interface="torch", diff_method="parameter-shift"
        )
        original_result = original_qnode(U)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary_two_qubit)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="torch", diff_method="parameter-shift"
        )
        transformed_result = transformed_qnode(U)

        assert qml.math.allclose(original_result, transformed_result)

        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(U.grad, transformed_U.grad)

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_tf(self, rot_angles, diff_method):
        """Tests differentiability in tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        def qfunc_with_qubit_unitary(angles):
            z = tf.cast(angles[0], tf.complex128)
            x = tf.cast(angles[1], tf.complex128)

            c = tf.cos(x / 2)
            s = tf.sin(x / 2) * 1j

            Z_mat = tf.convert_to_tensor([[tf.exp(-1j * z / 2), 0.0], [0.0, tf.exp(1j * z / 2)]])
            X_mat = tf.convert_to_tensor([[c, -s], [-s, c]])

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        original_qnode = qml.QNode(
            original_qfunc_for_grad, dev, interface="tf", diff_method=diff_method
        )
        original_input = tf.Variable(rot_angles, dtype=tf.float64)
        original_result = original_qnode(original_input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="tf", diff_method=diff_method
        )
        transformed_input = tf.Variable(rot_angles, dtype=tf.float64)
        transformed_result = transformed_qnode(transformed_input)

        assert qml.math.allclose(original_result, transformed_result)

        with tf.GradientTape() as tape:
            loss = original_qnode(original_input)
        original_grad = tape.gradient(loss, original_input)

        with tf.GradientTape() as tape:
            loss = transformed_qnode(transformed_input)

        transformed_grad = tape.gradient(loss, transformed_input)

        # For 64bit values, need to slightly increase the tolerance threshold
        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-7)

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_jax(self, rot_angles, diff_method):
        """Tests differentiability in jax interface."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        def qfunc_with_qubit_unitary(angles):
            z = angles[0]
            x = angles[1]

            Z_mat = jnp.array([[jnp.exp(-1j * z / 2), 0.0], [0.0, jnp.exp(1j * z / 2)]])

            c = jnp.cos(x / 2)
            s = jnp.sin(x / 2) * 1j
            X_mat = jnp.array([[c, -s], [-s, c]])

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        # Setting the dtype to complex64 causes the gradients to be complex...
        input = jnp.array(rot_angles, dtype=jnp.float64)

        original_qnode = qml.QNode(
            original_qfunc_for_grad, dev, interface="jax", diff_method=diff_method
        )
        original_result = original_qnode(input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="jax", diff_method=diff_method
        )
        transformed_result = transformed_qnode(input)
        assert qml.math.allclose(original_result, transformed_result)

        original_grad = jax.grad(original_qnode)(input)
        transformed_grad = jax.grad(transformed_qnode)(input)
        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-7)
