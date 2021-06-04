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
Unit tests for the qfunc transform decorators.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np


class TestSingleTapeTransform:
    """Tests for the single_tape_transform decorator"""

    def test_error_invalid_callable(self):
        """Test that an error is raised if the transform
        is applied to an invalid function"""

        with pytest.raises(ValueError, match="does not appear to be a valid Python function"):
            qml.single_tape_transform(5)

    def test_parametrized_transform(self):
        """Test that a parametrized transform can be applied
        to a tape"""

        @qml.single_tape_transform
        def my_transform(tape, a, b):
            for op in tape.operations + tape.measurements:
                if op.name == "CRX":
                    wires = op.wires
                    param = op.parameters[0]
                    qml.RX(a, wires=wires[1])
                    qml.RY(qml.math.sum(b) * param / 2, wires=wires[1])
                    qml.CZ(wires=[wires[1], wires[0]])
                else:
                    op.queue()

        a = 0.1
        b = np.array([0.2, 0.3])
        x = 0.543

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])

        ops = my_transform(tape, a, b).operations
        assert len(ops) == 4
        assert ops[0].name == "Hadamard"

        assert ops[1].name == "RX"
        assert ops[1].parameters == [a]

        assert ops[2].name == "RY"
        assert ops[2].parameters == [np.sum(b) * x / 2]

        assert ops[3].name == "CZ"


class TestQFuncTransforms:
    """Tests for the qfunc_transform decorator"""

    def test_error_invalid_transform_callable(self):
        """Test that an error is raised if the transform
        is applied to an invalid function"""

        with pytest.raises(
            ValueError, match="can only be applied to single tape transform functions"
        ):
            qml.qfunc_transform(5)

    def test_error_invalid_qfunc(self):
        """Test that an error is raised if the transform
        is applied to an invalid function"""

        def identity_transform(tape):
            for op in tape.operations + tape.measurements:
                op.queue()

        my_transform = qml.qfunc_transform(identity_transform)

        with pytest.raises(ValueError, match="does not appear to be a valid Python function"):
            my_transform(5)

    def test_unparametrized_transform(self):
        """Test that an unparametrized transform can be applied
        to a quantum function"""

        def my_transform(tape):
            for op in tape.operations + tape.measurements:
                if op.name == "CRX":
                    wires = op.wires
                    param = op.parameters[0]
                    qml.RX(param, wires=wires[1])
                    qml.RY(param / 2, wires=wires[1])
                    qml.CZ(wires=[wires[1], wires[0]])
                else:
                    op.queue()

        my_transform = qml.qfunc_transform(my_transform)

        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])

        new_qfunc = my_transform(qfunc)
        x = 0.543

        ops = qml.transforms.make_tape(new_qfunc)(x).operations
        assert len(ops) == 4
        assert ops[0].name == "Hadamard"

        assert ops[1].name == "RX"
        assert ops[1].parameters == [x]

        assert ops[2].name == "RY"
        assert ops[2].parameters == [x / 2]

        assert ops[3].name == "CZ"

    def test_unparametrized_transform_decorator(self):
        """Test that an unparametrized transform can be applied
        to a quantum function via a decorator"""

        @qml.qfunc_transform
        def my_transform(tape):
            for op in tape.operations + tape.measurements:
                if op.name == "CRX":
                    wires = op.wires
                    param = op.parameters[0]
                    qml.RX(param, wires=wires[1])
                    qml.RY(param / 2, wires=wires[1])
                    qml.CZ(wires=[wires[1], wires[0]])
                else:
                    op.queue()

        @my_transform
        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])

        x = 0.543
        ops = qml.transforms.make_tape(qfunc)(x).operations
        assert len(ops) == 4
        assert ops[0].name == "Hadamard"

        assert ops[1].name == "RX"
        assert ops[1].parameters == [x]

        assert ops[2].name == "RY"
        assert ops[2].parameters == [x / 2]

        assert ops[3].name == "CZ"

    def test_parametrized_transform(self):
        """Test that a parametrized transform can be applied
        to a quantum function"""

        def my_transform(tape, a, b):
            for op in tape.operations + tape.measurements:
                if op.name == "CRX":
                    wires = op.wires
                    param = op.parameters[0]
                    qml.RX(a, wires=wires[1])
                    qml.RY(qml.math.sum(b) * param / 2, wires=wires[1])
                    qml.CZ(wires=[wires[1], wires[0]])
                else:
                    op.queue()

        my_transform = qml.qfunc_transform(my_transform)

        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])

        a = 0.1
        b = np.array([0.2, 0.3])
        x = 0.543
        new_qfunc = my_transform(a, b)(qfunc)

        ops = qml.transforms.make_tape(new_qfunc)(x).operations
        assert len(ops) == 4
        assert ops[0].name == "Hadamard"

        assert ops[1].name == "RX"
        assert ops[1].parameters == [a]

        assert ops[2].name == "RY"
        assert ops[2].parameters == [np.sum(b) * x / 2]

        assert ops[3].name == "CZ"

    def test_parametrized_transform_decorator(self):
        """Test that a parametrized transform can be applied
        to a quantum function via a decorator"""

        @qml.qfunc_transform
        def my_transform(tape, a, b):
            for op in tape.operations + tape.measurements:
                if op.name == "CRX":
                    wires = op.wires
                    param = op.parameters[0]
                    qml.RX(a, wires=wires[1])
                    qml.RY(qml.math.sum(b) * param / 2, wires=wires[1])
                    qml.CZ(wires=[wires[1], wires[0]])
                else:
                    op.queue()

        a = 0.1
        b = np.array([0.2, 0.3])
        x = 0.543

        @my_transform(a, b)
        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])

        ops = qml.transforms.make_tape(qfunc)(x).operations
        assert len(ops) == 4
        assert ops[0].name == "Hadamard"

        assert ops[1].name == "RX"
        assert ops[1].parameters == [a]

        assert ops[2].name == "RY"
        assert ops[2].parameters == [np.sum(b) * x / 2]

        assert ops[3].name == "CZ"

    def test_nested_transforms(self):
        """Test that nesting multiple transforms works as expected"""

        @qml.qfunc_transform
        def convert_cnots(tape):
            for op in tape.operations + tape.measurements:
                if op.name == "CNOT":
                    wires = op.wires
                    qml.Hadamard(wires=wires[0])
                    qml.CZ(wires=[wires[0], wires[1]])
                else:
                    op.queue()

        @qml.qfunc_transform
        def expand_hadamards(tape, x):
            for op in tape.operations + tape.measurements:
                if op.name == "Hadamard":
                    qml.RZ(x, wires=op.wires)
                else:
                    op.queue()

        x = 0.5

        @expand_hadamards(x)
        @convert_cnots
        def ansatz():
            qml.CNOT(wires=[0, 1])

        ops = qml.transforms.make_tape(ansatz)().operations
        assert len(ops) == 2
        assert ops[0].name == "RZ"
        assert ops[0].parameters == [x]
        assert ops[1].name == "CZ"


############################################
# Test transform, ansatz, and qfunc function


@pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
class TestQFuncTransformGradients:
    """Tests for the qfunc_transform decorator differentiability"""

    @staticmethod
    @qml.qfunc_transform
    def my_transform(tape, a, b):
        """Test transform"""
        for op in tape.operations + tape.measurements:
            if op.name == "CRX":
                wires = op.wires
                param = op.parameters[0]
                qml.RX(a * param, wires=wires[1])
                qml.RY(qml.math.sum(b) * qml.math.sqrt(param), wires=wires[1])
                qml.CZ(wires=[wires[1], wires[0]])
            else:
                op.queue()

    @staticmethod
    def ansatz(x):
        """Test ansatz"""
        qml.Hadamard(wires=0)
        qml.CRX(x, wires=[0, 1])

    @staticmethod
    def circuit(param, *transform_weights):
        """Test QFunc"""
        qml.RX(0.1, wires=0)
        TestQFuncTransformGradients.my_transform(*transform_weights)(
            TestQFuncTransformGradients.ansatz
        )(param)
        return qml.expval(qml.PauliZ(1))

    @staticmethod
    def expval(x, a, b):
        """Analytic expectation value of the above circuit qfunc"""
        return np.cos(np.sum(b) * np.sqrt(x)) * np.cos(a * x)

    def test_differentiable_qfunc_autograd(self, diff_method):
        """Test that a qfunc transform is differentiable when using
        autograd"""
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(self.circuit, dev, interface="autograd", diff_method=diff_method)

        a = np.array(0.5, requires_grad=True)
        b = np.array([0.1, 0.2], requires_grad=True)
        x = np.array(0.543, requires_grad=True)

        res = qnode(x, a, b)
        assert np.allclose(res, self.expval(x, a, b))

        grad = qml.grad(qnode)(x, a, b)
        expected = qml.grad(self.expval)(x, a, b)
        assert all(np.allclose(g, e) for g, e in zip(grad, expected))

    def test_differentiable_qfunc_tf(self, diff_method):
        """Test that a qfunc transform is differentiable when using
        TensorFlow"""
        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(self.circuit, dev, interface="tf", diff_method=diff_method)

        a = tf.Variable(0.5, dtype=tf.float64)
        b = tf.Variable([0.1, 0.2], dtype=tf.float64)
        x = tf.Variable(0.543, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = qnode(x, a, b)

        assert np.allclose(res, self.expval(x, a, b))

        grad = tape.gradient(res, [x, a, b])
        expected = qml.grad(self.expval)(x.numpy(), a.numpy(), b.numpy())
        assert all(np.allclose(g, e) for g, e in zip(grad, expected))

    def test_differentiable_qfunc_torch(self, diff_method):
        """Test that a qfunc transform is differentiable when using
        PyTorch"""
        if diff_method == "backprop":
            pytest.skip("Does not support backprop mode")

        torch = pytest.importorskip("torch")
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(self.circuit, dev, interface="torch", diff_method=diff_method)

        a = torch.tensor(0.5, requires_grad=True)
        b = torch.tensor([0.1, 0.2], requires_grad=True)
        x = torch.tensor(0.543, requires_grad=True)

        res = qnode(x, a, b)
        expected = self.expval(x.detach().numpy(), a.detach().numpy(), b.detach().numpy())
        assert np.allclose(res.detach().numpy(), expected)

        res.backward()
        expected = qml.grad(self.expval)(x.detach().numpy(), a.detach().numpy(), b.detach().numpy())
        assert np.allclose(x.grad, expected[0])
        assert np.allclose(a.grad, expected[1])
        assert np.allclose(b.grad, expected[2])

    def test_differentiable_qfunc_jax(self, diff_method):
        """Test that a qfunc transform is differentiable when using
        jax"""
        jax = pytest.importorskip("jax")
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(self.circuit, dev, interface="jax", diff_method=diff_method)

        a = jax.numpy.array(0.5)
        b = jax.numpy.array([0.1, 0.2])
        x = jax.numpy.array(0.543)

        res = qnode(x, a, b)
        assert np.allclose(res, self.expval(x, a, b))

        grad = jax.grad(qnode, argnums=[0, 1, 2])(x, a, b)
        expected = qml.grad(self.expval)(np.array(x), np.array(a), np.array(b))
        print(grad, expected)
        assert all(np.allclose(g, e) for g, e in zip(grad, expected))
