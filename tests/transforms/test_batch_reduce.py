# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the batch reduce transform decorator.
"""


class TestBatchReduceTransforms:
    """Tests for the qnode_transform decorator"""

    def test_single_tape_transform(self):
        """Test that an unparametrized single tape transform can be applied
        to QNode"""

        @qml.single_tape_transform
        def tape_transform(tape):
            for op in tape.operations:
                if op.name == "CRX":
                    qml.CRY(op.parameters[0] ** 2, wires=op.wires)
                else:
                    op.queue()

        my_transform = qml.batch_reduce(tape_transform)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        new_qnode = my_transform(qnode)
        x = -0.543

        res = new_qnode(x)
        expected = np.cos(x ** 2 / 2) ** 2
        assert np.allclose(res, expected)

    def test_unparametrized_transform(self):
        """Test that an unparametrized transform can be applied
        to QNode"""

        @qml.single_tape_transform
        def tape_transform(tape):
            for op in tape.operations:
                if op.name == "CRX":
                    qml.CRY(op.parameters[0] ** 2, wires=op.wires)
                else:
                    op.queue()

        def my_transform(qnode):
            return [tape_transform(qnode.qtape)], lambda res: qml.math.sqrt(res)

        my_transform = qml.batch_reduce(my_transform)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        new_qnode = my_transform(qnode)
        x = -0.543

        res = new_qnode(x)
        expected = np.abs(np.cos(x ** 2 / 2))
        assert np.allclose(res, expected)

    def test_unparametrized_transform_decorator(self):
        """Test that an unparametrized transform can be applied
        to a QNode via a decorator"""

        @qml.single_tape_transform
        def tape_transform(tape):
            for op in tape.operations:
                if op.name == "CRX":
                    qml.CRY(op.parameters[0] ** 2, wires=op.wires)
                else:
                    op.queue()

        @qml.batch_reduce
        def my_transform(qnode):
            return [tape_transform(qnode.qtape)], lambda res: qml.math.sqrt(res)

        dev = qml.device("default.qubit", wires=2)

        @my_transform
        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        x = -0.543

        res = qnode(x)
        expected = np.abs(np.cos(x ** 2 / 2))
        assert np.allclose(res, expected)

    def test_parametrized_transform(self):
        """Test that a parametrized transform can be applied
        to a QNode"""

        @qml.single_tape_transform
        def tape_transform(tape, a, b):
            for op in tape.operations + tape.measurements:
                if op.name == "CRX":
                    wires = op.wires
                    param = op.parameters[0]
                    qml.RX(a * param, wires=wires[1])
                    qml.RY(qml.math.sum(b) * qml.math.sqrt(param), wires=wires[1])
                    qml.CZ(wires=[wires[1], wires[0]])
                else:
                    op.queue()

        @qml.batch_reduce
        def my_transform(qnode, a, b):
            tape1 = tape_transform(qnode.qtape, a, b)
            tape2 = tape_transform(qnode.qtape, b[0], -a)
            return [tape1, tape2], lambda res: qml.math.squeeze(res[1] - res[0])

        a = 0.1
        b = np.array([0.2, 0.3])

        dev = qml.device("default.qubit", wires=2)

        @my_transform(a, b)
        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        x = 0.543

        res = qnode(x)
        expected = -np.cos(np.sum(b) * np.sqrt(x)) * np.cos(a * x) + np.cos(
            a * np.sqrt(x)
        ) * np.cos(b[0] * x)
        assert np.allclose(res, expected)


class TestQNodeTransformGradients:
    """Tests for the qnode_transform decorator differentiability"""

    @staticmethod
    @qml.single_tape_transform
    def tape_transform(tape, a, b):
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
    @qml.batch_reduce
    def my_transform(qnode, a, b):
        tape1 = TestQNodeTransformGradients.tape_transform(qnode.qtape, a, b)
        tape2 = TestQNodeTransformGradients.tape_transform(qnode.qtape, b[0], -a)
        return [tape1, tape2], lambda res: qml.math.squeeze(res[1] - res[0])

    @staticmethod
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.CRX(x, wires=[0, 1])
        return qml.expval(qml.PauliZ(1))

    @staticmethod
    def expval(x, a, b):
        """Analytic expectation value of the above circuit qfunc"""
        return -np.cos(np.sum(b) * np.sqrt(x)) * np.cos(a * x) + np.cos(a * np.sqrt(x)) * np.cos(
            b[0] * x
        )

    def test_differentiable_QNode_autograd(self):
        """Test that a QNode transform is differentiable when using
        autograd"""
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(self.circuit, dev, interface="autograd")

        a = np.array(0.5, requires_grad=True)
        b = np.array([0.1, 0.2], requires_grad=True)
        x = np.array(0.543, requires_grad=True)

        def cost_fn(x, a, b):
            new_qnode = self.my_transform(a, b)(qnode)
            return new_qnode(x)

        res = cost_fn(x, a, b)
        assert np.allclose(res, self.expval(x, a, b))

        grad = qml.grad(cost_fn)(x, a, b)
        expected = qml.grad(self.expval)(x, a, b)
        assert all(np.allclose(g, e) for g, e in zip(grad, expected))

    def test_differentiable_QNode_tf(self):
        """Test that a QNode transform is differentiable when using
        TensorFlow"""
        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(self.circuit, dev, interface="tf")

        a = tf.Variable(0.5, dtype=tf.float64)
        b = tf.Variable([0.1, 0.2], dtype=tf.float64)
        x = tf.Variable(0.543, dtype=tf.float64)

        with tf.GradientTape() as tape:
            new_qnode = self.my_transform(a, b)(qnode)
            res = new_qnode(x)

        assert np.allclose(res, self.expval(x, a, b))

        grad = tape.gradient(res, [x, a, b])
        expected = qml.grad(self.expval)(x.numpy(), a.numpy(), b.numpy())
        assert all(np.allclose(g, e) for g, e in zip(grad, expected))

    def test_differentiable_QNode_torch(self):
        """Test that a QNode transform is differentiable when using
        TensorFlow"""
        torch = pytest.importorskip("torch")
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(self.circuit, dev, interface="torch")

        a = torch.tensor(0.5, requires_grad=True)
        b = torch.tensor([0.1, 0.2], requires_grad=True)
        x = torch.tensor(0.543, requires_grad=True)

        new_qnode = self.my_transform(a, b)(qnode)
        res = new_qnode(x)
        expected = self.expval(x.detach().numpy(), a.detach().numpy(), b.detach().numpy())
        assert np.allclose(res.detach().numpy(), expected)

        res.backward()
        expected = qml.grad(self.expval)(x.detach().numpy(), a.detach().numpy(), b.detach().numpy())
        assert np.allclose(x.grad, expected[0])
        assert np.allclose(a.grad, expected[1])
        assert np.allclose(b.grad, expected[2])
