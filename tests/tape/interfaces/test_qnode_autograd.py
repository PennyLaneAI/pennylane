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
"""Unit tests for the autograd interface"""
import pytest
from pennylane import numpy as np

import pennylane as qml
from pennylane.tape import JacobianTape, qnode, QNode, QubitParamShiftTape


@pytest.mark.parametrize(
    "dev_name,diff_method", [
        ["default.qubit", "finite-diff"],
        ["default.qubit", "parameter-shift"],
        ["default.qubit", "backprop"],
        ["default.qubit", "adjoint"]
    ],
)
class TestQNode:
    """Same tests as above, but this time via the QNode interface!"""

    def test_nondiff_param_unwrapping(self, dev_name, diff_method, mocker):
        """Test that non-differentiable parameters are correctly unwrapped
        to NumPy ndarrays or floats (if 0-dimensional)"""
        if diff_method != "parameter-shift":
            pytest.skip("Test only supports parameter-shift")

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="autograd", diff_method="parameter-shift")
        def circuit(x, y):
            qml.RX(x[0], wires=0)
            qml.Rot(*x[1:], wires=0)
            qml.RY(y[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=False)
        y = np.array([0.5], requires_grad=True)

        param_data = []
        def mock_apply(*args, **kwargs):
            for op in args[0]:
                param_data.extend(op.data.copy())

        mocker.patch.object(dev, "apply", side_effect=mock_apply)
        circuit(x, y)
        assert param_data == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert not any(isinstance(p, np.tensor) for p in param_data)

        # test the jacobian works correctly
        param_data = []
        qml.grad(circuit)(x, y)
        assert param_data == [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5 + np.pi/2, 0.1, 0.2, 0.3, 0.4, 0.5 - np.pi/2]
        assert not any(isinstance(p, np.tensor) for p in param_data)

    def test_execution_no_interface(self, dev_name, diff_method):
        """Test execution works without an interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface=None)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)

        res = circuit(a)

        assert circuit.qtape.interface == None

        # without the interface, the QNode simply returns a scalar array
        assert isinstance(res, np.ndarray)
        assert res.shape == tuple()

        # without the interface, the tape is unable to deduce
        # trainable parameters
        assert circuit.qtape.trainable_params == {0, 1}

        # gradients should cause an error
        with pytest.raises(TypeError, match="must be real number, not ArrayBox"):
            qml.grad(circuit)(a)

    def test_execution_with_interface(self, dev_name, diff_method):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)
        circuit(a)

        assert circuit.qtape.interface == "autograd"

        # the tape is able to deduce trainable parameters
        assert circuit.qtape.trainable_params == {0}

        # gradients should work
        grad = qml.grad(circuit)(a)
        assert isinstance(grad, float)
        assert grad.shape == tuple()

    def test_interface_swap(self, dev_name, diff_method, tol):
        """Test that the autograd interface can be applied to a QNode
        with a pre-existing interface"""
        tf = pytest.importorskip("tensorflow", minversion="2.1")

        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.1, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res_tf = circuit(a)

        grad_tf = tape.gradient(res_tf, a)

        # switch to autograd interface
        circuit.to_autograd()

        a = np.array(0.1, requires_grad=True)

        res = circuit(a)
        grad = qml.grad(circuit)(a)

        assert np.allclose(res, res_tf, atol=tol, rtol=0)
        assert np.allclose(grad, grad_tf, atol=tol, rtol=0)

    def test_jacobian(self, dev_name, diff_method, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(JacobianTape, "jacobian")
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        res = circuit(a, b)

        assert circuit.qtape.trainable_params == {0, 1}
        assert res.shape == (2,)

        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(circuit)(a, b)
        expected = [[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        if diff_method == "finite-diff":
            spy.assert_called()
        elif diff_method == "backprop":
            spy.assert_not_called()

    def test_jacobian_no_evaluate(self, dev_name, diff_method, mocker, tol):
        """Test jacobian calculation when no prior circuit evaluation has been performed"""
        spy = mocker.spy(JacobianTape, "jacobian")
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        jac_fn = qml.jacobian(circuit)
        res = jac_fn(a, b)
        expected = [[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        if diff_method == "finite-diff":
            spy.assert_called()
        elif diff_method == "backprop":
            spy.assert_not_called()

        # call the Jacobian with new parameters
        a = np.array(0.6, requires_grad=True)
        b = np.array(0.832, requires_grad=True)

        res = jac_fn(a, b)
        expected = [[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_options(self, dev_name, diff_method, mocker, tol):
        """Test setting jacobian options"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        spy = mocker.spy(JacobianTape, "numeric_pd")

        a = np.array([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="autograd", h=1e-8, order=2)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        qml.jacobian(circuit)(a)

        for args in spy.call_args_list:
            assert args[1]["order"] == 2
            assert args[1]["h"] == 1e-8

    def test_changing_trainability(self, dev_name, diff_method, mocker, tol):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method != "parameter-shift":
            pytest.skip("Test only supports parameter-shift")

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="autograd", diff_method="parameter-shift")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        def loss(a, b):
            return np.sum(circuit(a, b))

        grad_fn = qml.grad(loss)
        spy = mocker.spy(QubitParamShiftTape, "parameter_shift")

        res = grad_fn(a, b)

        # the tape has reported both arguments as trainable
        assert circuit.qtape.trainable_params == {0, 1}

        expected = [-np.sin(a) + np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # The parameter-shift rule has been called for each argument
        assert len(spy.call_args_list) == 2

        # make the second QNode argument a constant
        a = 0.54 # the QNode will treat a scalar as differentiable
        b = np.array(0.8, requires_grad=False)

        spy.call_args_list = []
        res = grad_fn(a, b)

        # the tape has reported only the first argument as trainable
        assert circuit.qtape.trainable_params == {0}

        expected = [-np.sin(a) + np.sin(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # JacobianTape.numeric_pd has been called only once
        assert len(spy.call_args_list) == 1

        # trainability also updates on evaluation
        a = np.array(0.54, requires_grad=False)
        b = np.array(0.8, requires_grad=True)
        circuit(a, b)
        assert circuit.qtape.trainable_params == {1}

    def test_classical_processing(self, dev_name, diff_method, tol):
        """Test classical processing within the quantum tape"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        c = np.array(0.3, requires_grad=True)

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c ** 2 + np.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        res = qml.jacobian(circuit)(a, b, c)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {0, 2}
            tape_params = np.array(circuit.qtape.get_parameters())
            assert np.all(tape_params == [a * c, c + c ** 2 + np.sin(a)])

        assert res.shape == (2,)

    def test_no_trainable_parameters(self, dev_name, diff_method, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = np.array(0.1, requires_grad=False)
        b = np.array(0.2, requires_grad=False)

        res = circuit(a, b)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == set()

        assert res.shape == (2,)
        assert isinstance(res, np.ndarray)

        assert not qml.jacobian(circuit)(a, b)

        def cost(a, b):
            return np.sum(circuit(a, b))

        with pytest.warns(UserWarning, match="Output seems independent of input"):
            grad = qml.grad(cost)(a, b)

        assert grad == tuple()

    def test_matrix_parameter(self, dev_name, diff_method, tol):
        """Test that the autograd interface works correctly
        with a matrix parameter"""
        U = np.array([[0, 1], [1, 0]], requires_grad=False)
        a = np.array(0.1, requires_grad=True)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(U, a)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {1}

        res = qml.grad(circuit)(U, a)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, dev_name, diff_method, mocker, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""
        mock = mocker.patch.object(qml.operation.Operation, "do_check_domain", False)

        class U3(qml.U3):
            def expand(self):
                theta, phi, lam = self.data
                wires = self.wires

                with JacobianTape() as tape:
                    qml.Rot(lam, theta, -lam, wires=wires)
                    qml.PhaseShift(phi + lam, wires=wires)

                return tape

        dev = qml.device(dev_name, wires=1)
        a = np.array(0.1, requires_grad=False)
        p = np.array([0.1, 0.2, 0.3], requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        res = circuit(a, p)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {1, 2, 3, 4}
        elif diff_method == "backprop":
            # For a backprop device, no interface wrapping is performed, and JacobianTape.jacobian()
            # is never called. As a result, JacobianTape.trainable_params is never set --- the ML
            # framework uses its own backprop logic and its own bookkeeping re: trainable parameters.
            assert circuit.qtape.trainable_params == {0, 1, 2, 3, 4}

        assert [i.name for i in circuit.qtape.operations] == ["RX", "Rot", "PhaseShift"]

        if diff_method == "finite-diff":
            assert np.all(circuit.qtape.get_parameters() == [p[2], p[0], -p[2], p[1] + p[2]])
        elif diff_method == "backprop":
            # In backprop mode, all parameters are returned.
            assert np.all(circuit.qtape.get_parameters() == [a, p[2], p[0], -p[2], p[1] + p[2]])

        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.grad(circuit)(a, p)
        expected = np.array(
            [
                np.cos(p[1]) * (np.cos(a) * np.cos(p[0]) - np.sin(a) * np.sin(p[0]) * np.sin(p[2])),
                np.cos(p[1]) * np.cos(p[2]) * np.sin(a)
                - np.sin(p[1])
                * (np.cos(a) * np.sin(p[0]) + np.cos(p[0]) * np.sin(a) * np.sin(p[2])),
                np.sin(a)
                * (np.cos(p[0]) * np.cos(p[1]) * np.cos(p[2]) - np.sin(p[1]) * np.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_probability_differentiation(self, dev_name, diff_method, tol):
        """Tests correct output shape and evaluation for a tape
        with a single prob output"""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        res = qml.jacobian(circuit)(x, y)

        expected = np.array(
            [
                [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2],
                [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_probability_differentiation(self, dev_name, diff_method, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple prob outputs"""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1])

        res = circuit(x, y)

        expected = np.array(
            [
                [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2],
                [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(circuit)(x, y)
        expected = np.array(
            [
                [[-np.sin(x) / 2, 0], [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2]],
                [
                    [np.sin(x) / 2, 0],
                    [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
                ],
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_ragged_differentiation(self, dev_name, diff_method, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])]

        res = circuit(x, y)

        expected = np.array(
            [np.cos(x), (1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(circuit)(x, y)
        expected = np.array(
            [
                [-np.sin(x), 0],
                [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2],
                [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_ragged_differentiation_variance(self, dev_name, diff_method, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and variance outputs"""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return [qml.var(qml.PauliZ(0)), qml.probs(wires=[1])]

        res = circuit(x, y)

        expected = np.array(
            [np.sin(x) ** 2, (1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(circuit)(x, y)
        expected = np.array(
            [
                [2 * np.cos(x) * np.sin(x), 0],
                [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2],
                [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_sampling(self, dev_name, diff_method):
        """Test sampling works as expected"""
        dev = qml.device(dev_name, wires=2, shots=10)

        @qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return [qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))]

        res = circuit()

        assert res.shape == (2, 10)
        assert isinstance(res, np.ndarray)

    def test_gradient_non_differentiable_exception(self, dev_name, diff_method):
        """Test that an exception is raised if non-differentiable data is
        differentiated"""
        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(data1):
            qml.templates.AmplitudeEmbedding(data1, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        grad_fn = qml.grad(circuit, argnum=0)
        data1 = np.array([0, 1, 1, 0], requires_grad=False) / np.sqrt(2)

        with pytest.raises(qml.numpy.NonDifferentiableError, match="is non-differentiable"):
            grad_fn(data1)

    def test_chained_qnodes(self, dev_name, diff_method):
        """Test that the gradient of chained QNodes works without error"""
        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit1(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        @qml.qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit2(data, weights):
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        def cost(weights):
            w1, w2 = weights
            c1 = circuit1(w1)
            c2 = circuit2(c1, w2)
            return np.sum(c2) ** 2

        w1 = qml.init.strong_ent_layers_normal(n_wires=2, n_layers=3)
        w2 = qml.init.strong_ent_layers_normal(n_wires=2, n_layers=4)

        weights = [w1, w2]

        grad_fn = qml.grad(cost)
        res = grad_fn(weights)

        assert len(res) == 2

    def test_chained_gradient_value(self, dev_name, diff_method, tol):
        """Test that the returned gradient value for two chained qubit QNodes
        is correct."""
        dev1 = qml.device(dev_name, wires=3)

        @qml.qnode(dev1, diff_method=diff_method)
        def circuit1(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(2))

        dev2 = qml.device("default.qubit", wires=2)

        @qml.qnode(dev2, diff_method=diff_method)
        def circuit2(data, weights):
            qml.RX(data[0], wires=0)
            qml.RX(data[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(weights[0], wires=0)
            qml.RZ(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0) @ qml.PauliY(1))

        def cost(a, b, c, weights):
            return circuit2(circuit1(a, b, c), weights)

        grad_fn = qml.grad(cost)

        # Set the first parameter of circuit1 as non-differentiable.
        a = np.array(0.4, requires_grad=False)

        # The remaining free parameters are all differentiable.
        b = 0.5
        c = 0.1
        weights = np.array([0.2, 0.3])

        res = grad_fn(a, b, c, weights)

        # Output should have shape [dcost/db, dcost/dc, dcost/dw],
        # where b,c are scalars, and w is a vector of length 2.
        assert len(res) == 3
        assert res[0].shape == tuple() # scalar
        assert res[1].shape == tuple() # scalar
        assert res[2].shape == (2,)    # vector

        cacbsc = np.cos(a)*np.cos(b)*np.sin(c)

        expected = np.array([
            # analytic expression for dcost/db
            -np.cos(a)*np.sin(b)*np.sin(c)*np.cos(cacbsc)*np.sin(weights[0])*np.sin(np.cos(a)),
            # analytic expression for dcost/dc
            np.cos(a)*np.cos(b)*np.cos(c)*np.cos(cacbsc)*np.sin(weights[0])*np.sin(np.cos(a)),
            # analytic expression for dcost/dw[0]
            np.sin(cacbsc)*np.cos(weights[0])*np.sin(np.cos(a)),
            # analytic expression for dcost/dw[1]
            0
        ])

        # np.hstack 'flattens' the ragged gradient array allowing it
        # to be compared with the expected result
        assert np.allclose(np.hstack(res), expected, atol=tol, rtol=0)

        if diff_method != "backprop":
            # Check that the gradient was computed
            # for all parameters in circuit2
            assert circuit2.qtape.trainable_params == {0, 1, 2, 3}

            # Check that the parameter-shift rule was not applied
            # to the first parameter of circuit1.
            assert circuit1.qtape.trainable_params == {1, 2}


def qtransform(qnode, a, framework=np):
    """Transforms every RY(y) gate in a circuit to RX(-a*cos(y))"""

    def construct(self, args, kwargs):
        """New quantum tape construct method, that performs
        the transform on the tape in a define-by-run manner"""

        # the following global variable is defined simply for testing
        # purposes, so that we can easily extract the transformed operations
        # for verification.
        global t_op

        t_op = []

        QNode.construct(self, args, kwargs)

        new_ops = []
        for o in self.qtape.operations:
            # here, we loop through all tape operations, and make
            # the transformation if a RY gate is encountered.
            if isinstance(o, qml.RY):
                t_op.append(qml.RX(-a * framework.cos(o.data[0]), wires=o.wires))
                new_ops.append(t_op[-1])
            else:
                new_ops.append(o)

        self.qtape._ops = new_ops
        self.qtape._update()

    import copy

    new_qnode = copy.deepcopy(qnode)
    new_qnode.construct = construct.__get__(new_qnode, QNode)
    return new_qnode


@pytest.mark.parametrize(
    "dev_name,diff_method",
    [("default.qubit", "finite-diff"), ("default.qubit.autograd", "backprop")],
)
def test_transform(dev_name, diff_method, monkeypatch, tol):
    """Test an example transform"""
    monkeypatch.setattr(qml.operation.Operation, "do_check_domain", False)

    dev = qml.device(dev_name, wires=1)

    @qnode(dev, interface="autograd", diff_method=diff_method)
    def circuit(weights):
        # the following global variables are defined simply for testing
        # purposes, so that we can easily extract the operations for verification.
        global op1, op2
        op1 = qml.RY(weights[0], wires=0)
        op2 = qml.RX(weights[1], wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    weights = np.array([0.32, 0.543], requires_grad=True)
    a = np.array(0.5, requires_grad=True)

    def loss(weights, a):
        # the following global variable is defined simply for testing
        # purposes, so that we can easily extract the transformed QNode
        # for verification.
        global new_circuit

        # transform the circuit QNode with trainable weight 'a'
        new_circuit = qtransform(circuit, a)

        # evaluate the transformed QNode
        res = new_circuit(weights)

        # evaluate the original QNode with pre-processed parameters
        res2 = circuit(np.sin(weights))

        # return the sum of the two QNode evaluations
        return res + res2

    res = loss(weights, a)

    # verify that the transformed QNode has the expected operations
    assert circuit.qtape.operations == [op1, op2]
    # RY(y) gate is transformed to RX(-a*cos(y))
    assert new_circuit.qtape.operations[0] == t_op[0]
    # RX gate is is not transformed
    assert new_circuit.qtape.operations[1].name == op2.name
    assert new_circuit.qtape.operations[1].wires == op2.wires

    # check that the incident gate arguments of both QNode tapes are correct
    assert np.all(np.array(circuit.qtape.get_parameters()) == np.sin(weights))
    assert np.all(
        np.array(new_circuit.qtape.get_parameters()) == [-a * np.cos(weights[0]), weights[1]]
    )

    # verify that the gradient has the correct shape
    grad = qml.grad(loss)(weights, a)
    assert len(grad) == 2
    assert grad[0].shape == weights.shape
    assert grad[1].shape == a.shape

    # compare against the expected values
    assert np.allclose(res, 1.8244501889992706, atol=tol, rtol=0)
    assert np.allclose(grad[0], [-0.26610258, -0.47053553], atol=tol, rtol=0)
    assert np.allclose(grad[1], 0.06486032, atol=tol, rtol=0)
