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
"""Integration tests for using the autograd interface with a QNode"""
# pylint: disable=too-many-arguments,too-few-public-methods
import autograd
import autograd.numpy as anp
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnode

qubit_device_and_diff_method = [
    ["default.qubit.legacy", "finite-diff", False],
    ["default.qubit.legacy", "parameter-shift", False],
    ["default.qubit.legacy", "backprop", True],
    ["default.qubit.legacy", "adjoint", True],
    ["default.qubit.legacy", "adjoint", False],
    ["default.qubit.legacy", "spsa", False],
    ["default.qubit.legacy", "hadamard", False],
]

interface_qubit_device_and_diff_method = [
    ["autograd", "default.qubit.legacy", "finite-diff", False],
    ["autograd", "default.qubit.legacy", "parameter-shift", False],
    ["autograd", "default.qubit.legacy", "backprop", True],
    ["autograd", "default.qubit.legacy", "adjoint", True],
    ["autograd", "default.qubit.legacy", "adjoint", False],
    ["autograd", "default.qubit.legacy", "spsa", False],
    ["autograd", "default.qubit.legacy", "hadamard", False],
    ["auto", "default.qubit.legacy", "finite-diff", False],
    ["auto", "default.qubit.legacy", "parameter-shift", False],
    ["auto", "default.qubit.legacy", "backprop", True],
    ["auto", "default.qubit.legacy", "adjoint", True],
    ["auto", "default.qubit.legacy", "adjoint", False],
    ["auto", "default.qubit.legacy", "spsa", False],
    ["auto", "default.qubit.legacy", "hadamard", False],
]

pytestmark = pytest.mark.autograd

TOL_FOR_SPSA = 1.0
SEED_FOR_SPSA = 32651
H_FOR_SPSA = 0.01


@pytest.mark.parametrize(
    "interface,dev_name,diff_method,grad_on_execution", interface_qubit_device_and_diff_method
)
class TestQNode:
    """Test that using the QNode with Autograd integrates with the PennyLane stack"""

    # pylint: disable=unused-argument

    def test_nondiff_param_unwrapping(
        self, interface, dev_name, diff_method, grad_on_execution, mocker
    ):
        """Test that non-differentiable parameters are correctly unwrapped
        to NumPy ndarrays or floats (if 0-dimensional)"""
        if diff_method != "parameter-shift":
            pytest.skip("Test only supports parameter-shift")

        dev = qml.device("default.qubit.legacy", wires=1)

        @qnode(dev, interface=interface, diff_method=diff_method)
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
                param_data.extend(op.data)

        mocker.patch.object(dev, "apply", side_effect=mock_apply)
        circuit(x, y)
        assert param_data == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert not any(isinstance(p, np.tensor) for p in param_data)

        # test the jacobian works correctly
        param_data = []
        qml.grad(circuit)(x, y)
        assert param_data == [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5 + np.pi / 2,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5 - np.pi / 2,
        ]
        assert not any(isinstance(p, np.tensor) for p in param_data)

    def test_execution_no_interface(self, interface, dev_name, diff_method, grad_on_execution):
        """Test execution works without an interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        num_wires = 1
        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface=None, diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)

        res = circuit(a)

        assert circuit.qtape.interface is None

        # without the interface, the QNode simply returns a scalar array
        assert isinstance(res, np.ndarray)
        assert res.shape == tuple()

        # gradients should cause an error
        with pytest.raises(TypeError, match="must be real number, not ArrayBox"):
            qml.grad(circuit)(a)

    def test_execution_with_interface(self, interface, dev_name, diff_method, grad_on_execution):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        num_wires = 1
        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)
        assert circuit.interface == interface
        # gradients should work
        grad = qml.grad(circuit)(a)

        assert isinstance(grad, float)
        assert grad.shape == tuple()

    def test_jacobian(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test jacobian calculation"""
        num_wires = 2
        kwargs = dict(
            diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        if diff_method == "spsa":
            spsa_kwargs = dict(sampler_rng=np.random.default_rng(SEED_FOR_SPSA), num_directions=10)
            kwargs = {**kwargs, **spsa_kwargs}
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            num_wires = 3

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, **kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        res = circuit(a, b)

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        assert circuit.qtape.trainable_params == [0, 1]
        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(cost)(a, b)
        assert isinstance(res, tuple) and len(res) == 2
        expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (2,)
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == (2,)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_jacobian_no_evaluate(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test jacobian calculation when no prior circuit evaluation has been performed"""
        num_wires = 2
        kwargs = dict(
            diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )

        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            num_wires = 3

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, **kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        jac_fn = qml.jacobian(cost)
        res = jac_fn(a, b)
        expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        # call the Jacobian with new parameters
        a = np.array(0.6, requires_grad=True)
        b = np.array(0.832, requires_grad=True)

        res = jac_fn(a, b)
        expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_jacobian_options(self, interface, dev_name, diff_method, grad_on_execution):
        """Test setting jacobian options"""
        wires = [0]
        if diff_method in ["backprop", "adjoint"]:
            pytest.skip("Test does not support backprop or adjoint method")
        elif diff_method == "finite-diff":
            kwargs = {"h": 1e-8, "approx_order": 2}
        elif diff_method == "parameter-shift":
            kwargs = {"shifts": [(0.1,), (0.2,)]}
        elif diff_method == "hadamard":
            wires = [0, "aux"]
            kwargs = {"aux_wire": qml.wires.Wires("aux"), "device_wires": wires}
        else:
            kwargs = {}

        a = np.array([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit.legacy", wires=wires)

        @qnode(dev, interface=interface, diff_method=diff_method, **kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit(a)

        qml.jacobian(circuit)(a)

    def test_changing_trainability(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method != "parameter-shift":
            pytest.skip("Test only supports parameter-shift")

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device("default.qubit.legacy", wires=2)

        @qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        def loss(a, b):
            return np.sum(autograd.numpy.hstack(circuit(a, b)))

        grad_fn = qml.grad(loss)
        res = grad_fn(a, b)

        # the tape has reported both arguments as trainable
        assert circuit.qtape.trainable_params == [0, 1]

        expected = [-np.sin(a) + np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # make the second QNode argument a constant
        a = np.array(0.54, requires_grad=True)
        b = np.array(0.8, requires_grad=False)

        res = grad_fn(a, b)

        # the tape has reported only the first argument as trainable
        assert circuit.qtape.trainable_params == [0]

        expected = [-np.sin(a) + np.sin(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # trainability also updates on evaluation
        a = np.array(0.54, requires_grad=False)
        b = np.array(0.8, requires_grad=True)
        circuit(a, b)
        assert circuit.qtape.trainable_params == [1]

    def test_classical_processing(self, interface, dev_name, diff_method, grad_on_execution):
        """Test classical processing within the quantum tape"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        c = np.array(0.3, requires_grad=True)

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c**2 + np.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        res = qml.jacobian(circuit)(a, b, c)

        assert circuit.qtape.trainable_params == [0, 2]
        tape_params = np.array(circuit.qtape.get_parameters())
        assert np.all(tape_params == [a * c, c + c**2 + np.sin(a)])

        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

    def test_no_trainable_parameters(self, interface, dev_name, diff_method, grad_on_execution):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        dev = qml.device(dev_name, wires=2)

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = np.array(0.1, requires_grad=False)
        b = np.array(0.2, requires_grad=False)

        res = circuit(a, b)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == []

        assert len(res) == 2
        assert isinstance(res, tuple)

        def cost0(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            assert not qml.jacobian(cost0)(a, b)

        def cost1(a, b):
            return np.sum(circuit(a, b))

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            grad = qml.grad(cost1)(a, b)

        assert grad == tuple()

    def test_matrix_parameter(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test that the autograd interface works correctly
        with a matrix parameter"""
        U = np.array([[0, 1], [1, 0]], requires_grad=False)
        a = np.array(0.1, requires_grad=True)

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(U, a)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == [1]

        res = qml.grad(circuit)(U, a)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_gradient_non_differentiable_exception(
        self, interface, dev_name, diff_method, grad_on_execution
    ):
        """Test that an exception is raised if non-differentiable data is
        differentiated"""
        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(data1):
            qml.templates.AmplitudeEmbedding(data1, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        grad_fn = qml.grad(circuit, argnum=0)
        data1 = np.array([0, 1, 1, 0], requires_grad=False) / np.sqrt(2)

        with pytest.raises(qml.numpy.NonDifferentiableError, match="is non-differentiable"):
            grad_fn(data1)

    def test_differentiable_expand(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test that operation and nested tape expansion
        is differentiable"""
        kwargs = dict(
            diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        if diff_method == "spsa":
            spsa_kwargs = dict(sampler_rng=np.random.default_rng(SEED_FOR_SPSA), num_directions=20)
            kwargs = {**kwargs, **spsa_kwargs}
            tol = TOL_FOR_SPSA

        class U3(qml.U3):
            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        dev = qml.device(dev_name, wires=2)
        a = np.array(0.1, requires_grad=False)
        p = np.array([0.1, 0.2, 0.3], requires_grad=True)

        @qnode(dev, **kwargs)
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        res = circuit(a, p)
        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert isinstance(res, np.ndarray)
        assert res.shape == ()
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.grad(circuit)(a, p)

        assert isinstance(res, np.ndarray)
        assert len(res) == 3

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


class TestShotsIntegration:
    """Test that the QNode correctly changes shot value, and
    remains differentiable."""

    def test_changing_shots(self, mocker, tol):
        """Test that changing shots works on execution"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=None)
        a, b = np.array([0.543, -0.654], requires_grad=True)

        @qnode(dev, diff_method=qml.gradients.param_shift)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        spy = mocker.spy(dev, "sample")

        # execute with device default shots (None)
        res = circuit(a, b)
        assert np.allclose(res, -np.cos(a) * np.sin(b), atol=tol, rtol=0)
        spy.assert_not_called()

        # execute with shots=100
        res = circuit(a, b, shots=100)  # pylint: disable=unexpected-keyword-arg
        spy.assert_called_once()
        assert spy.spy_return.shape == (100,)

        # device state has been unaffected
        assert dev.shots is None
        res = circuit(a, b)
        assert np.allclose(res, -np.cos(a) * np.sin(b), atol=tol, rtol=0)
        spy.assert_called_once()  # same single call performed above

    @pytest.mark.xfail(reason="Param shift and shot vectors.")
    def test_gradient_integration(self):
        """Test that temporarily setting the shots works
        for gradient computations"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=None)
        a, b = np.array([0.543, -0.654], requires_grad=True)

        @qnode(dev, diff_method=qml.gradients.param_shift)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        # TODO: fix the shot vectors issue
        res = qml.jacobian(cost_fn)(a, b, shots=[10000, 10000, 10000])
        assert dev.shots is None
        assert isinstance(res, tuple) and len(res) == 2
        assert all(r.shape == (3,) for r in res)

        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert all(
            np.allclose(np.mean(r, axis=0), e, atol=0.1, rtol=0) for r, e in zip(res, expected)
        )

    def test_update_diff_method(self, mocker):
        """Test that temporarily setting the shots updates the diff method"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=100)
        a, b = np.array([0.543, -0.654], requires_grad=True)

        spy = mocker.spy(qml, "execute")

        @qnode(dev)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        cost_fn(a, b)
        # since we are using finite shots, parameter-shift will
        # be chosen
        assert cost_fn.gradient_fn is qml.gradients.param_shift
        assert spy.call_args[1]["gradient_fn"] is qml.gradients.param_shift

        # if we set the shots to None, backprop can now be used
        cost_fn(a, b, shots=None)  # pylint: disable=unexpected-keyword-arg
        assert spy.call_args[1]["gradient_fn"] == "backprop"
        assert cost_fn.gradient_fn == "backprop"

        cost_fn(a, b)
        assert cost_fn.gradient_fn is qml.gradients.param_shift
        assert spy.call_args[1]["gradient_fn"] is qml.gradients.param_shift


@pytest.mark.parametrize(
    "interface,dev_name,diff_method,grad_on_execution", interface_qubit_device_and_diff_method
)
class TestQubitIntegration:
    """Tests that ensure various qubit circuits integrate correctly"""

    # pylint: disable=unused-argument

    def test_probability_differentiation(
        self, interface, dev_name, diff_method, grad_on_execution, tol
    ):
        """Tests correct output shape and evaluation for a tape
        with a single prob output"""
        kwargs = dict(
            diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")
        elif diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, **kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        res = qml.jacobian(circuit)(x, y)
        assert isinstance(res, tuple) and len(res) == 2

        expected = (
            np.array([-np.sin(x) * np.cos(y) / 2, np.cos(y) * np.sin(x) / 2]),
            np.array([-np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )
        assert all(np.allclose(r, e, atol=tol, rtol=0) for r, e in zip(res, expected))

    def test_multiple_probability_differentiation(
        self, interface, dev_name, diff_method, grad_on_execution, tol
    ):
        """Tests correct output shape and evaluation for a tape
        with multiple prob outputs"""
        kwargs = dict(
            diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")
        elif diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, **kwargs)
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

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        res = qml.jacobian(cost)(x, y)

        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (4,)
        assert res[1].shape == (4,)

        expected = (
            np.array(
                [
                    [
                        -np.sin(x) / 2,
                        np.sin(x) / 2,
                        -np.sin(x) * np.cos(y) / 2,
                        np.sin(x) * np.cos(y) / 2,
                    ],
                ]
            ),
            np.array(
                [
                    [0, 0, -np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2],
                ]
            ),
        )
        assert all(np.allclose(r, e, atol=tol, rtol=0) for r, e in zip(res, expected))

    def test_ragged_differentiation(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        kwargs = dict(
            diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        num_wires = 2
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")
        elif diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, **kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        res = circuit(x, y)
        assert isinstance(res, tuple)
        expected = [np.cos(x), [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]]
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        res = qml.jacobian(cost)(x, y)
        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == (3,)
        assert res[1].shape == (3,)

        expected = (
            np.array([-np.sin(x), -np.sin(x) * np.cos(y) / 2, np.sin(x) * np.cos(y) / 2]),
            np.array([0, -np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_ragged_differentiation_variance(
        self, interface, dev_name, diff_method, grad_on_execution, tol
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and variance outputs"""
        kwargs = dict(
            diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")
        elif diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        dev = qml.device(dev_name, wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(dev, **kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0)), qml.probs(wires=[1])

        res = circuit(x, y)

        expected_var = np.array(np.sin(x) ** 2)
        expected_probs = np.array(
            [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]
        )

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()
        assert np.allclose(res[0], expected_var, atol=tol, rtol=0)

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == (2,)
        assert np.allclose(res[1], expected_probs, atol=tol, rtol=0)

        def cost(x, y):
            return autograd.numpy.hstack(circuit(x, y))

        jac = qml.jacobian(cost)(x, y)
        assert isinstance(res, tuple) and len(res) == 2

        expected = (
            np.array([np.sin(2 * x), -np.sin(x) * np.cos(y) / 2, np.sin(x) * np.cos(y) / 2]),
            np.array([0, -np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], np.ndarray)
        assert jac[0].shape == (3,)
        assert np.allclose(jac[0], expected[0], atol=tol, rtol=0)

        assert isinstance(jac[1], np.ndarray)
        assert jac[1].shape == (3,)
        assert np.allclose(jac[1], expected[1], atol=tol, rtol=0)

    def test_chained_qnodes(self, interface, dev_name, diff_method, grad_on_execution):
        """Test that the gradient of chained QNodes works without error"""
        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        class Template(qml.templates.StronglyEntanglingLayers):
            def decomposition(self):
                return [qml.templates.StronglyEntanglingLayers(*self.parameters, self.wires)]

        @qnode(dev, interface=interface, diff_method=diff_method)
        def circuit1(weights):
            Template(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        @qnode(dev, interface=interface, diff_method=diff_method)
        def circuit2(data, weights):
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            Template(weights, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        def cost(w1, w2):
            c1 = circuit1(w1)
            c2 = circuit2(c1, w2)
            return np.sum(c2) ** 2

        w1 = qml.templates.StronglyEntanglingLayers.shape(n_wires=2, n_layers=3)
        w2 = qml.templates.StronglyEntanglingLayers.shape(n_wires=2, n_layers=4)

        weights = [
            np.random.random(w1, requires_grad=True),
            np.random.random(w2, requires_grad=True),
        ]

        grad_fn = qml.grad(cost)
        res = grad_fn(*weights)

        assert len(res) == 2

    def test_chained_gradient_value(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test that the returned gradient value for two chained qubit QNodes
        is correct."""
        kwargs = dict(interface=interface, diff_method=diff_method)
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA
        num_wires = 3

        if diff_method == "hadamard":
            num_wires = 4

        dev1 = qml.device(dev_name, wires=num_wires)

        @qnode(dev1, **kwargs)
        def circuit1(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(2))

        dev2 = qml.device("default.qubit.legacy", wires=num_wires)

        @qnode(dev2, interface=interface, diff_method=diff_method)
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
        b = np.array(0.5, requires_grad=True)
        c = np.array(0.1, requires_grad=True)
        weights = np.array([0.2, 0.3], requires_grad=True)

        res = grad_fn(a, b, c, weights)

        # Output should have shape [dcost/db, dcost/dc, dcost/dw],
        # where b,c are scalars, and w is a vector of length 2.
        assert len(res) == 3
        assert res[0].shape == tuple()  # scalar
        assert res[1].shape == tuple()  # scalar
        assert res[2].shape == (2,)  # vector

        cacbsc = np.cos(a) * np.cos(b) * np.sin(c)

        expected = np.array(
            [
                # analytic expression for dcost/db
                -np.cos(a)
                * np.sin(b)
                * np.sin(c)
                * np.cos(cacbsc)
                * np.sin(weights[0])
                * np.sin(np.cos(a)),
                # analytic expression for dcost/dc
                np.cos(a)
                * np.cos(b)
                * np.cos(c)
                * np.cos(cacbsc)
                * np.sin(weights[0])
                * np.sin(np.cos(a)),
                # analytic expression for dcost/dw[0]
                np.sin(cacbsc) * np.cos(weights[0]) * np.sin(np.cos(a)),
                # analytic expression for dcost/dw[1]
                0,
            ]
        )

        # np.hstack 'flattens' the ragged gradient array allowing it
        # to be compared with the expected result
        assert np.allclose(np.hstack(res), expected, atol=tol, rtol=0)

        if diff_method != "backprop":
            # Check that the gradient was computed
            # for all parameters in circuit2
            assert circuit2.qtape.trainable_params == [0, 1, 2, 3]

            # Check that the parameter-shift rule was not applied
            # to the first parameter of circuit1.
            assert circuit1.qtape.trainable_params == [1, 2]

    def test_second_derivative(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test second derivative calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array([1.0, 2.0], requires_grad=True)
        res = circuit(x)
        g = qml.grad(circuit)(x)
        g2 = qml.grad(lambda x: np.sum(qml.grad(circuit)(x)))(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_g2 = [
            -np.cos(a) * np.cos(b) + np.sin(a) * np.sin(b),
            np.sin(a) * np.sin(b) - np.cos(a) * np.cos(b),
        ]

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(g2, expected_g2, atol=tol, rtol=0)

    def test_hessian(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array([1.0, 2.0], requires_grad=True)
        res = circuit(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)

        assert isinstance(res, np.ndarray)
        assert res.shape == ()
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        g = grad_fn(x)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]

        assert isinstance(g, np.ndarray)
        assert g.shape == (2,)
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        hess = qml.jacobian(grad_fn)(x)

        expected_hess = [
            [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
            [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
        ]

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (2, 2)

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_unused_parameter(
        self, interface, dev_name, diff_method, grad_on_execution, tol
    ):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array([1.0, 2.0], requires_grad=True)
        res = circuit(x)

        a, _ = x

        expected_res = np.cos(a)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)

        hess = qml.jacobian(grad_fn)(x)

        expected_hess = [
            [-np.cos(a), 0],
            [0, 0],
        ]

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test hessian calculation of a vector valued QNode"""

        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.probs(wires=0)

        x = np.array([1.0, 2.0], requires_grad=True)
        res = circuit(x)

        a, b = x

        expected_res = [0.5 + 0.5 * np.cos(a) * np.cos(b), 0.5 - 0.5 * np.cos(a) * np.cos(b)]

        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        jac_fn = qml.jacobian(circuit)
        jac = jac_fn(x)

        expected_res = [
            [-0.5 * np.sin(a) * np.cos(b), -0.5 * np.cos(a) * np.sin(b)],
            [0.5 * np.sin(a) * np.cos(b), 0.5 * np.cos(a) * np.sin(b)],
        ]

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (2, 2)
        assert np.allclose(jac, expected_res, atol=tol, rtol=0)

        hess = qml.jacobian(jac_fn)(x)

        expected_hess = [
            [
                [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.sin(a) * np.sin(b)],
                [0.5 * np.sin(a) * np.sin(b), -0.5 * np.cos(a) * np.cos(b)],
            ],
            [
                [0.5 * np.cos(a) * np.cos(b), -0.5 * np.sin(a) * np.sin(b)],
                [-0.5 * np.sin(a) * np.sin(b), 0.5 * np.cos(a) * np.cos(b)],
            ],
        ]

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (2, 2, 2)

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued_postprocessing(
        self, interface, dev_name, diff_method, grad_on_execution, tol
    ):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
        )
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))

        def cost_fn(x):
            return x @ autograd.numpy.hstack(circuit(x))

        x = np.array([0.76, -0.87], requires_grad=True)
        res = cost_fn(x)

        a, b = x

        expected_res = x @ [np.cos(a) * np.cos(b), np.cos(a) * np.cos(b)]

        assert isinstance(res, np.ndarray)
        assert res.shape == ()
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        hess = qml.jacobian(qml.grad(cost_fn))(x)

        expected_hess = [
            [
                -(np.cos(b) * ((a + b) * np.cos(a) + 2 * np.sin(a))),
                -(np.cos(b) * np.sin(a)) + (-np.cos(a) + (a + b) * np.sin(a)) * np.sin(b),
            ],
            [
                -(np.cos(b) * np.sin(a)) + (-np.cos(a) + (a + b) * np.sin(a)) * np.sin(b),
                -(np.cos(a) * ((a + b) * np.cos(b) + 2 * np.sin(b))),
            ],
        ]

        assert hess.shape == (2, 2)

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued_separate_args(
        self, interface, dev_name, diff_method, grad_on_execution, tol
    ):
        """Test hessian calculation of a vector valued QNode that has separate input arguments"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=0)

        a = np.array(1.0, requires_grad=True)
        b = np.array(2.0, requires_grad=True)
        res = circuit(a, b)

        expected_res = [0.5 + 0.5 * np.cos(a) * np.cos(b), 0.5 - 0.5 * np.cos(a) * np.cos(b)]
        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        jac_fn = qml.jacobian(circuit)
        g = jac_fn(a, b)
        assert isinstance(g, tuple) and len(g) == 2

        expected_g = (
            [-0.5 * np.sin(a) * np.cos(b), 0.5 * np.sin(a) * np.cos(b)],
            [-0.5 * np.cos(a) * np.sin(b), 0.5 * np.cos(a) * np.sin(b)],
        )
        assert g[0].shape == (2,)
        assert np.allclose(g[0], expected_g[0], atol=tol, rtol=0)

        assert g[1].shape == (2,)
        assert np.allclose(g[1], expected_g[1], atol=tol, rtol=0)

        jac_fn_a = lambda *args: jac_fn(*args)[0]
        jac_fn_b = lambda *args: jac_fn(*args)[1]
        hess_a = qml.jacobian(jac_fn_a)(a, b)
        hess_b = qml.jacobian(jac_fn_b)(a, b)
        assert isinstance(hess_a, tuple) and len(hess_a) == 2
        assert isinstance(hess_b, tuple) and len(hess_b) == 2

        exp_hess_a = (
            [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.cos(a) * np.cos(b)],
            [0.5 * np.sin(a) * np.sin(b), -0.5 * np.sin(a) * np.sin(b)],
        )
        exp_hess_b = (
            [0.5 * np.sin(a) * np.sin(b), -0.5 * np.sin(a) * np.sin(b)],
            [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.cos(a) * np.cos(b)],
        )

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        for hess, exp_hess in zip([hess_a, hess_b], [exp_hess_a, exp_hess_b]):
            assert np.allclose(hess[0], exp_hess[0], atol=tol, rtol=0)
            assert np.allclose(hess[1], exp_hess[1], atol=tol, rtol=0)

    def test_hessian_ragged(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test hessian calculation of a ragged QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=2)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=1)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=1)

        x = np.array([1.0, 2.0], requires_grad=True)

        a, b = x

        cos_prod = np.cos(a) * np.cos(b)
        expected_res = (cos_prod, [0.5 + 0.5 * cos_prod, 0.5 - 0.5 * cos_prod])
        res = circuit(x)
        assert all(qml.math.allclose(r, e) for r, e in zip(res, expected_res))

        def cost_fn(x):
            return autograd.numpy.hstack(circuit(x))

        jac_fn = qml.jacobian(cost_fn)

        hess = qml.jacobian(jac_fn)(x)
        expected_hess = [
            [
                [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
                [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
            ],
            [
                [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.sin(a) * np.sin(b)],
                [0.5 * np.sin(a) * np.sin(b), -0.5 * np.cos(a) * np.cos(b)],
            ],
            [
                [0.5 * np.cos(a) * np.cos(b), -0.5 * np.sin(a) * np.sin(b)],
                [-0.5 * np.sin(a) * np.sin(b), 0.5 * np.cos(a) * np.cos(b)],
            ],
        ]

        if diff_method in {"finite-diff"}:
            tol = 10e-2

        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_state(self, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test that the state can be returned and differentiated"""
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support states")

        dev = qml.device(dev_name, wires=2)

        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.state()

        def cost_fn(x, y):
            res = circuit(x, y)
            assert res.dtype is np.dtype("complex128")
            probs = np.abs(res) ** 2
            return probs[0] + probs[2]

        res = cost_fn(x, y)

        if diff_method not in {"backprop"}:
            pytest.skip("Test only supports backprop")

        res = qml.jacobian(cost_fn)(x, y)
        expected = np.array([-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2])
        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()
        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector(self, state, interface, dev_name, diff_method, grad_on_execution, tol):
        """Test that the variance of a projector is correctly returned"""
        kwargs = dict(
            diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support projectors")
        elif diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        dev = qml.device(dev_name, wires=2)
        P = np.array(state, requires_grad=False)
        x, y = np.array([0.765, -0.654], requires_grad=True)

        @qnode(dev, **kwargs)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Projector(P, wires=0) @ qml.PauliX(1))

        res = circuit(x, y)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert isinstance(res, np.ndarray)
        assert res.shape == ()
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac = qml.jacobian(circuit)(x, y)
        expected = np.array(
            [
                [
                    0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                    -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
                ]
            ]
        )

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], np.ndarray)
        assert jac[0].shape == ()

        assert isinstance(jac[1], np.ndarray)
        assert jac[1].shape == ()

        assert np.allclose(jac, expected, atol=tol, rtol=0)


@pytest.mark.parametrize(
    "diff_method,kwargs",
    [
        ["finite-diff", {}],
        ["spsa", {"num_directions": 100, "h": 0.05}],
        ("parameter-shift", {}),
        ("parameter-shift", {"force_order2": True}),
    ],
)
class TestCV:
    """Tests for CV integration"""

    def test_first_order_observable(self, diff_method, kwargs, tol):
        """Test variance of a first order CV observable"""
        dev = qml.device("default.gaussian", wires=1)
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        r = np.array(0.543, requires_grad=True)
        phi = np.array(-0.654, requires_grad=True)

        @qnode(dev, diff_method=diff_method, **kwargs)
        def circuit(r, phi):
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.var(qml.QuadX(0))

        res = circuit(r, phi)
        expected = np.exp(2 * r) * np.sin(phi) ** 2 + np.exp(-2 * r) * np.cos(phi) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        res = qml.jacobian(circuit)(r, phi)
        expected = np.array(
            [
                [
                    2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                    2 * np.sinh(2 * r) * np.sin(2 * phi),
                ]
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_second_order_observable(self, diff_method, kwargs, tol):
        """Test variance of a second order CV expectation value"""
        dev = qml.device("default.gaussian", wires=1)
        if diff_method == "spsa":
            tol = TOL_FOR_SPSA
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        n = np.array(0.12, requires_grad=True)
        a = np.array(0.765, requires_grad=True)

        @qnode(dev, diff_method=diff_method, **kwargs)
        def circuit(n, a):
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            return qml.var(qml.NumberOperator(0))

        res = circuit(n, a)
        expected = n**2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        res = qml.jacobian(circuit)(n, a)
        expected = np.array([[2 * a**2 + 2 * n + 1, 2 * a * (2 * n + 1)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)


def test_adjoint_reuse_device_state(mocker):
    """Tests that the autograd interface reuses the device state for adjoint differentiation"""
    dev = qml.device("default.qubit.legacy", wires=1)

    @qnode(dev, diff_method="adjoint")
    def circ(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    spy = mocker.spy(dev, "adjoint_jacobian")

    qml.grad(circ, argnum=0)(1.0)
    assert circ.device.num_executions == 1

    spy.assert_called_with(mocker.ANY, use_device_state=True)


@pytest.mark.parametrize("dev_name,diff_method,grad_on_execution", qubit_device_and_diff_method)
class TestTapeExpansion:
    """Test that tape expansion within the QNode integrates correctly
    with the Autograd interface"""

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_gradient_expansion_trainable_only(
        self, dev_name, diff_method, grad_on_execution, max_diff
    ):
        """Test that a *supported* operation with no gradient recipe is only
        expanded for parameter-shift and finite-differences when it is trainable."""
        if diff_method not in ("parameter-shift", "finite-diff", "spsa", "hadamard"):
            pytest.skip("Only supports gradient transforms")
        if max_diff == 2 and diff_method == "hadamard":
            pytest.skip("Max diff > 1 not supported for Hadamard gradient.")

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        class PhaseShift(qml.PhaseShift):
            grad_method = None

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(dev, diff_method=diff_method, grad_on_execution=grad_on_execution, max_diff=max_diff)
        def circuit(x, y):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            PhaseShift(2 * y, wires=0)
            return qml.expval(qml.PauliX(0))

        x = np.array(0.5, requires_grad=True)
        y = np.array(0.7, requires_grad=False)
        circuit(x, y)

        qml.grad(circuit)(x, y)

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_expansion_analytic(
        self, dev_name, diff_method, grad_on_execution, max_diff, tol
    ):
        """Test that if there are non-commuting groups and the number of shots is None
        the first and second order gradients are correctly evaluated"""
        kwargs = dict(
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
        )
        if diff_method in ["adjoint", "hadamard"]:
            pytest.skip("The diff method requested does not yet support Hamiltonians")
        elif diff_method == "spsa":
            tol = TOL_FOR_SPSA
            spsa_kwargs = dict(sampler_rng=np.random.default_rng(SEED_FOR_SPSA), num_directions=10)
            kwargs = {**kwargs, **spsa_kwargs}

        dev = qml.device(dev_name, wires=3, shots=None)
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(dev, **kwargs)
        def circuit(data, weights, coeffs):
            weights = weights.reshape(1, -1)
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        d = np.array([0.1, 0.2], requires_grad=False)
        w = np.array([0.654, -0.734], requires_grad=True)
        c = np.array([-0.6543, 0.24, 0.54], requires_grad=True)

        # test output
        res = circuit(d, w, c)
        expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
        assert np.allclose(res, expected, atol=tol)

        # test gradients
        grad = qml.grad(circuit)(d, w, c)
        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[0], expected_w, atol=tol)
        assert np.allclose(grad[1], expected_c, atol=tol)

        # test second-order derivatives
        if diff_method in ("parameter-shift", "backprop") and max_diff == 2:
            if diff_method == "backprop":
                with pytest.warns(UserWarning, match=r"Output seems independent of input."):
                    grad2_c = qml.jacobian(qml.grad(circuit, argnum=2), argnum=2)(d, w, c)
            else:
                grad2_c = qml.jacobian(qml.grad(circuit, argnum=2), argnum=2)(d, w, c)
            assert np.allclose(grad2_c, 0, atol=tol)

            grad2_w_c = qml.jacobian(qml.grad(circuit, argnum=1), argnum=2)(d, w, c)
            expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
                0,
                -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
                -np.sin(d[1] + w[1]),
            ]
            assert np.allclose(grad2_w_c, expected, atol=tol)

    @pytest.mark.slow
    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_expansion_finite_shots(
        self, dev_name, diff_method, grad_on_execution, max_diff, mocker
    ):
        """Test that the Hamiltonian is expanded if there
        are non-commuting groups and the number of shots is finite
        and the first and second order gradients are correctly evaluated"""
        gradient_kwargs = {}
        tol = 0.3
        if diff_method in ("adjoint", "backprop", "hadamard"):
            pytest.skip("The adjoint and backprop methods do not yet support sampling")
        elif diff_method == "spsa":
            gradient_kwargs = dict(
                h=H_FOR_SPSA,
                sampler_rng=np.random.default_rng(SEED_FOR_SPSA),
                num_directions=20,
            )
            tol = TOL_FOR_SPSA
        elif diff_method == "finite-diff":
            gradient_kwargs = {"h": 0.05}

        dev = qml.device(dev_name, wires=3, shots=50000)
        spy = mocker.spy(qml.transforms, "hamiltonian_expand")
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            **gradient_kwargs,
        )
        def circuit(data, weights, coeffs):
            weights = weights.reshape(1, -1)
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            H = qml.Hamiltonian(coeffs, obs)
            H.compute_grouping()
            return qml.expval(H)

        d = np.array([0.1, 0.2], requires_grad=False)
        w = np.array([0.654, -0.734], requires_grad=True)
        c = np.array([-0.6543, 0.24, 0.54], requires_grad=True)

        # test output
        res = circuit(d, w, c)
        expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
        assert np.allclose(res, expected, atol=tol)
        spy.assert_called()

        # test gradients
        grad = qml.grad(circuit)(d, w, c)
        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[0], expected_w, atol=tol)
        assert np.allclose(grad[1], expected_c, atol=tol)

        # test second-order derivatives
        if diff_method == "parameter-shift" and max_diff == 2:
            with pytest.warns(UserWarning, match=r"Output seems independent of input."):
                grad2_c = qml.jacobian(qml.grad(circuit, argnum=2), argnum=2)(d, w, c)
            assert np.allclose(grad2_c, 0, atol=tol)

            grad2_w_c = qml.jacobian(qml.grad(circuit, argnum=1), argnum=2)(d, w, c)
            expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
                0,
                -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
                -np.sin(d[1] + w[1]),
            ]
            assert np.allclose(grad2_w_c, expected, atol=tol)


class TestSample:
    """Tests for the sample integration"""

    def test_backprop_error(self):
        """Test that sampling in backpropagation grad_on_execution raises an error"""
        dev = qml.device("default.qubit.legacy", wires=2)

        @qnode(dev, diff_method="backprop")
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        with pytest.raises(qml.QuantumFunctionError, match="only supported when shots=None"):
            circuit(shots=10)  # pylint: disable=unexpected-keyword-arg

    def test_sample_dimension(self):
        """Test that the sample function outputs samples of the right size"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=2, shots=n_sample)

        @qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        res = circuit()

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == (10,)
        assert isinstance(res[0], np.ndarray)

        assert res[1].shape == (10,)
        assert isinstance(res[1], np.ndarray)

    def test_sample_combination(self):
        """Test the output of combining expval, var and sample"""

        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=3, shots=n_sample)

        @qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert isinstance(result, tuple)
        assert len(result) == 3

        assert np.array_equal(result[0].shape, (n_sample,))
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)
        assert result[0].dtype == np.dtype("int")

    def test_single_wire_sample(self):
        """Test the return type and shape of sampling a single wire"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=1, shots=n_sample)

        @qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (n_sample,))

    def test_multi_wire_sample_regular_shape(self):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=3, shots=n_sample)

        @qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, tuple)
        assert len(result) == 3

        assert result[0].shape == (10,)
        assert isinstance(result[0], np.ndarray)

        assert result[1].shape == (10,)
        assert isinstance(result[1], np.ndarray)

        assert result[2].shape == (10,)
        assert isinstance(result[2], np.ndarray)


@pytest.mark.parametrize("dev_name,diff_method,grad_on_execution", qubit_device_and_diff_method)
class TestReturn:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types."""

    # pylint: disable=unused-argument

    def test_grad_single_measurement_param(self, dev_name, diff_method, grad_on_execution):
        """For one measurement and one param, the gradient is a float."""
        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)

        grad = qml.grad(circuit)(a)

        import sys

        python_version = sys.version_info.minor
        if diff_method == "backprop" and python_version > 7:
            # Since numpy 1.23.0
            assert isinstance(grad, np.ndarray)
        else:
            assert isinstance(grad, float)

    def test_grad_single_measurement_multiple_param(self, dev_name, diff_method, grad_on_execution):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""
        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        grad = qml.grad(circuit)(a, b)

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        assert grad[0].shape == ()
        assert grad[1].shape == ()

    def test_grad_single_measurement_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""
        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array([0.1, 0.2], requires_grad=True)

        grad = qml.grad(circuit)(a)

        assert isinstance(grad, np.ndarray)
        assert len(grad) == 2
        assert grad.shape == (2,)

    def test_jacobian_single_measurement_param_probs(
        self, dev_name, diff_method, grad_on_execution
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)

        jac = qml.jacobian(circuit)(a)

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev_name, diff_method, grad_on_execution
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        jac = qml.jacobian(circuit)(a, b)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], np.ndarray)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], np.ndarray)
        assert jac[1].shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev_name, diff_method, grad_on_execution
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned."""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = np.array([0.1, 0.2], requires_grad=True)
        jac = qml.jacobian(circuit)(a)

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (4, 2)

    def test_jacobian_multiple_measurement_single_param(
        self, dev_name, diff_method, grad_on_execution
    ):
        """The jacobian of multiple measurements with a single params return an array."""
        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)

        def cost(x):
            return anp.hstack(circuit(x))

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (5,)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev_name, diff_method, grad_on_execution
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        def cost(x, y):
            return anp.hstack(circuit(x, y))

        jac = qml.jacobian(cost)(a, b)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], np.ndarray)
        assert jac[0].shape == (5,)

        assert isinstance(jac[1], np.ndarray)
        assert jac[1].shape == (5,)

    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array([0.1, 0.2], requires_grad=True)

        def cost(x):
            return anp.hstack(circuit(x))

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        assert jac.shape == (5, 2)

    def test_hessian_expval_multiple_params(self, dev_name, diff_method, grad_on_execution):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 4

        dev = qml.device(dev_name, wires=num_wires)

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")

        par_0 = qml.numpy.array(0.1, requires_grad=True)
        par_1 = qml.numpy.array(0.2, requires_grad=True)

        @qnode(dev, interface="autograd", diff_method=diff_method, max_diff=2)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x, y):
            return anp.hstack(qml.grad(circuit)(x, y))

        hess = qml.jacobian(cost)(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], np.ndarray)
        assert hess[0].shape == (2,)

        assert isinstance(hess[1], np.ndarray)
        assert hess[1].shape == (2,)

    def test_hessian_expval_multiple_param_array(self, dev_name, diff_method, grad_on_execution):
        """The hessian of single measurement with a multiple params array return a single array."""

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 4

        dev = qml.device(dev_name, wires=num_wires)

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")

        params = qml.numpy.array([0.1, 0.2], requires_grad=True)

        @qnode(dev, interface="autograd", diff_method=diff_method, max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = qml.jacobian(qml.grad(circuit))(params)

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (2, 2)

    def test_hessian_var_multiple_params(self, dev_name, diff_method, grad_on_execution):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        par_0 = qml.numpy.array(0.1, requires_grad=True)
        par_1 = qml.numpy.array(0.2, requires_grad=True)

        @qnode(dev, interface="autograd", diff_method=diff_method, max_diff=2)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x, y):
            return anp.hstack(qml.grad(circuit)(x, y))

        hess = qml.jacobian(cost)(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], np.ndarray)
        assert hess[0].shape == (2,)

        assert isinstance(hess[1], np.ndarray)
        assert hess[1].shape == (2,)

    def test_hessian_var_multiple_param_array(self, dev_name, diff_method, grad_on_execution):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        dev = qml.device(dev_name, wires=2)

        params = qml.numpy.array([0.1, 0.2], requires_grad=True)

        @qnode(dev, interface="autograd", diff_method=diff_method, max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = qml.jacobian(qml.grad(circuit))(params)

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (2, 2)

    def test_hessian_probs_expval_multiple_params(self, dev_name, diff_method, grad_on_execution):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        if diff_method in ["adjoint", "hadamard"]:
            pytest.skip("The adjoint method does not currently support second-order diff.")

        par_0 = qml.numpy.array(0.1, requires_grad=True)
        par_1 = qml.numpy.array(0.2, requires_grad=True)

        @qnode(dev, interface="autograd", diff_method=diff_method, max_diff=2)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        def circuit_stack(x, y):
            return anp.hstack(circuit(x, y))

        def cost(x, y):
            return anp.hstack(qml.jacobian(circuit_stack)(x, y))

        hess = qml.jacobian(cost)(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], np.ndarray)
        assert hess[0].shape == (6,)

        assert isinstance(hess[1], np.ndarray)
        assert hess[1].shape == (6,)

    def test_hessian_expval_probs_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""

        if diff_method in ["adjoint", "hadamard"]:
            pytest.skip("The adjoint method does not currently support second-order diff.")

        dev = qml.device(dev_name, wires=2)

        params = qml.numpy.array([0.1, 0.2], requires_grad=True)

        @qnode(dev, interface="autograd", diff_method=diff_method, max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        def cost(x):
            return anp.hstack(circuit(x))

        hess = qml.jacobian(qml.jacobian(cost))(params)

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (3, 2, 2)

    def test_hessian_probs_var_multiple_params(self, dev_name, diff_method, grad_on_execution):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        par_0 = qml.numpy.array(0.1, requires_grad=True)
        par_1 = qml.numpy.array(0.2, requires_grad=True)

        @qnode(dev, interface="autograd", diff_method=diff_method, max_diff=2)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        def circuit_stack(x, y):
            return anp.hstack(circuit(x, y))

        def cost(x, y):
            return anp.hstack(qml.jacobian(circuit_stack)(x, y))

        hess = qml.jacobian(cost)(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], np.ndarray)
        assert hess[0].shape == (6,)

        assert isinstance(hess[1], np.ndarray)
        assert hess[1].shape == (6,)

    def test_hessian_var_probs_multiple_param_array(self, dev_name, diff_method, grad_on_execution):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support second-order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard gradient does not support variances.")

        dev = qml.device(dev_name, wires=2)

        params = qml.numpy.array([0.1, 0.2], requires_grad=True)

        @qnode(dev, interface="autograd", diff_method=diff_method, max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        def cost(x):
            return anp.hstack(circuit(x))

        hess = qml.jacobian(qml.jacobian(cost))(params)

        assert isinstance(hess, np.ndarray)
        assert hess.shape == (3, 2, 2)


@pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
def test_no_ops(dev_name):
    """Test that the return value of the QNode matches in the interface
    even if there are no ops"""

    dev = qml.device(dev_name, wires=1)

    @qml.qnode(dev, interface="autograd")
    def circuit():
        qml.Hadamard(wires=0)
        return qml.state()

    res = circuit()
    assert isinstance(res, np.tensor)
