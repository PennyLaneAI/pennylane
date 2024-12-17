# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Tests for the ``default.mixed`` device for the Autograd interface
"""
# pylint: disable=protected-access
import pytest

import pennylane as qml
from pennylane import DeviceError
from pennylane import numpy as pnp
from pennylane.devices.default_mixed import DefaultMixed

pytestmark = pytest.mark.autograd


def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(
        DeviceError,
        match=msg,
    ):
        qml.device("default.mixed", wires=1, shots=1, analytic=True)


class TestQNodeIntegration:
    """Integration tests for default.mixed.autograd. This test ensures it integrates
    properly with the PennyLane UI, in particular the QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("default.mixed", wires=1)
        cap = dev.target_device.capabilities()
        capabilities = {
            "model": "qubit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "supports_broadcasting": False,
            "returns_probs": True,
            "returns_state": True,
            "passthru_devices": {
                "autograd": "default.mixed",
                "tf": "default.mixed",
                "torch": "default.mixed",
                "jax": "default.mixed",
            },
        }

        assert cap == capabilities

    def test_load_device(self):
        """Test that the plugin device loads correctly"""
        dev = qml.device("default.mixed", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == qml.measurements.Shots(None)
        assert dev.short_name == "default.mixed"
        assert dev.target_device.capabilities()["passthru_devices"]["autograd"] == "default.mixed"

    def test_qubit_circuit(self, tol):
        """Test that the device provides the correct
        result for a simple circuit."""
        p = pnp.array(0.543)

        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -pnp.sin(p)

        assert pnp.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, tol):
        """Test that the device state is correct after evaluating a
        quantum function on the device"""

        dev = qml.device("default.mixed", wires=2)

        state = dev.state
        expected = pnp.zeros((4, 4))
        expected[0, 0] = 1
        assert pnp.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            qml.RZ(pnp.pi / 4, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev.state

        amplitude = pnp.exp(-1j * pnp.pi / 4) / 2
        expected = pnp.array(
            [[0.5, 0, amplitude, 0], [0, 0, 0, 0], [pnp.conj(amplitude), 0, 0.5, 0], [0, 0, 0, 0]]
        )

        assert pnp.allclose(state, expected, atol=tol, rtol=0)


class TestDtypePreserved:
    """Test that the user-defined dtype of the device is preserved for QNode
    evaluation"""

    @pytest.mark.parametrize("r_dtype", [pnp.float32, pnp.float64])
    @pytest.mark.parametrize(
        "measurement",
        [
            qml.expval(qml.PauliY(0)),
            qml.var(qml.PauliY(0)),
            qml.probs(wires=[1]),
            qml.probs(wires=[2, 0]),
        ],
    )
    def test_real_dtype(self, r_dtype, measurement):
        """Test that the user-defined dtype of the device is preserved
        for QNodes with real-valued outputs"""
        p = 0.543

        dev = qml.device("default.mixed", wires=3)
        dev.target_device.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == r_dtype

    @pytest.mark.parametrize("c_dtype_name", ["complex64", "complex128"])
    @pytest.mark.parametrize(
        "measurement",
        [qml.state(), qml.density_matrix(wires=[1]), qml.density_matrix(wires=[2, 0])],
    )
    def test_complex_dtype(self, c_dtype_name, measurement):
        """Test that the user-defined dtype of the device is preserved
        for QNodes with complex-valued outputs"""
        p = 0.543
        c_dtype = pnp.dtype(c_dtype_name)

        dev = qml.device("default.mixed", wires=3, c_dtype=c_dtype)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == c_dtype


class TestOps:
    """Unit tests for operations supported by the default.mixed.autograd device"""

    def test_multirz_jacobian(self):
        """Test that the patched numpy functions are used for the MultiRZ
        operation and the jacobian can be computed."""
        wires = 4
        dev = qml.device("default.mixed", wires=wires)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(param):
            qml.MultiRZ(param, wires=[0, 1])
            return qml.probs(wires=list(range(wires)))

        param = pnp.array(0.3, requires_grad=True)
        res = qml.jacobian(circuit)(param)
        assert pnp.allclose(res, pnp.zeros(wires**2))

    def test_full_subsystem(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultMixed(wires=["a", "b", "c"])
        state = pnp.array([1, 0, 0, 0, 1, 0, 1, 1]) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(qml.math, "scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        state = pnp.outer(state, pnp.conj(state))

        assert pnp.all(dev._state.flatten() == state.flatten())
        spy.assert_not_called()

    def test_partial_subsystem(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = DefaultMixed(wires=["a", "b", "c"])
        state = pnp.array([1, 0, 1, 0]) / pnp.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(qml.math, "scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        state = pnp.kron(pnp.outer(state, pnp.conj(state)), pnp.array([[1, 0], [0, 0]]))

        assert pnp.all(pnp.reshape(dev._state, (8, 8)) == state)
        spy.assert_called()


@pytest.mark.parametrize(
    "op, exp_method, dev_wires",
    [
        (qml.RX(pnp.array(0.2), 0), "_apply_channel", 1),
        (qml.RX(pnp.array(0.2), 0), "_apply_channel", 8),
        (qml.CNOT([0, 1]), "_apply_channel", 3),
        (qml.CNOT([0, 1]), "_apply_channel", 8),
        (qml.MultiControlledX(wires=list(range(2))), "_apply_channel", 3),
        (qml.MultiControlledX(wires=list(range(3))), "_apply_channel_tensordot", 3),
        (qml.MultiControlledX(wires=list(range(8))), "_apply_channel_tensordot", 8),
        (qml.PauliError("X", pnp.array(0.5), 0), "_apply_channel", 2),
        (qml.PauliError("XXX", pnp.array(0.5), [0, 1, 2]), "_apply_channel_tensordot", 4),
        (qml.PauliError("X" * 8, pnp.array(0.5), list(range(8))), "_apply_channel_tensordot", 8),
    ],
)
def test_method_choice(mocker, op, exp_method, dev_wires):
    """Test that the right method between _apply_channel and _apply_channel_tensordot
    is chosen."""

    methods = ["_apply_channel", "_apply_channel_tensordot"]
    del methods[methods.index(exp_method)]
    unexp_method = methods[0]
    spy_exp = mocker.spy(DefaultMixed, exp_method)
    spy_unexp = mocker.spy(DefaultMixed, unexp_method)
    dev = qml.device("default.mixed", wires=dev_wires)
    dev._apply_operation(op)

    spy_unexp.assert_not_called()
    spy_exp.assert_called_once()


class TestPassthruIntegration:
    """Tests for integration with the PassthruQNode"""

    def test_jacobian_variable_multiply(self, tol):
        """Test that jacobian of a QNode with an attached default.mixed.autograd device
        gives the correct result in the case of parameters multiplied by scalars"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        weights = pnp.array([x, y, z], requires_grad=True)

        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(weights)

        expected = pnp.cos(3 * x) * pnp.cos(y) * pnp.cos(z / 2) - pnp.sin(3 * x) * pnp.sin(z / 2)
        assert pnp.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = qml.jacobian(circuit, 0)
        res = grad_fn(pnp.array(weights))

        expected = pnp.array(
            [
                -3 * (pnp.sin(3 * x) * pnp.cos(y) * pnp.cos(z / 2) + pnp.cos(3 * x) * pnp.sin(z / 2)),
                -pnp.cos(3 * x) * pnp.sin(y) * pnp.cos(z / 2),
                -0.5 * (pnp.sin(3 * x) * pnp.cos(z / 2) + pnp.cos(3 * x) * pnp.cos(y) * pnp.sin(z / 2)),
            ]
        )

        assert pnp.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_repeated(self, tol):
        """Test that jacobian of a QNode with an attached default.mixed.autograd device
        gives the correct result in the case of repeated parameters"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        p = pnp.array([x, y, z], requires_grad=True)
        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(p)

        expected = pnp.cos(y) ** 2 - pnp.sin(x) * pnp.sin(y) ** 2
        assert pnp.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = qml.jacobian(circuit, 0)
        res = grad_fn(p)

        expected = pnp.array(
            [-pnp.cos(x) * pnp.sin(y) ** 2, -2 * (pnp.sin(x) + 1) * pnp.sin(y) * pnp.cos(y), 0]
        )
        assert pnp.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_agrees_backprop_parameter_shift(self, tol):
        """Test that jacobian of a QNode with an attached default.mixed.autograd device
        gives the correct result with respect to the parameter-shift method"""
        p = pnp.array([0.43316321, 0.2162158, 0.75110998, 0.94714242], requires_grad=True)

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("default.mixed", wires=3)
        dev2 = qml.device("default.mixed", wires=3)

        def cost(x):
            return qml.math.stack(circuit(x))

        circuit1 = qml.QNode(cost, dev1, diff_method="backprop", interface="autograd")
        circuit2 = qml.QNode(cost, dev2, diff_method="parameter-shift")

        res = circuit1(p)

        assert pnp.allclose(res, circuit2(p), atol=tol, rtol=0)

        grad_fn = qml.jacobian(circuit1, 0)
        res = grad_fn(p)
        assert pnp.allclose(res, qml.jacobian(circuit2)(p), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op, wire_ids, exp_fn",
        [
            (qml.RY, [0], lambda a: -pnp.sin(a)),
            (qml.AmplitudeDamping, [0], lambda a: -2),
            (qml.DepolarizingChannel, [-1], lambda a: -4 / 3),
            (lambda a, wires: qml.ResetError(p0=a, p1=0.1, wires=wires), [0], lambda a: -2),
            (lambda a, wires: qml.ResetError(p0=0.1, p1=a, wires=wires), [0], lambda a: 0),
        ],
    )
    @pytest.mark.parametrize("wires", [[0], ["abc"]])
    def test_state_differentiability(self, wires, op, wire_ids, exp_fn, tol):
        """Test that the device state can be differentiated"""
        # pylint: disable=too-many-arguments
        dev = qml.device("default.mixed", wires=wires)

        @qml.qnode(dev, diff_method="backprop", interface="autograd")
        def circuit(a):
            qml.PauliX(wires[wire_ids[0]])
            op(a, wires=[wires[idx] for idx in wire_ids])
            return qml.state()

        a = pnp.array(0.23, requires_grad=True)

        def cost(a):
            """A function of the device quantum state, as a function
            of input QNode parameters."""
            state = circuit(a)
            res = pnp.abs(state) ** 2
            return res[1][1] - res[0][0]

        grad = qml.grad(cost)(a)
        expected = exp_fn(a)
        assert pnp.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("wires", [range(2), [-12.32, "abc"]])
    def test_density_matrix_differentiability(self, wires, tol):
        """Test that the density matrix can be differentiated"""
        dev = qml.device("default.mixed", wires=wires)

        @qml.qnode(dev, diff_method="backprop", interface="autograd")
        def circuit(a):
            qml.RY(a, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            return qml.density_matrix(wires=wires[1])

        a = pnp.array(0.54, requires_grad=True)

        def cost(a):
            """A function of the device quantum state, as a function
            of input QNode parameters."""
            state = circuit(a)
            res = pnp.abs(state) ** 2
            return res[1][1] - res[0][0]

        grad = qml.grad(cost)(a)
        expected = pnp.sin(a)
        assert pnp.allclose(grad, expected, atol=tol, rtol=0)

    def test_prob_differentiability(self, tol):
        """Test that the device probability can be differentiated"""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="autograd")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = pnp.array(0.54, requires_grad=True)
        b = pnp.array(0.12, requires_grad=True)

        def cost(a, b):
            prob_wire_1 = circuit(a, b)
            return prob_wire_1[1] - prob_wire_1[0]

        res = cost(a, b)
        expected = -pnp.cos(a) * pnp.cos(b)
        assert pnp.allclose(res, expected, atol=tol, rtol=0)

        grad = qml.grad(cost)(a, b)
        expected = [pnp.sin(a) * pnp.cos(b), pnp.cos(a) * pnp.sin(b)]
        assert pnp.allclose(grad, expected, atol=tol, rtol=0)

    def test_prob_vector_differentiability(self, tol):
        """Test that the device probability vector can be differentiated directly"""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="autograd")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = pnp.array(0.54, requires_grad=True)
        b = pnp.array(0.12, requires_grad=True)

        res = circuit(a, b)
        expected = [
            pnp.cos(a / 2) ** 2 * pnp.cos(b / 2) ** 2 + pnp.sin(a / 2) ** 2 * pnp.sin(b / 2) ** 2,
            pnp.cos(a / 2) ** 2 * pnp.sin(b / 2) ** 2 + pnp.sin(a / 2) ** 2 * pnp.cos(b / 2) ** 2,
        ]
        assert pnp.allclose(res, expected, atol=tol, rtol=0)

        grad = qml.jacobian(circuit)(a, b)
        expected = 0.5 * pnp.array(
            [
                [-pnp.sin(a) * pnp.cos(b), pnp.sin(a) * pnp.cos(b)],
                [-pnp.cos(a) * pnp.sin(b), pnp.cos(a) * pnp.sin(b)],
            ]
        )

        assert pnp.allclose(grad, expected, atol=tol, rtol=0)

    def test_sample_backprop_error(self):
        """Test that sampling in backpropagation mode raises an error"""
        # pylint: disable=unused-variable
        dev = qml.device("default.mixed", wires=1, shots=100)

        msg = "does not support backprop with requested circuit"

        with pytest.raises(qml.QuantumFunctionError, match=msg):

            @qml.qnode(dev, diff_method="backprop", interface="autograd")
            def circuit(a):
                qml.RY(a, wires=0)
                return qml.sample(qml.PauliZ(0))

    def test_expval_gradient(self, tol):
        """Tests that the gradient of expval is correct"""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="autograd")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = pnp.array(-0.234, requires_grad=True)
        b = pnp.array(0.654, requires_grad=True)

        res = circuit(a, b)
        expected_cost = 0.5 * (pnp.cos(a) * pnp.cos(b) + pnp.cos(a) - pnp.cos(b) + 1)
        assert pnp.allclose(res, expected_cost, atol=tol, rtol=0)

        res = qml.grad(circuit)(a, b)
        expected_grad = pnp.array(
            [-0.5 * pnp.sin(a) * (pnp.cos(b) + 1), 0.5 * pnp.sin(b) * (1 - pnp.cos(a))]
        )
        assert pnp.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "x, shift",
        [pnp.array((0.0, 0.0), requires_grad=True), pnp.array((0.5, -0.5), requires_grad=True)],
    )
    def test_hessian_at_zero(self, x, shift):
        """Tests that the Hessian at vanishing state vector amplitudes
        is correct."""
        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit(x):
            qml.RY(shift, wires=0)
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert qml.math.isclose(qml.jacobian(circuit)(x), 0.0)
        assert qml.math.isclose(qml.jacobian(qml.jacobian(circuit))(x), -1.0)
        assert qml.math.isclose(qml.grad(qml.grad(circuit))(x), -1.0)

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.compute_decomposition])
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
    def test_autograd_interface_gradient(self, operation, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the Autograd interface, using a variety of differentiation methods."""
        dev = qml.device("default.mixed", wires=1)
        state = pnp.array(1j * pnp.array([1, -1]) / pnp.sqrt(2), requires_grad=False)

        @qml.qnode(dev, diff_method=diff_method, interface="autograd")
        def circuit(x, weights, w):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.StatePrep(state, wires=w)
            operation(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        def cost(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        theta = 0.543
        phi = -0.234
        lam = 0.654

        params = pnp.array([theta, phi, lam], requires_grad=True)

        res = cost(params)
        expected_cost = (pnp.sin(lam) * pnp.sin(phi) - pnp.cos(theta) * pnp.cos(lam) * pnp.cos(phi)) ** 2
        assert pnp.allclose(res, expected_cost, atol=tol, rtol=0)

        res = qml.grad(cost)(params)
        expected_grad = (
            pnp.array(
                [
                    pnp.sin(theta) * pnp.cos(lam) * pnp.cos(phi),
                    pnp.cos(theta) * pnp.cos(lam) * pnp.sin(phi) + pnp.sin(lam) * pnp.cos(phi),
                    pnp.cos(theta) * pnp.sin(lam) * pnp.cos(phi) + pnp.cos(lam) * pnp.sin(phi),
                ]
            )
            * 2
            * (pnp.sin(lam) * pnp.sin(phi) - pnp.cos(theta) * pnp.cos(lam) * pnp.cos(phi))
        )
        assert pnp.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "dev_name,diff_method,mode",
        [
            ["default.mixed", "finite-diff", False],
            ["default.mixed", "parameter-shift", False],
            ["default.mixed", "backprop", True],
        ],
    )
    def test_multiple_measurements_differentiation(self, dev_name, diff_method, mode, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device(dev_name, wires=2)
        x = pnp.array(0.543, requires_grad=True)
        y = pnp.array(-0.654, requires_grad=True)

        @qml.qnode(dev, diff_method=diff_method, interface="autograd", grad_on_execution=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        res = circuit(x, y)

        expected = pnp.array(
            [pnp.cos(x), (1 + pnp.cos(x) * pnp.cos(y)) / 2, (1 - pnp.cos(x) * pnp.cos(y)) / 2]
        )
        assert pnp.allclose(qml.math.hstack(res), expected, atol=tol, rtol=0)

        def cost(x, y):
            return qml.math.hstack(circuit(x, y))

        res = qml.jacobian(cost)(x, y)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (3,)
        assert res[1].shape == (3,)

        expected = (
            pnp.array([-pnp.sin(x), -pnp.sin(x) * pnp.cos(y) / 2, pnp.sin(x) * pnp.cos(y) / 2]),
            pnp.array([0, -pnp.cos(x) * pnp.sin(y) / 2, pnp.cos(x) * pnp.sin(y) / 2]),
        )
        assert pnp.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert pnp.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_batching(self, tol):
        """Tests that the gradient of the qnode is correct with batching"""
        dev = qml.device("default.mixed", wires=2)

        @qml.batch_params
        @qml.qnode(dev, diff_method="backprop", interface="autograd")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = pnp.array([-0.234, 0.678], requires_grad=True)
        b = pnp.array([0.654, 1.236], requires_grad=True)

        res = circuit(a, b)
        expected_cost = 0.5 * (pnp.cos(a) * pnp.cos(b) + pnp.cos(a) - pnp.cos(b) + 1)
        assert pnp.allclose(res, expected_cost, atol=tol, rtol=0)

        res_a, res_b = qml.jacobian(circuit)(a, b)
        expected_a, expected_b = [
            -0.5 * pnp.sin(a) * (pnp.cos(b) + 1),
            0.5 * pnp.sin(b) * (1 - pnp.cos(a)),
        ]

        assert pnp.allclose(pnp.diag(res_a), expected_a, atol=tol, rtol=0)
        assert pnp.allclose(pnp.diag(res_b), expected_b, atol=tol, rtol=0)


# pylint: disable=too-few-public-methods
class TestHighLevelIntegration:
    """Tests for integration with higher level components of PennyLane."""

    def test_template_integration(self):
        """Test that a PassthruQNode default.mixed.autograd works with templates."""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        weights = pnp.random.random(shape, requires_grad=True)

        grad = qml.grad(circuit)(weights)
        assert grad.shape == weights.shape


class TestMeasurements:
    """Tests for measurements with default.mixed"""

    @pytest.mark.parametrize(
        "measurement",
        [
            qml.counts(qml.PauliZ(0)),
            qml.counts(wires=[0]),
            qml.sample(qml.PauliX(0)),
            qml.sample(wires=[1]),
        ],
    )
    def test_measurements_tf(self, measurement):
        """Test sampling-based measurements work with `default.mixed` for trainable interfaces"""
        num_shots = 1024
        dev = qml.device("default.mixed", wires=2, shots=num_shots)

        @qml.qnode(dev, interface="autograd")
        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        res = circuit(pnp.array(0.5))

        assert len(res) == 2 if isinstance(measurement, qml.measurements.CountsMP) else num_shots

    @pytest.mark.parametrize(
        "meas_op",
        [qml.PauliX(0), qml.PauliZ(0)],
    )
    def test_measurement_diff(self, meas_op):
        """Test sequence of single-shot expectation values work for derivatives"""
        num_shots = 64
        dev = qml.device("default.mixed", shots=[(1, num_shots)], wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(angle):
            qml.RX(angle, wires=0)
            return qml.expval(meas_op)

        def cost(angle):
            return qml.math.hstack(circuit(angle))

        angle = pnp.array(0.1234)

        assert isinstance(qml.jacobian(cost)(angle), pnp.ndarray)
        assert len(cost(angle)) == num_shots
