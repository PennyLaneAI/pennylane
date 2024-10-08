# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the SPSA optimizer"""
import pytest

import pennylane as qml
from pennylane import numpy as np

univariate = [(np.sin), (lambda x: np.exp(x / 10.0)), (lambda x: x**2)]

multivariate = [
    (lambda x: np.sin(x[0]) + np.cos(x[1])),
    (lambda x: np.exp(x[0] / 3) * np.tanh(x[1])),
    (lambda x: np.sum([x_**2 for x_ in x])),
]


class TestSPSAOptimizer:
    """Test the SPSA optimizer."""

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f", univariate)
    def test_apply_grad(self, args, f):
        """Test that a gradient step can be applied correctly with a univariate
        function."""
        gamma = 0.3
        c = 0.1
        spsa_opt = qml.SPSAOptimizer(maxiter=10, c=c, gamma=gamma)
        args = np.array([args], requires_grad=True)

        alpha = 0.602
        A = 1.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / k**gamma
        ak = a / (A + k) ** alpha
        # assume delta is 1.
        di = 1
        multiplier = ck * di
        yplus = f(args + multiplier)
        yminus = f(args - multiplier)
        grad = (yplus - yminus) / (2 * multiplier)

        res = spsa_opt.apply_grad(grad, args)
        expected = args - ak * grad
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("f", multivariate)
    def test_apply_grad_multivar(self, f):
        """Test that a gradient step can be applied correctly to a multivariate
        function."""
        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 10.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / k**gamma
        ak = a / (A + k) ** alpha
        deltas = np.array(np.meshgrid([1, -1], [1, -1])).T.reshape(-1, 2)

        spsa_opt = qml.SPSAOptimizer(A=A)

        x_vals = np.linspace(-10, 10, 16, endpoint=False)

        for jdx in range(len(x_vals[:-1])):
            args = x_vals[jdx : jdx + 2]
            y_pm = []
            for delta in deltas:
                thetaplus = list(args)
                thetaminus = list(args)
                for index, arg in enumerate(args):
                    multiplier = ck * delta[index]
                    thetaplus[index] = arg + multiplier
                    thetaminus[index] = arg - multiplier
                yplus = f(thetaplus)
                yminus = f(thetaminus)
                y_pm.append(yplus - yminus)
            # choose one delta
            d = 0
            grad = np.array([y_pm[d] / (2 * ck * di) for di in deltas[d]])
            tol = ak * max(np.abs(y_pm)) / ck
            args_res = spsa_opt.apply_grad(grad, args)
            expected = args - ak * grad
            assert np.allclose(args_res, expected, atol=tol)

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f", univariate)
    def test_step_and_cost_supplied_univar_cost(self, args, f):
        """Test that returned cost is correct."""
        spsa_opt = qml.SPSAOptimizer(10)
        args = np.array(args, requires_grad=True)

        _, res = spsa_opt.step_and_cost(f, args)
        expected = f(args)
        assert np.all(res == expected)

    def test_step_and_cost_supplied_cost(self):
        """Test that the correct cost is returned via the step_and_cost method
        for the SPSA optimizer."""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(variables):
            qml.RX(variables[0], wires=[0])
            qml.RY(variables[1], wires=[0])
            qml.RY(variables[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        inputs = np.array([0.4, 0.2, 0.4], requires_grad=True)

        expected = quant_fun(inputs)
        _, res = spsa_opt.step_and_cost(quant_fun, inputs)

        assert np.all(res == expected)

    def test_step_and_cost_hamiltonian(self):
        """Test that the correct cost is returned via the step_and_cost method
        for the SPSA optimizer."""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(variables):
            obs = [qml.PauliX(0) @ qml.PauliZ(0), qml.PauliZ(0) @ qml.Hadamard(0)]
            return qml.expval(qml.Hamiltonian(variables, obs))

        inputs = np.array([0.2, -0.543], requires_grad=True)

        expected = quant_fun(inputs)
        _, res = spsa_opt.step_and_cost(quant_fun, inputs)

        assert np.all(res == expected)

    def test_step_spsa(self):
        """Test that the correct param is returned via the step method."""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(params):
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[0])
            qml.RY(params[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        args = np.array([0.4, 0.2, 0.8], requires_grad=True)

        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 1.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / k**gamma
        ak = a / (A + k) ** alpha
        deltas = np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3)

        y_pm = []
        for delta in deltas:
            thetaplus = list(args)
            thetaminus = list(args)
            for index, arg in enumerate(args):
                multiplier = ck * delta[index]
                thetaplus[index] = arg + multiplier
                thetaminus[index] = arg - multiplier
            yplus = quant_fun(thetaplus)
            yminus = quant_fun(thetaminus)
            y_pm.append(yplus - yminus)
        # choose one delta
        d = 0
        grad = np.array([y_pm[d] / (2 * ck * di) for di in deltas[d]])
        tol = ak * max(np.abs(y_pm)) / ck

        expected = args - ak * grad

        res = spsa_opt.step(quant_fun, args)

        assert np.allclose(res, expected, atol=tol)

    def test_step_and_cost_spsa_single_multid_input(self):
        """Test that the correct cost is returned via the step_and_cost method
        with a multidimensional input."""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        multid_array = np.array([[0.1, 0.2], [-0.1, -0.4]])

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun_mdarr(var):
            qml.RX(var[0, 1], wires=[0])
            qml.RY(var[1, 0], wires=[0])
            qml.RY(var[1, 1], wires=[0])
            return qml.expval(qml.PauliZ(0))

        _, res = spsa_opt.step_and_cost(quant_fun_mdarr, multid_array)
        expected = quant_fun_mdarr(multid_array)

        assert np.all(res == expected)

    def test_step_spsa_single_multid_input(self):
        """Test that the correct param is returned via the step method
        with a multidimensional input."""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        multid_array = np.array([[0.1, 0.2], [-0.1, -0.4]])

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun_mdarr(var):
            qml.RX(var[0, 0], wires=[0])
            qml.RX(var[0, 1], wires=[0])
            qml.RY(var[1, 0], wires=[0])
            qml.RY(var[1, 1], wires=[0])
            return qml.expval(qml.PauliZ(0))

        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 1.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / k**gamma
        ak = a / (A + k) ** alpha
        # pylint:disable=too-many-function-args
        deltas = np.array(np.meshgrid([1, -1], [1, -1], [1, -1], [1, -1])).T.reshape(-1, 2, 2)

        args = (multid_array,)
        y_pm = []
        for delta in deltas:
            thetaplus = list(args)
            thetaminus = list(args)
            for index, arg in enumerate(args):
                multiplier = ck * delta
                thetaplus[index] = arg + multiplier
                thetaminus[index] = arg - multiplier
            yplus = np.array([quant_fun_mdarr(p) for p in thetaplus])
            yminus = np.array([quant_fun_mdarr(p) for p in thetaminus])
            y_pm.append(yplus - yminus)
        # choose one delta
        d = 0
        grad = np.array([y_pm[d] / (2 * ck * deltas[d])])
        tol = ak * max(np.abs(y_pm)) / ck

        expected = multid_array - ak * grad

        res = spsa_opt.step(quant_fun_mdarr, multid_array)

        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f", univariate)
    def test_step_for_univar_cost(self, args, f):
        """Test that a gradient step can be applied correctly with a univariate
        function."""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        args = np.array([args], requires_grad=True)

        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 1.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / k**gamma
        ak = a / (A + k) ** alpha

        # assume delta is 1.
        di = 1
        multiplier = ck * di
        yplus = f(args + multiplier)
        yminus = f(args - multiplier)
        grad = (yplus - yminus) / (2 * multiplier)

        res = spsa_opt.step(f, args)
        expected = args - ak * grad
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f", univariate)
    def test_step_and_cost(self, args, f):
        """Test that a gradient step can be applied correctly with a univariate
        function."""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        args = np.array([args], requires_grad=True)

        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 1.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / k**gamma
        ak = a / (A + k) ** alpha
        # assume delta is 1.
        di = 1
        multiplier = ck * di
        yplus = f(args + multiplier)
        yminus = f(args - multiplier)
        grad = (yplus - yminus) / (2 * multiplier)
        y = f(args)

        res, rescost = spsa_opt.step_and_cost(f, args)
        expected = args - ak * grad
        assert np.allclose(res, expected)
        assert np.allclose(y, rescost)

    def test_parameters_not_a_tensor_and_not_all_require_grad(self):
        """Test execution of list of parameters of different sizes
        and not all require grad."""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(*variables):
            qml.RX(variables[0][1], wires=[0])
            qml.RY(variables[1][2], wires=[0])
            qml.RY(variables[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        inputs = [
            np.array((0.2, 0.3), requires_grad=True),
            np.array([0.4, 0.2, 0.4], requires_grad=False),
            np.array(0.1, requires_grad=True),
        ]

        res, _ = spsa_opt.step_and_cost(quant_fun, *inputs)
        assert isinstance(res, list)
        assert np.all(res[1] == inputs[1])
        assert np.all(res[0] != inputs[0])

    def test_parameters_in_step(self):
        """Test execution of list of parameters of different sizes
        and not all require grad."""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(*variables):
            qml.RX(variables[0][1], wires=[0])
            qml.RY(variables[1][2], wires=[0])
            qml.RY(variables[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        inputs = [
            np.array((0.2, 0.3), requires_grad=True),
            np.array([0.4, 0.2, 0.4], requires_grad=False),
            np.array(0.1, requires_grad=True),
        ]

        res = spsa_opt.step(quant_fun, *inputs)
        assert isinstance(res, list)
        assert np.all(res[1] == inputs[1])
        assert np.all(res[0] != inputs[0])

    def test_parameter_not_an_array(self):
        """Test function when there is only one float parameter that doesn't
        require grad."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost(a):
            return circuit(a)

        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        params = 0.5

        res = spsa_opt.step(cost, params)

        assert isinstance(res, float)
        assert res == params

    def test_obj_func_with_probs_not_a_scalar_function(self):
        """Test that if the objective function is not a
        scalar function, like qml.probs(), an error is raised."""

        n_wires = 4
        n_layers = 3
        dev = qml.device("default.qubit", wires=n_wires)

        def circuit(params):
            qml.StronglyEntanglingLayers(params, wires=list(range(n_wires)))

        @qml.qnode(dev)
        def cost(params):
            circuit(params)
            return qml.probs(wires=[0, 1, 2, 3])

        opt = qml.SPSAOptimizer(maxiter=10)
        params = np.random.normal(scale=0.1, size=(n_layers, n_wires, 3), requires_grad=True)

        with pytest.raises(
            ValueError,
            match="The objective function must be a scalar function for the gradient ",
        ):
            opt.step(cost, params)

    @pytest.mark.parametrize("f", univariate)
    def test_obj_func_not_a_scalar_function(self, f):
        """Test that if the objective function is not a
        scalar function, an error is raised."""
        spsa_opt = qml.SPSAOptimizer(10)
        args = np.array([[0.1, 0.2], [-0.1, -0.4]])

        with pytest.raises(
            ValueError,
            match="The objective function must be a scalar function for the gradient ",
        ):
            spsa_opt.step(f, args)

    @pytest.mark.tf
    def test_obj_func_not_a_scalar_function_with_tensorflow_interface(self):
        """Test that if the objective function is not a
        scalar function, an error is raised using tensorflow_interface."""

        n_wires = 4
        n_layers = 3
        dev = qml.device("default.qubit", wires=n_wires)

        def circuit(params):
            qml.StronglyEntanglingLayers(params, wires=list(range(n_wires)))

        @qml.qnode(dev)
        def cost(params):
            circuit(params)
            return qml.probs(wires=[0, 1, 2, 3])

        opt = qml.SPSAOptimizer(maxiter=10)
        params = np.random.normal(scale=0.1, size=(n_layers, n_wires, 3), requires_grad=True)

        with pytest.raises(
            ValueError,
            match="The objective function must be a scalar function for the gradient ",
        ):
            opt.step(cost, params)

    @pytest.mark.slow
    def test_lightning_device(self):
        """Test SPSAOptimizer implementation with lightning.qubit device."""
        coeffs = [0.2, -0.543, 0.4514]
        obs = [
            qml.PauliX(0) @ qml.PauliZ(1),
            qml.PauliZ(0) @ qml.Hadamard(2),
            qml.PauliX(3) @ qml.PauliZ(1),
        ]
        H = qml.Hamiltonian(coeffs, obs)
        num_qubits = 4
        dev = qml.device("lightning.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def cost_fun(params, num_qubits=1):
            qml.BasisState([1, 1, 0, 0], wires=range(num_qubits))

            assert num_qubits == 4

            for i in range(num_qubits):
                qml.Rot(*params[i], wires=0)
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[2, 0])
                qml.CNOT(wires=[3, 1])
            return qml.expval(H)

        init_params = np.random.normal(0, np.pi, (num_qubits, 3), requires_grad=True)
        params = init_params

        init_energy = cost_fun(init_params, num_qubits)

        max_iterations = 100
        opt = qml.SPSAOptimizer(maxiter=max_iterations)
        for _ in range(max_iterations):
            params, energy = opt.step_and_cost(cost_fun, params, num_qubits=num_qubits)

        assert np.all(params != init_params)
        assert energy < init_energy

    @pytest.mark.slow
    def test_default_mixed_device(self):
        """Test SPSAOptimizer implementation with default.mixed device."""
        n_qubits = 1
        max_iterations = 400
        dev = qml.device("default.mixed", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.AmplitudeDamping(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        opt = qml.SPSAOptimizer(maxiter=max_iterations, c=0.3)

        init_params = np.random.normal(scale=0.1, size=(2,), requires_grad=True)
        params = init_params
        init_circuit_res = circuit(params)

        for _ in range(max_iterations):
            params, circuit_res = opt.step_and_cost(circuit, params)

        assert np.all(params != init_params)
        assert circuit_res < init_circuit_res

    def test_not_A_nor_maxiter_provided(self):
        """Test that if the objective function is not a
        scalar function, an error is raised."""
        with pytest.raises(
            TypeError,
            match="One of the parameters maxiter or A must be provided.",
        ):
            qml.SPSAOptimizer()
