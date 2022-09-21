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
"""Unit tests for the autograd interface"""
import functools
import importlib
import sys

import autograd
import pytest
from pennylane import numpy as np
from pennylane.operation import Observable, AnyWires

import pennylane as qml
from pennylane.devices import DefaultQubit
from pennylane.gradients import finite_diff, param_shift
from pennylane.interfaces import execute

pytestmark = pytest.mark.autograd


class TestAutogradExecuteUnitTests:
    """Unit tests for autograd execution"""

    def test_import_error(self, mocker):
        """Test that an exception is caught on import error"""

        mock = mocker.patch.object(autograd.extend, "defvjp")
        mock.side_effect = ImportError()

        try:
            del sys.modules["pennylane.interfaces.autograd"]
        except:
            pass

        dev = qml.device("default.qubit", wires=2, shots=None)

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliY(1))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="autograd not found. Please install the latest version "
            "of autograd to enable the 'autograd' interface",
        ):
            qml.execute([tape], dev, gradient_fn=param_shift, interface="autograd")

    def test_jacobian_options(self, mocker, tol):
        """Test setting jacobian options"""
        spy = mocker.spy(qml.gradients, "param_shift")

        a = np.array([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        def cost(a, device):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                device,
                gradient_fn=param_shift,
                gradient_kwargs={"shifts": [(np.pi / 4,)] * 2},
            )[0]

        res = qml.jacobian(cost)(a, device=dev)

        for args in spy.call_args_list:
            assert args[1]["shifts"] == [(np.pi / 4,)] * 2

    def test_incorrect_mode(self):
        """Test that an error is raised if a gradient transform
        is used with mode=forward"""
        a = np.array([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        def cost(a, device):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute([tape], device, gradient_fn=param_shift, mode="forward")[0]

        with pytest.raises(
            ValueError, match="Gradient transforms cannot be used with mode='forward'"
        ):
            res = qml.jacobian(cost)(a, device=dev)

    def test_unknown_interface(self):
        """Test that an error is raised if the interface is unknown"""
        a = np.array([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        def cost(a, device):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute([tape], device, gradient_fn=param_shift, interface="None")[0]

        with pytest.raises(ValueError, match="Unknown interface"):
            cost(a, device=dev)

    def test_forward_mode(self, mocker):
        """Test that forward mode uses the `device.execute_and_gradients` pathway"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(dev, "execute_and_gradients")

        def cost(a):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                dev,
                gradient_fn="device",
                gradient_kwargs={"method": "adjoint_jacobian", "use_device_state": True},
            )[0]

        a = np.array([0.1, 0.2], requires_grad=True)
        cost(a)

        # adjoint method only performs a single device execution, but gets both result and gradient
        assert dev.num_executions == 1
        spy.assert_called()

    def test_backward_mode(self, mocker):
        """Test that backward mode uses the `device.batch_execute` and `device.gradients` pathway"""
        dev = qml.device("default.qubit", wires=1)
        spy_execute = mocker.spy(qml.devices.DefaultQubit, "batch_execute")
        spy_gradients = mocker.spy(qml.devices.DefaultQubit, "gradients")

        def cost(a):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                dev,
                gradient_fn="device",
                mode="backward",
                gradient_kwargs={"method": "adjoint_jacobian"},
            )[0]

        a = np.array([0.1, 0.2], requires_grad=True)
        cost(a)

        assert dev.num_executions == 1
        spy_execute.assert_called()
        spy_gradients.assert_not_called()

        qml.jacobian(cost)(a)
        spy_gradients.assert_called()


class TestBatchTransformExecution:
    """Tests to ensure batch transforms can be correctly executed
    via qml.execute and map_batch_transform"""

    def test_no_batch_transform(self, mocker):
        """Test that batch transforms can be disabled and enabled"""
        dev = qml.device("default.qubit", wires=2, shots=100000)

        H = qml.PauliZ(0) @ qml.PauliZ(1) - qml.PauliX(0)
        x = 0.6
        y = 0.2

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        spy = mocker.spy(dev, "batch_transform")

        with pytest.raises(AssertionError, match="Hamiltonian must be used with shots=None"):
            qml.execute([tape], dev, None, device_batch_transform=False)

        spy.assert_not_called()

        res = qml.execute([tape], dev, None, device_batch_transform=True)
        spy.assert_called()
        assert np.allclose(res[0], np.cos(y), atol=0.1)


class TestCaching:
    """Test for caching behaviour"""

    def test_cache_maxsize(self, mocker):
        """Test the cachesize property of the cache"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(qml.interfaces, "cache_execute")

        def cost(a, cachesize):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.probs(wires=0)

            return execute([tape], dev, gradient_fn=param_shift, cachesize=cachesize)[0]

        params = np.array([0.1, 0.2])
        qml.jacobian(cost)(params, cachesize=2)
        cache = spy.call_args[0][1]

        assert cache.maxsize == 2
        assert cache.currsize == 2
        assert len(cache) == 2

    def test_custom_cache(self, mocker):
        """Test the use of a custom cache object"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(qml.interfaces, "cache_execute")

        def cost(a, cache):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.probs(wires=0)

            return execute([tape], dev, gradient_fn=param_shift, cache=cache)[0]

        custom_cache = {}
        params = np.array([0.1, 0.2])
        qml.jacobian(cost)(params, cache=custom_cache)

        cache = spy.call_args[0][1]
        assert cache is custom_cache

    def test_caching_param_shift(self, tol):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum."""
        dev = qml.device("default.qubit", wires=1)

        def cost(a, cache):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.probs(wires=0)

            return execute([tape], dev, gradient_fn=param_shift, cache=cache)[0]

        # Without caching, 9 evaluations would be required to compute
        # the Jacobian: 1 (forward pass) + 2 (backward pass) * (2 shifts * 2 params)
        #
        # However, the jacobian is being cached in the interface by default,
        # hence we do 5 evaluations
        params = np.array([0.1, 0.2])
        qml.jacobian(cost)(params, cache=None)
        assert dev.num_executions == 5

        # With caching, 5 evaluations are required to compute
        # the Jacobian: 1 (forward pass) + (2 shifts * 2 params)
        dev._num_executions = 0
        jac_fn = qml.jacobian(cost)
        grad1 = jac_fn(params, cache=True)
        assert dev.num_executions == 5

        # Check that calling the cost function again
        # continues to evaluate the device (that is, the cache
        # is emptied between calls)
        grad2 = jac_fn(params, cache=True)
        assert dev.num_executions == 10
        assert np.allclose(grad1, grad2, atol=tol, rtol=0)

        # Check that calling the cost function again
        # with different parameters produces a different Jacobian
        grad2 = jac_fn(2 * params, cache=True)
        assert dev.num_executions == 15
        assert not np.allclose(grad1, grad2, atol=tol, rtol=0)

    @pytest.mark.parametrize("num_params", [2, 3])
    def test_caching_param_shift_hessian(self, num_params, tol):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum
        when computing Hessians."""
        dev = qml.device("default.qubit", wires=2)
        params = np.arange(1, num_params + 1) / 10

        N = len(params)

        def cost(x, cache):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])

                for i in range(2, num_params):
                    qml.RZ(x[i], wires=[i % 2])

                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            return execute([tape], dev, gradient_fn=param_shift, cache=cache, max_diff=2)[0]

        # No caching: number of executions is not ideal
        hess1 = qml.jacobian(qml.grad(cost))(params, cache=False)

        if num_params == 2:
            # compare to theoretical result
            x, y, *_ = params
            expected = np.array(
                [
                    [2 * np.cos(2 * x) * np.sin(y) ** 2, np.sin(2 * x) * np.sin(2 * y)],
                    [np.sin(2 * x) * np.sin(2 * y), -2 * np.cos(x) ** 2 * np.cos(2 * y)],
                ]
            )
            assert np.allclose(expected, hess1, atol=tol, rtol=0)

        expected_runs = 1  # forward pass

        # Jacobian of an involutory observable:
        # ------------------------------------
        #
        # 2 * N execs: evaluate the analytic derivative of <A>
        # 1 execs: Get <A>, the expectation value of the tape with unshifted parameters.
        num_shifted_evals = 2 * N
        runs_for_jacobian = num_shifted_evals + 1
        expected_runs += runs_for_jacobian

        # Each tape used to compute the Jacobian is then shifted again
        expected_runs += runs_for_jacobian * num_shifted_evals
        assert dev.num_executions == expected_runs

        # Use caching: number of executions is ideal
        dev._num_executions = 0
        hess2 = qml.jacobian(qml.grad(cost))(params, cache=True)
        assert np.allclose(hess1, hess2, atol=tol, rtol=0)

        expected_runs_ideal = 1  # forward pass
        expected_runs_ideal += 2 * N  # Jacobian
        expected_runs_ideal += N + 1  # Hessian diagonal
        expected_runs_ideal += 4 * N * (N - 1) // 2  # Hessian off-diagonal
        assert dev.num_executions == expected_runs_ideal
        assert expected_runs_ideal < expected_runs

    def test_caching_adjoint_backward(self):
        """Test that caching reduces the number of adjoint evaluations
        when mode=backward"""
        dev = qml.device("default.qubit", wires=2)
        params = np.array([0.1, 0.2, 0.3])

        def cost(a, cache):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            return execute(
                [tape],
                dev,
                gradient_fn="device",
                cache=cache,
                mode="backward",
                gradient_kwargs={"method": "adjoint_jacobian"},
            )[0]

        # Without caching, 3 evaluations are required.
        # 1 for the forward pass, and one per output dimension
        # on the backward pass.
        qml.jacobian(cost)(params, cache=None)
        assert dev.num_executions == 3

        # With caching, only 2 evaluations are required. One
        # for the forward pass, and one for the backward pass.
        dev._num_executions = 0
        jac_fn = qml.jacobian(cost)
        grad1 = jac_fn(params, cache=True)
        assert dev.num_executions == 2


execute_kwargs = [
    {"gradient_fn": param_shift},
    {
        "gradient_fn": "device",
        "mode": "forward",
        "gradient_kwargs": {"method": "adjoint_jacobian", "use_device_state": True},
    },
    {
        "gradient_fn": "device",
        "mode": "backward",
        "gradient_kwargs": {"method": "adjoint_jacobian"},
    },
]


@pytest.mark.parametrize("execute_kwargs", execute_kwargs)
class TestAutogradExecuteIntegration:
    """Test the autograd interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs):
        """Test execution"""
        dev = qml.device("default.qubit", wires=1)

        def cost(a, b):
            with qml.tape.QuantumTape() as tape1:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            with qml.tape.QuantumTape() as tape2:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            return execute([tape1, tape2], dev, **execute_kwargs)

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        res = cost(a, b)

        assert len(res) == 2
        assert res[0].shape == (1,)
        assert res[1].shape == (1,)

    def test_scalar_jacobian(self, execute_kwargs, tol):
        """Test scalar jacobian calculation"""
        a = np.array(0.1, requires_grad=True)
        dev = qml.device("default.qubit", wires=2)

        def cost(a):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.expval(qml.PauliZ(0))
            return execute([tape], dev, **execute_kwargs)[0]

        res = qml.jacobian(cost)(a)
        assert res.shape == (1,)

        # compare to standard tape jacobian
        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = [0]
        tapes, fn = param_shift(tape)
        expected = fn(dev.batch_execute(tapes))

        assert expected.shape == (1, 1)
        assert np.allclose(res, np.squeeze(expected), atol=tol, rtol=0)

    def test_jacobian(self, execute_kwargs, tol):
        """Test jacobian calculation"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        def cost(a, b, device):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliY(1))
            return execute([tape], device, **execute_kwargs)[0]

        dev = qml.device("default.qubit", wires=2)

        res = cost(a, b, device=dev)
        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(cost)(a, b, device=dev)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (2,)
        assert res[1].shape == (2,)

        expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
        assert all(np.allclose(_r, _e, atol=tol, rtol=0) for _r, _e in zip(res, expected))

    def test_tape_no_parameters(self, execute_kwargs, tol):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""
        dev = qml.device("default.qubit", wires=1)

        def cost(params):
            with qml.tape.QuantumTape() as tape1:
                qml.Hadamard(0)
                qml.expval(qml.PauliX(0))

            with qml.tape.QuantumTape() as tape2:
                qml.RY(np.array(0.5, requires_grad=False), wires=0)
                qml.expval(qml.PauliZ(0))

            with qml.tape.QuantumTape() as tape3:
                qml.RY(params[0], wires=0)
                qml.RX(params[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return sum(execute([tape1, tape2, tape3], dev, **execute_kwargs))

        params = np.array([0.1, 0.2], requires_grad=True)
        x, y = params

        res = cost(params)
        expected = 1 + np.cos(0.5) + np.cos(x) * np.cos(y)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = qml.grad(cost)(params)
        expected = [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_reusing_quantum_tape(self, execute_kwargs, tol):
        """Test re-using a quantum tape by passing new parameters"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliY(1))

        assert tape.trainable_params == [0, 1]

        def cost(a, b):
            tape.set_parameters([a, b])
            return execute([tape], dev, **execute_kwargs)[0]

        jac_fn = qml.jacobian(cost)
        jac = jac_fn(a, b)

        a = np.array(0.54, requires_grad=True)
        b = np.array(0.8, requires_grad=True)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls
        res2 = cost(2 * a, b)
        expected = [np.cos(2 * a), -np.cos(2 * a) * np.sin(b)]
        assert np.allclose(res2, expected, atol=tol, rtol=0)

        jac_fn = qml.jacobian(lambda a, b: cost(2 * a, b))
        jac = jac_fn(a, b)
        expected = (
            [-2 * np.sin(2 * a), 2 * np.sin(2 * a) * np.sin(b)],
            [0, -np.cos(2 * a) * np.cos(b)],
        )
        assert isinstance(jac, tuple) and len(jac) == 2
        assert all(np.allclose(_j, _e, atol=tol, rtol=0) for _j, _e in zip(jac, expected))

    def test_classical_processing(self, execute_kwargs, tol):
        """Test classical processing within the quantum tape"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        c = np.array(0.3, requires_grad=True)

        def cost(a, b, c, device):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a * c, wires=0)
                qml.RZ(b, wires=0)
                qml.RX(c + c**2 + np.sin(a), wires=0)
                qml.expval(qml.PauliZ(0))

            return execute([tape], device, **execute_kwargs)[0]

        dev = qml.device("default.qubit", wires=2)
        res = qml.jacobian(cost)(a, b, c, device=dev)
        # Only two arguments are trainable
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (1,)
        assert res[1].shape == (1,)

    def test_no_trainable_parameters(self, execute_kwargs, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        a = np.array(0.1, requires_grad=False)
        b = np.array(0.2, requires_grad=False)

        def cost(a, b, device):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))
            return execute([tape], device, **execute_kwargs)[0]

        dev = qml.device("default.qubit", wires=2)
        res = cost(a, b, device=dev)
        assert res.shape == (2,)

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            res = qml.jacobian(cost)(a, b, device=dev)
        assert len(res) == 0

        def loss(a, b):
            return np.sum(cost(a, b, device=dev))

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            res = qml.grad(loss)(a, b)

        assert np.allclose(res, 0)

    def test_matrix_parameter(self, execute_kwargs, tol):
        """Test that the autograd interface works correctly
        with a matrix parameter"""
        U = np.array([[0, 1], [1, 0]], requires_grad=False)
        a = np.array(0.1, requires_grad=True)

        def cost(a, U, device):
            with qml.tape.QuantumTape() as tape:
                qml.QubitUnitary(U, wires=0)
                qml.RY(a, wires=0)
                qml.expval(qml.PauliZ(0))
            return execute([tape], device, **execute_kwargs)[0]

        dev = qml.device("default.qubit", wires=2)
        res = cost(a, U, device=dev)
        assert np.allclose(res, -np.cos(a), atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(a, U, device=dev)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, execute_kwargs, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""

        class U3(qml.U3):
            def expand(self):
                tape = qml.tape.QuantumTape()
                theta, phi, lam = self.data
                wires = self.wires
                tape._ops += [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]
                return tape

        def cost_fn(a, p, device):
            tape = qml.tape.QuantumTape()

            with tape:
                qml.RX(a, wires=0)
                U3(*p, wires=0)
                qml.expval(qml.PauliX(0))

            tape = tape.expand(stop_at=lambda obj: device.supports_operation(obj.name))
            return execute([tape], device, **execute_kwargs)[0]

        a = np.array(0.1, requires_grad=False)
        p = np.array([0.1, 0.2, 0.3], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)
        res = cost_fn(a, p, device=dev)
        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost_fn)
        res = jac_fn(a, p, device=dev)
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

    def test_probability_differentiation(self, execute_kwargs, tol):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""

        if execute_kwargs["gradient_fn"] == "device":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        def cost(x, y, device):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=[0])
                qml.probs(wires=[1])

            return execute([tape], device, **execute_kwargs)[0]

        dev = qml.device("default.qubit", wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        res = cost(x, y, device=dev)
        expected = np.array(
            [
                [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2],
                [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(x, y, device=dev)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (2, 2)
        assert res[1].shape == (2, 2)

        expected = (
            np.array(
                [
                    [-np.sin(x) / 2, np.sin(x) / 2],
                    [-np.sin(x) * np.cos(y) / 2, np.sin(x) * np.cos(y) / 2],
                ]
            ),
            np.array(
                [
                    [0, 0],
                    [-np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2],
                ]
            ),
        )

        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_ragged_differentiation(self, execute_kwargs, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if execute_kwargs["gradient_fn"] == "device":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        def cost(x, y, device):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            return execute([tape], device, **execute_kwargs)[0]

        dev = qml.device("default.qubit", wires=2)
        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

        res = cost(x, y, device=dev)
        expected = np.array(
            [np.cos(x), (1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(x, y, device=dev)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (3,)
        assert res[1].shape == (3,)

        expected = (
            np.array([-np.sin(x), -np.sin(x) * np.cos(y) / 2, np.sin(x) * np.cos(y) / 2]),
            np.array([0, -np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_sampling(self, execute_kwargs):
        """Test sampling works as expected"""
        if execute_kwargs["gradient_fn"] == "device" and execute_kwargs["mode"] == "forward":
            pytest.skip("Adjoint differentiation does not support samples")

        def cost(x, device):
            with qml.tape.QuantumTape() as tape:
                qml.Hadamard(wires=[0])
                qml.CNOT(wires=[0, 1])
                qml.sample(qml.PauliZ(0))
                qml.sample(qml.PauliX(1))

            return execute([tape], device, **execute_kwargs)[0]

        dev = qml.device("default.qubit", wires=2, shots=10)
        x = np.array(0.543, requires_grad=True)
        res = cost(x, device=dev)
        assert res.shape == (2, 10)


class TestHigherOrderDerivatives:
    """Test that the autograd execute function can be differentiated"""

    @pytest.mark.parametrize(
        "params",
        [
            np.array([0.543, -0.654], requires_grad=True),
            np.array([0, -0.654], requires_grad=True),
            np.array([-2.0, 0], requires_grad=True),
        ],
    )
    def test_parameter_shift_hessian(self, params, tol):
        """Tests that the output of the parameter-shift transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape1:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            with qml.tape.QuantumTape() as tape2:
                qml.RX(x[0], wires=0)
                qml.RY(x[0], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=1)

            result = execute([tape1, tape2], dev, gradient_fn=param_shift, max_diff=2)
            return result[0] + result[1][0, 0]

        res = cost_fn(params)
        x, y = params
        expected = 0.5 * (3 + np.cos(x) ** 2 * np.cos(2 * y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.grad(cost_fn)(params)
        expected = np.array(
            [-np.cos(x) * np.cos(2 * y) * np.sin(x), -np.cos(x) ** 2 * np.sin(2 * y)]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(qml.grad(cost_fn))(params)
        expected = np.array(
            [
                [-np.cos(2 * x) * np.cos(2 * y), np.sin(2 * x) * np.sin(2 * y)],
                [np.sin(2 * x) * np.sin(2 * y), -2 * np.cos(x) ** 2 * np.cos(2 * y)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_adjoint_hessian(self, tol):
        """Since the adjoint hessian is not a differentiable transform,
        higher-order derivatives are not supported."""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                dev,
                gradient_fn="device",
                gradient_kwargs={"method": "adjoint_jacobian", "use_device_state": True},
            )[0]

        with pytest.warns(UserWarning, match="Output seems independent"):
            res = qml.jacobian(qml.grad(cost_fn))(params)

        assert np.allclose(res, np.zeros([2, 2]), atol=tol, rtol=0)

    def test_max_diff(self, tol):
        """Test that setting the max_diff parameter blocks higher-order
        derivatives"""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape1:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            with qml.tape.QuantumTape() as tape2:
                qml.RX(x[0], wires=0)
                qml.RY(x[0], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=1)

            result = execute([tape1, tape2], dev, gradient_fn=param_shift, max_diff=1)
            return result[0] + result[1][0, 0]

        res = cost_fn(params)
        x, y = params
        expected = 0.5 * (3 + np.cos(x) ** 2 * np.cos(2 * y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.grad(cost_fn)(params)
        expected = np.array(
            [-np.cos(x) * np.cos(2 * y) * np.sin(x), -np.cos(x) ** 2 * np.sin(2 * y)]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        with pytest.warns(UserWarning, match="Output seems independent"):
            res = qml.jacobian(qml.grad(cost_fn))(params)

        expected = np.zeros([2, 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestOverridingShots:
    """Test overriding shots on execution"""

    def test_changing_shots(self, mocker, tol):
        """Test that changing shots works on execution"""
        dev = qml.device("default.qubit", wires=2, shots=None)
        a, b = np.array([0.543, -0.654], requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliY(1))

        spy = mocker.spy(dev, "sample")

        # execute with device default shots (None)
        res = execute([tape], dev, gradient_fn=param_shift)
        assert np.allclose(res, -np.cos(a) * np.sin(b), atol=tol, rtol=0)
        spy.assert_not_called()

        # execute with shots=100
        res = execute([tape], dev, gradient_fn=param_shift, override_shots=100)
        spy.assert_called()
        assert spy.spy_return.shape == (100,)

        # device state has been unaffected
        assert dev.shots is None
        spy = mocker.spy(dev, "sample")
        res = execute([tape], dev, gradient_fn=param_shift)
        assert np.allclose(res, -np.cos(a) * np.sin(b), atol=tol, rtol=0)
        spy.assert_not_called()

    def test_overriding_shots_with_same_value(self, mocker):
        """Overriding shots with the same value as the device will have no effect"""
        dev = qml.device("default.qubit", wires=2, shots=123)
        a, b = np.array([0.543, -0.654], requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliY(1))

        spy = mocker.Mock(wraps=qml.Device.shots.fset)
        mock_property = qml.Device.shots.setter(spy)
        mocker.patch.object(qml.Device, "shots", mock_property)

        res = execute([tape], dev, gradient_fn=param_shift, override_shots=123)
        # overriden shots is the same, no change
        spy.assert_not_called()

        res = execute([tape], dev, gradient_fn=param_shift, override_shots=100)
        # overriden shots is not the same, shots were changed
        spy.assert_called()

        # shots were temporarily set to the overriden value
        assert spy.call_args_list[0][0] == (dev, 100)
        # shots were then returned to the built-in value
        assert spy.call_args_list[1][0] == (dev, 123)

    def test_overriding_device_with_shot_vector(self):
        """Overriding a device that has a batch of shots set
        results in original shots being returned after execution"""
        dev = qml.device("default.qubit", wires=2, shots=[10, (1, 3), 5])

        assert dev.shots == 18
        assert dev._shot_vector == [(10, 1), (1, 3), (5, 1)]

        a, b = np.array([0.543, -0.654], requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliY(1))

        res = execute([tape], dev, gradient_fn=param_shift, override_shots=100)[0]
        assert len(res) == 1

        # device is unchanged
        assert dev.shots == 18
        assert dev._shot_vector == [(10, 1), (1, 3), (5, 1)]

        res = execute([tape], dev, gradient_fn=param_shift)[0]
        assert len(res) == 5

    def test_gradient_integration(self, tol):
        """Test that temporarily setting the shots works
        for gradient computations"""
        dev = qml.device("default.qubit", wires=2, shots=None)
        a, b = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(a, b, shots):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliY(1))

            return execute([tape], dev, gradient_fn=param_shift, override_shots=shots)[0]

        res = qml.jacobian(cost_fn)(a, b, shots=[10000, 10000, 10000])
        assert dev.shots is None
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (3,)
        assert res[1].shape == (3,)

        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert all(
            np.allclose(np.mean(r, axis=0), e, atol=0.1, rtol=0) for r, e in zip(res, expected)
        )


execute_kwargs = [
    {"gradient_fn": param_shift},
    {"gradient_fn": finite_diff},
]


@pytest.mark.parametrize("execute_kwargs", execute_kwargs)
class TestHamiltonianWorkflows:
    """Test that tapes ending with expectations
    of Hamiltonians provide correct results and gradients"""

    @pytest.fixture
    def cost_fn(self, execute_kwargs):
        """Cost function for gradient tests"""

        def _cost_fn(weights, coeffs1, coeffs2, dev=None):
            obs1 = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
            H1 = qml.Hamiltonian(coeffs1, obs1)

            obs2 = [qml.PauliZ(0)]
            H2 = qml.Hamiltonian(coeffs2, obs2)

            with qml.tape.QuantumTape() as tape:
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(H1)
                qml.expval(H2)

            return execute([tape], dev, **execute_kwargs)[0]

        return _cost_fn

    @staticmethod
    def cost_fn_expected(weights, coeffs1, coeffs2):
        """Analytic value of cost_fn above"""
        a, b, c = coeffs1
        d = coeffs2[0]
        x, y = weights
        return [-c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y)), d * np.cos(x)]

    @staticmethod
    def cost_fn_jacobian(weights, coeffs1, coeffs2):
        """Analytic jacobian of cost_fn above"""
        a, b, c = coeffs1
        d = coeffs2[0]
        x, y = weights
        return np.array(
            [
                [
                    -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
                    b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
                    np.cos(x),
                    np.cos(x) * np.sin(y),
                    -(np.sin(x) * np.sin(y)),
                    0,
                ],
                [-d * np.sin(x), 0, 0, 0, 0, np.cos(x)],
            ]
        )

    def test_multiple_hamiltonians_not_trainable(self, cost_fn, execute_kwargs, tol):
        coeffs1 = np.array([0.1, 0.2, 0.3], requires_grad=False)
        coeffs2 = np.array([0.7], requires_grad=False)
        weights = np.array([0.4, 0.5], requires_grad=True)
        dev = qml.device("default.qubit", wires=2)

        res = cost_fn(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.jacobian(cost_fn)(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)[:, :2]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_hamiltonians_trainable(self, cost_fn, execute_kwargs, tol):
        coeffs1 = np.array([0.1, 0.2, 0.3], requires_grad=True)
        coeffs2 = np.array([0.7], requires_grad=True)
        weights = np.array([0.4, 0.5], requires_grad=True)
        dev = qml.device("default.qubit", wires=2)

        res = cost_fn(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = np.hstack(qml.jacobian(cost_fn)(weights, coeffs1, coeffs2, dev=dev))
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestCustomJacobian:
    def test_custom_jacobians(self):
        class CustomJacobianDevice(DefaultQubit):
            @classmethod
            def capabilities(cls):
                capabilities = super().capabilities()
                capabilities["provides_jacobian"] = True
                return capabilities

            def jacobian(self, tape):
                return np.array([1.0, 2.0, 3.0, 4.0])

        dev = CustomJacobianDevice(wires=2)

        @qml.qnode(dev, diff_method="device")
        def circuit(v):
            qml.RX(v, wires=0)
            return qml.probs(wires=[0, 1])

        d_circuit = qml.jacobian(circuit, argnum=0)

        params = np.array(1.0, requires_grad=True)

        d_out = d_circuit(params)
        assert np.allclose(d_out, np.array([1.0, 2.0, 3.0, 4.0]))

    def test_custom_jacobians_2(self):
        """Test computing the gradient using the parameter-shift
        rule with a device that provides a jacobian"""

        class MyQubit(DefaultQubit):
            @classmethod
            def capabilities(cls):
                capabilities = super().capabilities().copy()
                capabilities.update(
                    provides_jacobian=True,
                )
                return capabilities

            def jacobian(self, *args, **kwargs):
                raise NotImplementedError()

        dev = MyQubit(wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", mode="backward")
        def qnode(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        res = qml.jacobian(qnode)(a, b)
        expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])

        assert np.allclose(res[0], expected[0])
        assert np.allclose(res[1], expected[1])


class SpecialObject:
    """SpecialObject
    A special object that conveniently encapsulates the return value of
    a special observable supported by a special device and which supports
    multiplication with scalars and addition.
    """

    def __init__(self, val):
        self.val = val

    def __mul__(self, other):
        new = SpecialObject(self.val)
        new *= other
        return new

    def __imul__(self, other):
        self.val *= other
        return self

    def __rmul__(self, other):
        return self * other

    def __iadd__(self, other):
        self.val += other.val if isinstance(other, self.__class__) else other
        return self

    def __add__(self, other):
        new = SpecialObject(self.val)
        new += other.val if isinstance(other, self.__class__) else other
        return new

    def __radd__(self, other):
        return self + other


class SpecialObservable(Observable):
    """SpecialObservable"""

    num_wires = AnyWires
    num_params = 0
    par_domain = None

    def diagonalizing_gates(self):
        """Diagonalizing gates"""
        return []


class DeviceSupportingSpecialObservable(DefaultQubit):
    name = "Device supporting SpecialObservable"
    short_name = "default.qubit.specialobservable"
    observables = DefaultQubit.observables.union({"SpecialObservable"})

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            provides_jacobian=True,
        )
        return capabilities

    def expval(self, observable, **kwargs):
        if self.analytic and isinstance(observable, SpecialObservable):
            val = super().expval(qml.PauliZ(wires=0), **kwargs)
            return SpecialObject(val)

        return super().expval(observable, **kwargs)

    def jacobian(self, tape):
        # we actually let pennylane do the work of computing the
        # jacobian for us but return it as a device jacobian
        gradient_tapes, fn = qml.gradients.param_shift(tape)
        tape_jacobian = fn(qml.execute(gradient_tapes, self, None))
        return tape_jacobian


@pytest.mark.autograd
class TestObservableWithObjectReturnType:
    """Unit tests for qnode returning a custom object"""

    def test_custom_return_type(self):
        """Test custom return values for a qnode"""

        dev = DeviceSupportingSpecialObservable(wires=1, shots=None)

        # force diff_method='parameter-shift' because otherwise
        # PennyLane swaps out dev for default.qubit.autograd
        @qml.qnode(dev, diff_method="parameter-shift")
        def qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(SpecialObservable(wires=0))

        @qml.qnode(dev, diff_method="parameter-shift")
        def reference_qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        out = qnode(0.2)
        assert isinstance(out, np.ndarray)
        assert isinstance(out.item(), SpecialObject)
        assert np.isclose(out.item().val, reference_qnode(0.2))

    def test_jacobian_with_custom_return_type(self):
        """Test differentiation of a QNode on a device supporting a
        special observable that returns an object rather than a number."""

        dev = DeviceSupportingSpecialObservable(wires=1, shots=None)

        # force diff_method='parameter-shift' because otherwise
        # PennyLane swaps out dev for default.qubit.autograd
        @qml.qnode(dev, diff_method="parameter-shift")
        def qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(SpecialObservable(wires=0))

        @qml.qnode(dev, diff_method="parameter-shift")
        def reference_qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        reference_jac = (qml.jacobian(reference_qnode)(np.array(0.2, requires_grad=True)),)

        assert np.isclose(
            reference_jac,
            qml.jacobian(qnode)(np.array(0.2, requires_grad=True)).item().val,
        )

        # now check that also the device jacobian works with a custom return type
        @qml.qnode(dev, diff_method="device")
        def device_gradient_qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(SpecialObservable(wires=0))

        assert np.isclose(
            reference_jac,
            qml.jacobian(device_gradient_qnode)(np.array(0.2, requires_grad=True)).item().val,
        )
