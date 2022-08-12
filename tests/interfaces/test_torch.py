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
"""Unit tests for the Torch interface"""
import functools

import numpy as np
import pytest

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")

import pennylane as qml
from pennylane.gradients import finite_diff, param_shift
from pennylane.interfaces import execute


class TestTorchExecuteUnitTests:
    """Unit tests for torch execution"""

    def test_jacobian_options(self, mocker, tol):
        """Test setting jacobian options"""
        spy = mocker.spy(qml.gradients, "param_shift")

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            qml.expval(qml.PauliZ(0))

        res = execute(
            [tape],
            dev,
            gradient_fn=param_shift,
            gradient_kwargs={"shifts": [(np.pi / 4,)] * 2},
            interface="torch",
        )[0]

        res.backward()

        for args in spy.call_args_list:
            assert args[1]["shift"] == [(np.pi / 4,)] * 2

    def test_incorrect_mode(self):
        """Test that an error is raised if a gradient transform
        is used with mode=forward"""
        a = torch.tensor([0.1, 0.2], requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            qml.expval(qml.PauliZ(0))

        with pytest.raises(
            ValueError, match="Gradient transforms cannot be used with mode='forward'"
        ):
            execute([tape], dev, gradient_fn=param_shift, mode="forward", interface="torch")[0]

    def test_forward_mode_reuse_state(self, mocker):
        """Test that forward mode uses the `device.execute_and_gradients` pathway
        while reusing the quantum state."""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(dev, "execute_and_gradients")

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            qml.expval(qml.PauliZ(0))

        res = execute(
            [tape],
            dev,
            gradient_fn="device",
            gradient_kwargs={"method": "adjoint_jacobian", "use_device_state": True},
            interface="torch",
        )[0]

        # adjoint method only performs a single device execution, but gets both result and gradient
        assert dev.num_executions == 1
        spy.assert_called()

    def test_forward_mode(self, mocker):
        """Test that forward mode uses the `device.execute_and_gradients` pathway"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(dev, "execute_and_gradients")

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            qml.expval(qml.PauliZ(0))

        res = execute(
            [tape],
            dev,
            gradient_fn="device",
            gradient_kwargs={"method": "adjoint_jacobian"},
            interface="torch",
        )[0]

        # two device executions; one for the value, one for the Jacobian
        assert dev.num_executions == 2
        spy.assert_called()

    def test_backward_mode(self, mocker):
        """Test that backward mode uses the `device.batch_execute` and `device.gradients` pathway"""
        dev = qml.device("default.qubit", wires=1)
        spy_execute = mocker.spy(qml.devices.DefaultQubit, "batch_execute")
        spy_gradients = mocker.spy(qml.devices.DefaultQubit, "gradients")

        a = torch.tensor([0.1, 0.2], requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            qml.expval(qml.PauliZ(0))

        res = execute(
            [tape],
            dev,
            gradient_fn="device",
            mode="backward",
            gradient_kwargs={"method": "adjoint_jacobian"},
            interface="torch",
        )[0]

        assert dev.num_executions == 1
        spy_execute.assert_called()
        spy_gradients.assert_not_called()

        res.backward()
        spy_gradients.assert_called()


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

            return execute(
                [tape], dev, gradient_fn=param_shift, cachesize=cachesize, interface="torch"
            )[0][0, 0]

        params = torch.tensor([0.1, 0.2], requires_grad=True)
        res = cost(params, cachesize=2)
        res.backward()
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

            return execute([tape], dev, gradient_fn=param_shift, cache=cache, interface="torch")[0][
                0, 0
            ]

        custom_cache = {}
        params = torch.tensor([0.1, 0.2], requires_grad=True)
        res = cost(params, cache=custom_cache)
        res.backward()

        cache = spy.call_args[0][1]
        assert cache is custom_cache

    def test_caching_param_shift(self, tol):
        """Test that, with the parameter-shift transform,
        Torch always uses the optimum number of evals when computing the Jacobian."""
        dev = qml.device("default.qubit", wires=1)

        def cost(a, cache):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.probs(wires=0)

            return execute([tape], dev, gradient_fn=param_shift, cache=cache, interface="torch")[0][
                0, 0
            ]

        # Without caching, 5 evaluations are required to compute
        # the Jacobian: 1 (forward pass) + (2 shifts * 2 params)
        params = torch.tensor([0.1, 0.2], requires_grad=True)
        torch.autograd.functional.jacobian(lambda p: cost(p, cache=None), params)
        assert dev.num_executions == 5

        # With caching, 5 evaluations are required to compute
        # the Jacobian: 1 (forward pass) + (2 shifts * 2 params)
        dev._num_executions = 0
        torch.autograd.functional.jacobian(lambda p: cost(p, cache=True), params)
        assert dev.num_executions == 5

    @pytest.mark.parametrize("num_params", [2, 3])
    def test_caching_param_shift_hessian(self, num_params, tol):
        """Test that, with the parameter-shift transform,
        caching reduces the number of evaluations to their optimum
        when computing Hessians."""
        dev = qml.device("default.qubit", wires=2)
        params = torch.tensor(np.arange(1, num_params + 1) / 10, requires_grad=True)

        N = len(params)

        def cost(x, cache):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])

                for i in range(2, num_params):
                    qml.RZ(x[i], wires=[i % 2])

                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            return execute(
                [tape], dev, gradient_fn=param_shift, cache=cache, interface="torch", max_diff=2
            )[0]

        # No caching: number of executions is not ideal
        hess1 = torch.autograd.functional.hessian(lambda x: cost(x, cache=None), params)

        if num_params == 2:
            # compare to theoretical result
            x, y, *_ = params.detach()
            expected = torch.tensor(
                [
                    [2 * np.cos(2 * x) * np.sin(y) ** 2, np.sin(2 * x) * np.sin(2 * y)],
                    [np.sin(2 * x) * np.sin(2 * y), -2 * np.cos(x) ** 2 * np.cos(2 * y)],
                ]
            )
            assert np.allclose(expected, hess1, atol=tol, rtol=0)

        expected_runs = 1  # forward pass
        expected_runs += 2 * N  # Jacobian
        expected_runs += 4 * N + 1  # Hessian diagonal
        expected_runs += 4 * N**2  # Hessian off-diagonal
        assert dev.num_executions == expected_runs

        # Use caching: number of executions is ideal
        dev._num_executions = 0
        hess2 = torch.autograd.functional.hessian(lambda x: cost(x, cache=True), params)
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
        params = torch.tensor([0.1, 0.2, 0.3])

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
                interface="torch",
            )[0]

        # Without caching, 3 evaluations are required.
        # 1 for the forward pass, and one per output dimension
        # on the backward pass.
        torch.autograd.functional.jacobian(lambda x: cost(x, cache=None), params)
        assert dev.num_executions == 3

        # With caching, only 2 evaluations are required. One
        # for the forward pass, and one for the backward pass.
        dev._num_executions = 0
        torch.autograd.functional.jacobian(lambda x: cost(x, cache=True), params)
        assert dev.num_executions == 2


torch_devices = [None]

if torch.cuda.is_available():
    torch_devices.append(torch.device("cuda"))


execute_kwargs = [
    {"gradient_fn": param_shift, "interface": "torch"},
    {
        "gradient_fn": "device",
        "mode": "forward",
        "gradient_kwargs": {"method": "adjoint_jacobian", "use_device_state": False},
        "interface": "torch",
    },
    {
        "gradient_fn": "device",
        "mode": "forward",
        "gradient_kwargs": {"method": "adjoint_jacobian", "use_device_state": True},
        "interface": "torch",
    },
    {
        "gradient_fn": "device",
        "mode": "backward",
        "gradient_kwargs": {"method": "adjoint_jacobian"},
        "interface": "torch",
    },
]


@pytest.mark.gpu
@pytest.mark.parametrize("torch_device", torch_devices)
@pytest.mark.parametrize("execute_kwargs", execute_kwargs)
class TestTorchExecuteIntegration:
    """Test the torch interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, torch_device, execute_kwargs):
        """Test that the execute function produces results with the expected shapes"""
        dev = qml.device("default.qubit", wires=1)
        a = torch.tensor(0.1, requires_grad=True, device=torch_device)
        b = torch.tensor(0.2, requires_grad=False, device=torch_device)

        with qml.tape.QuantumTape() as tape1:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.expval(qml.PauliZ(0))

        res = execute([tape1, tape2], dev, **execute_kwargs)

        assert len(res) == 2
        assert res[0].shape == (1,)
        assert res[1].shape == (1,)

    def test_scalar_jacobian(self, torch_device, execute_kwargs, tol):
        """Test scalar jacobian calculation by comparing two types of pipelines"""
        a = torch.tensor(0.1, requires_grad=True, dtype=torch.float64, device=torch_device)
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.expval(qml.PauliZ(0))

        res = execute([tape], dev, **execute_kwargs)[0]
        res.backward()

        # compare to backprop gradient
        def cost(a):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.expval(qml.PauliZ(0))

            dev = qml.device("default.qubit.autograd", wires=2)
            return dev.batch_execute([tape])[0]

        expected = qml.grad(cost, argnum=0)(0.1)
        assert torch.allclose(a.grad, torch.tensor(expected, device=torch_device), atol=tol, rtol=0)

    def test_jacobian(self, torch_device, execute_kwargs, tol):
        """Test jacobian calculation by checking against analytic values"""
        a_val = 0.1
        b_val = 0.2

        a = torch.tensor(a_val, requires_grad=True, device=torch_device)
        b = torch.tensor(b_val, requires_grad=True, device=torch_device)

        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.RZ(torch.tensor(0.543, device=torch_device), wires=0)
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliY(1))

        res = execute([tape], dev, **execute_kwargs)[0]
        assert tape.trainable_params == [1, 2]

        assert isinstance(res, torch.Tensor)
        assert res.shape == (2,)

        expected = torch.tensor(
            [np.cos(a_val), -np.cos(a_val) * np.sin(b_val)], device=torch_device
        )
        assert torch.allclose(res.detach(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)

        loss.backward()
        expected = torch.tensor(
            [-np.sin(a_val) + np.sin(a_val) * np.sin(b_val), -np.cos(a_val) * np.cos(b_val)],
            dtype=a.dtype,
            device=torch_device,
        )
        assert torch.allclose(a.grad, expected[0], atol=tol, rtol=0)
        assert torch.allclose(b.grad, expected[1], atol=tol, rtol=0)

    def test_tape_no_parameters(self, torch_device, execute_kwargs, tol):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""
        dev = qml.device("default.qubit", wires=1)
        params = torch.tensor([0.1, 0.2], requires_grad=True, device=torch_device)
        x, y = params.detach()

        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(0)
            qml.expval(qml.PauliX(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RY(0.5, wires=0)
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape3:
            qml.RY(params[0], wires=0)
            qml.RX(params[1], wires=0)
            qml.expval(qml.PauliZ(0))

        res = sum(execute([tape1, tape2, tape3], dev, **execute_kwargs))
        expected = torch.tensor(1 + np.cos(0.5), dtype=res.dtype) + torch.cos(x) * torch.cos(y)
        expected = expected.to(device=res.device)

        assert torch.allclose(res, expected, atol=tol, rtol=0)

        res.backward()
        grad = params.grad.detach()
        expected = torch.tensor(
            [-torch.cos(y) * torch.sin(x), -torch.cos(x) * torch.sin(y)],
            dtype=grad.dtype,
            device=grad.device,
        )
        assert torch.allclose(grad, expected, atol=tol, rtol=0)

    def test_reusing_quantum_tape(self, torch_device, execute_kwargs, tol):
        """Test re-using a quantum tape by passing new parameters"""
        a = torch.tensor(0.1, requires_grad=True, device=torch_device)
        b = torch.tensor(0.2, requires_grad=True, device=torch_device)

        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliY(1))

        assert tape.trainable_params == [0, 1]

        res = execute([tape], dev, **execute_kwargs)[0]
        loss = torch.sum(res)
        loss.backward()

        a_val = 0.54
        b_val = 0.8
        a = torch.tensor(a_val, requires_grad=True, device=torch_device)
        b = torch.tensor(b_val, requires_grad=True, device=torch_device)

        tape.set_parameters([2 * a, b])
        res2 = execute([tape], dev, **execute_kwargs)[0]

        expected = torch.tensor(
            [np.cos(2 * a_val), -np.cos(2 * a_val) * np.sin(b_val)],
            device=torch_device,
            dtype=res2.dtype,
        )
        assert torch.allclose(res2.detach(), expected, atol=tol, rtol=0)

        loss = torch.sum(res2)
        loss.backward()

        expected = torch.tensor(
            [
                -2 * np.sin(2 * a_val) + 2 * np.sin(2 * a_val) * np.sin(b_val),
                -np.cos(2 * a_val) * np.cos(b_val),
            ],
            dtype=a.dtype,
            device=torch_device,
        )

        assert torch.allclose(a.grad, expected[0], atol=tol, rtol=0)
        assert torch.allclose(b.grad, expected[1], atol=tol, rtol=0)

    def test_classical_processing(self, torch_device, execute_kwargs, tol):
        """Test the classical processing of gate parameters within the quantum tape"""
        p_val = [0.1, 0.2]
        params = torch.tensor(p_val, requires_grad=True, device=torch_device)

        dev = qml.device("default.qubit", wires=1)

        with qml.tape.QuantumTape() as tape:
            qml.RY(params[0] * params[1], wires=0)
            qml.RZ(0.2, wires=0)
            qml.RX(params[1] + params[1] ** 2 + torch.sin(params[0]), wires=0)
            qml.expval(qml.PauliZ(0))

        res = execute([tape], dev, **execute_kwargs)[0]

        assert tape.trainable_params == [0, 2]

        tape_params = torch.tensor([i.detach() for i in tape.get_parameters()], device=torch_device)
        expected = torch.tensor(
            [p_val[0] * p_val[1], p_val[1] + p_val[1] ** 2 + np.sin(p_val[0])],
            dtype=tape_params.dtype,
            device=torch_device,
        )

        assert torch.allclose(
            tape_params,
            expected,
            atol=tol,
            rtol=0,
        )

        res.backward()

        assert isinstance(params.grad, torch.Tensor)
        assert params.shape == (2,)

    def test_no_trainable_parameters(self, torch_device, execute_kwargs, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.2, wires=0)
            qml.RX(torch.tensor(0.1, device=torch_device), wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))

        res = execute([tape], dev, **execute_kwargs)[0]
        assert tape.trainable_params == []

        assert res.shape == (2,)
        assert isinstance(res, torch.Tensor)

        with pytest.raises(
            RuntimeError,
            match="element 0 of tensors does not require grad and does not have a grad_fn",
        ):
            res.backward()

    @pytest.mark.parametrize(
        "U", [torch.tensor([[0.0, 1.0], [1.0, 0.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])]
    )
    def test_matrix_parameter(self, torch_device, U, execute_kwargs, tol):
        """Test that the torch interface works correctly
        with a matrix parameter"""
        a_val = 0.1
        a = torch.tensor(a_val, requires_grad=True, device=torch_device)

        if isinstance(U, torch.Tensor) and torch_device is not None:
            U = U.to(torch_device)

        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            qml.expval(qml.PauliZ(0))

        res = execute([tape], dev, **execute_kwargs)[0]
        assert tape.trainable_params == [1]

        expected = torch.tensor(-np.cos(a_val), dtype=res.dtype, device=torch_device)
        assert torch.allclose(res.detach(), expected, atol=tol, rtol=0)

        res.backward()
        expected = torch.tensor([np.sin(a_val)], dtype=a.grad.dtype, device=torch_device)
        assert torch.allclose(a.grad, expected, atol=tol, rtol=0)

    def test_differentiable_expand(self, torch_device, execute_kwargs, tol):
        """Test that operation and nested tape expansion
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

        dev = qml.device("default.qubit", wires=1)
        a = np.array(0.1)
        p_val = [0.1, 0.2, 0.3]
        p = torch.tensor(p_val, requires_grad=True, device=torch_device)

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            qml.expval(qml.PauliX(0))

        res = execute([tape], dev, **execute_kwargs)[0]

        expected = torch.tensor(
            np.cos(a) * np.cos(p_val[1]) * np.sin(p_val[0])
            + np.sin(a)
            * (
                np.cos(p_val[2]) * np.sin(p_val[1])
                + np.cos(p_val[0]) * np.cos(p_val[1]) * np.sin(p_val[2])
            ),
            dtype=res.dtype,
            device=torch_device,
        )
        assert torch.allclose(res.detach(), expected, atol=tol, rtol=0)

        res.backward()
        expected = torch.tensor(
            [
                np.cos(p_val[1])
                * (np.cos(a) * np.cos(p_val[0]) - np.sin(a) * np.sin(p_val[0]) * np.sin(p_val[2])),
                np.cos(p_val[1]) * np.cos(p_val[2]) * np.sin(a)
                - np.sin(p_val[1])
                * (np.cos(a) * np.sin(p_val[0]) + np.cos(p_val[0]) * np.sin(a) * np.sin(p_val[2])),
                np.sin(a)
                * (
                    np.cos(p_val[0]) * np.cos(p_val[1]) * np.cos(p_val[2])
                    - np.sin(p_val[1]) * np.sin(p_val[2])
                ),
            ],
            dtype=p.grad.dtype,
            device=torch_device,
        )
        assert torch.allclose(p.grad, expected, atol=tol, rtol=0)

    def test_probability_differentiation(self, torch_device, execute_kwargs, tol):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""

        if execute_kwargs["gradient_fn"] == "device":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        dev = qml.device("default.qubit", wires=2)
        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True, device=torch_device)
        y = torch.tensor(y_val, requires_grad=True, device=torch_device)

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0])
            qml.probs(wires=[1])

        res = execute([tape], dev, **execute_kwargs)[0]

        expected = torch.tensor(
            [
                [np.cos(x_val / 2) ** 2, np.sin(x_val / 2) ** 2],
                [(1 + np.cos(x_val) * np.cos(y_val)) / 2, (1 - np.cos(x_val) * np.cos(y_val)) / 2],
            ],
            dtype=res.dtype,
            device=torch_device,
        )
        assert torch.allclose(res.detach(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)
        loss.backward()
        expected = torch.tensor(
            [
                -np.sin(x_val) / 2
                + np.sin(x_val) / 2
                - np.sin(x_val) * np.cos(y_val) / 2
                + np.cos(y_val) * np.sin(x_val) / 2,
                -np.cos(x_val) * np.sin(y_val) / 2 + np.cos(x_val) * np.sin(y_val) / 2,
            ],
            dtype=x.grad.dtype,
            device=torch_device,
        )
        assert torch.allclose(x.grad, expected[0], atol=tol, rtol=0)
        assert torch.allclose(y.grad, expected[1], atol=tol, rtol=0)

    def test_ragged_differentiation(self, torch_device, execute_kwargs, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if execute_kwargs["gradient_fn"] == "device":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        dev = qml.device("default.qubit", wires=2)
        x_val = 0.543
        y_val = -0.654
        x = torch.tensor(x_val, requires_grad=True, device=torch_device)
        y = torch.tensor(y_val, requires_grad=True, device=torch_device)

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[1])

        res = execute([tape], dev, **execute_kwargs)[0]

        expected = torch.tensor(
            [
                np.cos(x_val),
                (1 + np.cos(x_val) * np.cos(y_val)) / 2,
                (1 - np.cos(x_val) * np.cos(y_val)) / 2,
            ],
            dtype=res.dtype,
            device=torch_device,
        )
        assert torch.allclose(res.detach(), expected, atol=tol, rtol=0)

        loss = torch.sum(res)
        loss.backward()
        expected = torch.tensor(
            [
                -np.sin(x_val)
                + -np.sin(x_val) * np.cos(y_val) / 2
                + np.cos(y_val) * np.sin(x_val) / 2,
                -np.cos(x_val) * np.sin(y_val) / 2 + np.cos(x_val) * np.sin(y_val) / 2,
            ],
            dtype=x.grad.dtype,
            device=torch_device,
        )
        assert torch.allclose(x.grad, expected[0], atol=tol, rtol=0)
        assert torch.allclose(y.grad, expected[1], atol=tol, rtol=0)

    def test_sampling(self, torch_device, execute_kwargs):
        """Test sampling works as expected"""
        if execute_kwargs["gradient_fn"] == "device" and execute_kwargs["mode"] == "forward":
            pytest.skip("Adjoint differentiation does not support samples")

        dev = qml.device("default.qubit", wires=2, shots=10)

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.sample(qml.PauliZ(0))
            qml.sample(qml.PauliX(1))

        res = execute([tape], dev, **execute_kwargs)[0]

        assert res.shape == (2, 10)
        assert isinstance(res, torch.Tensor)

    def test_sampling_expval(self, torch_device, execute_kwargs):
        """Test sampling works as expected if combined with expectation values"""
        if execute_kwargs["gradient_fn"] == "device" and execute_kwargs["mode"] == "forward":
            pytest.skip("Adjoint differentiation does not support samples")

        dev = qml.device("default.qubit", wires=2, shots=10)

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.sample(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        res = execute([tape], dev, **execute_kwargs)[0]

        assert len(res) == 2
        assert isinstance(res, tuple)
        assert res[0].shape == (10,)
        assert isinstance(res[0], torch.Tensor)
        assert isinstance(res[1], torch.Tensor)

    def test_sampling_gradient_error(self, torch_device, execute_kwargs):
        """Test differentiating a tape with sampling results in an error"""
        if execute_kwargs["gradient_fn"] == "device" and execute_kwargs["mode"] == "forward":
            pytest.skip("Adjoint differentiation does not support samples")

        dev = qml.device("default.qubit", wires=1, shots=10)

        x = torch.tensor(0.65, requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.sample(qml.PauliZ(0))

        res = execute([tape], dev, **execute_kwargs)[0]

        with pytest.raises(
            RuntimeError,
            match="element 0 of tensors does not require grad and does not have a grad_fn",
        ):
            res.backward()

    def test_repeated_application_after_expand(self, torch_device, execute_kwargs, tol):
        """Test that the Torch interface continues to work after
        tape expansions"""
        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        weights = torch.ones((3,))

        with qml.tape.QuantumTape() as tape:
            qml.U3(*weights, wires=0)
            qml.expval(qml.PauliZ(wires=0))

        tape = tape.expand()
        res1 = execute([tape], dev, **execute_kwargs)[0]


@pytest.mark.parametrize("torch_device", torch_devices)
class TestHigherOrderDerivatives:
    """Test that the torch execute function can be differentiated"""

    @pytest.mark.parametrize(
        "params",
        [
            torch.tensor([0.543, -0.654], requires_grad=True),
            torch.tensor([0, -0.654], requires_grad=True),
            torch.tensor([-2.0, 0], requires_grad=True),
        ],
    )
    def test_parameter_shift_hessian(self, torch_device, params, tol):
        """Tests that the output of the parameter-shift transform
        can be differentiated using torch, yielding second derivatives."""
        dev = qml.device("default.qubit", wires=2)
        params = torch.tensor([0.543, -0.654], requires_grad=True, dtype=torch.float64)

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

            result = execute(
                [tape1, tape2], dev, gradient_fn=param_shift, interface="torch", max_diff=2
            )
            return result[0] + result[1][0, 0]

        res = cost_fn(params)
        x, y = params.detach()
        expected = torch.as_tensor(0.5 * (3 + np.cos(x) ** 2 * np.cos(2 * y)))
        assert torch.allclose(res, expected, atol=tol, rtol=0)

        res.backward()
        expected = torch.tensor(
            [-np.cos(x) * np.cos(2 * y) * np.sin(x), -np.cos(x) ** 2 * np.sin(2 * y)]
        )
        assert torch.allclose(params.grad.detach(), expected, atol=tol, rtol=0)

        res = torch.autograd.functional.hessian(cost_fn, params)
        expected = torch.tensor(
            [
                [-np.cos(2 * x) * np.cos(2 * y), np.sin(2 * x) * np.sin(2 * y)],
                [np.sin(2 * x) * np.sin(2 * y), -2 * np.cos(x) ** 2 * np.cos(2 * y)],
            ]
        )
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_hessian_vector_valued(self, torch_device, tol):
        """Test hessian calculation of a vector valued tape"""
        dev = qml.device("default.qubit", wires=1)

        def circuit(x):
            with qml.tape.QuantumTape() as tape:
                qml.RY(x[0], wires=0)
                qml.RX(x[1], wires=0)
                qml.probs(wires=0)

            return torch.stack(
                execute([tape], dev, gradient_fn=param_shift, interface="torch", max_diff=2)
            )

        x = torch.tensor([1.0, 2.0], requires_grad=True, device=torch_device)
        res = circuit(x)

        if torch_device is not None:
            a, b = x.detach().cpu().numpy()
        else:
            a, b = x.detach().numpy()

        expected_res = torch.tensor(
            [
                0.5 + 0.5 * np.cos(a) * np.cos(b),
                0.5 - 0.5 * np.cos(a) * np.cos(b),
            ],
            dtype=res.dtype,
            device=torch_device,
        )
        assert torch.allclose(res.detach(), expected_res, atol=tol, rtol=0)

        jac_fn = lambda x: torch.autograd.functional.jacobian(circuit, x, create_graph=True)

        g = jac_fn(x)

        hess = torch.autograd.functional.jacobian(jac_fn, x)

        expected_g = torch.tensor(
            [
                [-0.5 * np.sin(a) * np.cos(b), -0.5 * np.cos(a) * np.sin(b)],
                [0.5 * np.sin(a) * np.cos(b), 0.5 * np.cos(a) * np.sin(b)],
            ],
            dtype=g.dtype,
            device=torch_device,
        )
        assert torch.allclose(g.detach(), expected_g, atol=tol, rtol=0)

        expected_hess = torch.tensor(
            [
                [
                    [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.sin(a) * np.sin(b)],
                    [0.5 * np.sin(a) * np.sin(b), -0.5 * np.cos(a) * np.cos(b)],
                ],
                [
                    [0.5 * np.cos(a) * np.cos(b), -0.5 * np.sin(a) * np.sin(b)],
                    [-0.5 * np.sin(a) * np.sin(b), 0.5 * np.cos(a) * np.cos(b)],
                ],
            ],
            dtype=hess.dtype,
            device=torch_device,
        )
        assert torch.allclose(hess.detach(), expected_hess, atol=tol, rtol=0)

    def test_adjoint_hessian(self, torch_device, tol):
        """Since the adjoint hessian is not a differentiable transform,
        higher-order derivatives are not supported."""
        dev = qml.device("default.qubit", wires=2)
        params = torch.tensor(
            [0.543, -0.654], requires_grad=True, dtype=torch.float64, device=torch_device
        )

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
                interface="torch",
            )[0]

        res = torch.autograd.functional.hessian(cost_fn, params)
        expected = torch.zeros([2, 2], dtype=torch.float64, device=torch_device)
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_max_diff(self, torch_device, tol):
        """Test that setting the max_diff parameter blocks higher-order
        derivatives"""
        dev = qml.device("default.qubit", wires=2)
        params = torch.tensor([0.543, -0.654], requires_grad=True, dtype=torch.float64)

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

            result = execute(
                [tape1, tape2], dev, gradient_fn=param_shift, max_diff=1, interface="torch"
            )
            return result[0] + result[1][0, 0]

        res = cost_fn(params)
        x, y = params.detach()
        expected = torch.as_tensor(0.5 * (3 + np.cos(x) ** 2 * np.cos(2 * y)))
        assert torch.allclose(res.to(torch_device), expected.to(torch_device), atol=tol, rtol=0)

        res.backward()
        expected = torch.tensor(
            [-np.cos(x) * np.cos(2 * y) * np.sin(x), -np.cos(x) ** 2 * np.sin(2 * y)]
        )
        assert torch.allclose(
            params.grad.detach().to(torch_device), expected.to(torch_device), atol=tol, rtol=0
        )

        res = torch.autograd.functional.hessian(cost_fn, params)
        expected = torch.zeros([2, 2], dtype=torch.float64)
        assert torch.allclose(res.to(torch_device), expected.to(torch_device), atol=tol, rtol=0)


execute_kwargs = [
    {"gradient_fn": param_shift, "interface": "torch"},
    {"gradient_fn": finite_diff, "interface": "torch"},
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
        a, b, c = coeffs1.detach().numpy()
        d = coeffs2.detach().numpy()[0]
        x, y = weights.detach().numpy()
        return [-c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y)), d * np.cos(x)]

    @staticmethod
    def cost_fn_jacobian(weights, coeffs1, coeffs2):
        """Analytic jacobian of cost_fn above"""
        a, b, c = coeffs1.detach().numpy()
        d = coeffs2.detach().numpy()[0]
        x, y = weights.detach().numpy()
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
        coeffs1 = torch.tensor([0.1, 0.2, 0.3], requires_grad=False, dtype=torch.float64)
        coeffs2 = torch.tensor([0.7], requires_grad=False, dtype=torch.float64)
        weights = torch.tensor([0.4, 0.5], requires_grad=True, dtype=torch.float64)
        dev = qml.device("default.qubit", wires=2)

        res = cost_fn(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res.detach(), expected, atol=tol, rtol=0)

        res = torch.hstack(
            torch.autograd.functional.jacobian(
                lambda *x: cost_fn(*x, dev=dev), (weights, coeffs1, coeffs2)
            )
        )
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        assert np.allclose(res.detach(), expected, atol=tol, rtol=0)

    def test_multiple_hamiltonians_trainable(self, cost_fn, execute_kwargs, tol):
        coeffs1 = torch.tensor([0.1, 0.2, 0.3], requires_grad=True, dtype=torch.float64)
        coeffs2 = torch.tensor([0.7], requires_grad=True, dtype=torch.float64)
        weights = torch.tensor([0.4, 0.5], requires_grad=True, dtype=torch.float64)
        dev = qml.device("default.qubit", wires=2)

        res = cost_fn(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res.detach(), expected, atol=tol, rtol=0)

        res = torch.hstack(
            torch.autograd.functional.jacobian(
                lambda *x: cost_fn(*x, dev=dev), (weights, coeffs1, coeffs2)
            )
        )
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        assert np.allclose(res.detach(), expected, atol=tol, rtol=0)
