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
"""Torch specific tests for execute and default qubit 2."""
import numpy as np
import pytest
from param_shift_dev import ParamShiftDerivativesDevice

import pennylane as qml
from pennylane import execute
from pennylane.devices import DefaultQubit
from pennylane.gradients import param_shift
from pennylane.measurements import Shots

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


# pylint: disable=too-few-public-methods
class TestCaching:
    """Tests for caching behaviour"""

    @pytest.mark.skip("caching is not implemented for torch")
    @pytest.mark.parametrize("num_params", [2, 3])
    def test_caching_param_shift_hessian(self, num_params):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum
        when computing Hessians."""
        dev = DefaultQubit()
        params = torch.arange(1, num_params + 1, requires_grad=True, dtype=torch.float64)

        N = len(params)

        def get_cost_tape(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])

                for i in range(2, num_params):
                    qml.RZ(x[i], wires=[i % 2])

                qml.CNOT(wires=[0, 1])
                qml.var(qml.prod(qml.PauliZ(0), qml.PauliX(1)))

            return qml.tape.QuantumScript.from_queue(q)

        def cost_no_cache(x):
            return qml.execute(
                [get_cost_tape(x)],
                dev,
                diff_method=qml.gradients.param_shift,
                cache=False,
                max_diff=2,
            )[0]

        def cost_cache(x):
            return qml.execute(
                [get_cost_tape(x)],
                dev,
                diff_method=qml.gradients.param_shift,
                cache=True,
                max_diff=2,
            )[0]

        # No caching: number of executions is not ideal
        with qml.Tracker(dev) as tracker:
            hess1 = torch.autograd.functional.hessian(cost_no_cache, params)

        if num_params == 2:
            # compare to theoretical result
            x, y, *_ = params.clone().detach()
            expected = torch.tensor(
                [
                    [2 * torch.cos(2 * x) * torch.sin(y) ** 2, torch.sin(2 * x) * torch.sin(2 * y)],
                    [
                        torch.sin(2 * x) * torch.sin(2 * y),
                        -2 * torch.cos(x) ** 2 * torch.cos(2 * y),
                    ],
                ]
            )
            assert torch.allclose(expected, hess1)

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
        assert tracker.totals["executions"] == expected_runs

        # Use caching: number of executions is ideal

        with qml.Tracker(dev) as tracker2:
            hess2 = torch.autograd.functional.hessian(cost_cache, params)
        assert torch.allclose(hess1, hess2)

        expected_runs_ideal = 1  # forward pass
        expected_runs_ideal += 2 * N  # Jacobian
        expected_runs_ideal += N + 1  # Hessian diagonal
        expected_runs_ideal += 4 * N * (N - 1) // 2  # Hessian off-diagonal
        assert tracker2.totals["executions"] == expected_runs_ideal
        assert expected_runs_ideal < expected_runs


def get_device(dev_name, seed):
    if dev_name == "param_shift.qubit":
        return ParamShiftDerivativesDevice(seed=seed)
    return qml.device(dev_name, seed=seed)


# add tests for lightning 2 when possible
# set rng for device when possible
test_matrix = [
    ({"diff_method": param_shift}, Shots(100000), "default.qubit"),
    ({"diff_method": param_shift}, Shots((100000, 100000)), "default.qubit"),
    ({"diff_method": param_shift}, Shots(None), "default.qubit"),
    ({"diff_method": "backprop"}, Shots(None), "default.qubit"),
    (
        {"diff_method": "adjoint", "grad_on_execution": True, "device_vjp": False},
        Shots(None),
        "default.qubit",
    ),
    (
        {
            "diff_method": "adjoint",
            "grad_on_execution": False,
            "device_vjp": False,
        },
        Shots(None),
        "default.qubit",
    ),
    ({"diff_method": "adjoint", "device_vjp": True}, Shots(None), "default.qubit"),
    ({"diff_method": "device", "device_vjp": False}, Shots((100000, 100000)), "param_shift.qubit"),
    ({"diff_method": "device", "device_vjp": True}, Shots((100000, 100000)), "param_shift.qubit"),
    (
        {"diff_method": param_shift},
        Shots(None),
        "reference.qubit",
    ),
    (
        {"diff_method": param_shift},
        Shots(100000),
        "reference.qubit",
    ),
    (
        {"diff_method": param_shift},
        Shots((100000, 100000)),
        "reference.qubit",
    ),
    ({"diff_method": "best"}, Shots(100000), "default.qubit"),
    ({"diff_method": "best"}, Shots(None), "default.qubit"),
    ({"diff_method": "best"}, Shots(None), "reference.qubit"),
]


def atol_for_shots(shots):
    """Return higher tolerance if finite shots."""
    return 1e-2 if shots else 1e-6


@pytest.mark.parametrize("execute_kwargs, shots, device_name", test_matrix)
class TestTorchExecuteIntegration:
    """Test the torch interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs, shots, device_name, seed):
        """Test execution"""

        device = get_device(device_name, seed)

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            return execute([tape1, tape2], device, **execute_kwargs)

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=False)

        with device.tracker:
            res = cost(a, b)

        if execute_kwargs.get("grad_on_execution", False):
            assert device.tracker.totals["execute_and_derivative_batches"] == 1
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2  # different wires so different hashes

        assert len(res) == 2
        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()
        exp = torch.cos(a) * torch.cos(b)
        if shots.has_partitioned_shots:
            for shot in range(2):
                for wire in range(2):
                    assert qml.math.allclose(res[shot][wire], exp, atol=atol_for_shots(shots))
        else:
            for wire in range(2):
                assert qml.math.allclose(res[wire], exp, atol=atol_for_shots(shots))

    def test_scalar_jacobian(self, execute_kwargs, shots, device_name, seed):
        """Test scalar jacobian calculation"""
        a = torch.tensor(0.1, requires_grad=True)
        device = get_device(device_name, seed)

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        res = torch.autograd.functional.jacobian(cost, a)
        if not shots.has_partitioned_shots:
            assert res.shape == ()  # pylint: disable=no-member

        expected = -qml.math.sin(a)

        assert expected.shape == ()
        if shots.has_partitioned_shots:
            for i in range(shots.num_copies):
                assert torch.allclose(res[i], expected, atol=atol_for_shots(shots), rtol=0)
                assert torch.allclose(res[i], -torch.sin(a), atol=atol_for_shots(shots))
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res, -torch.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, execute_kwargs, shots, device_name, seed):
        """Test jacobian calculation"""
        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)

        device = get_device(device_name, seed)

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            [res] = execute([tape], device, **execute_kwargs)
            if shots.has_partitioned_shots:
                return torch.hstack(res[0] + res[1])
            return torch.hstack(res)

        res = cost(a, b)
        expected = torch.tensor([torch.cos(a), -torch.cos(a) * torch.sin(b)])
        if shots.has_partitioned_shots:
            assert torch.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = torch.autograd.functional.jacobian(cost, (a, b))
        assert isinstance(res, tuple) and len(res) == 2

        expected = (
            torch.tensor([-torch.sin(a), torch.sin(a) * torch.sin(b)]),
            torch.tensor([0, -torch.cos(a) * torch.cos(b)]),
        )
        if shots.has_partitioned_shots:
            assert res[0].shape == (4,)
            assert res[1].shape == (4,)

            for _r, _e in zip(res, expected):
                assert torch.allclose(_r[:2], _e, atol=atol_for_shots(shots))
                assert torch.allclose(_r[2:], _e, atol=atol_for_shots(shots))

        else:
            assert res[0].shape == (2,)
            assert res[1].shape == (2,)

            for _r, _e in zip(res, expected):
                assert torch.allclose(_r, _e, atol=atol_for_shots(shots))

    def test_tape_no_parameters(self, execute_kwargs, shots, device_name, seed):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""

        device = get_device(device_name, seed)

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.Hadamard(0)], [qml.expval(qml.PauliX(0))], shots=shots
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(np.array(0.5), wires=0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape4 = qml.tape.QuantumScript(
                [qml.RY(np.array(0.5), 0)],
                [qml.probs(wires=[0, 1])],
                shots=shots,
            )
            res = execute([tape1, tape2, tape3, tape4], device, **execute_kwargs)
            if shots.has_partitioned_shots:
                res = [qml.math.asarray(ri, like="torch") for r in res for ri in r]
            else:
                res = [qml.math.asarray(r, like="torch") for r in res]
            return sum(torch.hstack(res))

        params = torch.tensor([0.1, 0.2], requires_grad=True)
        x, y = params.clone().detach()

        res = cost(params)
        expected = 2 + np.cos(0.5) + np.cos(x) * np.cos(y)

        if shots.has_partitioned_shots:
            assert torch.allclose(res, 2 * expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res.backward()
        expected = torch.tensor([-torch.cos(y) * torch.sin(x), -torch.cos(x) * torch.sin(y)])
        if shots.has_partitioned_shots:
            assert torch.allclose(params.grad, 2 * expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert torch.allclose(params.grad, expected, atol=atol_for_shots(shots), rtol=0)

    @pytest.mark.skip("torch cannot reuse tensors in various computations")
    def test_tapes_with_different_return_size(self, execute_kwargs, shots, device_name, seed):
        """Test that tapes wit different can be executed and differentiated."""

        if execute_kwargs["diff_method"] == "backprop":
            pytest.xfail("backprop is not compatible with something about this situation.")

        device = get_device(device_name, seed)

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))],
                shots=shots,
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(np.array(0.5), 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )
            res = execute([tape1, tape2, tape3], device, **execute_kwargs)
            return torch.hstack([qml.math.asarray(r, like="torch") for r in res])

        params = torch.tensor([0.1, 0.2], requires_grad=True)
        x, y = params.clone().detach()

        res = cost(params)
        assert isinstance(res, torch.Tensor)
        assert res.shape == (4,)

        assert torch.allclose(res[0], torch.cos(x) * torch.cos(y), atol=atol_for_shots(shots))
        assert torch.allclose(res[1], torch.tensor(1.0), atol=atol_for_shots(shots))
        assert torch.allclose(res[2], torch.cos(torch.tensor(0.5)), atol=atol_for_shots(shots))
        assert torch.allclose(res[3], torch.cos(x) * torch.cos(y), atol=atol_for_shots(shots))

        jac = torch.autograd.functional.jacobian(cost, params)
        assert isinstance(jac, torch.Tensor)
        assert jac.shape == (4, 2)  # pylint: disable=no-member

        assert torch.allclose(jac[1:3], torch.tensor(0.0), atol=atol_for_shots(shots))

        d1 = -torch.sin(x) * torch.cos(y)
        assert torch.allclose(jac[0, 0], d1, atol=atol_for_shots(shots))  # fails for torch
        assert torch.allclose(jac[3, 0], d1, atol=atol_for_shots(shots))

        d2 = -torch.cos(x) * torch.sin(y)
        assert torch.allclose(jac[0, 1], d2, atol=atol_for_shots(shots))  # fails for torch
        assert torch.allclose(jac[3, 1], d2, atol=atol_for_shots(shots))

    def test_reusing_quantum_tape(self, execute_kwargs, shots, device_name, seed):
        """Test re-using a quantum tape by passing new parameters"""
        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)
        device = get_device(device_name, seed)

        tape = qml.tape.QuantumScript(
            [qml.RY(a, 0), qml.RX(b, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))],
        )
        assert tape.trainable_params == [0, 1]

        def cost(a, b):
            new_tape = tape.bind_new_parameters([a, b], [0, 1])
            return torch.hstack(execute([new_tape], device, **execute_kwargs)[0])

        jac = torch.autograd.functional.jacobian(cost, (a, b))

        a = torch.tensor(0.54, requires_grad=True)
        b = torch.tensor(0.8, requires_grad=True)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls
        res2 = cost(2 * a, b)
        expected = torch.tensor([torch.cos(2 * a), -torch.cos(2 * a) * torch.sin(b)])
        assert torch.allclose(res2, expected, atol=atol_for_shots(shots), rtol=0)

        jac = torch.autograd.functional.jacobian(lambda a, b: cost(2 * a, b), (a, b))
        expected = (
            torch.tensor([-2 * torch.sin(2 * a), 2 * torch.sin(2 * a) * torch.sin(b)]),
            torch.tensor([0, -torch.cos(2 * a) * torch.cos(b)]),
        )
        assert isinstance(jac, tuple) and len(jac) == 2
        for _j, _e in zip(jac, expected):
            assert torch.allclose(_j, _e, atol=atol_for_shots(shots), rtol=0)

    def test_classical_processing(self, execute_kwargs, device_name, seed, shots):
        """Test classical processing within the quantum tape"""
        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=False)
        c = torch.tensor(0.3, requires_grad=True)
        device = get_device(device_name, seed)

        def cost(a, b, c):
            ops = [
                qml.RY(a * c, wires=0),
                qml.RZ(b, wires=0),
                qml.RX(c + c**2 + torch.sin(a), wires=0),
            ]

            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        # PyTorch docs suggest a lambda for cost functions with some non-trainable args
        # See for more: https://pytorch.org/docs/stable/autograd.html#functional-higher-level-api
        res = torch.autograd.functional.jacobian(lambda _a, _c: cost(_a, b, _c), (a, c))

        # Only two arguments are trainable
        assert isinstance(res, tuple) and len(res) == 2
        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        # I tried getting analytic results for this circuit but I kept being wrong and am giving up

    @pytest.mark.skip("torch handles gradients and jacobians differently")
    def test_no_trainable_parameters(self, execute_kwargs, shots, device_name, seed):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        a = torch.tensor(0.1, requires_grad=False)
        b = torch.tensor(0.2, requires_grad=False)

        device = get_device(device_name, seed)

        def cost(a, b):
            ops = [qml.RY(a, 0), qml.RX(b, 0), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            return torch.hstack(execute([tape], device, **execute_kwargs)[0])

        res = cost(a, b)
        assert res.shape == (2,)

        res = torch.autograd.functional.jacobian(cost, (a, b))
        assert len(res) == 0

        def loss(a, b):
            return torch.sum(cost(a, b))

        res = loss(a, b)
        res.backward()

        assert torch.allclose(torch.tensor([a.grad, b.grad]), 0)

    def test_matrix_parameter(self, execute_kwargs, device_name, seed, shots):
        """Test that the torch interface works correctly
        with a matrix parameter"""
        U = torch.tensor([[0, 1], [1, 0]], requires_grad=False, dtype=torch.float64)
        a = torch.tensor(0.1, requires_grad=True)
        device = get_device(device_name, seed)

        def cost(a, U):
            ops = [qml.QubitUnitary(U, wires=0), qml.RY(a, wires=0)]
            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
            return execute([tape], device, **execute_kwargs)[0]

        res = cost(a, U)
        assert torch.allclose(res, -torch.cos(a), atol=atol_for_shots(shots))

        jac = torch.autograd.functional.jacobian(lambda y: cost(y, U), a)
        assert isinstance(jac, torch.Tensor)
        assert torch.allclose(jac, torch.sin(a), atol=atol_for_shots(shots), rtol=0)

    def test_differentiable_expand(self, execute_kwargs, device_name, seed, shots):
        """Test that operation and nested tapes expansion
        is differentiable"""

        device = get_device(device_name, seed)

        class U3(qml.U3):
            """Dummy operator."""

            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        def cost_fn(a, p):
            tape = qml.tape.QuantumScript(
                [qml.RX(a, wires=0), U3(*p, wires=0)], [qml.expval(qml.PauliX(0))]
            )
            diff_method = execute_kwargs["diff_method"]
            if diff_method is None:
                _gradient_method = None
            elif isinstance(diff_method, str):
                _gradient_method = diff_method
            else:
                _gradient_method = "gradient-transform"
            config = qml.devices.ExecutionConfig(
                interface="autograd",
                gradient_method=_gradient_method,
                grad_on_execution=execute_kwargs.get("grad_on_execution", None),
            )
            program = device.preprocess_transforms(execution_config=config)
            return execute([tape], device, **execute_kwargs, transform_program=program)[0]

        a = torch.tensor(0.1, requires_grad=False)
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

        res = cost_fn(a, p)
        expected = torch.cos(a) * torch.cos(p[1]) * torch.sin(p[0]) + torch.sin(a) * (
            torch.cos(p[2]) * torch.sin(p[1]) + torch.cos(p[0]) * torch.cos(p[1]) * torch.sin(p[2])
        )
        assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = torch.autograd.functional.jacobian(lambda _p: cost_fn(a, _p), p)
        expected = torch.tensor(
            [
                torch.cos(p[1])
                * (
                    torch.cos(a) * torch.cos(p[0])
                    - torch.sin(a) * torch.sin(p[0]) * torch.sin(p[2])
                ),
                torch.cos(p[1]) * torch.cos(p[2]) * torch.sin(a)
                - torch.sin(p[1])
                * (
                    torch.cos(a) * torch.sin(p[0])
                    + torch.cos(p[0]) * torch.sin(a) * torch.sin(p[2])
                ),
                torch.sin(a)
                * (
                    torch.cos(p[0]) * torch.cos(p[1]) * torch.cos(p[2])
                    - torch.sin(p[1]) * torch.sin(p[2])
                ),
            ]
        )
        assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

    def test_probability_differentiation(self, execute_kwargs, device_name, seed, shots):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""
        device = get_device(device_name, seed)

        def cost(x, y):
            ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.probs(wires=0), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return torch.hstack(execute([tape], device, **execute_kwargs)[0])

        x = torch.tensor(0.543, requires_grad=True)
        y = torch.tensor(-0.654, requires_grad=True)

        res = cost(x, y)
        expected = torch.tensor(
            [
                [
                    torch.cos(x / 2) ** 2,
                    torch.sin(x / 2) ** 2,
                    (1 + torch.cos(x) * torch.cos(y)) / 2,
                    (1 - torch.cos(x) * torch.cos(y)) / 2,
                ],
            ]
        )
        assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = torch.autograd.functional.jacobian(cost, (x, y))
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (4,)
        assert res[1].shape == (4,)

        expected = (
            torch.tensor(
                [
                    [
                        -torch.sin(x) / 2,
                        torch.sin(x) / 2,
                        -torch.sin(x) * torch.cos(y) / 2,
                        torch.sin(x) * torch.cos(y) / 2,
                    ],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, -torch.cos(x) * torch.sin(y) / 2, torch.cos(x) * torch.sin(y) / 2],
                ]
            ),
        )

        assert torch.allclose(res[0], expected[0], atol=atol_for_shots(shots), rtol=0)
        assert torch.allclose(res[1], expected[1], atol=atol_for_shots(shots), rtol=0)

    def test_ragged_differentiation(self, execute_kwargs, device_name, seed, shots):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        device = get_device(device_name, seed)

        def cost(x, y):
            ops = [qml.RX(x, wires=0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return torch.hstack(execute([tape], device, **execute_kwargs)[0])

        x = torch.tensor(0.543, requires_grad=True)
        y = torch.tensor(-0.654, requires_grad=True)

        res = cost(x, y)
        expected = torch.tensor(
            [
                torch.cos(x),
                (1 + torch.cos(x) * torch.cos(y)) / 2,
                (1 - torch.cos(x) * torch.cos(y)) / 2,
            ]
        )
        assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = torch.autograd.functional.jacobian(cost, (x, y))
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (3,)
        assert res[1].shape == (3,)

        expected = (
            torch.tensor(
                [-torch.sin(x), -torch.sin(x) * torch.cos(y) / 2, torch.sin(x) * torch.cos(y) / 2]
            ),
            torch.tensor([0, -torch.cos(x) * torch.sin(y) / 2, torch.cos(x) * torch.sin(y) / 2]),
        )
        assert torch.allclose(res[0], expected[0], atol=atol_for_shots(shots), rtol=0)
        assert torch.allclose(res[1], expected[1], atol=atol_for_shots(shots), rtol=0)


class TestHigherOrderDerivatives:
    """Test that the torch execute function can be differentiated"""

    @pytest.mark.parametrize(
        "params",
        [
            torch.tensor([0.543, -0.654], requires_grad=True, dtype=torch.float64),
            torch.tensor([0, -0.654], requires_grad=True, dtype=torch.float64),
            torch.tensor([-2.0, 0], requires_grad=True, dtype=torch.float64),
        ],
    )
    def test_parameter_shift_hessian(self, params, tol):
        """Tests that the output of the parameter-shift transform
        can be differentiated using torch, yielding second derivatives."""
        dev = DefaultQubit()

        def cost_fn(x):
            ops1 = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops1, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])
            result = execute([tape1, tape2], dev, diff_method=param_shift, max_diff=2)
            return result[0] + result[1][0]

        res = cost_fn(params)
        x, y = params.clone().detach()
        expected = 0.5 * (3 + torch.cos(x) ** 2 * torch.cos(2 * y))
        assert torch.allclose(res, expected, atol=tol, rtol=0)

        res.backward()
        expected = torch.tensor(
            [-torch.cos(x) * torch.cos(2 * y) * torch.sin(x), -torch.cos(x) ** 2 * torch.sin(2 * y)]
        )
        assert torch.allclose(params.grad, expected, atol=tol, rtol=0)

        res = torch.autograd.functional.hessian(cost_fn, params)
        expected = torch.tensor(
            [
                [-torch.cos(2 * x) * torch.cos(2 * y), torch.sin(2 * x) * torch.sin(2 * y)],
                [torch.sin(2 * x) * torch.sin(2 * y), -2 * torch.cos(x) ** 2 * torch.cos(2 * y)],
            ]
        )
        assert torch.allclose(res, expected, atol=tol, rtol=0)

    def test_max_diff(self, tol):
        """Test that setting the max_diff parameter blocks higher-order
        derivatives"""
        dev = DefaultQubit()
        params = torch.tensor([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            ops = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])

            result = execute([tape1, tape2], dev, diff_method=param_shift, max_diff=1)
            return result[0] + result[1][0]

        res = cost_fn(params)
        x, y = params.clone().detach()
        expected = 0.5 * (3 + torch.cos(x) ** 2 * torch.cos(2 * y))
        assert torch.allclose(res, expected, atol=tol, rtol=0)

        res.backward()
        expected = torch.tensor(
            [-torch.cos(x) * torch.cos(2 * y) * torch.sin(x), -torch.cos(x) ** 2 * torch.sin(2 * y)]
        )
        assert torch.allclose(params.grad, expected, atol=tol, rtol=0)

        res = torch.autograd.functional.hessian(cost_fn, params)
        expected = torch.zeros([2, 2])
        assert torch.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("execute_kwargs, shots, device_name", test_matrix)
@pytest.mark.parametrize("constructor", (qml.Hamiltonian, qml.dot, "dunders"))
class TestHamiltonianWorkflows:
    """Test that tapes ending with expectations
    of Hamiltonians provide correct results and gradients"""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @pytest.fixture
    def cost_fn(self, execute_kwargs, shots, device_name, seed, constructor):
        """Cost function for gradient tests"""
        device = get_device(device_name, seed)

        def _cost_fn(weights, coeffs1, coeffs2):
            if constructor == "dunders":
                H1 = (
                    coeffs1[0] * qml.Z(0) + coeffs1[1] * qml.Z(0) @ qml.X(1) + coeffs1[2] * qml.Y(0)
                )
                H2 = coeffs2[0] * qml.Z(0)
            else:

                obs1 = [qml.Z(0), qml.Z(0) @ qml.X(1), qml.Y(0)]
                H1 = constructor(coeffs1, obs1)

                obs2 = [qml.PauliZ(0)]
                H2 = constructor(coeffs2, obs2)

            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(H1)
                qml.expval(H2)

            tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
            res = execute([tape], device, **execute_kwargs)[0]
            if shots.has_partitioned_shots:
                return torch.hstack(res[0] + res[1])
            return torch.hstack(res)

        return _cost_fn

    @staticmethod
    def cost_fn_expected(weights, coeffs1, coeffs2):
        """Analytic value of cost_fn above"""
        a, b, c = coeffs1.clone().detach()
        d = coeffs2[0].clone().detach()
        x, y = weights.clone().detach()
        return torch.tensor(
            [
                -c * torch.sin(x) * torch.sin(y) + torch.cos(x) * (a + b * torch.sin(y)),
                d * torch.cos(x),
            ]
        )

    @staticmethod
    def cost_fn_jacobian(weights, coeffs1, coeffs2):
        """Analytic jacobian of cost_fn above"""
        a, b, c = coeffs1.clone().detach()
        d = coeffs2[0].clone().detach()
        x, y = weights.clone().detach()
        return torch.tensor(
            [
                [
                    -c * torch.cos(x) * torch.sin(y) - torch.sin(x) * (a + b * torch.sin(y)),
                    b * torch.cos(x) * torch.cos(y) - c * torch.cos(y) * torch.sin(x),
                    torch.cos(x),
                    torch.cos(x) * torch.sin(y),
                    -(torch.sin(x) * torch.sin(y)),
                    0,
                ],
                [-d * torch.sin(x), 0, 0, 0, 0, torch.cos(x)],
            ]
        )

    def test_multiple_hamiltonians_not_trainable(self, cost_fn, shots):
        """Test hamiltonian with no trainable parameters."""

        coeffs1 = torch.tensor([0.1, 0.2, 0.3], requires_grad=False)
        coeffs2 = torch.tensor([0.7], requires_grad=False)
        weights = torch.tensor([0.4, 0.5], requires_grad=True)

        res = cost_fn(weights, coeffs1, coeffs2)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert torch.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = torch.autograd.functional.jacobian(lambda w: cost_fn(w, coeffs1, coeffs2), weights)
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)[:, :2]
        if shots.has_partitioned_shots:
            assert torch.allclose(res[:2, :], expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res[2:, :], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

    def test_multiple_hamiltonians_trainable(self, execute_kwargs, cost_fn, shots):
        """Test hamiltonian with trainable parameters."""
        if execute_kwargs["diff_method"] == "adjoint":
            pytest.skip("trainable hamiltonians not supported with adjoint")

        coeffs1 = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        coeffs2 = torch.tensor([0.7], requires_grad=True)
        weights = torch.tensor([0.4, 0.5], requires_grad=True)

        res = cost_fn(weights, coeffs1, coeffs2)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert torch.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = torch.hstack(torch.autograd.functional.jacobian(cost_fn, (weights, coeffs1, coeffs2)))
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert torch.allclose(res[:2, :], expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res[2:, :], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
