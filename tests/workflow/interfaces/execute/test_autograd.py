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
"""Autograd specific tests for execute and default qubit 2."""
import autograd
import numpy as np
import pytest
from param_shift_dev import ParamShiftDerivativesDevice

import pennylane as qml
from pennylane import execute
from pennylane import numpy as pnp
from pennylane.devices import DefaultQubit
from pennylane.gradients import param_shift
from pennylane.measurements import Shots

pytestmark = pytest.mark.autograd


def get_device(device_name, seed):
    if device_name == "param_shift.qubit":
        return ParamShiftDerivativesDevice(seed=seed)
    return qml.device(device_name, seed=seed)


# pylint: disable=too-few-public-methods
class TestCaching:
    """Tests for caching behaviour"""

    @pytest.mark.parametrize("num_params", [2, 3])
    def test_caching_param_shift_hessian(self, num_params):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum
        when computing Hessians."""
        dev = DefaultQubit()
        params = pnp.arange(1, num_params + 1) / 10

        N = len(params)

        def cost(x, cache):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])

                for i in range(2, num_params):
                    qml.RZ(x[i], wires=[i % 2])

                qml.CNOT(wires=[0, 1])
                qml.var(qml.prod(qml.PauliZ(0), qml.PauliX(1)))

            tape = qml.tape.QuantumScript.from_queue(q)
            return qml.execute(
                [tape], dev, diff_method=qml.gradients.param_shift, cache=cache, max_diff=2
            )[0]

        # No caching: number of executions is not ideal
        with qml.Tracker(dev) as tracker:
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
            assert np.allclose(expected, hess1)

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
            hess2 = qml.jacobian(qml.grad(cost))(params, cache=True)
        assert np.allclose(hess1, hess2)

        expected_runs_ideal = 1  # forward pass
        expected_runs_ideal += 2 * N  # Jacobian
        expected_runs_ideal += N + 1  # Hessian diagonal
        expected_runs_ideal += 4 * N * (N - 1) // 2  # Hessian off-diagonal
        assert tracker2.totals["executions"] == expected_runs_ideal
        assert expected_runs_ideal < expected_runs

    def test_single_backward_pass_batch(self):
        """Tests that the backward pass is one single batch, not a bunch of batches, when parameter shift
        is requested for multiple tapes."""

        dev = qml.device("default.qubit")

        def f(x):
            tape1 = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.probs(wires=0)])
            tape2 = qml.tape.QuantumScript([qml.RY(x, 0)], [qml.probs(wires=0)])

            results = qml.execute([tape1, tape2], dev, diff_method=qml.gradients.param_shift)
            return results[0] + results[1]

        x = qml.numpy.array(0.1)
        with dev.tracker:
            out = qml.jacobian(f)(x)

        assert dev.tracker.totals["batches"] == 2
        assert dev.tracker.history["simulations"] == [1, 1, 1, 1, 1, 1]
        expected = [-2 * np.cos(x / 2) * np.sin(x / 2), 2 * np.sin(x / 2) * np.cos(x / 2)]
        assert qml.math.allclose(out, expected)


# add tests for lightning 2 when possible
# set rng for device when possible
test_matrix = [
    ({"diff_method": param_shift}, Shots(50000), "default.qubit"),
    ({"diff_method": param_shift}, Shots((50000, 50000)), "default.qubit"),
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
    (
        {"diff_method": "device", "device_vjp": False},
        Shots((50000, 50000)),
        "param_shift.qubit",
    ),
    (
        {"diff_method": "device", "device_vjp": True},
        Shots((100000, 100000)),
        "param_shift.qubit",
    ),
    (
        {"diff_method": param_shift},
        Shots(None),
        "reference.qubit",
    ),
    (
        {"diff_method": param_shift},
        Shots(50000),
        "reference.qubit",
    ),
    (
        {"diff_method": param_shift},
        Shots((50000, 50000)),
        "reference.qubit",
    ),
    ({"diff_method": "best"}, Shots(10000), "default.qubit"),
    ({"diff_method": "best"}, Shots(None), "default.qubit"),
    ({"diff_method": "best"}, Shots(None), "reference.qubit"),
]


def atol_for_shots(shots):
    """Return higher tolerance if finite shots."""
    return 5e-2 if shots else 1e-6


@pytest.mark.parametrize("execute_kwargs, shots, device_name", test_matrix)
class TestAutogradExecuteIntegration:
    """Test the autograd interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs, shots, device_name, seed):
        """Test execution"""

        device = get_device(device_name, seed=seed)

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            return execute([tape1, tape2], device, **execute_kwargs)

        a = pnp.array(0.1, requires_grad=True)
        b = pnp.array(0.2, requires_grad=False)
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

        assert qml.math.allclose(res[0], np.cos(a) * np.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], np.cos(a) * np.cos(b), atol=atol_for_shots(shots))

    def test_scalar_jacobian(self, execute_kwargs, shots, device_name, seed):
        """Test scalar jacobian calculation"""

        device = get_device(device_name, seed=seed)
        a = pnp.array(0.1, requires_grad=True)

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        if shots.has_partitioned_shots:
            res = qml.jacobian(lambda x: qml.math.hstack(cost(x)))(a)
        else:
            res = qml.jacobian(cost)(a)
            assert res.shape == ()  # pylint: disable=no-member

        expected = -qml.math.sin(a)

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -np.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, execute_kwargs, shots, device_name, seed):
        """Test jacobian calculation"""
        a = pnp.array(0.1, requires_grad=True)
        b = pnp.array(0.2, requires_grad=True)

        device = get_device(device_name, seed=seed)

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            return autograd.numpy.hstack(execute([tape], device, **execute_kwargs)[0])

        res = cost(a, b)
        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = qml.jacobian(cost)(a, b)
        assert isinstance(res, tuple) and len(res) == 2
        if shots.has_partitioned_shots:
            assert res[0].shape == (4,)
            assert res[1].shape == (4,)

            expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
            for _r, _e in zip(res, expected):
                assert np.allclose(_r[:2], _e, atol=atol_for_shots(shots))
                assert np.allclose(_r[2:], _e, atol=atol_for_shots(shots))
        else:
            assert res[0].shape == (2,)
            assert res[1].shape == (2,)

            expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
            for _r, _e in zip(res, expected):
                assert np.allclose(_r, _e, atol=atol_for_shots(shots))

    @pytest.mark.filterwarnings("ignore:Attempted to compute the gradient")
    def test_tape_no_parameters(self, execute_kwargs, shots, device_name, seed):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""

        device = get_device(device_name, seed=seed)

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.Hadamard(0)], [qml.expval(qml.PauliX(0))], shots=shots
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(pnp.array(0.5, requires_grad=False), wires=0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape4 = qml.tape.QuantumScript(
                [qml.RY(pnp.array(0.5, requires_grad=False), 0)],
                [qml.probs(wires=[0, 1])],
                shots=shots,
            )
            res = qml.execute([tape1, tape2, tape3, tape4], device, **execute_kwargs)
            if shots.has_partitioned_shots:
                res = tuple(i for r in res for i in r)
            return sum(autograd.numpy.hstack(res))

        params = pnp.array([0.1, 0.2], requires_grad=True)
        x, y = params

        res = cost(params)
        expected = 2 + np.cos(0.5) + np.cos(x) * np.cos(y)
        if shots.has_partitioned_shots:
            expected = shots.num_copies * expected
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        grad = qml.grad(cost)(params)
        expected = np.array([-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)])
        if shots.has_partitioned_shots:
            expected = shots.num_copies * expected
        assert np.allclose(grad, expected, atol=atol_for_shots(shots), rtol=0)

    @pytest.mark.filterwarnings("ignore:Attempted to compute the gradient")
    def test_tapes_with_different_return_size(self, execute_kwargs, shots, device_name, seed):
        """Test that tapes wit different can be executed and differentiated."""

        if (
            execute_kwargs["diff_method"] == "backprop"
            or execute_kwargs["diff_method"] == "best"
            and not shots
        ):
            pytest.xfail("backprop is not compatible with something about this situation.")

        device = get_device(device_name, seed=seed)

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))],
                shots=shots,
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(pnp.array(0.5, requires_grad=False), 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )
            res = execute([tape1, tape2, tape3], device, **execute_kwargs)
            if shots.has_partitioned_shots:
                res = tuple(i for r in res for i in r)
            return autograd.numpy.hstack(res)

        params = pnp.array([0.1, 0.2], requires_grad=True)
        x, y = params

        res = cost(params)
        assert isinstance(res, np.ndarray)
        if not shots:
            assert res.shape == (4,)

        if shots.has_partitioned_shots:
            for i in (0, 1):
                assert np.allclose(res[2 * i], np.cos(x) * np.cos(y), atol=atol_for_shots(shots))
                assert np.allclose(res[2 * i + 1], 1, atol=atol_for_shots(shots))
                assert np.allclose(res[4 + i], np.cos(0.5), atol=atol_for_shots(shots))
                assert np.allclose(res[6 + i], np.cos(x) * np.cos(y), atol=atol_for_shots(shots))
        else:
            assert np.allclose(res[0], np.cos(x) * np.cos(y), atol=atol_for_shots(shots))
            assert np.allclose(res[1], 1, atol=atol_for_shots(shots))
            assert np.allclose(res[2], np.cos(0.5), atol=atol_for_shots(shots))
            assert np.allclose(res[3], np.cos(x) * np.cos(y), atol=atol_for_shots(shots))

        jac = qml.jacobian(cost)(params)
        assert isinstance(jac, np.ndarray)
        if not shots.has_partitioned_shots:
            assert jac.shape == (4, 2)  # pylint: disable=no-member

        d1 = -np.sin(x) * np.cos(y)
        d2 = -np.cos(x) * np.sin(y)

        if shots.has_partitioned_shots:
            assert np.allclose(jac[1], 0, atol=atol_for_shots(shots))
            assert np.allclose(jac[3:4], 0, atol=atol_for_shots(shots))

            assert np.allclose(jac[0, 0], d1, atol=atol_for_shots(shots))
            assert np.allclose(jac[2, 0], d1, atol=atol_for_shots(shots))

            assert np.allclose(jac[6, 0], d1, atol=atol_for_shots(shots))
            assert np.allclose(jac[7, 0], d1, atol=atol_for_shots(shots))

            assert np.allclose(jac[0, 1], d2, atol=atol_for_shots(shots))
            assert np.allclose(jac[6, 1], d2, atol=atol_for_shots(shots))
            assert np.allclose(jac[7, 1], d2, atol=atol_for_shots(shots))

        else:
            assert np.allclose(jac[1:3], 0, atol=atol_for_shots(shots))

            assert np.allclose(jac[0, 0], d1, atol=atol_for_shots(shots))
            assert np.allclose(jac[3, 0], d1, atol=atol_for_shots(shots))

            assert np.allclose(jac[0, 1], d2, atol=atol_for_shots(shots))
            assert np.allclose(jac[3, 1], d2, atol=atol_for_shots(shots))

    def test_reusing_quantum_tape(self, execute_kwargs, shots, device_name, seed):
        """Test re-using a quantum tape by passing new parameters"""

        device = get_device(device_name, seed=seed)

        a = pnp.array(0.1, requires_grad=True)
        b = pnp.array(0.2, requires_grad=True)

        tape = qml.tape.QuantumScript(
            [qml.RY(a, 0), qml.RX(b, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))],
        )
        assert tape.trainable_params == [0, 1]

        def cost(a, b):
            new_tape = tape.bind_new_parameters([a, b], [0, 1])
            return autograd.numpy.hstack(execute([new_tape], device, **execute_kwargs)[0])

        jac_fn = qml.jacobian(cost)
        jac = jac_fn(a, b)

        a = pnp.array(0.54, requires_grad=True)
        b = pnp.array(0.8, requires_grad=True)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls
        res2 = cost(2 * a, b)
        expected = [np.cos(2 * a), -np.cos(2 * a) * np.sin(b)]
        assert np.allclose(res2, expected, atol=atol_for_shots(shots), rtol=0)

        jac_fn = qml.jacobian(lambda a, b: cost(2 * a, b))
        jac = jac_fn(a, b)
        expected = (
            [-2 * np.sin(2 * a), 2 * np.sin(2 * a) * np.sin(b)],
            [0, -np.cos(2 * a) * np.cos(b)],
        )
        assert isinstance(jac, tuple) and len(jac) == 2
        for _j, _e in zip(jac, expected):
            assert np.allclose(_j, _e, atol=atol_for_shots(shots), rtol=0)

    def test_classical_processing(self, execute_kwargs, device_name, seed, shots):
        """Test classical processing within the quantum tape"""
        a = pnp.array(0.1, requires_grad=True)
        b = pnp.array(0.2, requires_grad=False)
        c = pnp.array(0.3, requires_grad=True)

        device = get_device(device_name, seed=seed)

        def cost(a, b, c):
            ops = [
                qml.RY(a * c, wires=0),
                qml.RZ(b, wires=0),
                qml.RX(c + c**2 + pnp.sin(a), wires=0),
            ]

            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))], shots=shots)
            if shots.has_partitioned_shots:
                return qml.math.hstack(execute([tape], device, **execute_kwargs)[0])
            return execute([tape], device, **execute_kwargs)[0]

        res = qml.jacobian(cost)(a, b, c)

        # Only two arguments are trainable
        assert isinstance(res, tuple) and len(res) == 2
        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        # I tried getting analytic results for this circuit but I kept being wrong and am giving up

    def test_no_trainable_parameters(self, execute_kwargs, shots, device_name, seed):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        a = pnp.array(0.1, requires_grad=False)
        b = pnp.array(0.2, requires_grad=False)

        device = get_device(device_name, seed=seed)

        def cost(a, b):
            ops = [qml.RY(a, 0), qml.RX(b, 0), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            return autograd.numpy.hstack(execute([tape], device, **execute_kwargs)[0])

        res = cost(a, b)
        assert res.shape == (2 * shots.num_copies,) if shots else (2,)

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            res = qml.jacobian(cost)(a, b)
        assert len(res) == 0

        def loss(a, b):
            return np.sum(cost(a, b))

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            res = qml.grad(loss)(a, b)

        assert np.allclose(res, 0)

    def test_matrix_parameter(self, execute_kwargs, device_name, seed, shots):
        """Test that the autograd interface works correctly
        with a matrix parameter"""
        device = get_device(device_name, seed=seed)
        U = pnp.array([[0, 1], [1, 0]], requires_grad=False)
        a = pnp.array(0.1, requires_grad=True)

        def cost(a, U):
            ops = [qml.QubitUnitary(U, wires=0), qml.RY(a, wires=0)]
            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
            return execute([tape], device, **execute_kwargs)[0]

        res = cost(a, U)
        assert np.allclose(res, -np.cos(a), atol=atol_for_shots(shots))

        jac_fn = qml.jacobian(cost)
        jac = jac_fn(a, U)
        assert isinstance(jac, np.ndarray)
        assert np.allclose(jac, np.sin(a), atol=atol_for_shots(shots), rtol=0)

    def test_differentiable_expand(self, execute_kwargs, device_name, seed, shots):
        """Test that operation and nested tapes expansion
        is differentiable"""

        device = get_device(device_name, seed=seed)

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

        a = pnp.array(0.1, requires_grad=False)
        p = pnp.array([0.1, 0.2, 0.3], requires_grad=True)

        res = cost_fn(a, p)
        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac_fn = qml.jacobian(cost_fn)
        res = jac_fn(a, p)
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
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

    def test_probability_differentiation(self, execute_kwargs, device_name, seed, shots):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""

        device = get_device(device_name, seed=seed)

        def cost(x, y):
            ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.probs(wires=0), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return autograd.numpy.hstack(execute([tape], device, **execute_kwargs)[0])

        x = pnp.array(0.543, requires_grad=True)
        y = pnp.array(-0.654, requires_grad=True)

        res = cost(x, y)
        expected = np.array(
            [
                [
                    np.cos(x / 2) ** 2,
                    np.sin(x / 2) ** 2,
                    (1 + np.cos(x) * np.cos(y)) / 2,
                    (1 - np.cos(x) * np.cos(y)) / 2,
                ],
            ]
        )
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(x, y)
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

        assert np.allclose(res[0], expected[0], atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res[1], expected[1], atol=atol_for_shots(shots), rtol=0)

    def test_ragged_differentiation(self, execute_kwargs, device_name, seed, shots):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        device = get_device(device_name, seed=seed)

        def cost(x, y):
            ops = [qml.RX(x, wires=0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return autograd.numpy.hstack(execute([tape], device, **execute_kwargs)[0])

        x = pnp.array(0.543, requires_grad=True)
        y = pnp.array(-0.654, requires_grad=True)

        res = cost(x, y)
        expected = np.array(
            [np.cos(x), (1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac_fn = qml.jacobian(cost)
        res = jac_fn(x, y)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (3,)
        assert res[1].shape == (3,)

        expected = (
            np.array([-np.sin(x), -np.sin(x) * np.cos(y) / 2, np.sin(x) * np.cos(y) / 2]),
            np.array([0, -np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )
        assert np.allclose(res[0], expected[0], atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res[1], expected[1], atol=atol_for_shots(shots), rtol=0)


class TestHigherOrderDerivatives:
    """Test that the autograd execute function can be differentiated"""

    @pytest.mark.parametrize(
        "params",
        [
            pnp.array([0.543, -0.654], requires_grad=True),
            pnp.array([0, -0.654], requires_grad=True),
            pnp.array([-2.0, 0], requires_grad=True),
        ],
    )
    def test_parameter_shift_hessian(self, params, tol):
        """Tests that the output of the parameter-shift transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = DefaultQubit()

        def cost_fn(x):
            ops1 = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops1, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])
            result = execute([tape1, tape2], dev, diff_method=param_shift, max_diff=2)
            return result[0] + result[1][0]

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

    def test_max_diff(self, tol):
        """Test that setting the max_diff parameter blocks higher-order
        derivatives"""
        dev = DefaultQubit()
        params = pnp.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            ops = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])

            result = execute([tape1, tape2], dev, diff_method=param_shift, max_diff=1)
            return result[0] + result[1][0]

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


@pytest.mark.parametrize("execute_kwargs, shots, device_name", test_matrix)
@pytest.mark.parametrize("constructor", (qml.Hamiltonian, qml.dot, "dunders"))
class TestHamiltonianWorkflows:
    """Test that tapes ending with expectations
    of Hamiltonians provide correct results and gradients"""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @pytest.fixture
    def cost_fn(self, execute_kwargs, shots, device_name, seed, constructor):
        """Cost function for gradient tests"""

        device = get_device(device_name, seed=seed)

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
            return autograd.numpy.hstack(execute([tape], device, **execute_kwargs)[0])

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

    def test_multiple_hamiltonians_not_trainable(self, cost_fn, shots):
        """Test hamiltonian with no trainable parameters."""

        coeffs1 = pnp.array([0.1, 0.2, 0.3], requires_grad=False)
        coeffs2 = pnp.array([0.7], requires_grad=False)
        weights = pnp.array([0.4, 0.5], requires_grad=True)

        res = cost_fn(weights, coeffs1, coeffs2)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = qml.jacobian(cost_fn)(weights, coeffs1, coeffs2)
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)[:, :2]
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2, :], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:, :], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

    def test_multiple_hamiltonians_trainable(self, execute_kwargs, cost_fn, shots, constructor):
        """Test hamiltonian with trainable parameters."""
        if execute_kwargs["diff_method"] == "adjoint":
            pytest.skip("trainable hamiltonians not supported with adjoint")
        if constructor == "dunders":
            pytest.xfail("autograd does not like constructing an sprod via dunder.")
        if shots.has_partitioned_shots:
            pytest.xfail(
                "multiple hamiltonians with shot vectors does not seem to be differentiable."
            )

        coeffs1 = pnp.array([0.1, 0.2, 0.3], requires_grad=True)
        coeffs2 = pnp.array([0.7], requires_grad=True)
        weights = pnp.array([0.4, 0.5], requires_grad=True)

        res = cost_fn(weights, coeffs1, coeffs2)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = pnp.hstack(qml.jacobian(cost_fn)(weights, coeffs1, coeffs2))
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert qml.math.allclose(res[:2, :], expected, atol=atol_for_shots(shots), rtol=0)
            assert qml.math.allclose(res[2:, :], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert qml.math.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
