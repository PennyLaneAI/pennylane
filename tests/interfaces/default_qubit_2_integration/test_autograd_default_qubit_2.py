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
import pytest

from param_shift_device import ParamShiftDerivativesDevice

from pennylane import numpy as np

import pennylane as qml
from pennylane.devices import DefaultQubit
from pennylane.gradients import param_shift
from pennylane.interfaces import execute
from pennylane.measurements import Shots

pytestmark = pytest.mark.autograd


# pylint: disable=too-few-public-methods
class TestCaching:
    """Tests for caching behaviour"""

    @pytest.mark.parametrize("num_params", [2, 3])
    def test_caching_param_shift_hessian(self, num_params):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum
        when computing Hessians."""
        dev = DefaultQubit()
        params = np.arange(1, num_params + 1) / 10

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
                [tape], dev, gradient_fn=qml.gradients.param_shift, cache=cache, max_diff=2
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
        expected_runs += 2 * N  # Jacobian
        expected_runs += 4 * N + 1  # Hessian diagonal
        expected_runs += 4 * N**2  # Hessian off-diagonal
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

            results = qml.execute([tape1, tape2], dev, gradient_fn=qml.gradients.param_shift)
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
    ({"gradient_fn": param_shift}, Shots(100000), DefaultQubit(seed=42)),
    ({"gradient_fn": param_shift}, Shots((100000, 100000)), DefaultQubit(seed=42)),
    ({"gradient_fn": param_shift}, Shots(None), DefaultQubit()),
    ({"gradient_fn": "backprop"}, Shots(None), DefaultQubit()),
    (
        {"gradient_fn": "adjoint", "grad_on_execution": True, "use_device_jacobian_product": False},
        Shots(None),
        DefaultQubit(),
    ),
    (
        {
            "gradient_fn": "adjoint",
            "grad_on_execution": False,
            "use_device_jacobian_product": False,
        },
        Shots(None),
        DefaultQubit(),
    ),
    ({"gradient_fn": "adjoint", "use_device_jacobian_product": True}, Shots(None), DefaultQubit()),
    (
        {"gradient_fn": "device", "use_device_jacobian_product": False},
        Shots((100000, 100000)),
        ParamShiftDerivativesDevice(),
    ),
    (
        {"gradient_fn": "device", "use_device_jacobian_product": True},
        Shots((100000, 100000)),
        ParamShiftDerivativesDevice(),
    ),
]


def atol_for_shots(shots):
    """Return higher tolerance if finite shots."""
    return 1e-2 if shots else 1e-6


@pytest.mark.parametrize("execute_kwargs, shots, device", test_matrix)
class TestAutogradExecuteIntegration:
    """Test the autograd interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs, shots, device):
        """Test execution"""

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            return execute([tape1, tape2], device, **execute_kwargs)

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
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

    def test_scalar_jacobian(self, execute_kwargs, shots, device):
        """Test scalar jacobian calculation"""
        a = np.array(0.1, requires_grad=True)

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        if shots.has_partitioned_shots:
            res = qml.jacobian(lambda x: qml.math.hstack(cost(x)))(a)
        else:
            res = qml.jacobian(cost)(a)
            assert res.shape == ()  # pylint: disable=no-member

        # compare to standard tape jacobian
        tape = qml.tape.QuantumScript([qml.RY(a, wires=0)], [qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        tapes, fn = param_shift(tape)
        expected = fn(device.execute(tapes))

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -np.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, execute_kwargs, shots, device):
        """Test jacobian calculation"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

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
    def test_tape_no_parameters(self, execute_kwargs, shots, device):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""

        if execute_kwargs["gradient_fn"] == "adjoint":
            pytest.skip("Adjoint differentiation does not yet support probabilities")
        if shots.has_partitioned_shots:
            pytest.skip("needs further investigation")

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.Hadamard(0)], [qml.expval(qml.PauliX(0))], shots=shots
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(np.array(0.5, requires_grad=False), wires=0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape4 = qml.tape.QuantumScript(
                [qml.RY(np.array(0.5, requires_grad=False), 0)],
                [qml.probs(wires=[0, 1])],
                shots=shots,
            )
            return sum(
                autograd.numpy.hstack(
                    execute([tape1, tape2, tape3, tape4], device, **execute_kwargs)
                )
            )

        params = np.array([0.1, 0.2], requires_grad=True)
        x, y = params

        res = cost(params)
        expected = 2 + np.cos(0.5) + np.cos(x) * np.cos(y)
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        grad = qml.grad(cost)(params)
        expected = [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)]
        assert np.allclose(grad, expected, atol=atol_for_shots(shots), rtol=0)

    @pytest.mark.filterwarnings("ignore:Attempted to compute the gradient")
    def test_tapes_with_different_return_size(self, execute_kwargs, shots, device):
        """Test that tapes wit different can be executed and differentiated."""

        if execute_kwargs["gradient_fn"] == "backprop":
            pytest.xfail("backprop is not compatible with something about this situation.")
        if shots.has_partitioned_shots:
            pytest.xfail("needs further investigation.")

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))],
                shots=shots,
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(np.array(0.5, requires_grad=False), 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )
            return autograd.numpy.hstack(execute([tape1, tape2, tape3], device, **execute_kwargs))

        params = np.array([0.1, 0.2], requires_grad=True)
        x, y = params

        res = cost(params)
        assert isinstance(res, np.ndarray)
        assert res.shape == (4,)

        assert np.allclose(res[0], np.cos(x) * np.cos(y), atol=atol_for_shots(shots))
        assert np.allclose(res[1], 1, atol=atol_for_shots(shots))
        assert np.allclose(res[2], np.cos(0.5), atol=atol_for_shots(shots))
        assert np.allclose(res[3], np.cos(x) * np.cos(y), atol=atol_for_shots(shots))

        jac = qml.jacobian(cost)(params)
        assert isinstance(jac, np.ndarray)
        assert jac.shape == (4, 2)  # pylint: disable=no-member

        assert np.allclose(jac[1:3], 0, atol=atol_for_shots(shots))

        d1 = -np.sin(x) * np.cos(y)
        assert np.allclose(jac[0, 0], d1, atol=atol_for_shots(shots))
        assert np.allclose(jac[3, 0], d1, atol=atol_for_shots(shots))

        d2 = -np.cos(x) * np.sin(y)
        assert np.allclose(jac[0, 1], d2, atol=atol_for_shots(shots))
        assert np.allclose(jac[3, 1], d2, atol=atol_for_shots(shots))

    def test_reusing_quantum_tape(self, execute_kwargs, shots, device):
        """Test re-using a quantum tape by passing new parameters"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

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

        a = np.array(0.54, requires_grad=True)
        b = np.array(0.8, requires_grad=True)

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

    def test_classical_processing(self, execute_kwargs, device, shots):
        """Test classical processing within the quantum tape"""
        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=False)
        c = np.array(0.3, requires_grad=True)

        def cost(a, b, c):
            ops = [
                qml.RY(a * c, wires=0),
                qml.RZ(b, wires=0),
                qml.RX(c + c**2 + np.sin(a), wires=0),
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

    def test_no_trainable_parameters(self, execute_kwargs, shots, device):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        a = np.array(0.1, requires_grad=False)
        b = np.array(0.2, requires_grad=False)

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

    def test_matrix_parameter(self, execute_kwargs, device, shots):
        """Test that the autograd interface works correctly
        with a matrix parameter"""
        U = np.array([[0, 1], [1, 0]], requires_grad=False)
        a = np.array(0.1, requires_grad=True)

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

    def test_differentiable_expand(self, execute_kwargs, device, shots):
        """Test that operation and nested tapes expansion
        is differentiable"""

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
            return execute([tape], device, **execute_kwargs)[0]

        a = np.array(0.1, requires_grad=False)
        p = np.array([0.1, 0.2, 0.3], requires_grad=True)

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

    def test_probability_differentiation(self, execute_kwargs, device, shots):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""

        if execute_kwargs["gradient_fn"] == "adjoint":
            pytest.skip("adjoint differentiation does not suppport probabilities.")

        def cost(x, y):
            ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.probs(wires=0), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return autograd.numpy.hstack(execute([tape], device, **execute_kwargs)[0])

        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

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

    def test_ragged_differentiation(self, execute_kwargs, device, shots):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if execute_kwargs["gradient_fn"] == "adjoint":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        def cost(x, y):
            ops = [qml.RX(x, wires=0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return autograd.numpy.hstack(execute([tape], device, **execute_kwargs)[0])

        x = np.array(0.543, requires_grad=True)
        y = np.array(-0.654, requires_grad=True)

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
            np.array([0.543, -0.654], requires_grad=True),
            np.array([0, -0.654], requires_grad=True),
            np.array([-2.0, 0], requires_grad=True),
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
            result = execute([tape1, tape2], dev, gradient_fn=param_shift, max_diff=2)
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
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            ops = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])

            result = execute([tape1, tape2], dev, gradient_fn=param_shift, max_diff=1)
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


@pytest.mark.parametrize("execute_kwargs, shots, device", test_matrix)
@pytest.mark.parametrize("use_new_op_math", (True, False))
class TestHamiltonianWorkflows:
    """Test that tapes ending with expectations
    of Hamiltonians provide correct results and gradients"""

    @pytest.fixture
    def cost_fn(self, execute_kwargs, shots, device, use_new_op_math):
        """Cost function for gradient tests"""

        def _cost_fn(weights, coeffs1, coeffs2):
            obs1 = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
            H1 = qml.Hamiltonian(coeffs1, obs1)
            if use_new_op_math:
                H1 = qml.pauli.pauli_sentence(H1).operation()

            obs2 = [qml.PauliZ(0)]
            H2 = qml.Hamiltonian(coeffs2, obs2)
            if use_new_op_math:
                H2 = qml.pauli.pauli_sentence(H2).operation()

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

    def test_multiple_hamiltonians_not_trainable(
        self, execute_kwargs, cost_fn, shots, use_new_op_math
    ):
        """Test hamiltonian with no trainable parameters."""

        if execute_kwargs["gradient_fn"] == "adjoint" and not use_new_op_math:
            pytest.skip("adjoint differentiation does not suppport hamiltonians.")

        coeffs1 = np.array([0.1, 0.2, 0.3], requires_grad=False)
        coeffs2 = np.array([0.7], requires_grad=False)
        weights = np.array([0.4, 0.5], requires_grad=True)

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

    def test_multiple_hamiltonians_trainable(self, execute_kwargs, cost_fn, shots, use_new_op_math):
        """Test hamiltonian with trainable parameters."""
        if execute_kwargs["gradient_fn"] == "adjoint":
            pytest.skip("trainable hamiltonians not supported with adjoint")
        if use_new_op_math:
            pytest.skip("parameter shift derivatives do not yet support sums.")

        coeffs1 = np.array([0.1, 0.2, 0.3], requires_grad=True)
        coeffs2 = np.array([0.7], requires_grad=True)
        weights = np.array([0.4, 0.5], requires_grad=True)

        res = cost_fn(weights, coeffs1, coeffs2)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = np.hstack(qml.jacobian(cost_fn)(weights, coeffs1, coeffs2))
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            # ?
            pass
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
