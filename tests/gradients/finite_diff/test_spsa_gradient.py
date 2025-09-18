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
Tests for the gradients.spsa_gradient module.
"""
import numpy as np
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.exceptions import QuantumFunctionError
from pennylane.gradients import spsa_grad
from pennylane.gradients.spsa_gradient import _rademacher_sampler

# pylint:disable = use-implicit-booleaness-not-comparison,abstract-method


def coordinate_sampler(indices, num_params, idx, rng=None):
    """Return a single canonical basis vector, corresponding
    to the index ``indices[idx]``. This is a sequential coordinate sampler
    that allows to exactly reproduce derivatives, instead of using SPSA in the
    intended way."""
    # pylint: disable=unused-argument
    idx = idx % len(indices)
    direction = np.zeros(num_params)
    direction[indices[idx]] = 1.0
    return direction


class TestRademacherSampler:
    """Test the Rademacher distribution sampler."""

    @pytest.mark.parametrize(
        "ids, num", [(list(range(5)), 5), ([0, 2, 4], 5), ([0], 1), ([2, 3], 5)]
    )
    def test_output_structure(self, ids, num):
        """Test that the sampled output has the right entries to be non-zero
        and attains the right values."""
        ids_mask = np.zeros(num, dtype=bool)
        ids_mask[ids] = True
        rng = np.random.default_rng()

        for _ in range(5):
            direction = _rademacher_sampler(ids, num, rng=rng)
            assert direction.shape == (num,)
            assert set(direction).issubset({0, -1, 1})
            assert np.allclose(np.abs(direction)[ids_mask], 1)
            assert np.allclose(direction[~ids_mask], 0)

    def test_call_with_third_arg(self):
        """Test that a third argument is ignored."""
        rng = np.random.default_rng()
        _rademacher_sampler([0, 1, 2], 4, "ignored dummy", rng=rng)

    def test_differing_seeds(self):
        """Test that the output differs for different seeds."""
        ids = [0, 1, 2, 3, 4]
        num = 5
        seeds = [42, 43]
        rng = np.random.default_rng(seeds[0])
        first_direction = _rademacher_sampler(ids, num, rng=rng)
        rng = np.random.default_rng(seeds[1])
        second_direction = _rademacher_sampler(ids, num, rng=rng)
        assert not np.allclose(first_direction, second_direction)

    def test_same_seeds(self):
        """Test that the output is the same for identical RNGs."""
        ids = [0, 1, 2, 3, 4]
        num = 5
        rng = np.random.default_rng(42)
        first_direction = _rademacher_sampler(ids, num, rng=rng)
        np.random.seed(0)  # Setting the global seed should have no effect.
        rng = np.random.default_rng(42)
        second_direction = _rademacher_sampler(ids, num, rng=rng)
        assert np.allclose(first_direction, second_direction)

    @pytest.mark.parametrize(
        "ids, num",
        [
            (list(range(5)), 5),
            ([0, 2, 4], 5),
            ([0], 1),
            ([2, 3], 5),
        ],
    )
    @pytest.mark.parametrize("N", [10000])
    def test_mean_and_var(self, ids, num, N, seed):
        """Test that the mean and variance of many produced samples are
        close to the theoretical values."""
        rng = np.random.default_rng(seed)
        ids_mask = np.zeros(num, dtype=bool)
        ids_mask[ids] = True
        outputs = [_rademacher_sampler(ids, num, rng=rng) for _ in range(N)]
        # Test that the mean of non-zero entries is approximately right
        assert np.allclose(np.mean(outputs, axis=0)[ids_mask], 0, atol=4 / np.sqrt(N))
        # Test that the variance of non-zero entries is approximately right
        assert np.allclose(np.var(outputs, axis=0)[ids_mask], 1, atol=4 / N)
        # Test that the mean of zero entries is exactly 0, because all entries should be
        assert np.allclose(np.mean(outputs, axis=0)[~ids_mask], 0, atol=1e-8)
        # Test that the variance of zero entries is exactly 0, because all entries are the same
        assert np.allclose(np.var(outputs, axis=0)[~ids_mask], 0, atol=1e-8)


class TestSpsaGradient:
    """Tests for the SPSA gradient transform"""

    def test_sampler_argument(self, seed):
        """Make sure that custom samplers can be created as defined in the docs of spsa_grad."""

        def sampler_required_kwarg(
            indices, num_params, *args, rng
        ):  # pylint:disable=unused-argument
            direction = np.zeros(num_params)
            direction[indices] = rng.choice([-1, 0, 1], size=len(indices))
            return direction

        def sampler_required_arg_or_kwarg(
            indices, num_params, idx_rep, rng
        ):  # pylint:disable=unused-argument
            direction = np.zeros(num_params)
            direction[indices] = rng.choice([-1, 0, 1], size=len(indices))
            return direction

        def sampler_required_arg(
            indices, num_params, foo, idx_rep, rng, /
        ):  # pylint:disable=unused-argument
            """This should fail since spsa_grad passes rng as a kwarg."""
            direction = np.zeros(num_params)
            direction[indices] = rng.choice([-1, 0, 1], size=len(indices))
            return direction

        dev = qml.device("default.qubit", wires=1)

        tape = qml.tape.QuantumTape([qml.RX(0.5, wires=0)], [qml.expval(qml.PauliZ(0))])

        results = []
        for sampler in [sampler_required_arg_or_kwarg, sampler_required_kwarg]:
            sampler_rng = np.random.default_rng(seed)
            tapes, proc_fn = spsa_grad(
                tape, sampler=sampler, num_directions=100, sampler_rng=sampler_rng
            )

            res = qml.execute(tapes, dev)
            results.append(proc_fn(res))

        assert np.isclose(results[0], results[1], atol=0.1)

        err = "got some positional-only arguments passed as keyword arguments: 'rng'"
        with pytest.raises(TypeError, match=err):
            tapes, proc_fn = spsa_grad(
                tape, sampler=sampler_required_arg, num_directions=100, sampler_rng=sampler_rng
            )

    def test_invalid_sampler_rng(self):
        """Tests that if sampler_rng has an unexpected type, an error is raised."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method="spsa", gradient_kwargs={"sampler_rng": "foo"})
        def circuit(param):
            qml.RX(param, wires=0)
            return qml.expval(qml.PauliZ(0))

        expected_message = "The argument sampler_rng is expected to be a NumPy PRNG"
        with pytest.raises(ValueError, match=expected_message):
            qml.grad(circuit)(pnp.array(1.0))

    def test_trainable_batched_tape_raises(self):
        """Test that an error is raised for a broadcasted/batched tape if the broadcasted
        parameter is differentiated."""
        tape = qml.tape.QuantumScript([qml.RX([0.4, 0.2], 0)], [qml.expval(qml.PauliZ(0))])
        _match = r"Computing the gradient of broadcasted tapes .* using the SPSA gradient transform"
        with pytest.raises(NotImplementedError, match=_match):
            spsa_grad(tape)

    def test_nontrainable_batched_tape(self):
        """Test that no error is raised for a broadcasted/batched tape if the broadcasted
        parameter is not differentiated, and that the results correspond to the stacked
        results of the single-tape derivatives."""
        dev = qml.device("default.qubit")
        x = [0.4, 0.2]
        tape = qml.tape.QuantumScript(
            [qml.RY(0.6, 0), qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))], trainable_params=[0]
        )
        batched_tapes, batched_fn = spsa_grad(tape)
        batched_grad = batched_fn(dev.execute(batched_tapes))
        separate_tapes = [
            qml.tape.QuantumScript(
                [qml.RY(0.6, 0), qml.RX(_x, 0)], [qml.expval(qml.PauliZ(0))], trainable_params=[0]
            )
            for _x in x
        ]
        separate_tapes_and_fns = [spsa_grad(t) for t in separate_tapes]
        separate_grad = [_fn(dev.execute(_tapes)) for _tapes, _fn in separate_tapes_and_fns]
        assert np.allclose(batched_grad, separate_grad)

    def test_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a non-differentiable argument"""
        psi = pnp.array([1, 0, 1, 0], requires_grad=False) / np.sqrt(2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        # by default all parameters are assumed to be trainable
        with pytest.raises(
            ValueError, match=r"Cannot differentiate with respect to parameter\(s\) {0}"
        ):
            spsa_grad(tape)

        # setting trainable parameters avoids this
        tape.trainable_params = {1, 2}
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = spsa_grad(tape)

        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (4,)

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == (4,)

    @pytest.mark.parametrize("num_directions", [1, 10])
    def test_independent_parameter(self, num_directions, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.spsa_gradient, "generate_multishifted_tapes")

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = spsa_grad(tape, num_directions=num_directions)
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert isinstance(res[1], np.ndarray)

        # 2 tapes per direction because the default strategy for SPSA is "center"
        assert len(spy.call_args_list) == num_directions

        # Never shift the independent parameter
        for i in range(num_directions):
            assert spy.call_args_list[i][0][0:2] == (tape, [0])
            assert spy.call_args_list[i][0][2].shape == (2, 1)

    def test_no_trainable_params_tape(self):
        """Test that the correct output and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # TODO: remove once #2155 is resolved
        tape.trainable_params = []
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = spsa_grad(tape)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, np.ndarray)
        assert res.shape == (0,)

    def test_no_trainable_params_multiple_return_tape(self):
        """Test that the correct output and warning is generated in the absence of any trainable
        parameters with multiple returns."""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = []
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = spsa_grad(tape)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert res[0].shape == (0,)
        assert res[1].shape == (0,)

    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd(self):
        """Test that the correct output and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            spsa_grad(circuit)(weights)

    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch(self):
        """Test that the correct output and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            spsa_grad(circuit)(weights)

    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf(self):
        """Test that the correct output and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            spsa_grad(circuit)(weights)

    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax(self):
        """Test that the correct output and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            spsa_grad(circuit)(weights)

    def test_all_zero_diff_methods(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.probs([2, 3])

        params = pnp.array([0.5, 0.5, 0.5], requires_grad=True)

        result = spsa_grad(circuit)(params)

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3)
        assert np.allclose(result, 0)

    def test_all_zero_diff_methods_multiple_returns(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated, with multiple return values."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.expval(qml.PauliZ(wires=2)), qml.probs([2, 3])

        params = pnp.array([0.5, 0.5, 0.5], requires_grad=True)

        result = spsa_grad(circuit)(params)

        assert isinstance(result, tuple)
        assert len(result) == 2

        for r, exp_shape in zip(result, [(3,), (4, 3)]):
            assert isinstance(r, np.ndarray)
            assert r.shape == exp_shape
            assert np.allclose(r, 0)

    def test_y0(self):
        """Test that if first order finite differences is underlying the SPSA, then
        the tape is executed only once using the current parameter values."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        n = 13
        tapes, _ = spsa_grad(tape, strategy="forward", approx_order=1, num_directions=n)

        # one tape per direction, plus one global call
        assert len(tapes) == n + 1

    def test_y0_provided(self):
        """Test that by providing y0 the number of tapes is equal the number of parameters."""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        f0 = dev.execute(tape)
        n = 9
        tapes, _ = spsa_grad(tape, strategy="forward", approx_order=1, num_directions=n, f0=f0)
        # one tape per direction, the unshifted one already was evaluated above
        assert len(tapes) == n

    def test_independent_parameters(self, seed):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        rng = np.random.default_rng(seed)
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(1))

        tape2 = qml.tape.QuantumScript.from_queue(q2)
        n1 = 5
        tapes, fn = spsa_grad(
            tape1, approx_order=1, strategy="forward", num_directions=n1, sampler_rng=rng
        )
        with qml.Tracker(dev) as tracker:
            j1 = fn(dev.execute(tapes))

        assert len(tapes) == tracker.totals["executions"] == n1 + 1

        n2 = 11
        tapes, fn = spsa_grad(tape2, num_directions=n2, sampler_rng=rng)
        j2 = fn(dev.execute(tapes))

        assert len(tapes) == 2 * n2

        # Because there is just a single gate parameter varied for these tapes, the gradient
        # approximation will actually be as good as finite differences.
        exp = -np.sin(1)

        assert np.allclose(j1, [exp, 0])
        assert np.allclose(j2, [0, exp])

    def test_output_shape_matches_qnode(self):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=4)

        def cost1(x):
            qml.Rot(x[0], 0.3 * x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost2(x):
            qml.Rot(*x, wires=0)
            return [qml.expval(qml.PauliZ(0))]

        def cost3(x):
            qml.Rot(*x, wires=0)
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        def cost4(x):
            qml.Rot(*x, wires=0)
            return qml.probs([0, 1])

        def cost5(x):
            qml.Rot(*x, wires=0)
            return [qml.probs([0, 1])]

        def cost6(x):
            qml.Rot(*x, wires=0)
            return qml.probs([0, 1]), qml.probs([2, 3])

        x = pnp.random.rand(3)
        circuits = [qml.QNode(cost, dev) for cost in (cost1, cost2, cost3, cost4, cost5, cost6)]

        transform = [qml.math.shape(spsa_grad(c)(x)) for c in circuits]

        expected = [
            (3,),
            (1, 3),
            (2, 3),
            (4, 3),
            (1, 4, 3),
            (2, 4, 3),
        ]

        assert all(t == q for t, q in zip(transform, expected))

    def test_special_observable_qnode_differentiation(self):
        """Test differentiation of a QNode on a device supporting a
        special observable that returns an object rather than a number."""

        class SpecialObject:
            """SpecialObject

            A special object that conveniently encapsulates the return value of
            a special observable supported by a special device and which supports
            multiplication with scalars and addition.
            """

            def __init__(self, val):
                self.val = val

            def __mul__(self, other):
                return SpecialObject(self.val * other)

            def __rmul__(self, other):
                return self * other

            def __add__(self, other):
                new = self.val + (other.val if isinstance(other, self.__class__) else other)
                return SpecialObject(new)

        class SpecialObservable(qml.operation.Operator):
            """SpecialObservable"""

            # pylint:disable=too-few-public-methods

            def diagonalizing_gates(self):
                """Diagonalizing gates"""
                return []

        class DeviceSupportingSpecialObservable(DefaultQubitLegacy):
            """A device class supporting SpecialObservable."""

            # pylint:disable=too-few-public-methods
            name = "Device supporting SpecialObservable"
            short_name = "default.qubit.specialobservable"
            observables = DefaultQubitLegacy.observables.union({"SpecialObservable"})

            @staticmethod
            def _asarray(arr, dtype=None):
                return pnp.array(arr, dtype=dtype)

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.R_DTYPE = SpecialObservable

            def expval(self, observable, **kwargs):
                """Compute the expectation value by returning a SpecialObject
                if the observable is a SpecialObservable and the device is analytic."""
                if self.analytic and isinstance(observable, SpecialObservable):
                    val = super().expval(qml.PauliZ(wires=0), **kwargs)
                    return SpecialObject(val)

                return super().expval(observable, **kwargs)

        dev = DeviceSupportingSpecialObservable(wires=1, shots=None)

        @qml.qnode(dev, diff_method="spsa")
        def qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(SpecialObservable(wires=0))

        @qml.qnode(dev, diff_method="spsa")
        def reference_qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        par = pnp.array(0.2, requires_grad=True)
        assert np.isclose(qnode(par).item().val, reference_qnode(par).item())
        assert np.isclose(
            qml.jacobian(qnode)(par).item().val, qml.jacobian(reference_qnode)(par).item()
        )


@pytest.mark.parametrize("approx_order", [2, 4])
@pytest.mark.parametrize("strategy", ["forward", "backward", "center"])
@pytest.mark.parametrize("validate", [True, False])
class TestSpsaGradientIntegration:
    """Tests for the SPSA gradient transform"""

    def test_ragged_output(self, approx_order, strategy, validate):
        """Test that the Jacobian is correctly returned for a tape with ragged output"""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1, 2])

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=11,
            validate_params=validate,
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)

        assert len(res) == 2

        assert len(res[0]) == 3
        assert res[0][0].shape == (2,)
        assert res[0][1].shape == (2,)
        assert res[0][2].shape == (2,)

        assert len(res[1]) == 3
        assert res[1][0].shape == (4,)
        assert res[1][1].shape == (4,)
        assert res[1][2].shape == (4,)

    def test_single_expectation_value(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            h=1e-6,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=2,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res = tuple(qml.math.convert_like(r * 2, r) for r in res)

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value_with_argnum_all(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where all parameters are chosen to compute
        the jacobian"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # we choose both trainable parameters
        tapes, fn = spsa_grad(
            tape,
            argnum=[0, 1],
            approx_order=approx_order,
            strategy=strategy,
            num_directions=2,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res = tuple(qml.math.convert_like(r * 2, r) for r in res)

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value_with_argnum_one(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where only one parameter is chosen to
        estimate the jacobian.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # we choose only 1 trainable parameter - do not need to account for the multiplicative
        # error of using the coordinate_sampler
        tapes, fn = spsa_grad(
            tape,
            argnum=1,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=3,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()

        expected = [0, np.cos(y) * np.cos(x)]

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_value_with_argnum_one(self, approx_order, strategy, validate):
        """Tests correct output shape and evaluation for a tape
        with a multiple measurement, where only one parameter is chosen to
        be trainable.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        # we choose only 1 trainable parameter - do not need to account for the multiplicative
        # error of using the coordinate_sampler
        tapes, fn = spsa_grad(
            tape,
            argnum=1,
            approx_order=approx_order,
            strategy=strategy,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert isinstance(res[0], tuple)
        assert np.allclose(res[0][0], 0)
        assert isinstance(res[1], tuple)
        assert np.allclose(res[1][0], 0)

    def test_multiple_expectation_values(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=2,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res = tuple(tuple(qml.math.convert_like(_r * 2, _r) for _r in r) for r in res)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2
        assert np.allclose(res[0], [-np.sin(x), 0], atol=tol, rtol=0)
        assert isinstance(res[0][0], np.ndarray)
        assert isinstance(res[0][1], np.ndarray)

        assert isinstance(res[1], tuple)
        assert len(res[1]) == 2
        assert np.allclose(res[1], [0, np.cos(y)], atol=tol, rtol=0)
        assert isinstance(res[1][0], np.ndarray)
        assert isinstance(res[1][1], np.ndarray)

    def test_var_expectation_values(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            sampler=coordinate_sampler,
            num_directions=2,
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res = tuple(tuple(qml.math.convert_like(_r * 2, _r) for _r in r) for r in res)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2
        assert np.allclose(res[0], [-np.sin(x), 0], atol=tol, rtol=0)
        assert isinstance(res[0][0], np.ndarray)
        assert isinstance(res[0][1], np.ndarray)

        assert isinstance(res[1], tuple)
        assert len(res[1]) == 2
        assert np.allclose(res[1], [0, -2 * np.cos(y) * np.sin(y)], atol=tol, rtol=0)
        assert isinstance(res[1][0], np.ndarray)
        assert isinstance(res[1][1], np.ndarray)

    def test_prob_expectation_values(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            sampler=coordinate_sampler,
            num_directions=2,
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res = tuple(tuple(qml.math.convert_like(_r * 2, _r) for _r in r) for r in res)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2
        assert np.allclose(res[0][0], -np.sin(x), atol=tol, rtol=0)
        assert isinstance(res[0][0], np.ndarray)
        assert np.allclose(res[0][1], 0, atol=tol, rtol=0)
        assert isinstance(res[0][1], np.ndarray)

        assert isinstance(res[1], tuple)
        assert len(res[1]) == 2
        assert np.allclose(
            res[1][0],
            [
                -(np.cos(y / 2) ** 2 * np.sin(x)) / 2,
                -(np.sin(x) * np.sin(y / 2) ** 2) / 2,
                (np.sin(x) * np.sin(y / 2) ** 2) / 2,
                (np.cos(y / 2) ** 2 * np.sin(x)) / 2,
            ],
            atol=tol,
            rtol=0,
        )
        assert isinstance(res[1][0], np.ndarray)
        assert np.allclose(
            res[1][1],
            [
                -(np.cos(x / 2) ** 2 * np.sin(y)) / 2,
                (np.cos(x / 2) ** 2 * np.sin(y)) / 2,
                (np.sin(x / 2) ** 2 * np.sin(y)) / 2,
                -(np.sin(x / 2) ** 2 * np.sin(y)) / 2,
            ],
            atol=tol,
            rtol=0,
        )
        assert isinstance(res[1][1], np.ndarray)


@pytest.mark.parametrize(
    "sampler, num_directions, atol", [(_rademacher_sampler, 10, 0.5), (coordinate_sampler, 2, 1e-3)]
)
class TestSpsaGradientDifferentiation:
    """Test that the transform is differentiable"""

    @pytest.mark.autograd
    def test_autograd(self, sampler, num_directions, atol, seed):
        """Tests that the output of the SPSA gradient transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit", wires=2)
        params = pnp.array([0.543, -0.654], requires_grad=True)
        rng = np.random.default_rng(seed)

        def cost_fn(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = spsa_grad(
                tape, n=1, num_directions=num_directions, sampler=sampler, sampler_rng=rng
            )
            jac = pnp.array(fn(dev.execute(tapes)))
            if sampler is coordinate_sampler:
                jac *= 2
            return jac

        res = qml.jacobian(cost_fn)(params)
        x, y = params
        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )

        assert np.allclose(res, expected, atol=atol, rtol=0)

    @pytest.mark.autograd
    def test_autograd_ragged(self, sampler, num_directions, atol, seed):
        """Tests that the output of the SPSA gradient transform
        of a ragged tape can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit", wires=2)
        params = pnp.array([0.543, -0.654], requires_grad=True)
        rng = np.random.default_rng(seed)

        def cost_fn(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = spsa_grad(
                tape, n=1, num_directions=num_directions, sampler=sampler, sampler_rng=rng
            )
            jac = fn(dev.execute(tapes))
            if sampler is coordinate_sampler:
                jac = tuple(tuple(2 * _j for _j in _jac) for _jac in jac)
            return jac[1][0]

        x, y = params
        res = qml.jacobian(cost_fn)(params)[0]
        expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])
        assert np.allclose(res, expected, atol=atol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_tf(self, sampler, num_directions, atol, seed):
        """Tests that the output of the SPSA gradient transform
        can be differentiated using TF, yielding second derivatives."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)
        rng = np.random.default_rng(seed)

        with tf.GradientTape(persistent=True) as t:
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = spsa_grad(
                tape, n=1, num_directions=num_directions, sampler=sampler, sampler_rng=rng
            )
            jac_0, jac_1 = fn(dev.execute(tapes))
            if sampler is coordinate_sampler:
                jac_0 *= 2
                jac_1 *= 2

        x, y = 1.0 * params

        res_0 = t.jacobian(jac_0, params)
        res_1 = t.jacobian(jac_1, params)

        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        assert np.allclose([res_0, res_1], expected, atol=atol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_tf_ragged(self, sampler, num_directions, atol, seed):
        """Tests that the output of the SPSA gradient transform
        of a ragged tape can be differentiated using TF, yielding second derivatives."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)
        rng = np.random.default_rng(seed)

        with tf.GradientTape(persistent=True) as t:
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = spsa_grad(
                tape, n=1, num_directions=num_directions, sampler=sampler, sampler_rng=rng
            )

            jac_01 = fn(dev.execute(tapes))[1][0]
            if sampler is coordinate_sampler:
                jac_01 *= 2

        x, y = 1.0 * params

        res_01 = t.jacobian(jac_01, params)

        expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])

        assert np.allclose(res_01[0], expected, atol=atol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, sampler, num_directions, atol, seed):
        """Tests that the output of the SPSA gradient transform
        can be differentiated using Torch, yielding second derivatives."""
        import torch

        dev = qml.device("default.qubit", wires=2)
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)
        rng = np.random.default_rng(seed)

        def cost_fn(params):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = spsa_grad(
                tape, n=1, num_directions=num_directions, sampler=sampler, sampler_rng=rng
            )
            jac = fn(dev.execute(tapes))
            if sampler is coordinate_sampler:
                jac = tuple(2 * _jac for _jac in jac)
            return jac

        hess = torch.autograd.functional.jacobian(cost_fn, params)

        x, y = params.detach().numpy()

        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )

        assert np.allclose(hess[0].detach().numpy(), expected[0], atol=atol, rtol=0)
        assert np.allclose(hess[1].detach().numpy(), expected[1], atol=atol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, sampler, num_directions, atol, seed):
        """Tests that the output of the SPSA gradient transform
        can be differentiated using JAX, yielding second derivatives."""
        import jax
        from jax import numpy as jnp

        dev = qml.device("default.qubit", wires=2)
        params = jnp.array([0.543, -0.654])
        rng = np.random.default_rng(seed)

        def cost_fn(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = spsa_grad(
                tape, n=1, num_directions=num_directions, sampler=sampler, sampler_rng=rng
            )
            jac = fn(dev.execute(tapes))
            if sampler is coordinate_sampler:
                jac = tuple(2 * _jac for _jac in jac)
            return jac

        res = jax.jacobian(cost_fn)(params)
        assert isinstance(res, tuple)
        x, y = params
        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        assert np.allclose(res, expected, atol=atol, rtol=0)
