# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for executing circuits with mid-circuit measurements"""
# pylint:disable=wrong-import-position, protected-access
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# We import sampling so that we can correctly spy function calls for unit testing
from pennylane.devices.qubit import sampling
from pennylane.transforms.defer_measurements import DeferMeasurementsInterpreter

pytestmark = [
    pytest.mark.capture,
    pytest.mark.integration,
]


def create_execution_config(postselect_mode=None):
    return qml.devices.ExecutionConfig(
        mcm_config={"mcm_method": "deferred", "postselect_mode": postselect_mode}
    )


class TestExecutionAnalytic:
    """Tests for executing analytic circuits that are transformed by qml.defer_measurements
    with default.qubit."""

    def test_single_mcm(self):
        """Test that applying a single MCM works."""

        dev = qml.device("default.qubit", wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f():
            qml.Hadamard(0)
            qml.measure(0)
            qml.Hadamard(0)
            return qml.expval(qml.PauliX(0))

        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=create_execution_config())
        assert qml.math.allclose(res, 0)

    def test_qubit_reset(self):
        """Test that resetting a qubit works as expected."""

        dev = qml.device("default.qubit", wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f():
            qml.PauliX(0)
            qml.measure(0, reset=True)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=create_execution_config())
        assert qml.math.allclose(res, 1)

    @pytest.mark.parametrize("reset", [False, True])
    @pytest.mark.parametrize("postselect", [0, 1])
    def test_postselection(self, reset, postselect):
        """Test that postselection works as expected."""

        dev = qml.device("default.qubit", wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.measure(0, reset=reset, postselect=postselect)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.Z(1))

        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=create_execution_config())

        eigval = -2 * postselect + 1
        if reset:
            assert qml.math.allclose(res, [1, eigval])
        else:
            assert qml.math.allclose(res, [eigval, eigval])

    def test_mcms_as_gate_parameters(self):
        """Test that using MCMs as gate parameters works as expected."""

        dev = qml.device("default.qubit", wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f():
            qml.Hadamard(0)
            m = qml.measure(0)
            qml.RX(m * jnp.pi, 0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=create_execution_config())
        # If 0 measured, RX does nothing, so state is |0>. If 1 measured, RX(pi)
        # makes state |1> -> |0>, so <Z> will always be 1
        assert qml.math.allclose(res, 1)

    def test_cond(self):
        """Test that using qml.cond with MCM predicates works as expected."""

        dev = qml.device("default.qubit", wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f(x):
            qml.Hadamard(0)
            qml.Hadamard(1)
            m0 = qml.measure(0)
            m1 = qml.measure(1)

            @qml.cond(m0 == 0)
            def cond_fn(y):
                qml.RY(y, 0)

            @cond_fn.else_if(m1 == 0)
            def _(y):
                qml.RY(2 * y, 0)

            @cond_fn.otherwise
            def _(y):
                qml.RY(3 * y, 0)

            cond_fn(x)

            return qml.expval(qml.PauliZ(0))

        phi = jnp.pi / 3
        jaxpr = jax.make_jaxpr(f)(phi)
        res = dev.eval_jaxpr(
            jaxpr.jaxpr, jaxpr.consts, phi, execution_config=create_execution_config()
        )
        expected = 0.5 * (jnp.cos(phi) + jnp.sin(phi) ** 2)
        assert qml.math.allclose(res, expected)

    def test_cond_non_mcm(self):
        """Test that using qml.cond with non-MCM predicates works as expected."""

        dev = qml.device("default.qubit", wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f(x):
            qml.Hadamard(0)
            m0 = qml.measure(0)

            @qml.cond(x > 2.5)
            def cond_fn():
                qml.RX(m0 * jnp.pi, 0)
                # Final state |0>

            @cond_fn.else_if(x > 1.5)
            def _():
                qml.PauliZ(0)
                # Equal prob of |0> and |1>

            @cond_fn.otherwise
            def _():
                qml.Hadamard(0)
                m1 = qml.measure(0)
                qml.RX(m1 * jnp.pi, 0)
                qml.X(0)
                # Final state |1>

            cond_fn()

            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        arg_true = 3.0
        res = dev.eval_jaxpr(
            jaxpr.jaxpr, jaxpr.consts, arg_true, execution_config=create_execution_config()
        )
        assert qml.math.allclose(res, 1)  # Final state |0>; <Z> = 1

        arg_elif = 2.0
        res = dev.eval_jaxpr(
            jaxpr.jaxpr, jaxpr.consts, arg_elif, execution_config=create_execution_config()
        )
        assert qml.math.allclose(res, 0)  # Equal prob of |0>, |1>; <Z> = 1

        arg_true = 1.0
        res = dev.eval_jaxpr(
            jaxpr.jaxpr, jaxpr.consts, arg_true, execution_config=create_execution_config()
        )
        assert qml.math.allclose(res, -1)  # Final state |1>, <Z> = -1

    @pytest.mark.parametrize(
        "mp_fn",
        [qml.expval, qml.var, qml.probs],
    )
    def test_mcm_statistics(self, mp_fn):
        """Test that collecting statistics on MCMs is handled correctly."""

        dev = qml.device("default.qubit", wires=5)

        def processing_fn(m1, m2):
            return 2.5 * m1 - m2

        def f():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.Hadamard(0)
            m1 = qml.measure(0)
            qml.Hadamard(0)
            m2 = qml.measure(0)

            outs = (mp_fn(op=m0),)
            if mp_fn is qml.probs:
                outs += (mp_fn(op=[m0, m1, m2]),)
            else:
                outs += (mp_fn(op=processing_fn(m1, m2)),)

            return outs

        transformed_f = DeferMeasurementsInterpreter(num_wires=5)(f)

        jaxpr = jax.make_jaxpr(transformed_f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=create_execution_config())

        with qml.capture.pause():
            qnode_f = qml.QNode(f, dev, mcm_method="deferred")
            expected = qnode_f()

        for r, e in zip(res, expected, strict=True):
            assert qml.math.allclose(r, e)


# pylint: disable=too-few-public-methods
@pytest.mark.slow
class TestExecutionFiniteShots:
    """Tests for executing circuits with finite shots."""

    @pytest.mark.parametrize("n_postselects", [1, 2, 3])
    @pytest.mark.parametrize("postselect", [0, 1])
    def test_hw_like_samples(self, postselect, n_postselects):
        """Test that postselect_mode="hw-like" updates the number of samples as expected."""
        num_wires = 5
        dev = qml.device("default.qubit", wires=num_wires, seed=jax.random.PRNGKey(5432))
        config = create_execution_config(postselect_mode="hw-like")

        @DeferMeasurementsInterpreter(num_wires=num_wires)
        def f():
            for _ in range(n_postselects):
                qml.Hadamard(0)
                qml.measure(0, postselect=postselect)
            return qml.sample(wires=[0])

        jaxpr = jax.make_jaxpr(f)()
        res = tuple(
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=config, shots=1000)[0]
            for _ in range(10)
        )
        assert all(qml.math.allclose(r, postselect) for r in res)
        lens = [len(r) for r in res]
        assert qml.math.allclose(
            qml.math.mean(lens), int(1000 / (2**n_postselects)), atol=5 + 2 * n_postselects, rtol=0
        )

    @pytest.mark.parametrize("n_postselects", [1, 2, 3])
    @pytest.mark.parametrize("postselect", [0, 1])
    def test_fill_shots_samples(self, postselect, n_postselects):
        """Test that postselect_mode="fill-shots" updates the number of samples as expected."""
        num_wires = 5
        dev = qml.device("default.qubit", wires=num_wires, seed=jax.random.PRNGKey(1234))
        config = create_execution_config(postselect_mode="fill-shots")

        @DeferMeasurementsInterpreter(num_wires=num_wires)
        def f():
            for _ in range(n_postselects):
                qml.Hadamard(0)
                qml.measure(0, postselect=postselect)
            return qml.sample(wires=[0])

        jaxpr = jax.make_jaxpr(f)()
        res = tuple(
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=config, shots=1000)[0]
            for _ in range(5)
        )
        assert all(qml.math.allclose(r, postselect) for r in res)
        lens = [len(r) for r in res]
        assert all(l == 1000 for l in lens)

    def test_correct_sampling(self, mocker):
        """Test that sampling is performed with the correct pipeline."""
        num_wires = 5
        dev = qml.device("default.qubit", wires=num_wires, seed=jax.random.PRNGKey(1234))
        config = create_execution_config(postselect_mode="fill-shots")

        @DeferMeasurementsInterpreter(num_wires=num_wires)
        def f():
            for i in range(4):
                qml.Hadamard(0)
                qml.measure(0, reset=bool(i % 2))

            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()

        expected_state = qml.math.zeros(2**num_wires, dtype=complex)
        # Last MCM resets state, so only first half of state-vector will be non-zero,
        # corresponding to |0> on wire 0. Uniform superposition with real amplitudes
        # because we only used Hadamard.
        expected_state[:16] = 1
        # Computed which indices will have negative amplitude by hand
        expected_state[[3, 7, 11, 12, 13, 14]] = -1
        expected_state /= qml.math.norm(expected_state)
        expected_state = qml.math.reshape(expected_state, (2,) * num_wires)
        measure_spy = mocker.spy(sampling, "sample_state")

        _ = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=config, shots=100)
        measure_spy.assert_called()
        state = measure_spy.call_args.args[0]
        shots = measure_spy.call_args.kwargs["shots"]
        assert qml.math.allclose(state, expected_state)
        assert shots == 100

    @pytest.mark.parametrize("n_postselects", [1, 2, 3])
    @pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
    def test_correct_sampling_postselection(self, postselect_mode, n_postselects, mocker):
        """Test that sampling is performed using the correct pipeline with postselection."""
        num_wires = 4
        dev = qml.device("default.qubit", wires=num_wires, seed=jax.random.PRNGKey(5432))
        config = create_execution_config(postselect_mode=postselect_mode)

        @DeferMeasurementsInterpreter(num_wires=num_wires)
        def f():
            for _ in range(n_postselects):
                qml.Hadamard(0)
                qml.measure(0, postselect=1)

            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()

        spied_shots = []
        # Map n_postselects to tuple (state index with non-zero amplitude, amplitude)
        postselect_inds_and_vals = {1: (9, 1), 2: (11, -1), 3: (15, 1)}
        ind, val = postselect_inds_and_vals[n_postselects]

        expected_state = qml.math.zeros(2**num_wires, dtype=complex)
        expected_state[ind] = val
        expected_state = qml.math.reshape(expected_state, (2,) * num_wires)
        measure_spy = mocker.spy(sampling, "sample_state")

        for _ in range(5):
            _ = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=config, shots=1000)

            measure_spy.assert_called()
            state = measure_spy.call_args.args[0]
            shots = measure_spy.call_args.kwargs["shots"]
            spied_shots.append(shots)
            assert qml.math.allclose(state, expected_state)

            measure_spy.reset_mock()

        if postselect_mode == "fill-shots":
            assert all(s == 1000 for s in spied_shots)
        else:
            assert qml.math.allclose(
                qml.math.mean(spied_shots),
                1000 / (2**n_postselects),
                atol=5 + 2 * n_postselects,
                rtol=0,
            )

    @pytest.mark.parametrize("postselect_mode", ["fill-shots", "hw-like"])
    @pytest.mark.parametrize("n_iters", [1, 2, 3])
    def test_mcm_samples_shape(self, postselect_mode, n_iters):
        """Test that returning samples of mcms has the correct shape."""
        num_wires = 4
        dev = qml.device("default.qubit", wires=num_wires, seed=jax.random.PRNGKey(1234))
        config = create_execution_config(postselect_mode=postselect_mode)

        @DeferMeasurementsInterpreter(num_wires=num_wires)
        def f():
            ms = []
            for _ in range(n_iters):
                qml.Hadamard(0)
                ms.append(qml.measure(0, postselect=1))

            return qml.sample(op=ms)

        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=config, shots=100)

        if postselect_mode == "fill-shots":
            assert qml.math.shape(res[0]) == (100, n_iters)

        if postselect_mode == "hw-like":
            shape = qml.math.shape(res[0])
            # Other tests have verified that the _number_ of samples
            # will be consisent with the expected behaviour of hw-like
            # execution, so we only verify the n_mcms dimension
            assert len(shape) == 2
            assert shape[1] == n_iters
