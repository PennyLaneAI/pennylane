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

from pennylane.transforms.defer_measurements import DeferMeasurementsInterpreter

pytestmark = [
    pytest.mark.jax,
    pytest.mark.usefixtures("enable_disable_plxpr"),
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
class TestExecutionFiniteShots:
    """Tests for executing circuits with finite shots."""

    @pytest.mark.parametrize("postselect", [0, 1])
    def test_hw_like_samples(self, postselect):
        """Test that postselect_mode="hw-like" updates the number of samples as expected."""
        num_wires = 5
        dev = qml.device(
            "default.qubit", wires=num_wires, shots=1000, seed=jax.random.PRNGKey(1234)
        )
        config = create_execution_config(postselect_mode="hw-like")

        @DeferMeasurementsInterpreter(num_wires=num_wires)
        def f():
            qml.Hadamard(0)
            qml.measure(0, postselect=postselect)
            return qml.sample(wires=[0])

        jaxpr = jax.make_jaxpr(f)()
        res = tuple(
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=config)[0] for _ in range(10)
        )
        assert all(qml.math.allclose(r, postselect) for r in res)
        lens = [len(r) for r in res]
        assert qml.math.allclose(qml.math.mean(lens), 500, atol=10, rtol=0)

    @pytest.mark.parametrize("postselect", [0, 1])
    def test_fill_shots_samples(self, postselect):
        """Test that postselect_mode="fill-shots" updates the number of samples as expected."""
        num_wires = 5
        dev = qml.device(
            "default.qubit", wires=num_wires, shots=1000, seed=jax.random.PRNGKey(1234)
        )
        config = create_execution_config(postselect_mode="fill-shots")

        @DeferMeasurementsInterpreter(num_wires=num_wires)
        def f():
            qml.Hadamard(0)
            qml.measure(0, postselect=postselect)
            return qml.sample(wires=[0])

        jaxpr = jax.make_jaxpr(f)()
        res = tuple(
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, execution_config=config)[0] for _ in range(5)
        )
        assert all(qml.math.allclose(r, postselect) for r in res)
        lens = [len(r) for r in res]
        assert all(l == 1000 for l in lens)

    @pytest.mark.parametrize("mp_fn", [qml.sample, qml.expval, qml.probs, qml.var])
    def test_mcm_statistics(self, mp_fn):
        """Test that collecting statistics on MCMs works as expected with finite shots."""

    @pytest.mark.parametrize("mp_fn", [qml.sample, qml.expval, qml.probs, qml.var])
    def test_non_mcm_terminal_measurements(self, mp_fn):
        """Test that non-MCM terminal measurement results are correct with finite shots."""
