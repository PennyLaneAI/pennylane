# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module tests the default qubit interpreter.
"""
import pytest

jax = pytest.importorskip("jax")
pytestmark = [pytest.mark.jax, pytest.mark.capture]

from jax import numpy as jnp  # pylint: disable=wrong-import-position

import pennylane as qml  # pylint: disable=wrong-import-position

# must be below the importorskip
# pylint: disable=wrong-import-position
from pennylane.devices import ExecutionConfig
from pennylane.devices.qubit.dq_interpreter import DefaultQubitInterpreter


def test_initialization():
    """Test that relevant properties are set on initialization."""
    dq = DefaultQubitInterpreter(num_wires=3, shots=None)
    assert dq.num_wires == 3
    assert dq.original_shots == qml.measurements.Shots(None)
    assert isinstance(dq.initial_key, jax.numpy.ndarray)
    assert dq.stateref is None


def test_no_partitioned_shots():
    """Test that an error is raised if partitioned shots is requested."""

    with pytest.raises(NotImplementedError, match="does not yet support partitioned shots"):
        DefaultQubitInterpreter(num_wires=1, shots=(100, 100, 100))


def test_setup_and_cleanup():
    """Test setup initializes the stateref dictionary and cleanup removes it."""
    key = jax.random.PRNGKey(1234)
    dq = DefaultQubitInterpreter(num_wires=2, shots=2, key=key)
    assert dq.stateref is None

    with pytest.raises(AttributeError, match="execution not yet initialized"):
        _ = dq.state

    dq.setup()
    assert isinstance(dq.stateref, dict)
    assert list(dq.stateref.keys()) == ["state", "shots", "key", "is_state_batched"]

    assert dq.stateref["key"] is key
    assert dq.key is key

    assert dq.stateref["shots"] == qml.measurements.Shots(2)
    assert dq.shots == qml.measurements.Shots(2)

    assert dq.state is dq.stateref["state"]
    assert dq.is_state_batched is False
    assert dq.stateref["is_state_batched"] is False
    expected = jax.numpy.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    assert qml.math.allclose(dq.state, expected)

    dq.cleanup()
    assert dq.stateref is None


@pytest.mark.parametrize("name", ("state", "key", "is_state_batched", "shots"))
def test_working_state_key_before_setup(name):
    """Test that state and key can't be accessed before setup."""

    key = jax.random.PRNGKey(9876)

    dq = DefaultQubitInterpreter(num_wires=1, key=key)

    with pytest.raises(AttributeError, match="execution not yet initialized"):
        setattr(dq, name, [1.0, 0.0])

    with pytest.raises(AttributeError, match="execution not yet initialized"):
        getattr(dq, name)


def test_simple_execution():
    """Test the execution, jitting, and gradient of a simple quantum circuit."""

    @DefaultQubitInterpreter(num_wires=1, shots=None)
    def f(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    res = f(0.5)
    assert qml.math.allclose(res, jax.numpy.cos(0.5))

    jit_res = jax.jit(f)(0.5)
    assert qml.math.allclose(jit_res, res)

    g = jax.grad(f)(jax.numpy.array(0.5))
    assert qml.math.allclose(g, -jax.numpy.sin(0.5))


def test_capture_remains_enabled_if_measurement_error():
    """Test that capture remains enabled if there is a measurement error."""

    @DefaultQubitInterpreter(num_wires=1, shots=None)
    def g():
        return qml.sample(wires=0)  # sampling with analytic execution.

    with pytest.raises(NotImplementedError):
        g()

    assert qml.capture.enabled()


def test_pytree_function_output():
    """Test that the results respect the pytree output of the function."""

    @DefaultQubitInterpreter(num_wires=1, shots=None)
    def g():
        return {
            "probs": qml.probs(wires=0),
            "state": qml.state(),
            "var_Z": qml.var(qml.Z(0)),
            "var_X": qml.var(qml.X(0)),
        }

    res = g()
    assert qml.math.allclose(res["probs"], [1.0, 0.0])
    assert qml.math.allclose(res["state"], [1.0, 0.0 + 0j])
    assert qml.math.allclose(res["var_Z"], 0.0)
    assert qml.math.allclose(res["var_X"], 1.0)


def test_mcm_reset():
    """Test that mid circuit measurements can reset the state."""

    @DefaultQubitInterpreter(num_wires=1)
    def f():
        qml.X(0)
        qml.measure(0, reset=True)
        return qml.state()

    out = f()
    assert qml.math.allclose(out, jnp.array([1.0, 0.0]))  # reset into zero state.


def test_mcm_postselect_on_opposite_value():
    """Test that the results are nan's if we postselect on the opposite of the mcm."""

    @DefaultQubitInterpreter(num_wires=1)
    def f():
        qml.measure(0, postselect=1)
        return qml.expval(qml.Z(0)), qml.state()

    expval, state = f()
    assert jax.numpy.isnan(expval)
    assert jax.numpy.isnan(state).all()


def test_operator_arithmetic():
    """Test that dq can execute operator arithmetic."""

    @DefaultQubitInterpreter(num_wires=2)
    def f(x):
        qml.RY(1.0, 0)
        qml.adjoint(qml.RY(x, 0))
        _ = qml.SX(1) ** 2
        return qml.expval(qml.Z(0) + 2 * qml.Z(1))

    output = f(0.5)
    expected = jnp.cos(1 - 0.5) - 2 * 1
    assert qml.math.allclose(output, expected)


def test_parameter_broadcasting():
    """Test that DefaultQubit can execute a circuit with parameter broadcasting."""

    @DefaultQubitInterpreter(num_wires=3)
    def f(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    x = jax.numpy.array([1.2, 2.3, 3.4])
    output = f(x)
    expected = jax.numpy.cos(x)
    assert qml.math.allclose(output, expected)


@pytest.mark.parametrize("basis_state", [0, 1])
def test_projector(basis_state):
    """Test that Projectors are applied correctly as operations."""
    config = ExecutionConfig()

    @DefaultQubitInterpreter(num_wires=1, shots=None, execution_config=config)
    def circuit(x):
        qml.RX(x, 0)
        qml.Projector(jnp.array([basis_state]), 0)
        return qml.state()

    x = 1.5
    expected_state = qml.math.array([jnp.cos(x / 2), -1j * jnp.sin(x / 2)])
    expected_state[int(not basis_state)] = 0
    expected_state = expected_state / qml.math.norm(expected_state)

    assert qml.math.allclose(circuit(x), expected_state)


def test_informative_error_if_jitting_abstract_conditionals():
    """Test that an informative error is raised if jitting is attempted with abtract conditionals."""

    config = ExecutionConfig()

    @DefaultQubitInterpreter(num_wires=1, shots=None, execution_config=config)
    def circuit(val):
        qml.cond(val, qml.X, qml.Y)(0)
        return qml.state()

    with pytest.raises(
        NotImplementedError, match="does not yet support jitting cond with abstract conditions"
    ):
        _ = jax.jit(circuit)(True)


class TestSampling:
    """Test cases for generating samples."""

    def test_known_sampling(self, seed):
        """Test sampling output with deterministic sampling output"""

        @DefaultQubitInterpreter(num_wires=2, shots=10, key=jax.random.PRNGKey(seed))
        def sampler():
            qml.X(0)
            return qml.sample(wires=(0, 1))

        results = sampler()

        expected0 = jax.numpy.ones((10,))  # zero wire
        expected1 = jax.numpy.zeros((10,))  # one wire
        expected = jax.numpy.vstack([expected0, expected1]).T

        assert qml.math.allclose(results, expected)

    def test_same_key_same_results(self, seed):
        """Test that two circuits with the same key give identical results."""
        key = jax.random.PRNGKey(seed)

        @DefaultQubitInterpreter(num_wires=1, shots=100, key=key)
        def circuit1():
            qml.Hadamard(0)
            return qml.sample(wires=0)

        @DefaultQubitInterpreter(num_wires=1, shots=100, key=key)
        def circuit2():
            qml.Hadamard(0)
            return qml.sample(wires=0)

        res1_first_exec = circuit1()
        res2_first_exec = circuit2()
        res1_second_exec = circuit1()
        res2_second_exec = circuit2()

        assert qml.math.allclose(res1_first_exec, res2_first_exec)
        assert qml.math.allclose(res1_second_exec, res2_second_exec)

    @pytest.mark.parametrize("mcm_value", (0, 1))
    def test_return_mcm(self, mcm_value):
        """Test that the interpreter can return the result of mid circuit measurements"""

        @DefaultQubitInterpreter(num_wires=1)
        def f():
            if mcm_value:
                qml.X(0)
            return qml.measure(0)

        output = f()
        assert qml.math.allclose(output, mcm_value)

    def test_mcm_depends_on_key(self):
        """Test that the value of an mcm depends on the key."""

        def get_mcm_from_key(key):
            @DefaultQubitInterpreter(num_wires=1, key=key)
            def f():
                qml.H(0)
                return qml.measure(0)

            return f()

        for key in range(0, 100, 10):
            m1 = get_mcm_from_key(jax.random.PRNGKey(key))
            m2 = get_mcm_from_key(jax.random.PRNGKey(key))
            assert qml.math.allclose(m1, m2)

        samples = [int(get_mcm_from_key(jax.random.PRNGKey(key))) for key in range(0, 100, 1)]
        assert set(samples) == {0, 1}

    def test_classical_transformation_mcm_value(self):
        """Test that mid circuit measurements can be used in classical manipulations."""

        @DefaultQubitInterpreter(num_wires=1)
        def f():
            qml.X(0)
            m0 = qml.measure(0)  # 1
            qml.X(0)  # reset to 0
            qml.RX(2 * m0, wires=0)
            return qml.expval(qml.Z(0))

        expected = jax.numpy.cos(2.0)
        assert qml.math.allclose(f(), expected)

    @pytest.mark.parametrize("mp_type", (qml.sample, qml.expval, qml.probs))
    def test_mcm_measurements_not_yet_implemented(self, mp_type):
        """Test that measurements of mcms are not yet implemented"""

        @DefaultQubitInterpreter(num_wires=1)
        def f():
            m0 = qml.measure(0)
            if mp_type == qml.probs:
                return mp_type(op=m0)
            return mp_type(m0)

        with pytest.raises(NotImplementedError):
            f()

    def test_mcms_not_all_same_key(self, seed):
        """Test that each mid circuit measurement has a different key."""

        @DefaultQubitInterpreter(num_wires=1, shots=None, key=jax.random.PRNGKey(seed))
        def g():
            ms = []
            for _ in range(33):
                qml.Hadamard(0)
                ms.append(qml.measure(0, reset=0))
            return ms

        output = g()
        assert not all(qml.math.allclose(output[0], output[i]) for i in range(1, 33))
        # only way we could get different values between the mcms is if they had different seeds

    def test_each_measurement_has_different_key(self, seed):
        """Test that each sampling measurement is performed with a different key."""

        @DefaultQubitInterpreter(num_wires=1, shots=100, key=jax.random.PRNGKey(seed))
        def g():
            qml.Hadamard(0)
            return qml.sample(wires=0), qml.sample(wires=0)

        res1, res2 = g()
        assert not qml.math.allclose(res1, res2)

    def test_more_executions_same_interpreter_different_results(self, seed):
        """Test that if multiple executions occur with the same interpreter, they will have different results."""

        @DefaultQubitInterpreter(num_wires=1, shots=100, key=jax.random.PRNGKey(seed))
        def f():
            qml.Hadamard(0)
            return qml.sample(wires=0)

        s1 = f()
        s2 = f()  # should be done with different key, leading to different results.
        assert not qml.math.allclose(s1, s2)

    # 20 % failure rate; need to revise and fix soon
    # FIXME: [sc-95722]
    @pytest.mark.local_salt(8)
    @pytest.mark.parametrize("n_postselects", [1, 2, 3])
    def test_projector_samples_hw_like(self, seed, n_postselects):
        """Test that hw-like postselect_mode causes the number of samples to change as expected."""
        config = ExecutionConfig(mcm_config={"postselect_mode": "hw-like"})

        @DefaultQubitInterpreter(
            num_wires=1, shots=1000, key=jax.random.PRNGKey(seed), execution_config=config
        )
        def f():
            for _ in range(n_postselects):
                qml.Hadamard(0)
                qml.Projector(jnp.array([1]), 0)
            return qml.sample(wires=0)

        lens = [len(f()) for _ in range(10)]
        assert qml.math.allclose(
            qml.math.mean(lens), int(1000 / (2**n_postselects)), atol=5 + 2 * n_postselects, rtol=0
        )

    @pytest.mark.parametrize("n_postselects", [1, 2, 3])
    def test_projector_samples_fill_shots(self, seed, n_postselects):
        """Test that hw-like postselect_mode causes the number of samples to change as expected."""
        config = ExecutionConfig(mcm_config={"postselect_mode": "fill-shots"})

        @DefaultQubitInterpreter(
            num_wires=1, shots=1000, key=jax.random.PRNGKey(seed), execution_config=config
        )
        def f():
            for _ in range(n_postselects):
                qml.Hadamard(0)
                qml.Projector(jnp.array([1]), 0)
            return qml.sample(wires=0)

        lens = [len(f()) for _ in range(10)]
        assert all(l == 1000 for l in lens)


class TestCustomPrimitiveRegistrations:
    """Tests for primitives with custom primitive registrations."""

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_transform(self, lazy):
        """Test that the adjoint_transform is interpreted correctly."""

        @DefaultQubitInterpreter(num_wires=1, shots=None)
        def circuit(x):

            def adjoint_fn(y):
                phi = y * jnp.pi / 2
                qml.RZ(phi, 0)
                qml.RX(phi - jnp.pi, 0)

            qml.adjoint(adjoint_fn, lazy=lazy)(x)
            return qml.state()

        x = 1.5
        rz_phi = -x * jnp.pi / 2
        rx_phi = rz_phi + jnp.pi
        expected_state = jnp.array(
            [
                jnp.cos(rx_phi / 2) * jnp.exp(-rz_phi * 1j / 2),
                -1j * jnp.sin(rx_phi / 2) * jnp.exp(rz_phi * 1j / 2),
            ]
        )
        assert jnp.allclose(circuit(x), expected_state)

    def test_ctrl_transform(self):
        """Test that the ctrl_transform is interpreted correctly."""

        @DefaultQubitInterpreter(num_wires=3, shots=None)
        def circuit(x):
            qml.X(0)

            def ctrl_fn(y):
                phi = y * jnp.pi / 2
                qml.RZ(phi, 2)
                qml.RX(phi - jnp.pi, 2)

            qml.ctrl(ctrl_fn, control=[0, 1], control_values=[1, 0])(x)
            return qml.state()

        x = 1.5
        rz_phi = x * jnp.pi / 2
        rx_phi = rz_phi - jnp.pi
        expected_state = qml.math.zeros(8, dtype=complex)
        expected_state[4] = jnp.cos(rx_phi / 2) * jnp.exp(-rz_phi * 1j / 2)
        expected_state[5] = -1j * jnp.sin(rx_phi / 2) * jnp.exp(-rz_phi * 1j / 2)

        assert jnp.allclose(circuit(x), expected_state)


class TestClassicalComponents:
    """Test execution of classical components."""

    def test_classical_operations_in_circuit(self):
        """Test that we can have classical operations in the circuit."""

        @DefaultQubitInterpreter(num_wires=1)
        def f(x, y, w):
            qml.RX(2 * x + y, wires=w - 1)
            return qml.expval(qml.Z(0))

        x = jax.numpy.array(0.5)
        y = jax.numpy.array(1.2)
        w = jax.numpy.array(1)

        output = f(x, y, w)
        expected = jax.numpy.cos(2 * x + y)
        assert qml.math.allclose(output, expected)

    def test_for_loop(self):
        """Test that the for loop can be executed."""

        @DefaultQubitInterpreter(num_wires=4)
        def f(y):
            @qml.for_loop(4)
            def f(i, x):
                qml.RX(x, i)
                return x + 0.1

            f(y)
            return [qml.expval(qml.Z(i)) for i in range(4)]

        output = f(1.0)
        assert len(output) == 4
        assert qml.math.allclose(output[0], jax.numpy.cos(1.0))
        assert qml.math.allclose(output[1], jax.numpy.cos(1.1))
        assert qml.math.allclose(output[2], jax.numpy.cos(1.2))
        assert qml.math.allclose(output[3], jax.numpy.cos(1.3))

    def test_for_loop_consts(self):
        """Test that the for_loop can be executed properly when it has closure variables."""

        @DefaultQubitInterpreter(num_wires=2)
        def g(x):
            @qml.for_loop(2)
            def f(i):
                qml.RX(x, i)  # x is closure variable

            f()
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res1, res2 = g(jax.numpy.array(-0.654))
        expected = jnp.cos(-0.654)
        assert qml.math.allclose(res1, expected)
        assert qml.math.allclose(res2, expected)

    def test_while_loop(self):
        """Test that the while loop can be executed."""

        @DefaultQubitInterpreter(num_wires=4)
        def f():
            def cond_fn(i):
                return i < 4

            @qml.while_loop(cond_fn)
            def f(i):
                qml.X(i)
                return i + 1

            f(0)
            return [qml.expval(qml.Z(i)) for i in range(4)]

        output = f()
        assert qml.math.allclose(output, [-1, -1, -1, -1])

    def test_while_loop_with_consts(self):
        """Test that both the cond_fn and body_fn can contain constants with the while loop."""

        @DefaultQubitInterpreter(num_wires=2, shots=None, key=jax.random.PRNGKey(87665))
        def g(x, target):
            def cond_fn(i):
                return i < target

            @qml.while_loop(cond_fn)
            def f(i):
                qml.RX(x, 0)
                return i + 1

            f(0)
            return qml.expval(qml.Z(0))

        output = g(jnp.array(1.2), jnp.array(2))

        assert qml.math.allclose(output, jnp.cos(2 * 1.2))

    def test_cond_boolean(self):
        """Test that cond can be used with normal classical values."""

        def true_fn(x):
            qml.RX(x, 0)
            return x + 1

        def false_fn(x):
            return 2 * x

        @DefaultQubitInterpreter(num_wires=1)
        def f(x, val):
            out = qml.cond(val, true_fn, false_fn)(x)
            return qml.probs(wires=0), out

        output_true = f(0.5, True)
        expected0 = [jax.numpy.cos(0.5 / 2) ** 2, jax.numpy.sin(0.5 / 2) ** 2]
        assert qml.math.allclose(output_true[0], expected0)
        assert qml.math.allclose(output_true[1], 1.5)  # 0.5 + 1

        output_false = f(0.5, False)
        assert qml.math.allclose(output_false[0], [1.0, 0.0])
        assert qml.math.allclose(output_false[1], 1.0)  # 2 * 0.5

    def test_cond_mcm(self):
        """Test that cond can be used with the output of mcms."""

        def true_fn(y):
            qml.RX(y, 0)

        # pylint: disable=unused-argument
        def false_fn(y):
            qml.X(0)

        @DefaultQubitInterpreter(num_wires=1, shots=None)
        def g(x):
            qml.X(0)
            m0 = qml.measure(0)
            qml.X(0)
            qml.cond(m0, true_fn, false_fn)(x)
            return qml.probs(wires=0)

        output = g(0.5)
        expected = [jnp.cos(0.5 / 2) ** 2, jnp.sin(0.5 / 2) ** 2]
        assert qml.math.allclose(output, expected)

    def test_cond_false_no_false_fn(self):
        """Test nothing is returned when the false_fn is not provided but the condition is false."""

        def true_fn(w):
            qml.X(w)

        @DefaultQubitInterpreter(num_wires=1)
        def g(condition):
            qml.cond(condition, true_fn)(0)
            return qml.expval(qml.Z(0))

        out = g(False)
        assert qml.math.allclose(out, 1.0)

    def test_condition_with_consts(self):
        """Test that each branch in a condition can contain consts."""

        @DefaultQubitInterpreter(num_wires=1)
        def circuit(x, y, z, condition0, condition1):

            def true_fn():
                qml.RX(x, 0)

            def false_fn():
                qml.RX(y, 0)

            def elif_fn():
                qml.RX(z, 0)

            qml.cond(condition0, true_fn, false_fn=false_fn, elifs=((condition1, elif_fn),))()

            return qml.expval(qml.Z(0))

        x = jax.numpy.array(0.3)
        y = jax.numpy.array(0.6)
        z = jax.numpy.array(1.2)

        res0 = circuit(x, y, z, True, False)
        assert qml.math.allclose(res0, jnp.cos(x))

        res1 = circuit(x, y, z, False, True)
        assert qml.math.allclose(res1, jnp.cos(z))  # elif branch = z

        res2 = circuit(x, y, z, False, False)
        assert qml.math.allclose(res2, jnp.cos(y))  # false fn = y


@pytest.mark.usefixtures("enable_disable_dynamic_shapes")
class TestDynamicShapes:
    """Tests for creating arrays with a dynamic input."""

    @pytest.mark.parametrize(
        "creation_fn", [jax.numpy.ones, jax.numpy.zeros, lambda s: jax.numpy.full(s, 0.5)]
    )
    def test_broadcast_in_dim(self, creation_fn):
        """Test that DefaultQubitInterpreter can handle jax.numpy.ones and the associated broadcast_in_dim primitive."""

        @DefaultQubitInterpreter(num_wires=1)
        def f(n):
            ones = creation_fn((n + 1,))
            qml.RX(ones, wires=0)
            return qml.expval(qml.Z(0))

        output = f(3)
        assert output.shape == (4,)
        ones = creation_fn(4)
        assert qml.math.allclose(output, jax.numpy.cos(ones))

    def test_dynamic_shape_arange(self):
        """Test that DefaultQubitInterpreter can handle jnp.arange."""

        @DefaultQubitInterpreter(num_wires=1)
        def f(n):
            x = jax.numpy.arange(n)
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        output = f(4)
        assert output.shape == (4,)
        x = jax.numpy.arange(4)
        assert qml.math.allclose(output, jax.numpy.cos(x))
