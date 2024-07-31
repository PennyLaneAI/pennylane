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
"""Tests for capturing mid-circuit measurements."""
# pylint: disable=ungrouped-imports, wrong-import-order, wrong-import-position
import pytest

import pennylane as qml
from pennylane.measurements.mid_measure import MeasurementValue, MidMeasureMP

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pennylane.capture import AbstractOperator

pytestmark = pytest.mark.jax


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


@pytest.mark.unit
@pytest.mark.parametrize("reset", [True, False])
@pytest.mark.parametrize("postselect", [None, 0, 1])
class TestMidMeasureUnit:
    """Unit tests for capturing mid-circuit measurements."""

    def test_mid_measure(self, reset, postselect):
        """Test that qml.measure returns expected results."""
        with qml.queuing.AnnotatedQueue() as q:
            m = qml.measure(0, reset=reset, postselect=postselect)

        assert len(q) == 1
        mp = list(q.keys())[0].obj
        assert isinstance(mp, MidMeasureMP)
        assert mp.reset == reset
        assert mp.postselect == postselect
        assert isinstance(m, MeasurementValue)
        assert len(m.measurements) == 1
        assert m.measurements[0] is mp

    @pytest.mark.parametrize("x64_mode", [True, False])
    def test_mid_measure_capture(self, reset, postselect, x64_mode):
        """Test that qml.measure is captured correctly."""
        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        jaxpr = jax.make_jaxpr(qml.measure)(0, reset=reset, postselect=postselect)
        assert len(jaxpr.eqns) == 1
        invars = jaxpr.eqns[0].invars
        outvars = jaxpr.eqns[0].outvars
        assert len(invars) == len(outvars) == 1
        expected_dtype = jnp.int64 if x64_mode else jnp.int32
        assert invars[0].aval == jax.core.ShapedArray((), expected_dtype, weak_type=True)
        assert outvars[0].aval == jax.core.ShapedArray((), expected_dtype)
        assert set(jaxpr.eqns[0].params.keys()) == {"reset", "postselect"}

        jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.integration
class TestMidMeasureCapture:
    """Integration tests for capturing mid-circuit measurements in quantum functions."""

    def test_simple_circuit_capture(self):
        """Test that circuits with mid-circuit measurements can be captured."""

        def f(x):
            qml.RX(x, 0)
            qml.measure(0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.0)
        assert jaxpr.eqns[1].primitive.name == "measure"
        assert isinstance(jaxpr.eqns[1].outvars[0], jax.core.DropVar)

    @pytest.mark.parametrize("measurement", ["expval", "sample", "var", "probs"])
    @pytest.mark.parametrize("multi_mcm", [True, False])
    def test_circuit_with_terminal_measurement_capture(self, measurement, multi_mcm):
        """Test that circuits with mid-circuit measurements that also collect statistics
        on the mid-circuit measurements can be captured."""

        if multi_mcm and measurement in {"expval", "var"}:
            pytest.skip("Cannot use sequence of MCMs with expval or var.")

        p_name = measurement + "_mcm"
        measure_fn = getattr(qml, measurement)

        def f(x):
            qml.RX(x, 0)
            m1 = qml.measure(0)
            m2 = qml.measure(0)
            return measure_fn(op=[m1, m2] if multi_mcm else m1)

        jaxpr = jax.make_jaxpr(f)(1.0)

        assert jaxpr.eqns[1].primitive.name == "measure"
        assert isinstance(mcm1 := jaxpr.eqns[1].outvars[0], jax.core.Var)
        assert jaxpr.eqns[2].primitive.name == "measure"
        assert isinstance(mcm2 := jaxpr.eqns[2].outvars[0], jax.core.Var)

        assert jaxpr.eqns[3].primitive.name == p_name
        assert jaxpr.eqns[3].invars == [mcm1, mcm2] if multi_mcm else [mcm1]

    def test_circuit_with_boolean_arithmetic_capture(self):
        """Test that circuits that apply boolean logic to mid-circuit measurement values
        can be captured."""

        def f(x):
            qml.RX(x, 0)
            m1 = qml.measure(0)
            m2 = qml.measure(0)
            a = ~m1
            b = m2 > m1
            _ = m1 | m2
            c = a != m2
            _ = b & c
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.0)

        # MCMs
        mcm1 = jaxpr.eqns[1].outvars[0]
        mcm2 = jaxpr.eqns[2].outvars[0]

        # ~m1
        assert jaxpr.eqns[3].primitive.name == "not"
        assert jaxpr.eqns[3].invars == [mcm1]
        a = jaxpr.eqns[3].outvars[0]

        # m2 > m1
        assert jaxpr.eqns[4].primitive.name == "gt"
        assert jaxpr.eqns[4].invars == [mcm2, mcm1]
        b = jaxpr.eqns[4].outvars[0]

        # m1 | m2
        assert jaxpr.eqns[5].primitive.name == "or"
        assert jaxpr.eqns[5].invars == [mcm1, mcm2]

        # a != m2
        assert jaxpr.eqns[6].primitive.name == "ne"
        assert jaxpr.eqns[6].invars == [a, mcm2]
        c = jaxpr.eqns[6].outvars[0]

        # b & c
        assert jaxpr.eqns[7].primitive.name == "and"
        assert jaxpr.eqns[7].invars == [b, c]

    def test_circuit_with_classical_processing_capture(self):
        """Test that circuits that apply non-boolean operations to mid-circuit measurement
        values can be captured."""

        def f(x):
            qml.RX(x, 0)
            m1 = qml.measure(0)
            m2 = qml.measure(0)
            a = 3.1 * m1
            b = m2 / a
            _ = a**b
            _ = a - m2
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.0)

        # MCMs
        mcm1 = jaxpr.eqns[1].outvars[0]
        mcm2 = jaxpr.eqns[2].outvars[0]

        # Type promotion primitives
        assert jaxpr.eqns[3].primitive.name == "convert_element_type"
        assert jaxpr.eqns[3].invars == [mcm1]
        mcm1_f = jaxpr.eqns[3].outvars[0]
        assert jaxpr.eqns[5].primitive.name == "convert_element_type"
        assert jaxpr.eqns[5].invars == [mcm2]
        mcm2_f1 = jaxpr.eqns[5].outvars[0]
        assert jaxpr.eqns[8].primitive.name == "convert_element_type"
        assert jaxpr.eqns[8].invars == [mcm2]
        mcm2_f2 = jaxpr.eqns[8].outvars[0]

        # 3.1 * m1
        assert jaxpr.eqns[4].primitive.name == "mul"
        assert isinstance(jaxpr.eqns[4].invars[0], jax.core.Literal)
        assert jaxpr.eqns[4].invars[1] == mcm1_f
        a = jaxpr.eqns[4].outvars[0]

        # m2 / a
        assert jaxpr.eqns[6].primitive.name == "div"
        assert jaxpr.eqns[6].invars == [mcm2_f1, a]
        b = jaxpr.eqns[6].outvars[0]

        # a**b
        assert jaxpr.eqns[7].primitive.name == "pow"
        assert jaxpr.eqns[7].invars == [a, b]

        # a - m2
        assert jaxpr.eqns[9].primitive.name == "sub"
        assert jaxpr.eqns[9].invars == [a, mcm2_f2]

    @pytest.mark.parametrize("fn", [jnp.sin, jnp.log, jnp.exp, jnp.sqrt])
    def mid_measure_processed_with_jax_numpy_capture(self, fn):
        """Test that a circuit containing mid-circuit measurements processed using jax.numpy
        can be captured."""

        def f(x):
            qml.RX(x, 0)
            m = qml.measure(0)
            _ = fn(m)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.0)

        mcm = jaxpr.eqns[1].outvars[0]

        # Type promotion primitive
        assert jaxpr.eqns[2].primitive.name == "convert_element_type"
        assert jaxpr.eqns[2].invars == [mcm]
        mcm_f = jaxpr.eqns[2].outvars[0]

        # Arithmetic function primitive
        assert jaxpr.eqns[3].primitive.name == fn.__name__
        assert jaxpr.eqns[3].invars == [mcm_f]

    def mid_measure_broadcast_capture(self):
        """Test that creating and using arrays of mid-circuit measurements can be captured."""

        def f(x):
            qml.RX(x, 0)
            m1 = qml.measure(0)
            m2 = qml.measure(0)
            arr = jnp.array([m1, m2])
            arr = jnp.reshape(arr, (1, 1, -1))
            jnp.max(arr)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.0)

        mcm1 = jaxpr.eqns[1].outvars[0]
        mcm2 = jaxpr.eqns[2].outvars[0]

        # jnp.array([m1, m2])
        assert jaxpr.eqns[3].primitive.name == "broadcast_in_dim"
        assert jaxpr.eqns[3].invars == [mcm1]
        a = jaxpr.eqns[3].outvars[0]

        assert jaxpr.eqns[4].primitive.name == "broadcast_in_dim"
        assert jaxpr.eqns[4].invars == [mcm2]
        b = jaxpr.eqns[4].outvars[0]

        assert jaxpr.eqns[5].primitive.name == "concatenate"
        assert jaxpr.eqns[5].invars == [a, b]
        arr = jaxpr.eqns[5].outvars[0]
        assert arr.aval.shape == (2,)

        # jnp.reshape(arr, (1, 1, -1))
        assert jaxpr.eqns[6].primitive.name == "reshape"
        assert jaxpr.eqns[6].invars == [arr]
        arr = jaxpr.eqns[6].outvars[0]
        assert arr.aval.shape == (1, 1, 2)

        # jnp.max(arr)
        assert jaxpr.eqns[7].primitive.name == "reduce_max"
        assert jaxpr.eqns[7].invars == [arr]

    def test_mid_measure_as_gate_parameter_capture(self):
        """Test that mid-circuit measurements (simple or classical processed) used as gate
        parameters can be captured."""

        def f(x):
            qml.RX(x, 0)
            m = qml.measure(0)
            qml.RX(m, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.0)

        mcm = jaxpr.eqns[1].outvars[0]

        # qml.RX(m, 0)
        assert jaxpr.eqns[2].primitive.name == "RX"
        assert jaxpr.eqns[2].invars[0] == mcm
        assert isinstance(jaxpr.eqns[2].outvars[0].aval, AbstractOperator)


def compare_with_capture_disabled(qnode, *args, **kwargs):
    """Helper function for comparing execution results with capture disabled."""
    res = qnode(*args, **kwargs)
    qml.capture.disable()
    expected = qnode(*args, **kwargs)
    return jnp.allclose(res, expected)


@pytest.fixture(scope="function")
def get_device():
    def get_qubit_device(*args, **kwargs):
        return qml.device("default.qubit", *args, **kwargs)

    yield get_qubit_device


# pylint: disable=too-many-arguments, redefined-outer-name
@pytest.mark.system
@pytest.mark.parametrize("shots", [None, 50])
@pytest.mark.parametrize("mp_fn", [qml.expval, qml.var, qml.sample, qml.probs])
class TestMidMeasureExecute:
    """System-level tests for executing circuits with mid-circuit measurements with program
    capture enabled."""

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("phi", jnp.arange(1.0, 2 * jnp.pi, 1.5))
    def test_simple_circuit_execution(self, phi, reset, postselect, get_device, shots, mp_fn):
        """Test that circuits with mid-circuit measurements can be executed in a QNode."""
        if shots is None and mp_fn is qml.sample:
            pytest.skip("Cannot measure samples in analytic mode")

        dev = get_device(wires=2, shots=shots, seed=jax.random.PRNGKey(12345))

        @qml.qnode(dev)
        def f(x):
            qml.RX(x, 0)
            qml.measure(0, reset=reset, postselect=postselect)
            return mp_fn(op=qml.Z(0))

        assert compare_with_capture_disabled(f, phi)

    @pytest.mark.parametrize("phi", jnp.arange(1.0, 2 * jnp.pi, 1.5))
    @pytest.mark.parametrize("multi_mcm", [True, False])
    def test_circuit_with_terminal_measurement_execution(
        self, phi, get_device, shots, mp_fn, multi_mcm
    ):
        """Test that circuits with mid-circuit measurements that also collect statistics
        on the mid-circuit measurements can be executed in a QNode."""
        if shots is None and mp_fn is qml.sample:
            pytest.skip("Cannot measure samples in analytic mode")

        if multi_mcm and mp_fn in (qml.expval, qml.var):
            pytest.skip("Cannot measure sequences of MCMs with expval or var")

        dev = get_device(wires=2, shots=shots, seed=jax.random.PRNGKey(12345))

        @qml.qnode(dev)
        def f(x, y):
            qml.RX(x, 0)
            m1 = qml.measure(0)
            qml.RX(y, 0)
            m2 = qml.measure(0)
            return mp_fn(op=[m1, m2] if multi_mcm else m1)

        assert compare_with_capture_disabled(f, phi, phi + 1.5)

    @pytest.mark.xfail
    @pytest.mark.parametrize("phi", jnp.arange(1.0, 2 * jnp.pi, 1.5))
    def test_circuit_with_boolean_arithmetic_execution(self, phi, get_device, shots, mp_fn):
        """Test that circuits that apply boolean logic to mid-circuit measurement values
        can be executed."""
        if shots is None and mp_fn is qml.sample:
            pytest.skip("Cannot measure samples in analytic mode")

        dev = get_device(wires=2, shots=shots, seed=jax.random.PRNGKey(12345))

        @qml.qnode(dev)
        def f(x, y):
            qml.RX(x, 0)
            m1 = qml.measure(0)
            qml.RX(y, 0)
            m2 = qml.measure(0)
            a = m1 & m2
            _ = a > m2
            return mp_fn(op=qml.Z(0))

        assert compare_with_capture_disabled(f, phi, phi + 1.5)

    @pytest.mark.xfail
    @pytest.mark.parametrize("phi", jnp.arange(1.0, 2 * jnp.pi, 1.5))
    def test_circuit_with_classical_processing_execution(self, phi, get_device, shots, mp_fn):
        """Test that circuits that apply non-boolean operations to mid-circuit measurement
        values can be executed."""
        if shots is None and mp_fn is qml.sample:
            pytest.skip("Cannot measure samples in analytic mode")

        dev = get_device(wires=2, shots=shots, seed=jax.random.PRNGKey(12345))

        @qml.qnode(dev)
        def f(x, y):
            qml.RX(x, 0)
            m1 = qml.measure(0)
            qml.RX(y, 0)
            m2 = qml.measure(0)
            a = 3.1 * m1
            _ = a ** (m2 / 5)
            return mp_fn(op=qml.Z(0))

        assert f(phi, phi + 1.5)

    @pytest.mark.xfail
    @pytest.mark.parametrize("phi", jnp.arange(1.0, 2 * jnp.pi, 1.5))
    @pytest.mark.parametrize("fn", [jnp.sin, jnp.sqrt, jnp.log, jnp.exp])
    def mid_measure_processed_with_jax_numpy_execution(self, phi, fn, get_device, shots, mp_fn):
        """Test that a circuit containing mid-circuit measurements processed using jax.numpy
        can be executed."""
        if shots is None and mp_fn is qml.sample:
            pytest.skip("Cannot measure samples in analytic mode")

        dev = get_device(wires=2, shots=shots, seed=jax.random.PRNGKey(12345))

        @qml.qnode(dev)
        def f(x):
            qml.RX(x, 0)
            m = qml.measure(0)
            _ = fn(m)
            return mp_fn(op=qml.Z(0))

        assert f(phi)

    @pytest.mark.xfail
    @pytest.mark.parametrize("phi", jnp.arange(1.0, 2 * jnp.pi, 1.5))
    def test_mid_measure_as_gate_parameter_execution(self, phi, get_device, shots, mp_fn):
        """Test that mid-circuit measurements (simple or classical processed) used as gate
        parameters can be executed."""
        if shots is None and mp_fn is qml.sample:
            pytest.skip("Cannot measure samples in analytic mode")

        dev = get_device(wires=2, shots=shots, seed=jax.random.PRNGKey(12345))

        @qml.qnode(dev)
        def f(x):
            qml.RX(x, 0)
            m = qml.measure(0)
            qml.RX(m, 0)
            return mp_fn(op=qml.Z(0))

        assert f(phi)
