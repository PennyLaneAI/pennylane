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
Tests for capturing mid-circuit measurements.
"""
import pytest

import pennylane as qml
from pennylane.measurements.mid_measure import MeasurementValue, MidMeasureMP

jax = pytest.importorskip("jax")
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

    def test_mid_measure_capture(self, reset, postselect):
        """Test that qml.measure is captured correctly."""
        jaxpr = jax.make_jaxpr(qml.measure)(0, reset=reset, postselect=postselect)
        assert len(jaxpr.eqns) == 1
        invars = jaxpr.eqns[0].invars
        outvars = jaxpr.eqns[0].outvars
        assert len(invars) == len(outvars) == 1
        assert isinstance(invars[0].aval, jax.core.ShapedArray)
        assert invars[0].aval.shape == ()
        assert (
            invars[0].aval.dtype == jax.numpy.int64
            if jax.config.jax_enable_x64
            else jax.numpy.int32
        )
        assert isinstance(outvars[0].aval, jax.core.ShapedArray)
        assert outvars[0].aval.shape == ()
        assert outvars[0].aval.dtype == jax.numpy.bool_
        assert set(jaxpr.eqns[0].params.keys()) == {"reset", "postselect"}


# pylint: disable=unused-argument
@pytest.mark.integration
class TestMidMeasureCapture:
    """Integration tests for capturing mid-circuit measurements in quantum functions."""

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_simple_circuit_capture(self, reset, postselect):
        """Test that circuits with mid-circuit measurements can be captured."""
        assert True

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_circuit_with_terminal_measurement_capture(self, reset, postselect):
        """Test that circuits with mid-circuit measurements that also collect statistics
        on the mid-circuit measurements can be captured."""
        assert True

    def test_circuit_with_boolean_arithmetic_capture(self):
        """Test that circuits that apply boolean logic to mid-circuit measurement values
        can be captured."""
        assert True

    def test_circuit_with_classical_processing_capture(self):
        """Test that circuits that apply non-boolean operations to mid-circuit measurement
        values can be captured."""
        assert True

    def mid_measure_processed_with_jax_numpy_capture(self):
        """Test that a circuit containing mid-circuit measurements processed using jax.numpy
        can be captured."""

    def test_mid_measure_as_gate_parameter_capture(self):
        """Test that mid-circuit measurements (simple or classical processed) used as gate
        parameters can be captured."""
        assert True


# pylint: disable=unused-argument
@pytest.mark.system
class TestMidMeasureExecute:
    """System-level tests for executing circuits with mid-circuit measurements with program
    capture enabled."""

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_simple_circuit_execution(self, reset, postselect):
        """Test that circuits with mid-circuit measurements can be executed in a QNode."""
        assert True

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_circuit_with_terminal_measurement_execution(self, reset, postselect):
        """Test that circuits with mid-circuit measurements that also collect statistics
        on the mid-circuit measurements can be executed in a QNode."""
        assert True

    @pytest.mark.xfail
    def test_circuit_with_boolean_arithmetic_execution(self):
        """Test that circuits that apply boolean logic to mid-circuit measurement values
        can be executed."""
        assert False

    @pytest.mark.xfail
    def test_circuit_with_classical_processing_execution(self):
        """Test that circuits that apply non-boolean operations to mid-circuit measurement
        values can be executed."""
        assert False

    @pytest.mark.xfail
    def mid_measure_processed_with_jax_numpy_execution(self):
        """Test that a circuit containing mid-circuit measurements processed using jax.numpy
        can be executed."""

    @pytest.mark.xfail
    def test_mid_measure_as_gate_parameter_execution(self):
        """Test that mid-circuit measurements (simple or classical processed) used as gate
        parameters can be executed."""
        assert False
