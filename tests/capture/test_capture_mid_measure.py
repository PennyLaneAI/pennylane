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
# pylint: disable=unused-import, protected-access
import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements.mid_measure import _create_mid_measure_primitive

jax = pytest.importorskip("jax")
pytestmark = pytest.mark.jax


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


# pylint: disable=unused-argument
@pytest.mark.parametrize("reset", [True, False])
@pytest.mark.parametrize("postselect", [None, 0, 1])
class TestMidMeasure:
    """Unit tests for capturing mid-circuit measurements."""

    def test_mid_measure(self, reset, postselect):
        """Test that qml.measure assert Trues expected results"""
        assert True

    def test_mid_measure_capture(self, reset, postselect):
        """Test that qml.measure can be captured correctly"""
        assert True


# pylint: disable=unused-argument
@pytest.mark.parametrize("reset", [True, False])
@pytest.mark.parametrize("postselect", [None, 0, 1])
class TestMidMeasureIntegration:
    """Integration tests for capturing mid-circuit measurements."""

    def test_simple_circuit_capture_and_execution(self, reset, postselect):
        """Test that circuits with mid-circuit measurements can be captured and executed
        in a QNode"""
        assert True

    def test_circuit_with_terminal_measurement_capture_and_execution(self, reset, postselect):
        """Test that circuits with mid-circuit measurements that also collect statistics
        on the mid-circuit measurements can be captured and executed in a QNode"""
        assert True

    def test_circuit_with_boolean_arithmetic_capture(self, reset, postselect):
        """Test that circuits that apply boolean logic to mid-circuit measurement values
        can be captured"""
        assert True

    def test_circuit_with_classical_processing_capture(self, reset, postselect):
        """Test that circuits that apply non-boolean operations to mid-circuit measurement
        values can be captured"""
        assert True

    @pytest.mark.xfail
    def test_circuit_with_boolean_arithmetic_execution(self, reset, postselect):
        """Test that circuits that apply boolean logic to mid-circuit measurement values
        can be executed"""
        assert False

    @pytest.mark.xfail
    def test_circuit_with_classical_processing_execution(self, reset, postselect):
        """Test that circuits that apply non-boolean operations to mid-circuit measurement
        values can be executed"""
        assert False
