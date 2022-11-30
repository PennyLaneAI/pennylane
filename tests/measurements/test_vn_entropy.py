# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the vn_entropy module"""
import copy

import pytest

import pennylane as qml
from pennylane.interfaces import INTERFACE_MAP
from pennylane.measurements import VnEntropy
from pennylane.measurements.vn_entropy import _VnEntropy
from pennylane.wires import Wires


class TestVnEntropy:
    """Unit tests for the ``qml.vn_entropy`` function."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "state_vector,expected",
        [([1.0, 0.0, 0.0, 1.0] / qml.math.sqrt(2), qml.math.log(2)), ([1.0, 0.0, 0.0, 0.0], 0)],
    )
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_vn_entropy(self, interface, state_vector, expected):
        """Tests the output of qml.vn_entropy"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.QubitStateVector(state_vector, wires=[0, 1])
            return qml.vn_entropy(wires=0)

        res = circuit()
        new_res = qml.vn_entropy(wires=0).process_state(
            state=circuit.device.state, wire_order=circuit.device.wires
        )
        assert qml.math.allclose(res, expected)
        assert qml.math.allclose(new_res, expected)
        assert INTERFACE_MAP.get(qml.math.get_interface(new_res)) == interface
        assert res.dtype == new_res.dtype

    def test_queue(self):
        """Test that the right measurement class is queued."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.vn_entropy(wires=0, log_base=2)

        circuit()

        assert isinstance(circuit.tape[0], _VnEntropy)

    def test_copy(self):
        """Test that the ``__copy__`` method also copies the ``log_base`` information."""
        meas = qml.vn_entropy(wires=0, log_base=2)
        meas_copy = copy.copy(meas)
        assert meas_copy.log_base == 2
        assert meas_copy.wires == Wires(0)

    def test_properties(self):
        """Test that the properties are correct."""
        meas = qml.vn_entropy(wires=0)
        assert meas.return_type == VnEntropy
