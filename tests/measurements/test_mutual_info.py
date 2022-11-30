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
"""Unit tests for the mutual_info module"""
import copy

import numpy as np
import pytest

import pennylane as qml
from pennylane.interfaces import INTERFACE_MAP
from pennylane.measurements import MutualInfo
from pennylane.measurements.mutual_info import _MutualInfo
from pennylane.wires import Wires


class TestMutualInfo:
    """Tests for the mutual_info function"""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1.0, 0.0, 0.0, 0.0], 0),
            ([qml.math.sqrt(2) / 2, 0.0, qml.math.sqrt(2) / 2, 0.0], 0),
            ([qml.math.sqrt(2) / 2, 0.0, 0.0, qml.math.sqrt(2) / 2], 2 * qml.math.log(2)),
            (qml.math.ones(4) * 0.5, 0.0),
        ],
    )
    def test_mutual_info_output(self, interface, state, expected):
        """Test the output of qml.mutual_info"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.QubitStateVector(state, wires=[0, 1])
            return qml.mutual_info(wires0=[0, 2], wires1=[1, 3])

        res = circuit()
        new_res = qml.mutual_info(wires0=[0, 2], wires1=[1, 3]).process_state(
            state=circuit.device.state, wire_order=circuit.device.wires
        )
        assert np.allclose(res, expected, atol=1e-6)
        assert np.allclose(new_res, expected, atol=1e-6)
        assert INTERFACE_MAP.get(qml.math.get_interface(new_res)) == interface
        assert res.dtype == new_res.dtype

    def test_queue(self):
        """Test that the right measurement class is queued."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.mutual_info(wires0=[0], wires1=[1])

        circuit()

        assert isinstance(circuit.tape[0], _MutualInfo)

    def test_properties(self):
        """Test that the properties are correct."""
        meas = qml.mutual_info(wires0=[0], wires1=[1])
        assert meas.numeric_type == float
        assert meas.return_type == MutualInfo

    def test_copy(self):
        """Test that the ``__copy__`` method also copies the ``log_base`` information."""
        meas = qml.mutual_info(wires0=[0], wires1=[1], log_base=2)
        meas_copy = copy.copy(meas)
        assert meas_copy.log_base == 2
        assert meas_copy.wires == Wires([0, 1])
