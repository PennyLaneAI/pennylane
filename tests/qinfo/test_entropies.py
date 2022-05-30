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
Tests for the entropies in the pennylane.qinfo
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import quantum_info as qinfo


class TestMutualInformation:
    """Tests for the mutual information functions"""

    @pytest.mark.parametrize("interface", ["autograd", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1, 0, 0, 0], 0),
            ([np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0], 0),
            ([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2], 2 * np.log(2)),
            (np.ones(4) * 0.5, 0),
        ],
    )
    def test_state(self, interface, state, expected):
        """Test that mutual information works for states"""
        state = qml.math.asarray(state, like=interface)
        actual = qinfo.to_mutual_info(state, wires0=[0], wires1=[1])
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("interface", ["autograd", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 0),
            ([[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]], 0),
            ([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], 2 * np.log(2)),
            (np.ones((4, 4)) * 0.25, 0),
        ],
    )
    def test_density_matrix(self, interface, state, expected):
        """Test that mutual information works for density matrices"""
        state = qml.math.asarray(state, like=interface)
        actual = qinfo.to_mutual_info(state, wires0=[0], wires1=[1])
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("interface", ["autograd", "tf", "torch"])
    @pytest.mark.parametrize(
        "params", [np.array([0, 0]), np.array([0.3, 0.4]), np.array([0.6, 0.8])]
    )
    def test_qnode_state(self, device, interface, params):
        """Test that mutual information works for QNodes that return the state"""
        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = qinfo.to_mutual_info(circuit, wires0=[0], wires1=[1])(params)

        # compare QNode results with the results of computing directly from the state
        state = circuit(params)
        expected = qinfo.to_mutual_info(state, wires0=[0], wires1=[1])

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("interface", ["autograd", "tf", "torch"])
    @pytest.mark.parametrize(
        "params", [np.array([0, 0]), np.array([0.3, 0.4]), np.array([0.6, 0.8])]
    )
    def test_qnode_mutual_info(self, device, interface, params):
        """Test that mutual information works for QNodes that directly return it"""
        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit_mutual_info(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        @qml.qnode(dev, interface=interface)
        def circuit_state(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = circuit_mutual_info(params)

        # compare QNode results with the results of computing directly from the state
        state = circuit_state(params)
        expected = qinfo.to_mutual_info(state, wires0=[0], wires1=[1])

        assert np.allclose(actual, expected)

    def test_grad_state(self):
        """Test that the gradient of mutual information works for states"""

    def test_grad_density_matrix(self):
        """Test that the gradient of mutual information works for density matrices"""

    def test_grad_qnode(self):
        """Test that the gradient of mutual information works for QNodes"""

    @pytest.mark.parametrize(
        "state, wires0, wires1",
        [
            (np.array([1, 0, 0, 0]), [0], [0]),
            (np.array([1, 0, 0, 0]), [0], [0, 1]),
            (np.array([1, 0, 0, 0, 0, 0, 0, 0]), [0, 1], [1]),
            (np.array([1, 0, 0, 0, 0, 0, 0, 0]), [0, 1], [1, 2]),
        ],
    )
    def test_subsystem_overlap(self, state, wires0, wires1):
        """Test that an error is raised when the subsystems overlap"""
        with pytest.raises(
            ValueError, match="Subsystems for computing mutual information must not overlap"
        ):
            qinfo.to_mutual_info(state, wires0=wires0, wires1=wires1)

    @pytest.mark.parametrize("state", [np.array(5), np.ones((3, 4)), np.ones((2, 2, 2))])
    def test_invalid_type(self, state):
        """Test that an error is raised when an unsupported type is passed"""
        with pytest.raises(
            ValueError, match="The state is not a QNode, a state vector or a density matrix."
        ):
            qinfo.to_mutual_info(state, wires0=[0], wires1=[1])
