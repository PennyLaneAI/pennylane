# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the specs transform"""
from typing import Sequence, Callable
from collections import defaultdict
from contextlib import nullcontext
import pytest

import pennylane as qml
from pennylane import numpy as np


class TestSpecsTransform:
    """Tests for the transform specs using the QNode"""

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 11), ("parameter-shift", 12), ("adjoint", 11)]
    )
    def test_empty(self, diff_method, len_info):
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circ():
            return qml.expval(qml.PauliZ(0))

        with (
            pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters")
            if diff_method == "parameter-shift"
            else nullcontext()
        ):
            info_func = qml.specs(circ)
            info = info_func()
        assert len(info) == len_info

        expected_resources = qml.resource.Resources(num_wires=1, gate_types=defaultdict(int))
        assert info["resources"] == expected_resources
        assert info["num_observables"] == 1
        assert info["num_diagonalizing_gates"] == 0
        assert info["num_device_wires"] == 1
        assert info["diff_method"] == diff_method
        assert info["num_trainable_params"] == 0
        assert info["device_name"] == dev.name

        if diff_method == "parameter-shift":
            assert info["num_gradient_executions"] == 0
            assert info["gradient_fn"] == "pennylane.gradients.parameter_shift.param_shift"

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 11), ("parameter-shift", 12), ("adjoint", 11)]
    )
    def test_specs(self, diff_method, len_info):
        """Test the specs transforms works in standard situations"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y, add_RY=True):
            qml.RX(x[0], wires=0)
            qml.Toffoli(wires=(0, 1, 2))
            qml.CRY(x[1], wires=(0, 1))
            qml.Rot(x[2], x[3], y, wires=2)
            if add_RY:
                qml.RY(x[4], wires=1)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = np.array([0.05, 0.1, 0.2, 0.3, 0.5], requires_grad=True)
        y = np.array(0.1, requires_grad=False)

        info_func = qml.specs(circuit)

        info = info_func(x, y, add_RY=False)

        circuit(x, y, add_RY=False)

        assert len(info) == len_info

        gate_sizes = defaultdict(int, {1: 2, 3: 1, 2: 1})
        gate_types = defaultdict(int, {"RX": 1, "Toffoli": 1, "CRY": 1, "Rot": 1})
        expected_resources = qml.resource.Resources(
            num_wires=3, num_gates=4, gate_types=gate_types, gate_sizes=gate_sizes, depth=3
        )
        assert info["resources"] == expected_resources

        assert info["num_observables"] == 2
        assert info["num_diagonalizing_gates"] == 1
        assert info["num_device_wires"] == 3
        assert info["diff_method"] == diff_method
        assert info["num_trainable_params"] == 4
        assert info["device_name"] == dev.name

        if diff_method == "parameter-shift":
            assert info["num_gradient_executions"] == 6

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 11), ("parameter-shift", 12), ("adjoint", 11)]
    )
    def test_specs_state(self, diff_method, len_info):
        """Test specs works when state returned"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit():
            return qml.state()

        info_func = qml.specs(circuit)
        info = info_func()
        assert len(info) == len_info

        assert info["resources"] == qml.resource.Resources(gate_types=defaultdict(int))

        assert info["num_observables"] == 1
        assert info["num_diagonalizing_gates"] == 0

    def make_qnode_and_params(self, initial_expansion_strategy):
        """Generates a qnode and params for use in other tests"""
        n_layers = 2
        n_wires = 5

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, expansion_strategy=initial_expansion_strategy)
        def circuit(params):
            qml.BasicEntanglerLayers(params, wires=range(n_wires))
            return qml.expval(qml.PauliZ(0))

        params_shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_wires)
        rng = np.random.default_rng(seed=10)
        params = rng.standard_normal(params_shape)  # pylint:disable=no-member

        return circuit, params

    @pytest.mark.xfail(reason="DefaultQubit2 does not support custom expansion depths")
    def test_max_expansion(self):
        """Test that a user can calculation specifications for a different max
        expansion parameter."""

        circuit, params = self.make_qnode_and_params("device")

        assert circuit.max_expansion == 10
        info = qml.specs(circuit, max_expansion=0)(params)
        assert circuit.max_expansion == 10

        assert len(info) == 11

        gate_sizes = defaultdict(int, {5: 1})
        gate_types = defaultdict(int, {"BasicEntanglerLayers": 1})
        expected_resources = qml.resource.Resources(
            num_wires=5, num_gates=1, gate_types=gate_types, gate_sizes=gate_sizes, depth=1
        )
        assert info["resources"] == expected_resources
        assert info["num_observables"] == 1
        assert info["num_device_wires"] == 5
        assert info["device_name"] == "default.qubit"
        assert info["diff_method"] == "best"
        assert info["gradient_fn"] == "backprop"

    def test_expansion_strategy(self):
        """Test that a user can calculate specs for different expansion strategies."""
        circuit, params = self.make_qnode_and_params("gradient")

        assert circuit.expansion_strategy == "gradient"
        info = qml.specs(circuit, expansion_strategy="device")(params)
        assert circuit.expansion_strategy == "gradient"

        assert len(info) == 11

    def test_gradient_transform(self):
        """Test that a gradient transform is properly labelled"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=qml.gradients.param_shift)
        def circuit():
            return qml.probs(wires=0)

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            info = qml.specs(circuit)()
        assert info["diff_method"] == "pennylane.gradients.parameter_shift.param_shift"
        assert info["gradient_fn"] == "pennylane.gradients.parameter_shift.param_shift"

    def test_custom_gradient_transform(self):
        """Test that a custom gradient transform is properly labelled"""
        dev = qml.device("default.qubit", wires=2)

        @qml.transforms.core.transform
        def my_transform(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
            return tape, None

        @qml.qnode(dev, diff_method=my_transform)
        def circuit():
            return qml.probs(wires=0)

        info = qml.specs(circuit)()
        assert info["diff_method"] == "test_specs.my_transform"
        assert info["gradient_fn"] == "test_specs.my_transform"

    @pytest.mark.parametrize(
        "device,num_wires",
        [
            (qml.device("default.qubit"), 1),
            (qml.device("default.qubit", wires=2), 1),
            (qml.device("default.qubit.legacy", wires=2), 2),
        ],
    )
    def test_num_wires_source_of_truth(self, device, num_wires):
        """Tests that num_wires behaves differently on old and new devices."""

        @qml.qnode(device)
        def circuit():
            qml.PauliX(0)
            return qml.state()

        info = qml.specs(circuit)()
        assert info["num_device_wires"] == num_wires
