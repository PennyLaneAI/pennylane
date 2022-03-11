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
import pytest
from collections import defaultdict
from contextlib import nullcontext

import pennylane as qml
from pennylane import numpy as np


class TestSpecsTransform:
    """Tests for the transform specs using the old QNode. This can be
    removed when `qml.beta.QNode is made default."""

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 10), ("parameter-shift", 12), ("adjoint", 11)]
    )
    def test_empty(self, diff_method, len_info):

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode_old.qnode(dev, diff_method=diff_method)
        def circ():
            return qml.expval(qml.PauliZ(0))

        info_func = qml.specs(circ)
        info = info_func()

        circ()
        assert info == circ.specs
        assert len(info) == len_info

        assert info["gate_sizes"] == defaultdict(int)
        assert info["gate_types"] == defaultdict(int)
        assert info["num_observables"] == 1
        assert info["num_operations"] == 0
        assert info["num_diagonalizing_gates"] == 0
        assert info["num_used_wires"] == 1
        assert info["depth"] == 0
        assert info["num_device_wires"] == 1
        assert info["diff_method"] == diff_method

        if diff_method == "parameter-shift":
            assert info["num_parameter_shift_executions"] == 1

        if diff_method != "backprop":
            assert info["device_name"] == "default.qubit"
            assert info["num_trainable_params"] == 0
        else:
            assert info["device_name"] == "default.qubit.autograd"

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 10), ("parameter-shift", 12), ("adjoint", 11)]
    )
    def test_specs(self, diff_method, len_info):
        """Test the specs transforms works in standard situations"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode_old.qnode(dev, diff_method=diff_method)
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

        assert info == circuit.specs

        assert len(info) == len_info

        assert info["gate_sizes"] == defaultdict(int, {1: 2, 3: 1, 2: 1})
        assert info["gate_types"] == defaultdict(int, {"RX": 1, "Toffoli": 1, "CRY": 1, "Rot": 1})
        assert info["num_operations"] == 4
        assert info["num_observables"] == 2
        assert info["num_diagonalizing_gates"] == 1
        assert info["num_used_wires"] == 3
        assert info["depth"] == 3
        assert info["num_device_wires"] == 4
        assert info["diff_method"] == diff_method

        if diff_method == "parameter-shift":
            assert info["num_parameter_shift_executions"] == 7

        if diff_method != "backprop":
            assert info["device_name"] == "default.qubit"
            assert info["num_trainable_params"] == 4
        else:
            assert info["device_name"] == "default.qubit.autograd"

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 10), ("parameter-shift", 11), ("adjoint", 11)]
    )
    def test_specs_state(self, diff_method, len_info):
        """Test specs works when state returned"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode_old.qnode(dev, diff_method=diff_method)
        def circuit():
            return qml.state()

        info_func = qml.specs(circuit)
        info = info_func()

        circuit()
        assert info == circuit.specs
        assert len(info) == len_info

        assert info["num_observables"] == 1
        assert info["num_diagonalizing_gates"] == 0

    def test_max_expansion(self):
        """Test that a user can calculation specifications for a different max
        expansion parameter."""

        n_layers = 2
        n_wires = 5

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode_old.qnode(dev)
        def circuit(params):
            qml.templates.BasicEntanglerLayers(params, wires=range(n_wires))
            return qml.expval(qml.PauliZ(0))

        params_shape = qml.templates.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_wires)
        rng = np.random.default_rng(seed=10)
        params = rng.standard_normal(params_shape)

        assert circuit.max_expansion == 10
        info = qml.specs(circuit, max_expansion=0)(params)
        assert circuit.max_expansion == 10

        assert len(info) == 10

        assert info["gate_sizes"] == defaultdict(int, {5: 1})
        assert info["gate_types"] == defaultdict(int, {"BasicEntanglerLayers": 1})
        assert info["num_operations"] == 1
        assert info["num_observables"] == 1
        assert info["num_used_wires"] == 5
        assert info["depth"] == 1
        assert info["num_device_wires"] == 5
        assert info["device_name"] == "default.qubit.autograd"
        assert info["diff_method"] == "backprop"


class TestSpecsTransformBetaQNode:
    """Tests for the transform specs using the new QNode"""

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 15), ("parameter-shift", 16), ("adjoint", 15)]
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

        assert info["gate_sizes"] == defaultdict(int)
        assert info["gate_types"] == defaultdict(int)
        assert info["num_observables"] == 1
        assert info["num_operations"] == 0
        assert info["num_diagonalizing_gates"] == 0
        assert info["num_used_wires"] == 1
        assert info["depth"] == 0
        assert info["num_device_wires"] == 1
        assert info["diff_method"] == diff_method
        assert info["num_trainable_params"] == 0

        if diff_method == "parameter-shift":
            assert info["num_gradient_executions"] == 0
            assert info["gradient_fn"] == "pennylane.gradients.parameter_shift.param_shift"

        if diff_method != "backprop":
            assert info["device_name"] == "default.qubit"
        else:
            assert info["device_name"] == "default.qubit.autograd"

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 15), ("parameter-shift", 16), ("adjoint", 15)]
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

        assert info["gate_sizes"] == defaultdict(int, {1: 2, 3: 1, 2: 1})
        assert info["gate_types"] == defaultdict(int, {"RX": 1, "Toffoli": 1, "CRY": 1, "Rot": 1})
        assert info["num_operations"] == 4
        assert info["num_observables"] == 2
        assert info["num_diagonalizing_gates"] == 1
        assert info["num_used_wires"] == 3
        assert info["depth"] == 3
        assert info["num_device_wires"] == 4
        assert info["diff_method"] == diff_method
        assert info["num_trainable_params"] == 4

        if diff_method == "parameter-shift":
            assert info["num_gradient_executions"] == 6

        if diff_method != "backprop":
            assert info["device_name"] == "default.qubit"
        else:
            assert info["device_name"] == "default.qubit.autograd"

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 15), ("parameter-shift", 16), ("adjoint", 15)]
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

        assert info["num_observables"] == 1
        assert info["num_diagonalizing_gates"] == 0

    def test_max_expansion(self):
        """Test that a user can calculation specifications for a different max
        expansion parameter."""

        n_layers = 2
        n_wires = 5

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(params):
            qml.templates.BasicEntanglerLayers(params, wires=range(n_wires))
            return qml.expval(qml.PauliZ(0))

        params_shape = qml.templates.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_wires)
        rng = np.random.default_rng(seed=10)
        params = rng.standard_normal(params_shape)

        assert circuit.max_expansion == 10
        info = qml.specs(circuit, max_expansion=0)(params)
        assert circuit.max_expansion == 10

        assert len(info) == 15

        assert info["gate_sizes"] == defaultdict(int, {5: 1})
        assert info["gate_types"] == defaultdict(int, {"BasicEntanglerLayers": 1})
        assert info["num_operations"] == 1
        assert info["num_observables"] == 1
        assert info["num_used_wires"] == 5
        assert info["depth"] == 1
        assert info["num_device_wires"] == 5
        assert info["device_name"] == "default.qubit.autograd"
        assert info["diff_method"] == "best"
        assert info["gradient_fn"] == "backprop"

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

        @qml.gradients.gradient_transform
        def my_transform(tape):
            return tape, None

        @qml.qnode(dev, diff_method=my_transform)
        def circuit():
            return qml.probs(wires=0)

        info = qml.specs(circuit)()
        assert info["diff_method"] == "test_specs.my_transform"
        assert info["gradient_fn"] == "test_specs.my_transform"
