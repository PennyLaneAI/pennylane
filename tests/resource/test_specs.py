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
# pylint: disable=invalid-sequence-index
from collections import defaultdict
from contextlib import nullcontext

import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn

devices_list = [
    (qml.device("default.qubit"), 1),
    (qml.device("default.qubit", wires=2), 2),
]


class TestSpecsTransform:
    """Tests for the transform specs using the QNode"""

    def sample_circuit(self):
        @qml.transforms.merge_rotations
        @qml.transforms.undo_swaps
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift", shifts=pnp.pi / 4)
        def circuit(x):
            qml.RandomLayers(qml.numpy.array([[1.0, 2.0]]), wires=(0, 1))
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.SWAP((0, 1))
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.sum(qml.X(0), qml.Y(1)))

        return circuit

    @pytest.mark.parametrize(
        "level,expected_gates,exptected_train_params",
        [(0, 6, 1), (1, 4, 3), (2, 3, 3), (3, 1, 1), (None, 2, 2)],
    )
    def test_int_specs_level(self, level, expected_gates, exptected_train_params):
        circ = self.sample_circuit()
        specs = qml.specs(circ, level=level)(0.1)

        assert specs["level"] == level
        assert specs["resources"].num_gates == expected_gates

        assert specs["num_trainable_params"] == exptected_train_params

    @pytest.mark.parametrize(
        "level1,level2",
        [
            ("top", 0),
            (0, slice(0, 0)),
            ("user", 3),
            ("user", slice(0, 3)),
            (None, slice(0, None)),
            (-1, slice(0, -1)),
            ("device", slice(0, None)),
        ],
    )
    def test_equivalent_levels(self, level1, level2):
        circ = self.sample_circuit()

        specs1 = qml.specs(circ, level=level1)(0.1)
        specs2 = qml.specs(circ, level=level2)(0.1)

        del specs1["level"]
        del specs2["level"]

        assert specs1 == specs2

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 13), ("parameter-shift", 14), ("adjoint", 13)]
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
        assert info["level"] == "gradient"

        if diff_method == "parameter-shift":
            assert info["num_gradient_executions"] == 0
            assert info["gradient_fn"] == "pennylane.gradients.parameter_shift.param_shift"

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 13), ("parameter-shift", 14), ("adjoint", 13)]
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

        x = pnp.array([0.05, 0.1, 0.2, 0.3, 0.5], requires_grad=True)
        y = pnp.array(0.1, requires_grad=False)

        info = qml.specs(circuit)(x, y, add_RY=False)

        assert len(info) == len_info

        gate_sizes = defaultdict(int, {1: 2, 3: 1, 2: 1})
        gate_types = defaultdict(int, {"RX": 1, "Toffoli": 1, "CRY": 1, "Rot": 1})
        expected_resources = qml.resource.Resources(
            num_wires=3, num_gates=4, gate_types=gate_types, gate_sizes=gate_sizes, depth=3
        )
        assert info["resources"] == expected_resources

        assert info["num_observables"] == 2
        assert info["num_diagonalizing_gates"] == 1
        assert info["num_device_wires"] == 4
        assert info["diff_method"] == diff_method
        assert info["num_trainable_params"] == 4
        assert info["device_name"] == dev.name
        assert info["level"] == "gradient"

        if diff_method == "parameter-shift":
            assert info["num_gradient_executions"] == 6

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 13), ("parameter-shift", 14), ("adjoint", 13)]
    )
    def test_specs_state(self, diff_method, len_info):
        """Test specs works when state returned"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit():
            return qml.state()

        info = qml.specs(circuit)()
        assert len(info) == len_info

        assert info["resources"] == qml.resource.Resources(gate_types=defaultdict(int))

        assert info["num_observables"] == 1
        assert info["num_diagonalizing_gates"] == 0
        assert info["level"] == "gradient"

    def test_level_with_diagonalizing_gates(self):
        """Test that when diagonalizing gates includes gates that are decomposed in
        device preprocess, for level=device, any unsupported diagonalizing gates are
        decomposed like the tape.operations."""

        class TestDevice(qml.devices.DefaultQubit):

            def stopping_condition(self, op):
                if isinstance(op, qml.QubitUnitary):
                    return False
                return True

            def preprocess(self, execution_config=qml.devices.DefaultExecutionConfig):
                program, config = super().preprocess(execution_config)
                program.add_transform(
                    qml.devices.preprocess.decompose, stopping_condition=self.stopping_condition
                )
                return program, config

        dev = TestDevice(wires=2)
        matrix = qml.matrix(qml.RX(1.2, 0))

        @qml.qnode(dev)
        def circ():
            qml.QubitUnitary(matrix, wires=0)
            return qml.expval(qml.X(0) + qml.Y(0))

        specs = qml.specs(circ)()
        assert specs["resources"].num_gates == 1
        assert specs["num_diagonalizing_gates"] == 1

        specs = qml.specs(circ, level="device")()
        assert specs["resources"].num_gates == 3
        assert specs["num_diagonalizing_gates"] == 3

    def test_splitting_transforms(self):
        coeffs = [0.2, -0.543, 0.1]
        obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Y(2), qml.Y(0) @ qml.X(2)]
        H = qml.Hamiltonian(coeffs, obs)

        @qml.transforms.split_non_commuting
        @qml.transforms.merge_rotations
        @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift", shifts=pnp.pi / 4)
        def circuit(x):
            qml.RandomLayers(qml.numpy.array([[1.0, 2.0]]), wires=(0, 1))
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.SWAP((0, 1))
            qml.X(0)
            qml.X(0)
            return qml.expval(H)

        specs_instance = qml.specs(circuit, level=1)(pnp.array([1.23, -1]))

        assert isinstance(specs_instance, dict)

        specs_list = qml.specs(circuit, level=2)(pnp.array([1.23, -1]))

        assert len(specs_list) == len(H)

        assert specs_list[0]["num_diagonalizing_gates"] == 1
        assert specs_list[1]["num_diagonalizing_gates"] == 3
        assert specs_list[2]["num_diagonalizing_gates"] == 4

        assert specs_list[0]["num_device_wires"] == specs_list[0]["num_tape_wires"] == 2
        assert specs_list[1]["num_device_wires"] == specs_list[1]["num_tape_wires"] == 3
        assert specs_list[2]["num_device_wires"] == specs_list[1]["num_tape_wires"] == 3

    def make_qnode_and_params(self, seed):
        """Generates a qnode and params for use in other tests"""
        n_layers = 2
        n_wires = 5

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(params):
            qml.BasicEntanglerLayers(params, wires=range(n_wires))
            return qml.expval(qml.PauliZ(0))

        params_shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_wires)
        rng = pnp.random.default_rng(seed=seed)
        params = rng.standard_normal(params_shape)  # pylint:disable=no-member

        return circuit, params

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

        @qml.transform
        def my_transform(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return tape, None

        @qml.qnode(dev, diff_method=my_transform)
        def circuit():
            return qml.probs(wires=0)

        info = qml.specs(circuit)()
        assert info["diff_method"] == "test_specs.my_transform"
        assert info["gradient_fn"] == "test_specs.my_transform"

    @pytest.mark.parametrize(
        "device,num_wires",
        devices_list,
    )
    def test_num_wires_source_of_truth(self, device, num_wires):
        """Tests that num_wires behaves differently on old and new devices."""

        @qml.qnode(device)
        def circuit():
            qml.PauliX(0)
            return qml.state()

        info = qml.specs(circuit)()
        assert info["num_device_wires"] == num_wires

    def test_no_error_contents_on_device_level(self):
        coeffs = [0.25, 0.75]
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.Hadamard(0)
            qml.TrotterProduct(H, time=2.4, order=2)

            return qml.state()

        top_specs = qml.specs(circuit, level="top")()
        dev_specs = qml.specs(circuit, level="device")()

        assert "SpectralNormError" in top_specs["errors"]
        assert pnp.allclose(top_specs["errors"]["SpectralNormError"].error, 13.824)

        # At the device level, approximations don't exist anymore and therefore
        # we should expect an empty errors dictionary.
        assert dev_specs["errors"] == {}
