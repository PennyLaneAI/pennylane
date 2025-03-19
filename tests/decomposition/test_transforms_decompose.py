# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the new experimental graph-based decomposition system integrated with `qml.transforms.decompose`."""

# pylint: disable=no-name-in-module

from functools import partial
from unittest.mock import patch

import pytest

import pennylane as qml
from pennylane.decomposition import DecompositionError, DecompositionGraph


class TestGraphToggle:

    @pytest.mark.unit
    def test_toggle_graph_decomposition(self):
        """Tests the toggling of the graph-based decomposition system."""

        assert not qml.decomposition.enabled_graph()

        qml.decomposition.enable_graph()
        assert qml.decomposition.enabled_graph()

        qml.decomposition.disable_graph()
        assert not qml.decomposition.enabled_graph()

        qml.decomposition.enable_graph()
        assert qml.decomposition.enabled_graph()

        qml.decomposition.disable_graph()
        assert not qml.decomposition.enabled_graph()

        qml.decomposition.enable_graph()
        assert qml.decomposition.enabled_graph()

        qml.decomposition.disable_graph()
        assert not qml.decomposition.enabled_graph()


class TestTransformDecompose:

    @pytest.mark.unit
    def test_valueerror_graph_disabled_fixed(self):
        """Tests that a ValueError is raised when graph is disabled and fixed_decomps is used."""

        qml.decomposition.disable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @partial(qml.transforms.decompose, fixed_decomps={qml.CNOT: my_cnot})
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        with pytest.raises(
            ValueError,
            match="The fixed_decomps and alt_decomps arguments can be used with the experimental graph-based decomposition.",
        ):
            circuit()

    @pytest.mark.unit
    def test_valueerror_graph_disabled_alt(self):
        """Tests that a ValueError is raised when the graph-based decomposition and alt_decomps is used."""

        qml.decomposition.disable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @partial(qml.transforms.decompose, alt_decomps={qml.CNOT: [my_cnot]})
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        with pytest.raises(
            ValueError,
            match="The fixed_decomps and alt_decomps arguments can be used with the experimental graph-based decomposition.",
        ):
            circuit()

    @pytest.mark.unit
    def test_valueerror_graph_disabled_fixed_alt(self):
        """Tests that a ValueError is raised when the graph-based decomposition and fixed and alt decomps are used."""

        qml.decomposition.disable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @partial(
            qml.transforms.decompose,
            fixed_decomps={qml.CNOT: my_cnot},
            alt_decomps={qml.CNOT: [my_cnot]},
        )
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        with pytest.raises(
            ValueError,
            match="The fixed_decomps and alt_decomps arguments can be used with the experimental graph-based decomposition.",
        ):
            circuit()

    def test_gate_names_gateset(self):
        """Tests that the string representation of the gate_set works with the new system."""

        qml.decomposition.enable_graph()

        @partial(
            qml.transforms.decompose,
            gate_set={"GlobalPhase", "RX", "RZ", "CNOT"},
        )
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        expected_resources = {"RZ": 2, "RX": 1, "GlobalPhase": 1, "CNOT": 1}
        assert qml.specs(circuit)()["resources"].gate_types == expected_resources

    def test_gate_types_gateset(self):
        """Tests that the PennyLane's Operators does not work with the new system."""

        qml.decomposition.enable_graph()

        @partial(
            qml.transforms.decompose,
            gate_set={qml.GlobalPhase, "RX", "RZ", "CNOT"},
        )
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(
            ValueError,
            match="The graph-based decomposition doesn't support Operator types",
        ):
            circuit()

    def test_fixed_decomps(self):
        """Test the fixed_decomps argument with the new system."""

        qml.decomposition.enable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @partial(qml.transforms.decompose, fixed_decomps={qml.CNOT: my_cnot})
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        expected_resources = {"Hadamard": 2, "CZ": 1}
        assert qml.specs(circuit)()["resources"].gate_types == expected_resources

    def test_alt_decomps_single(self):

        qml.decomposition.enable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @partial(qml.transforms.decompose, alt_decomps={qml.CNOT: [my_cnot]})
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        expected_resources = {"Hadamard": 2, "CZ": 1}
        assert qml.specs(circuit)()["resources"].gate_types == expected_resources

    def test_alt_decomps_multiple(self):

        qml.decomposition.enable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot1(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @qml.register_resources({qml.RY: 2, qml.CZ: 1, qml.Z: 2})
        def my_cnot2(wires, **__):
            qml.RY(np.pi / 2, wires[1])
            qml.Z(wires[1])
            qml.CZ(wires=wires)
            qml.RY(np.pi / 2, wires[1])
            qml.Z(wires[1])

        @partial(qml.transforms.decompose, alt_decomps={qml.CNOT: [my_cnot1, my_cnot2]})
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        expected_resources = {"Hadamard": 2, "CZ": 1}
        assert qml.specs(circuit)()["resources"].gate_types == expected_resources

    def test_custom_op(self):

        qml.decomposition.enable_graph()

        class CustomOp(qml.operation.Operation):

            resource_params = set()

            @property
            def resource_params(self):
                return {}

        @qml.register_resources({qml.RZ: 2, qml.CNOT: 1})
        def custom_decomp(theta, wires, **__):
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RZ(theta, wires=wires[0])

        @qml.register_resources({qml.RX: 3, qml.CNOT: 2})
        def custom_decomp2(theta, wires, **__):
            qml.RX(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RX(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RX(theta, wires=wires[0])

        @partial(
            qml.transforms.decompose,
            gate_set={"GlobalPhase", "RX", "RZ", "CNOT"},
            alt_decomps={CustomOp: [custom_decomp, custom_decomp2]},
        )
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            CustomOp(0.5, wires=[0, 1])  # not supported on default.qubit
            return qml.expval(qml.PauliZ(0))

        expected_resources = {"RZ": 2, "CNOT": 1}
        assert qml.specs(circuit)()["resources"].gate_types == expected_resources

    def test_qft_template(self):

        qml.decomposition.enable_graph()

        @partial(
            qml.transforms.decompose,
            gate_set={"GlobalPhase", "RX", "RZ", "CNOT"},
        )
        @qml.qnode(qml.device("default.qubit"))
        def circuit(wires):
            qml.QFT(wires=wires)
            return qml.expval(qml.PauliZ(0))

        expected_resources = {"RZ": 57, "RX": 6, "GlobalPhase": 21, "CNOT": 39}
        assert qml.specs(circuit)([*range(6)])["resources"].gate_types == expected_resources
