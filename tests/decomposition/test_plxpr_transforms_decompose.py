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

"""
Unit tests for the new experimental graph-based decomposition system integrated with `qml.transforms.decompose` with the program capture.
"""

# pylint: disable=no-name-in-module, too-few-public-methods

from functools import partial

import numpy as np
import pytest

import pennylane as qml

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]

jax = pytest.importorskip("jax")

from pennylane.tape.plxpr_conversion import (  # pylint: disable=wrong-import-position
    CollectOpsandMeas,
)


class TestPLxPRTransformDecompose:

    def test_plxpr_gate_names_gateset(self):
        """Test that the string representation of the gate_set works with the new decomposition system."""

        qml.capture.enable()
        qml.decomposition.enable_graph()

        @qml.capture.expand_plxpr_transforms
        @partial(
            qml.transforms.decompose,
            gate_set={"GlobalPhase", "RX", "RZ", "CNOT"},
        )
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        obj = CollectOpsandMeas()
        obj(circuit)()

        qml.assert_equal(obj.state["ops"][0], qml.RZ(np.pi / 2, 0))
        qml.assert_equal(obj.state["ops"][1], qml.RX(np.pi / 2, 0))
        qml.assert_equal(obj.state["ops"][2], qml.RZ(np.pi / 2, 0))
        qml.assert_equal(obj.state["ops"][3], qml.GlobalPhase(-np.pi / 2, 0))
        qml.assert_equal(obj.state["ops"][4], qml.CNOT([0, 1]))
        assert len(obj.state["ops"]) == 5

        qml.decomposition.disable_graph()
        qml.capture.disable()

    def test_plxpr_gate_types_gateset(self):
        """Test that the PennyLane's Operators does not work with the new decomposition system."""

        qml.capture.enable()
        qml.decomposition.enable_graph()

        @qml.capture.expand_plxpr_transforms
        @partial(
            qml.transforms.decompose,
            gate_set={qml.GlobalPhase, "RX", "RZ", "CNOT"},
        )
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(
            TypeError,
            match="The graph-based decomposition doesn't support Operator types",
        ):
            circuit()

        qml.decomposition.disable_graph()
        qml.capture.disable()

    def test_plxpr_fixed_decomps(self):
        """Test the fixed_decomps argument with the new decomposition system."""

        qml.capture.enable()
        qml.decomposition.enable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @qml.capture.expand_plxpr_transforms
        @partial(qml.transforms.decompose, fixed_decomps={qml.CNOT: my_cnot})
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        obj = CollectOpsandMeas()
        obj(circuit)()
        qml.assert_equal(obj.state["ops"][0], qml.Hadamard(1))
        qml.assert_equal(obj.state["ops"][1], qml.CZ([0, 1]))
        qml.assert_equal(obj.state["ops"][2], qml.Hadamard(1))
        assert len(obj.state["ops"]) == 3

        qml.decomposition.disable_graph()
        qml.capture.disable()

    def test_plxpr_alt_decomps_single(self):
        """Test the alt_decomps argument with a single decomposition rules."""

        qml.capture.enable()
        qml.decomposition.enable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @qml.capture.expand_plxpr_transforms
        @partial(qml.transforms.decompose, fixed_decomps={qml.CNOT: my_cnot})
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        obj = CollectOpsandMeas()
        obj(circuit)()
        qml.assert_equal(obj.state["ops"][0], qml.Hadamard(1))
        qml.assert_equal(obj.state["ops"][1], qml.CZ([0, 1]))
        qml.assert_equal(obj.state["ops"][2], qml.Hadamard(1))
        assert len(obj.state["ops"]) == 3

        qml.decomposition.disable_graph()
        qml.capture.disable()

    def test_plxpr_alt_decomps_multiple(self):
        """Test the alt_decomps argument with multiple decomposition rules."""

        qml.capture.enable()
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

        @qml.capture.expand_plxpr_transforms
        @partial(qml.transforms.decompose, alt_decomps={qml.CNOT: [my_cnot1, my_cnot2]})
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        obj = CollectOpsandMeas()
        obj(circuit)()
        qml.assert_equal(obj.state["ops"][0], qml.Hadamard(1))
        qml.assert_equal(obj.state["ops"][1], qml.CZ([0, 1]))
        qml.assert_equal(obj.state["ops"][2], qml.Hadamard(1))
        assert len(obj.state["ops"]) == 3

        qml.decomposition.disable_graph()
        qml.capture.disable()

    def test_plxpr_alt_decomps_custom_op(self):
        """Test the custom operator with the new decomposition system."""

        qml.capture.enable()
        qml.decomposition.enable_graph()

        class CustomOp(qml.operation.Operation):

            resource_params = set()

            @property
            def resource_params(self):  # pylint: disable=function-redefined
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

        @qml.capture.expand_plxpr_transforms
        @partial(
            qml.transforms.decompose,
            gate_set={"GlobalPhase", "RX", "RZ", "CNOT"},
            alt_decomps={CustomOp: [custom_decomp, custom_decomp2]},
        )
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            CustomOp(0.5, wires=[0, 1])  # not supported on default.qubit
            return qml.expval(qml.PauliZ(0))

        obj = CollectOpsandMeas()
        obj(circuit)()
        qml.assert_equal(obj.state["ops"][0], qml.RZ(0.5, 0))
        qml.assert_equal(obj.state["ops"][1], qml.CNOT([0, 1]))
        qml.assert_equal(obj.state["ops"][2], qml.RZ(0.5, 0))
        assert len(obj.state["ops"]) == 3

        qml.decomposition.disable_graph()
        qml.capture.disable()

    def test_plxpr_alt_decomps_custom_op_params(self):
        """Test the custom operator with the new decomposition system."""

        qml.capture.enable()
        qml.decomposition.enable_graph()

        class CustomOp(qml.operation.Operation):

            resource_params = set()

            @property
            def resource_params(self):  # pylint: disable=function-redefined
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

        @qml.capture.expand_plxpr_transforms
        @partial(
            qml.transforms.decompose,
            gate_set={"GlobalPhase", "RX", "RZ", "CNOT"},
            alt_decomps={CustomOp: [custom_decomp, custom_decomp2]},
        )
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(theta, phi):
            CustomOp(theta, wires=[0, 1])  # not supported on default.qubit
            qml.RX(phi, wires=0)
            return qml.expval(qml.PauliZ(0))

        obj = CollectOpsandMeas()
        obj(circuit)(0.5, 0.7)
        qml.assert_equal(obj.state["ops"][0], qml.RZ(0.5, 0))
        qml.assert_equal(obj.state["ops"][1], qml.CNOT([0, 1]))
        qml.assert_equal(obj.state["ops"][2], qml.RZ(0.5, 0))
        qml.assert_equal(obj.state["ops"][3], qml.RX(0.7, 0))
        assert len(obj.state["ops"]) == 4

        qml.decomposition.disable_graph()
        qml.capture.disable()

    def test_plxpr_error_disable_graphd_fixed(self):
        """Test that a TypeError is raised when graph is disabled and fixed_decomps is used."""

        qml.decomposition.disable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @qml.capture.expand_plxpr_transforms
        @partial(qml.transforms.decompose, fixed_decomps={qml.CNOT: my_cnot})
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        with pytest.raises(
            TypeError,
            match="The fixed_decomps and alt_decomps arguments must be used with the experimental graph-based decomposition.",
        ):
            circuit()

    def test_plxpr_error_disable_graphd_alt(self):
        """Test that a TypeError is raised when the graph-based decomposition and alt_decomps is used."""

        qml.capture.enable()
        qml.decomposition.disable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @qml.capture.expand_plxpr_transforms
        @partial(qml.transforms.decompose, alt_decomps={qml.CNOT: [my_cnot]})
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        with pytest.raises(
            TypeError,
            match="The fixed_decomps and alt_decomps arguments must be used with the experimental graph-based decomposition.",
        ):
            circuit()

        qml.capture.disable()

    def test_plxpr_error_disable_graphd_decomps(self):
        """Test that a TypeError is raised when the graph-based decomposition with fixed and alt decomps is used."""

        qml.capture.enable()
        qml.decomposition.disable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **__):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @qml.capture.expand_plxpr_transforms
        @partial(
            qml.transforms.decompose,
            fixed_decomps={qml.CNOT: my_cnot},
            alt_decomps={qml.CNOT: [my_cnot]},
        )
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        with pytest.raises(
            TypeError,
            match="The fixed_decomps and alt_decomps arguments must be used with the experimental graph-based decomposition.",
        ):
            circuit()

        qml.capture.disable()
