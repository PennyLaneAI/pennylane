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
"""Unit tests for the ``CommuteControlledInterpreter`` class"""

# pylint:disable=wrong-import-position, unused-argument
import numpy as np
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]

from pennylane.tape.plxpr_conversion import CollectOpsandMeas

from pennylane.transforms.optimization.commute_controlled import (
    CommuteControlledInterpreter,
    commute_controlled_plxpr_to_plxpr,
    commute_controlled,
)


class TestCommuteControlledInterpreter:
    """Unit tests for the CommuteControlledInterpreter for canceling adjacent inverse
    operations in plxpr."""

    def test_gate_with_no_basis(self):
        """Test that gates with no basis specified are ignored."""

        def circuit():
            qml.PauliX(wires=2)
            qml.ControlledQubitUnitary(jax.numpy.array([[0, 1], [1, 0]]), wires=[0, 2])
            qml.PauliX(wires=2)

        transformed_circuit = CommuteControlledInterpreter()(circuit)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = commute_controlled(circuit, direction="left")
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_gate_with_basis(self):
        """Test that gates with a basis specified are correctly transformed."""

        def circuit():
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[2, 0])
            qml.PauliY(wires=1)

        transformed_circuit = CommuteControlledInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = commute_controlled(circuit, direction="left")
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_x_gates(self):
        """Test that X-basis gates before controlled-X-type gates on targets get pushed ahead."""

        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[0, 2])
            qml.RX(0.2, wires=2)
            qml.Toffoli(wires=[0, 1, 2])
            qml.SX(wires=1)
            qml.PauliX(wires=1)
            qml.CRX(0.1, wires=[0, 1])

        transformed_circuit = CommuteControlledInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.jaxpr.eqns) == 7

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = commute_controlled(circuit, direction="left")
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_dont_push_x_gates(self):
        """Test that X-basis gates before controlled-X-type gates on controls don't get pushed."""

        def circuit():
            qml.PauliX(wires=0)
            qml.CNOT(wires=[0, 2])
            qml.RX(0.2, wires=0)
            qml.Toffoli(wires=[2, 0, 1])

        transformed_circuit = CommuteControlledInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.jaxpr.eqns) == 4

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = commute_controlled(circuit, direction="left")
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_y_gates_left(self):
        """Test that Y-basis gates after controlled-Y-type gates on targets get pushed behind."""

        def circuit():
            qml.CRY(-0.5, wires=[0, 2])
            qml.PauliY(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.CY(wires=[0, 1])
            qml.RY(0.3, wires=1)

        transformed_circuit = CommuteControlledInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = commute_controlled(circuit, direction="left")
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_dont_push_y_gates(self):
        """Test that Y-basis gates next to controlled-Y-type gates on controls don't get pushed."""

        def circuit():
            qml.CRY(-0.2, wires=[0, 2])
            qml.PauliY(wires=0)
            qml.CNOT(wires=[1, 2])
            qml.CY(wires=[0, 1])
            qml.RY(0.3, wires=0)

        transformed_circuit = CommuteControlledInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = commute_controlled(circuit, direction="left")
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_z_gates(self):
        """Test that Z-basis gates before controlled-Z-type gates on controls *and* targets get pushed behind."""

        def circuit():
            qml.PauliZ(wires=2)
            qml.S(wires=0)
            qml.CZ(wires=[0, 2])
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(0.2, wires=2)
            qml.T(wires=0)
            qml.PauliZ(wires=0)
            qml.CRZ(0.5, wires=[0, 1])

        transformed_circuit = CommuteControlledInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.jaxpr.eqns) == 8

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = commute_controlled(circuit, direction="left")
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_mixed_with_matrix(self):
        """Test that arbitrary gates after controlled gates on controls *and*
        targets get properly pushed."""
        # pylint:disable=too-many-function-args

        def circuit():
            qml.PauliX(wires=1)
            qml.S(wires=0)
            qml.CZ(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            qml.PauliY(wires=1)
            qml.CRY(0.5, wires=[1, 0])
            qml.PhaseShift(0.2, wires=0)
            qml.PauliY(wires=1)
            qml.T(wires=0)
            qml.CRZ(-0.3, wires=[0, 1])
            qml.RZ(0.2, wires=0)
            qml.PauliZ(wires=0)
            qml.PauliX(wires=1)
            qml.CRY(0.2, wires=[1, 0])

        transformed_circuit = CommuteControlledInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.jaxpr.eqns) == 14

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = commute_controlled(circuit, direction="left")
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)
