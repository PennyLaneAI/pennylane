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

    def test_invalid_direction(self):
        """Test that any direction other than 'left' or 'right' raises an error."""

        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[0, 2])
            qml.RX(0.2, wires=2)

        with pytest.raises(ValueError, match="must be 'left' or 'right'"):
            CommuteControlledInterpreter(direction="sideways")(circuit)

    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_gate_with_no_basis(self, direction):
        """Test that gates with no basis specified are ignored."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qml.PauliX(wires=2)
            qml.ControlledQubitUnitary(jax.numpy.array([[0, 1], [1, 0]]), wires=[0, 2])
            qml.PauliX(wires=2)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliX(wires=2),
            qml.ControlledQubitUnitary(jax.numpy.array([[0, 1], [1, 0]]), wires=[0, 2]),
            qml.PauliX(wires=2),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            # jax inserts a dtype for the array which confuses qml.equal
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    @pytest.mark.parametrize("direction", [("left"), ("right")])
    def test_gate_blocked_different_basis(self, direction):
        """Test that gates do not get pushed through controlled gates whose target bases don't match."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 1])
            qml.PauliY(wires=1)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliZ(wires=1),
            qml.CNOT(wires=[2, 1]),
            qml.PauliY(wires=1),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_x_gates_right(self):
        """Test that X-basis gates before controlled-X-type gates on targets get pushed ahead."""

        @CommuteControlledInterpreter(direction="right")
        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[0, 2])
            qml.RX(0.2, wires=2)
            qml.Toffoli(wires=[0, 1, 2])
            qml.SX(wires=1)
            qml.PauliX(wires=1)
            qml.CRX(0.1, wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 7

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.CNOT(wires=[0, 2]),
            qml.Toffoli(wires=[0, 1, 2]),
            qml.PauliX(wires=2),
            qml.RX(0.2, wires=2),
            qml.CRX(0.1, wires=[0, 1]),
            qml.SX(wires=1),
            qml.PauliX(wires=1),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_x_gates_left(self):
        """Test that X-basis gates after controlled-X-type gates on targets get pushed back."""

        @CommuteControlledInterpreter(direction="left")
        def circuit():
            qml.PauliX(wires=2)
            qml.RX(0.2, wires=2)
            qml.CNOT(wires=[0, 2])
            qml.Toffoli(wires=[0, 1, 2])
            qml.SX(wires=1)
            qml.PauliX(wires=1)
            qml.CRX(0.1, wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 7

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliX(wires=2),
            qml.RX(0.2, wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Toffoli(wires=[0, 1, 2]),
            qml.SX(wires=1),
            qml.PauliX(wires=1),
            qml.CRX(0.1, wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    @pytest.mark.parametrize("direction", [("left"), ("right")])
    def test_dont_push_x_gates(self, direction):
        """Test that X-basis gates before controlled-X-type gates on controls don't get pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qml.PauliX(wires=0)
            qml.CNOT(wires=[0, 2])
            qml.RX(0.2, wires=0)
            qml.Toffoli(wires=[2, 0, 1])

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 4

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliX(wires=0),
            qml.CNOT(wires=[0, 2]),
            qml.RX(0.2, wires=0),
            qml.Toffoli(wires=[2, 0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_y_gates_right(self):
        """Test that Y-basis gates before controlled-Y-type gates on targets get pushed ahead."""

        @CommuteControlledInterpreter(direction="right")
        def circuit():
            qml.PauliY(wires=2)
            qml.CRY(-0.5, wires=[0, 2])
            qml.CNOT(wires=[1, 2])
            qml.RY(0.3, wires=1)
            qml.CY(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.CRY(-0.5, wires=[0, 2]),
            qml.PauliY(wires=2),
            qml.CNOT(wires=[1, 2]),
            qml.CY(wires=[0, 1]),
            qml.RY(0.3, wires=1),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_y_gates_left(self):
        """Test that Y-basis gates after controlled-Y-type gates on targets get pushed behind."""

        @CommuteControlledInterpreter(direction="left")
        def circuit():
            qml.CRY(-0.5, wires=[0, 2])
            qml.PauliY(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.CY(wires=[0, 1])
            qml.RY(0.3, wires=1)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliY(wires=2),
            qml.CRY(-0.5, wires=[0, 2]),
            qml.CNOT(wires=[1, 2]),
            qml.RY(0.3, wires=1),
            qml.CY(wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    @pytest.mark.parametrize("direction", [("left"), ("right")])
    def test_dont_push_y_gates(self, direction):
        """Test that Y-basis gates next to controlled-Y-type gates on controls don't get pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qml.CRY(-0.2, wires=[0, 2])
            qml.PauliY(wires=0)
            qml.CNOT(wires=[1, 2])
            qml.CY(wires=[0, 1])
            qml.RY(0.3, wires=0)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.CRY(-0.2, wires=[0, 2]),
            qml.PauliY(wires=0),
            qml.CNOT(wires=[1, 2]),
            qml.CY(wires=[0, 1]),
            qml.RY(0.3, wires=0),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_z_gates_right(self):
        """Test that Z-basis gates before controlled-Z-type gates on controls *and* targets get pushed ahead."""

        @CommuteControlledInterpreter(direction="right")
        def circuit():
            qml.PauliZ(wires=2)
            qml.S(wires=0)
            qml.CZ(wires=[0, 2])

            qml.CNOT(wires=[0, 1])

            qml.PhaseShift(0.2, wires=2)
            qml.T(wires=0)
            qml.PauliZ(wires=0)
            qml.CRZ(0.5, wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 8

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.CZ(wires=[0, 2]),
            qml.PauliZ(wires=2),
            qml.CNOT(wires=[0, 1]),
            qml.PhaseShift(0.2, wires=2),
            qml.CRZ(0.5, wires=[0, 1]),
            qml.S(wires=0),
            qml.T(wires=0),
            qml.PauliZ(wires=0),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_z_gates_left(self):
        """Test that Z-basis after before controlled-Z-type gates on controls *and*
        targets get pushed behind."""

        @CommuteControlledInterpreter(direction="left")
        def circuit():
            qml.CZ(wires=[0, 2])
            qml.PauliZ(wires=2)
            qml.S(wires=0)

            qml.CNOT(wires=[0, 1])

            qml.CRZ(0.5, wires=[0, 1])
            qml.RZ(0.2, wires=2)
            qml.T(wires=0)
            qml.PauliZ(wires=0)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 8

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliZ(wires=2),
            qml.S(wires=0),
            qml.RZ(0.2, wires=2),
            qml.T(wires=0),
            qml.PauliZ(wires=0),
            qml.CZ(wires=[0, 2]),
            qml.CNOT(wires=[0, 1]),
            qml.CRZ(0.5, wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    @pytest.mark.parametrize(
        "direction, expected_ops",
        [
            (
                "right",
                [
                    qml.X(1),
                    qml.CZ(wires=[0, 1]),
                    qml.S(0),
                    qml.CNOT(wires=[1, 0]),
                    qml.Y(1),
                    qml.CRY(0.5, wires=[1, 0]),
                    qml.Y(1),
                    qml.CRZ(-0.3, wires=[0, 1]),
                    qml.PhaseShift(0.2, wires=[0]),
                    qml.T(0),
                    qml.RZ(0.2, wires=[0]),
                    qml.Z(0),
                    qml.X(1),
                    qml.CRY(0.2, wires=[1, 0]),
                ],
            ),
            (
                "left",
                [
                    qml.X(1),
                    qml.S(0),
                    qml.CZ(wires=[0, 1]),
                    qml.CNOT(wires=[1, 0]),
                    qml.Y(1),
                    qml.CRY(0.5, wires=[1, 0]),
                    qml.PhaseShift(0.2, wires=[0]),
                    qml.Y(1),
                    qml.T(0),
                    qml.RZ(0.2, wires=[0]),
                    qml.Z(0),
                    qml.CRZ(-0.3, wires=[0, 1]),
                    qml.X(1),
                    qml.CRY(0.2, wires=[1, 0]),
                ],
            ),
        ],
    )
    def test_push_mixed_with_matrix(self, direction, expected_ops):
        """Test that arbitrary gates after controlled gates on controls *and*
        targets get properly pushed."""
        # pylint:disable=too-many-function-args

        @CommuteControlledInterpreter(direction=direction)
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

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 14

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)
