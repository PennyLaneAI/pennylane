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

# pylint:disable=wrong-import-position, unused-argument, protected-access

import numpy as np
import pytest

import pennylane as qp

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]

from pennylane.capture.primitives import cond_prim, for_loop_prim, measure_prim, while_loop_prim
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.optimization.commute_controlled import (
    CommuteControlledInterpreter,
    commute_controlled,
    commute_controlled_plxpr_to_plxpr,
)


class TestCommuteControlledInterpreter:
    """Unit tests for the CommuteControlledInterpreter for commuting controlled gates."""

    def test_invalid_direction(self):
        """Test that any direction other than 'left' or 'right' raises an error."""

        def circuit():
            qp.PauliX(wires=2)
            qp.CNOT(wires=[0, 2])
            qp.RX(0.2, wires=2)

        with pytest.raises(ValueError, match="must be 'left' or 'right'"):
            CommuteControlledInterpreter(direction="sideways")(circuit)

    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_gate_with_no_basis(self, direction):
        """Test that gates with no basis specified are ignored."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qp.PauliX(wires=2)
            qp.ControlledQubitUnitary(jax.numpy.array([[0, 1], [1, 0]]), wires=[0, 2])
            qp.PauliX(wires=2)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.PauliX(wires=2),
            qp.ControlledQubitUnitary(jax.numpy.array([[0, 1], [1, 0]]), wires=[0, 2]),
            qp.PauliX(wires=2),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            # jax inserts a dtype for the array which confuses qp.equal
            qp.assert_equal(op1, op2, check_interface=False)

    @pytest.mark.parametrize("direction", [("left"), ("right")])
    def test_gate_blocked_different_basis(self, direction):
        """Test that gates do not get pushed through controlled gates whose target bases don't match."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qp.PauliZ(wires=1)
            qp.CNOT(wires=[2, 1])
            qp.PauliY(wires=1)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.PauliZ(wires=1),
            qp.CNOT(wires=[2, 1]),
            qp.PauliY(wires=1),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    def test_push_x_gates_right(self):
        """Test that X-basis gates before controlled-X-type gates on targets get pushed ahead."""

        @CommuteControlledInterpreter(direction="right")
        def circuit():
            qp.PauliX(wires=2)
            qp.CNOT(wires=[0, 2])
            qp.RX(0.2, wires=2)
            qp.Toffoli(wires=[0, 1, 2])
            qp.SX(wires=1)
            qp.PauliX(wires=1)
            qp.CRX(0.1, wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 7

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.CNOT(wires=[0, 2]),
            qp.Toffoli(wires=[0, 1, 2]),
            qp.PauliX(wires=2),
            qp.RX(0.2, wires=2),
            qp.CRX(0.1, wires=[0, 1]),
            qp.SX(wires=1),
            qp.PauliX(wires=1),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    def test_push_x_gates_left(self):
        """Test that X-basis gates after controlled-X-type gates on targets get pushed back."""

        @CommuteControlledInterpreter(direction="left")
        def circuit():
            qp.CNOT(wires=[0, 2])
            qp.PauliX(wires=2)
            qp.RX(0.2, wires=2)
            qp.Toffoli(wires=[0, 1, 2])
            qp.CRX(0.1, wires=[0, 1])
            qp.SX(wires=1)
            qp.PauliX(wires=1)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 7

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.PauliX(wires=2),
            qp.RX(0.2, wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.Toffoli(wires=[0, 1, 2]),
            qp.SX(wires=1),
            qp.PauliX(wires=1),
            qp.CRX(0.1, wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    @pytest.mark.parametrize("direction", [("left"), ("right")])
    def test_dont_push_x_gates(self, direction):
        """Test that X-basis gates before controlled-X-type gates on controls don't get pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qp.PauliX(wires=0)
            qp.CNOT(wires=[0, 2])
            qp.RX(0.2, wires=0)
            qp.Toffoli(wires=[2, 0, 1])

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 4

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.PauliX(wires=0),
            qp.CNOT(wires=[0, 2]),
            qp.RX(0.2, wires=0),
            qp.Toffoli(wires=[2, 0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    def test_push_y_gates_right(self):
        """Test that Y-basis gates before controlled-Y-type gates on targets get pushed ahead."""

        @CommuteControlledInterpreter(direction="right")
        def circuit():
            qp.PauliY(wires=2)
            qp.CRY(-0.5, wires=[0, 2])
            qp.CNOT(wires=[1, 2])
            qp.RY(0.3, wires=1)
            qp.CY(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.CRY(-0.5, wires=[0, 2]),
            qp.PauliY(wires=2),
            qp.CNOT(wires=[1, 2]),
            qp.CY(wires=[0, 1]),
            qp.RY(0.3, wires=1),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    def test_push_y_gates_left(self):
        """Test that Y-basis gates after controlled-Y-type gates on targets get pushed behind."""

        @CommuteControlledInterpreter(direction="left")
        def circuit():
            qp.CRY(-0.5, wires=[0, 2])
            qp.PauliY(wires=2)
            qp.CNOT(wires=[1, 2])
            qp.CY(wires=[0, 1])
            qp.RY(0.3, wires=1)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.PauliY(wires=2),
            qp.CRY(-0.5, wires=[0, 2]),
            qp.CNOT(wires=[1, 2]),
            qp.RY(0.3, wires=1),
            qp.CY(wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    @pytest.mark.parametrize("direction", [("left"), ("right")])
    def test_dont_push_y_gates(self, direction):
        """Test that Y-basis gates next to controlled-Y-type gates on controls don't get pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qp.CRY(-0.2, wires=[0, 2])
            qp.PauliY(wires=0)
            qp.CNOT(wires=[1, 2])
            qp.CY(wires=[0, 1])
            qp.RY(0.3, wires=0)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.CRY(-0.2, wires=[0, 2]),
            qp.PauliY(wires=0),
            qp.CNOT(wires=[1, 2]),
            qp.CY(wires=[0, 1]),
            qp.RY(0.3, wires=0),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    def test_push_z_gates_right(self):
        """Test that Z-basis gates before controlled-Z-type gates on controls *and* targets get pushed ahead."""

        @CommuteControlledInterpreter(direction="right")
        def circuit():
            qp.PauliZ(wires=2)
            qp.S(wires=0)
            qp.CZ(wires=[0, 2])

            qp.CNOT(wires=[0, 1])

            qp.PhaseShift(0.2, wires=2)
            qp.T(wires=0)
            qp.PauliZ(wires=0)
            qp.CRZ(0.5, wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 8

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.CZ(wires=[0, 2]),
            qp.PauliZ(wires=2),
            qp.CNOT(wires=[0, 1]),
            qp.PhaseShift(0.2, wires=2),
            qp.CRZ(0.5, wires=[0, 1]),
            qp.S(wires=0),
            qp.T(wires=0),
            qp.PauliZ(wires=0),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    def test_push_z_gates_left(self):
        """Test that Z-basis after controlled-Z-type gates on controls *and*
        targets get pushed behind."""

        @CommuteControlledInterpreter(direction="left")
        def circuit():
            qp.CZ(wires=[0, 2])
            qp.PauliZ(wires=2)
            qp.S(wires=0)

            qp.CNOT(wires=[0, 1])

            qp.CRZ(0.5, wires=[0, 1])
            qp.RZ(0.2, wires=2)
            qp.T(wires=0)
            qp.PauliZ(wires=0)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 8

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.PauliZ(wires=2),
            qp.S(wires=0),
            qp.RZ(0.2, wires=2),
            qp.T(wires=0),
            qp.PauliZ(wires=0),
            qp.CZ(wires=[0, 2]),
            qp.CNOT(wires=[0, 1]),
            qp.CRZ(0.5, wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    @pytest.mark.parametrize(
        "direction, expected_ops",
        [
            (
                "right",
                [
                    qp.X(1),
                    qp.CZ(wires=[0, 1]),
                    qp.S(0),
                    qp.CNOT(wires=[1, 0]),
                    qp.Y(1),
                    qp.CRY(0.5, wires=[1, 0]),
                    qp.Y(1),
                    qp.CRZ(-0.3, wires=[0, 1]),
                    qp.PhaseShift(0.2, wires=[0]),
                    qp.T(0),
                    qp.RZ(0.2, wires=[0]),
                    qp.Z(0),
                    qp.X(1),
                    qp.CRY(0.2, wires=[1, 0]),
                ],
            ),
            (
                "left",
                [
                    qp.X(1),
                    qp.S(0),
                    qp.CZ(wires=[0, 1]),
                    qp.CNOT(wires=[1, 0]),
                    qp.Y(1),
                    qp.CRY(0.5, wires=[1, 0]),
                    qp.PhaseShift(0.2, wires=[0]),
                    qp.Y(1),
                    qp.T(0),
                    qp.RZ(0.2, wires=[0]),
                    qp.Z(0),
                    qp.CRZ(-0.3, wires=[0, 1]),
                    qp.X(1),
                    qp.CRY(0.2, wires=[1, 0]),
                ],
            ),
        ],
    )
    def test_push_mixed(self, direction, expected_ops):
        """Test that arbitrary gates after controlled gates on controls *and*
        targets get properly pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qp.PauliX(wires=1)
            qp.S(wires=0)
            qp.CZ(wires=[0, 1])
            qp.CNOT(wires=[1, 0])
            qp.PauliY(wires=1)
            qp.CRY(0.5, wires=[1, 0])
            qp.PhaseShift(0.2, wires=0)
            qp.PauliY(wires=1)
            qp.T(wires=0)
            qp.CRZ(-0.3, wires=[0, 1])
            qp.RZ(0.2, wires=0)
            qp.PauliZ(wires=0)
            qp.PauliX(wires=1)
            qp.CRY(0.2, wires=[1, 0])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 14

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    def test_returned_ops_not_pushed(self):
        """Test that operations returned from a circuit are not pushed."""

        @CommuteControlledInterpreter()
        def circuit():
            return qp.RX(0.1, wires=2), qp.CNOT(wires=[0, 2])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 2

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qp.RX(0.1, wires=2),
            qp.CNOT(wires=[0, 2]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)


class TestCommuteControlledHigherOrderPrimitives:
    """Unit tests for the CommuteControlledInterpreter for higher order primitives."""

    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_qnode(self, direction):
        """Test that the CommuteControlledInterpreter can be used with a QNode."""

        @qp.qnode(device=qp.device("default.qubit", wires=2))
        def circuit():
            qp.PauliX(wires=1)
            qp.S(wires=0)
            qp.CZ(wires=[0, 1])
            qp.CNOT(wires=[1, 0])
            qp.PauliY(wires=1)
            qp.CRY(0.5, wires=[1, 0])
            qp.PhaseShift(0.2, wires=0)
            qp.PauliY(wires=1)
            qp.T(wires=0)
            qp.CRZ(-0.3, wires=[0, 1])
            qp.RZ(0.2, wires=0)
            qp.PauliZ(wires=0)
            qp.PauliX(wires=1)
            qp.CRY(0.2, wires=[1, 0])
            return qp.expval(qp.PauliZ(0))

        transformed_circuit = CommuteControlledInterpreter(direction=direction)(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 1
        circuit_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert len(circuit_jaxpr.eqns) == 16

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        with qp.capture.pause():
            # pylint: disable=not-callable
            expected_result = circuit()

        assert qp.math.allclose(result, expected_result)

    @pytest.mark.parametrize(
        "selector, expected_cond_ops",
        [
            (
                0.2,
                {
                    "right": [
                        qp.CNOT(wires=[0, 2]),
                        qp.Toffoli(wires=[0, 1, 2]),
                        qp.RX(np.pi, wires=2),
                    ],
                    "left": [
                        qp.RX(np.pi, wires=2),
                        qp.CNOT(wires=[0, 2]),
                        qp.Toffoli(wires=[0, 1, 2]),
                    ],
                },
            ),
            (
                0.8,
                {
                    "right": [
                        qp.CNOT(wires=[0, 2]),
                        qp.Toffoli(wires=[0, 1, 2]),
                        qp.RZ(np.pi, wires=0),
                    ],
                    "left": [
                        qp.RZ(np.pi, wires=0),
                        qp.CNOT(wires=[0, 2]),
                        qp.Toffoli(wires=[0, 1, 2]),
                    ],
                },
            ),
        ],
    )
    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_cond(self, selector, expected_cond_ops, direction):
        """Test that operations inside a conditional block are correctly pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit(selector, x):

            def true_branch(x):
                qp.CNOT(wires=[0, 2])
                qp.RZ(x, wires=0)
                qp.Toffoli(wires=[0, 1, 2])

            # pylint: disable=unused-argument
            def false_branch(x):
                qp.CNOT(wires=[0, 2])
                qp.RX(x, wires=2)
                qp.Toffoli(wires=[0, 1, 2])

            qp.Z(0)
            qp.CNOT(wires=[0, 1])
            qp.T(0)
            qp.cond(selector > 0.5, true_branch, false_branch)(x)
            qp.Z(0)
            qp.CNOT(wires=[0, 1])
            qp.T(0)

        jaxpr = jax.make_jaxpr(circuit)(selector, np.pi)
        initial_gates = (
            [qp.CNOT([0, 1]), qp.Z(0), qp.T(0)]
            if direction == "right"
            else [qp.Z(0), qp.T(0), qp.CNOT([0, 1])]
        )

        assert len(jaxpr.eqns) == 8
        assert jaxpr.eqns[0].primitive == jax.lax.gt_p
        for e, i in enumerate(range(1, 4)):
            assert jaxpr.eqns[i].primitive == initial_gates[e]._primitive

        assert jaxpr.eqns[4].primitive == cond_prim

        for e, i in enumerate(range(5, 8)):
            assert jaxpr.eqns[i].primitive == initial_gates[e]._primitive

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, selector, np.pi)
        jaxpr_ops = collector.state["ops"]
        expected_ops = initial_gates + expected_cond_ops[direction] + initial_gates
        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_for_loop(self, direction):
        """Test that operators inside a for loop are correctly pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit(x):

            qp.Z(0)
            qp.CNOT(wires=[0, 1])
            qp.T(0)

            @qp.for_loop(0, 2)
            # pylint: disable=unused-argument
            def loop(i, x):
                qp.CNOT(wires=[0, 2])
                qp.RX(x, wires=2)
                qp.Toffoli(wires=[0, 1, 2])
                return x

            # pylint: disable=no-value-for-parameter
            loop(x)
            qp.Z(0)
            qp.CNOT(wires=[0, 1])
            qp.T(0)

        jaxpr = jax.make_jaxpr(circuit)(np.pi)
        assert len(jaxpr.eqns) == 7
        assert jaxpr.eqns[3].primitive == for_loop_prim

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, np.pi)
        jaxpr_ops = collector.state["ops"]
        assert len(jaxpr_ops) == 12

        initial_gates = (
            [qp.CNOT([0, 1]), qp.Z(0), qp.T(0)]
            if direction == "right"
            else [qp.Z(0), qp.T(0), qp.CNOT([0, 1])]
        )
        loop_ctrls = [qp.CNOT(wires=[0, 2]), qp.Toffoli(wires=[0, 1, 2])]
        loop_rx = [qp.RX(np.pi, 2)]
        expected_loop_ops = loop_ctrls + loop_rx if direction == "right" else loop_rx + loop_ctrls
        expected_ops = initial_gates + expected_loop_ops * 2 + initial_gates

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_while_loop(self, direction):
        """Test that operators inside a while loop are correctly pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit(x):

            qp.Z(0)
            qp.CNOT(wires=[0, 1])
            qp.T(0)

            # pylint: disable=unused-argument
            @qp.while_loop(lambda i, x: i < 2)
            def loop(i, x):
                qp.CNOT(wires=[0, 2])
                qp.RX(x, wires=2)
                qp.Toffoli(wires=[0, 1, 2])
                return i + 1, x

            # pylint: disable=no-value-for-parameter
            loop(0, x)
            qp.Z(0)
            qp.CNOT(wires=[0, 1])
            qp.T(0)

        jaxpr = jax.make_jaxpr(circuit)(np.pi)
        assert len(jaxpr.eqns) == 7
        assert jaxpr.eqns[3].primitive == while_loop_prim

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, np.pi)
        jaxpr_ops = collector.state["ops"]
        assert len(jaxpr_ops) == 12

        initial_gates = (
            [qp.CNOT([0, 1]), qp.Z(0), qp.T(0)]
            if direction == "right"
            else [qp.Z(0), qp.T(0), qp.CNOT([0, 1])]
        )
        loop_ctrls = [qp.CNOT(wires=[0, 2]), qp.Toffoli(wires=[0, 1, 2])]
        loop_rx = [qp.RX(np.pi, 2)]
        expected_loop_ops = loop_ctrls + loop_rx if direction == "right" else loop_rx + loop_ctrls
        expected_ops = initial_gates + expected_loop_ops * 2 + initial_gates

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qp.assert_equal(op1, op2)

    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_mid_circuit_measurement(self, direction):
        """Test that mid-circuit measurements are correctly handled."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit(x):
            qp.CNOT(wires=[0, 2])
            qp.RX(x, wires=2)
            qp.Toffoli(wires=[0, 1, 2])
            qp.measure(0)
            qp.CNOT(wires=[0, 2])
            qp.RX(x, wires=2)
            qp.Toffoli(wires=[0, 1, 2])
            return qp.expval(qp.PauliZ(0))

        jaxpr = jax.make_jaxpr(circuit)(np.pi)
        assert len(jaxpr.eqns) == 9

        jaxpr_controlled_ops = [qp.CNOT([0, 2]), qp.Toffoli([0, 1, 2])]
        rx_op = [qp.RX(np.pi, 2)]
        initial_gates = (
            jaxpr_controlled_ops + rx_op if direction == "right" else rx_op + jaxpr_controlled_ops
        )

        # I test the jaxpr like this because `CollectOpsandMeas`
        # currently interprets a mid-circuit measurement as an operator, not a measurement
        for e, i in enumerate(range(3)):
            assert jaxpr.eqns[i].primitive == initial_gates[e]._primitive
        assert jaxpr.eqns[3].primitive == measure_prim
        for e, i in enumerate(range(4, 7)):
            assert jaxpr.eqns[i].primitive == initial_gates[e]._primitive
        assert jaxpr.eqns[7].primitive == qp.PauliZ._primitive


class TestCommuteControlledPLXPR:
    """Unit tests for the commute-controlled transformation on PLXPRs."""

    def test_single_qubit_fusion_plxpr_to_plxpr(self):
        """Test that the commute-controlled transformation works on a plxpr."""

        def circuit():
            qp.PauliX(wires=1)
            qp.S(wires=0)
            qp.CZ(wires=[0, 1])
            qp.CNOT(wires=[1, 0])
            qp.PauliY(wires=1)
            qp.CRY(0.5, wires=[1, 0])
            qp.PhaseShift(0.2, wires=0)
            qp.PauliY(wires=1)
            qp.T(wires=0)
            qp.CRZ(-0.3, wires=[0, 1])
            qp.RZ(0.2, wires=0)
            qp.PauliZ(wires=0)
            qp.PauliX(wires=1)
            qp.CRY(0.2, wires=[1, 0])

        jaxpr = jax.make_jaxpr(circuit)()
        transformed_jaxpr = commute_controlled_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {})
        assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
        assert len(transformed_jaxpr.eqns) == 14

        expected_ops = [
            qp.X,
            qp.CZ,
            qp.S,
            qp.CNOT,
            qp.Y,
            qp.CRY,
            qp.Y,
            qp.CRZ,
            qp.PhaseShift,
            qp.T,
            qp.RZ,
            qp.Z,
            qp.X,
            qp.CRY,
        ]
        assert all(
            eqn.primitive == cls._primitive
            for eqn, cls in zip(transformed_jaxpr.eqns, expected_ops, strict=True)
        )

    def test_applying_plxpr_decorator(self):
        """Test that the commute-controlled transformation works when applying the plxpr decorator."""

        @qp.capture.expand_plxpr_transforms
        @commute_controlled
        def circuit():
            qp.PauliX(wires=1)
            qp.S(wires=0)
            qp.CZ(wires=[0, 1])
            qp.CNOT(wires=[1, 0])
            qp.PauliY(wires=1)
            qp.CRY(0.5, wires=[1, 0])
            qp.PhaseShift(0.2, wires=0)
            qp.PauliY(wires=1)
            qp.T(wires=0)
            qp.CRZ(-0.3, wires=[0, 1])
            qp.RZ(0.2, wires=0)
            qp.PauliZ(wires=0)
            qp.PauliX(wires=1)
            qp.CRY(0.2, wires=[1, 0])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 14

        expected_ops = [
            qp.X,
            qp.CZ,
            qp.S,
            qp.CNOT,
            qp.Y,
            qp.CRY,
            qp.Y,
            qp.CRZ,
            qp.PhaseShift,
            qp.T,
            qp.RZ,
            qp.Z,
            qp.X,
            qp.CRY,
        ]
        assert all(
            eqn.primitive == cls._primitive
            for eqn, cls in zip(jaxpr.eqns, expected_ops, strict=True)
        )
