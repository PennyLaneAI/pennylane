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
"""Unit tests for the ``SingleQubitFusionInterpreter`` class."""

# pylint:disable=wrong-import-position,protected-access
import pytest

import pennylane as qml

import numpy as np

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    cond_prim,
    for_loop_prim,
    while_loop_prim,
)
from pennylane.transforms.optimization.single_qubit_fusion import (
    SingleQubitFusionInterpreter,
    single_qubit_plxpr_to_plxpr,
)

from pennylane.tape.plxpr_conversion import CollectOpsandMeas

from pennylane.transforms.optimization import single_qubit_fusion

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class TestSingleQubitFusionInterpreter:
    """Unit tests for the SingleQubitFusionInterpreter class"""

    def test_single_qubit_full_fusion(self):
        """Test that a sequence of single-qubit gates all fuse."""

        def circuit():
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.RX(0.1, wires=0)
            qml.SX(wires=0)
            qml.T(wires=0)
            qml.PauliX(wires=0)

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)

        # This circuit should be transformed to a single Rot(-4.37,1.98,-0.96) gate

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 1

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)()

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = single_qubit_fusion(circuit)
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert qml.equal(op1, op2)

    def test_single_qubit_full_fusion_qnode(self):
        """Test that a sequence of single-qubit gates fuse when inside a QNode."""

        @qml.qnode(device=qml.device("default.qubit", wires=1))
        def circuit():
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.RX(0.1, wires=0)
            qml.SX(wires=0)
            qml.T(wires=0)
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)

        # This circuit should be transformed to a single Rot(-4.37,1.98,-0.96) gate

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        circuit_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert len(circuit_jaxpr.eqns) == 3
        assert circuit_jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert circuit_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert circuit_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.literals)

        with qml.capture.pause():
            # pylint: disable=not-callable
            expected_result = single_qubit_fusion(circuit)()

        assert qml.math.allclose(result, expected_result)

    def test_single_qubit_CNOT_fusion(self):
        """Test that a sequence of single-qubit gates and a CNOT gate fuse."""

        def circuit():
            # Excluded gate at the start
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=0)
            qml.PauliX(wires=0)
            qml.RZ(0.1, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            # Excluded gate after another gate
            qml.RZ(0.1, wires=0)
            qml.PauliX(wires=1)
            qml.PauliZ(wires=1)
            # Excluded gate after multiple others
            qml.RZ(0.2, wires=1)

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)()

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = single_qubit_fusion(circuit)
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert qml.equal(op1, op2)

    def test_single_qubit_fusion_no_gates_after(self):
        """Test that gates with nothing after are applied without modification."""

        def circuit():
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=1)

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)

        # This circuit should remain unchanged

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 2

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = single_qubit_fusion(circuit)
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert qml.equal(op1, op2)

    def test_single_qubit_cancelled_fusion(self):
        """Test if a sequence of single-qubit gates that all cancel yields no operations."""

        def circuit():
            qml.RZ(0.1, wires=0)
            qml.RX(0.2, wires=0)
            qml.RX(-0.2, wires=0)
            qml.RZ(-0.1, wires=0)

        # This circuit should be transformed to no operations as all gates cancel.
        # With program capture enabled, this corresponds to a single rotation gate with angles 0.

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 1

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        transformed_ops_check = [qml.Rot(0.0, 0.0, 0.0, wires=[0])]

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert qml.equal(op1, op2)

    def test_single_qubit_fusion_not_implemented(self):
        """Test that fusion is correctly skipped for single-qubit gates where the rotation angles are not specified."""

        def circuit():
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=0)
            # No rotation angles specified for PauliRot since it is a gate that
            # in principle acts on an arbitrary number of wires.
            qml.PauliRot(0.2, "X", wires=0)
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=0)

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = single_qubit_fusion(circuit)
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert qml.equal(op1, op2)

    def test_single_qubit_fusion_multiple_qubits(self):
        """Test that all sequences of single-qubit gates across multiple qubits fuse properly."""

        def circuit():
            qml.RZ(0.3, wires=0)
            qml.RY(0.5, wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=1)
            qml.RX(0.1, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.SX(wires=1)
            qml.S(wires=1)
            qml.PhaseShift(0.3, wires=1)

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 4

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        with qml.capture.pause():
            transformed_circuit_check = single_qubit_fusion(circuit)
            transformed_ops_check = qml.tape.make_qscript(transformed_circuit_check)().operations
            # The order of the first two operations is different with program capture enabled
            assert qml.wires.Wires.shared_wires(
                [transformed_ops_check[0].wires, transformed_ops_check[1].wires]
            ) == qml.wires.Wires([])
            transformed_ops_check[0], transformed_ops_check[1] = (
                transformed_ops_check[1],
                transformed_ops_check[0],
            )

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert qml.equal(op1, op2)

    def test_returned_op_is_not_fused(self):
        """Test that ops that are returned by the function being transformed are not fused."""

        def circuit():
            qml.H(0)
            return qml.H(0)

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 2

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        transformed_ops_check = [qml.H(0), qml.H(0)]

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            assert qml.equal(op1, op2)

    def test_no_wire_ops_not_fused(self):
        """Test that inverse operations with no wires are not fused."""

        def circuit():
            qml.Identity()
            qml.PauliX(wires=0)
            qml.PauliY(wires=0)
            qml.Identity()

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        transformed_ops_check = [
            qml.Identity(),
            qml.Identity(),
            qml.Rot(-0.7853981633974485, 0.0, -2.356194490192345, wires=[0]),
        ]

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check):
            # The qml.equal function does not recognize two qml.Rot operators
            # as equivalent unless the input angles are exactly the same
            assert op1.name == op2.name
            assert qml.math.allclose(op1.parameters, op2.parameters)
            assert op1.wires == op2.wires

    def test_transform_higher_order_primitive(self):
        """Test that the inner_jaxpr of transform primitives is not transformed."""

        @qml.transform
        def fictitious_transform(tape):
            """Fictitious transform"""
            return [tape], lambda res: res[0]

        def circuit():
            @fictitious_transform
            def g():
                qml.S(0)
                qml.T(0)

            qml.RX(0.1, 0)
            g()
            qml.RY(0.2, 0)

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)()

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == fictitious_transform._primitive
        assert jaxpr.eqns[2].primitive == qml.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["inner_jaxpr"]
        assert len(inner_jaxpr.eqns) == 2
        assert inner_jaxpr.eqns[0].primitive == qml.S._primitive
        assert inner_jaxpr.eqns[1].primitive == qml.T._primitive

    @pytest.mark.parametrize(
        "selector, expected_ops",
        [
            (
                0.2,
                [
                    qml.CNOT(wires=[0, 1]),
                    qml.Hadamard(wires=[0]),  # H(0) and H(1) are not fused
                    qml.Hadamard(wires=[1]),  # because they act on different wires
                    qml.CNOT(wires=[0, 1]),
                ],
            ),
            (
                0.6,
                [
                    qml.CNOT(wires=[0, 1]),
                    qml.Rot(np.pi / 2, 0.6, 0.0, wires=[0]),  # RX and S fused
                    qml.CNOT(wires=[0, 1]),
                ],
            ),
            (
                0.8,
                [
                    qml.CNOT(wires=[0, 1]),
                    qml.Rot(np.pi / 2, 0.8, 0.0, wires=[0]),  # RX and S fused
                    qml.CNOT(wires=[0, 1]),
                ],
            ),
        ],
    )
    def test_cond(self, selector, expected_ops):
        """Test that operations inside a conditional block are correctly fused."""

        def circuit(x):

            qml.CNOT(wires=[0, 1])

            def true_branch(x):
                qml.RX(x, wires=0)
                qml.S(wires=0)

            def false_branch(x):
                qml.H(0)
                qml.H(1)

            qml.cond(x > 0.5, true_branch, false_branch)(x)
            qml.CNOT(wires=[0, 1])

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)(selector)

        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[1].primitive == jax.lax.gt_p
        assert jaxpr.eqns[2].primitive == cond_prim
        assert jaxpr.eqns[3].primitive == qml.CNOT._primitive

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, selector)
        jaxpr_ops = collector.state["ops"]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            # The qml.equal function does not recognize two qml.Rot operators
            # as equivalent unless the input angles are exactly the same
            assert op1.name == op2.name
            assert qml.math.allclose(op1.parameters, op2.parameters)
            assert op1.wires == op2.wires

    def test_for_loop(self):
        """Test that for operators inside a for loop are correctly fused."""

        def circuit(x):

            qml.CNOT(wires=[0, 1])

            @qml.for_loop(0, 1)
            def loop(i, x):
                qml.RX(x, wires=0)
                qml.RZ(x, wires=0)
                return qml.Hadamard(wires=0)

            loop(x)

            qml.CNOT(wires=[0, 1])

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)(np.pi)

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[1].primitive == for_loop_prim
        assert jaxpr.eqns[2].primitive == qml.CNOT._primitive

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, np.pi)
        jaxpr_ops = collector.state["ops"]
        assert len(jaxpr_ops) == 4

        expected_ops = [
            qml.CNOT(wires=[0, 1]),
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.Hadamard(wires=[0]),  # Should not be fused because is returned
            qml.CNOT(wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            # The qml.equal function does not recognize two qml.Rot operators
            # as equivalent unless the input angles are exactly the same
            assert op1.name == op2.name
            assert qml.math.allclose(op1.parameters, op2.parameters)
            assert op1.wires == op2.wires

    def test_while_loop(self):
        """Test that while operators inside a while loop are correctly fused."""

        def circuit(x):

            qml.CNOT(wires=[0, 1])

            def while_f(i):
                return i < 3

            @qml.while_loop(while_f)
            def loop(i):
                qml.RX(np.pi, wires=0)
                qml.RZ(np.pi, wires=0)
                return i + 1

            loop(0)

            qml.CNOT(wires=[0, 1])

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)(np.pi)

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[1].primitive == while_loop_prim
        assert jaxpr.eqns[2].primitive == qml.CNOT._primitive

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, np.pi)
        jaxpr_ops = collector.state["ops"]
        assert len(jaxpr_ops) == 5

        print(jaxpr_ops)

        expected_ops = [
            qml.CNOT(wires=[0, 1]),
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.CNOT(wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            # The qml.equal function does not recognize two qml.Rot operators
            # as equivalent unless the input angles are exactly the same
            assert op1.name == op2.name
            assert qml.math.allclose(op1.parameters, op2.parameters)
            assert op1.wires == op2.wires

    def test_single_qubit_fusion_traced_wires(self):
        """Test that single-qubit gates with traced wires are fused correctly."""

        def circuit(w1, w2, w3, w4):

            qml.X(wires=w1)
            qml.Y(wires=w2)
            qml.Z(wires=w2)
            qml.CNOT(wires=[w1, w2])
            qml.Z(wires=w3)
            qml.Hadamard(wires=w4)
            qml.CNOT(wires=[w1, w3])

        wires = [0, 1, 2, 3]

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)(*wires)

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *wires)
        jaxpr_ops = collector.state["ops"]

        assert len(jaxpr_ops) == 6

        expected_ops = [
            qml.Rot(np.pi / 2, np.pi, -np.pi / 2, wires=[0]),  # X not fused converted to Rot
            qml.Rot(0, np.pi, np.pi, wires=[1]),  # Y and Z fused
            qml.CNOT(wires=[0, 1]),
            qml.Rot(np.pi, 0, 0, wires=[2]),  # Z not fused converted to Rot
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=[3]),  # H not fused and not converted to Rot
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            # The qml.equal function currently does not distinguish between
            # two qml.Rot operators as equivalent unless the input angles are exactly the same
            assert op1.name == op2.name
            assert qml.math.allclose(op1.parameters, op2.parameters)
            assert op1.wires == op2.wires

    def test_single_qubit_fusion_traced_params(self):
        """Test that single-qubit gates with traced parameters are fused correctly."""

        def circuit(params):
            qml.Hadamard(wires=0)
            qml.RZ(params[0], wires=0)
            qml.PauliY(wires=1)
            qml.RZ(params[1], wires=0)
            qml.CNOT(wires=[1, 2])
            qml.CRY(params[2], wires=[1, 2])
            qml.PauliZ(wires=0)
            qml.CRY(params[3], wires=[1, 2])
            qml.Rot(params[0], params[1], params[2], wires=1)
            qml.Rot(params[2], params[3], params[0], wires=1)

        params = jax.numpy.array([0.1, 0.2, 0.3, 0.4])

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)
        jaxpr = jax.make_jaxpr(transformed_circuit)(params)

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, params)
        jaxpr_ops = collector.state["ops"]
        assert len(jaxpr_ops) == 6

        expected_ops = [
            qml.Rot(0.0, np.pi, 0.0, wires=[1]),
            qml.CNOT(wires=[1, 2]),
            qml.CRY(0.3, wires=[1, 2]),
            qml.CRY(0.4, wires=[1, 2]),
            qml.Rot(-np.pi, np.pi / 2, -2.8415926, wires=[0]),
            qml.Rot(0.51580256, 0.57563156, 0.30755687, wires=[1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            # The qml.equal function currently does not distinguish between
            # CRY(0.30000001192092896, wires=[1, 2]) and CRY(0.3, wires=[1, 2])
            assert op1.name == op2.name
            assert qml.math.allclose(op1.parameters, op2.parameters)
            assert op1.wires == op2.wires

    def test_single_qubit_fusion_plxpr_to_plxpr(self):

        def circuit():
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.RX(0.1, wires=0)
            qml.SX(wires=0)
            qml.T(wires=0)
            qml.PauliX(wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)()
        transformed_jaxpr = single_qubit_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {})
        assert isinstance(transformed_jaxpr, jax.core.ClosedJaxpr)
        assert len(transformed_jaxpr.eqns) == 3
        assert transformed_jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive
