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

import numpy as np

# pylint:disable=wrong-import-position,protected-access, too-few-public-methods
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    cond_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    measure_prim,
    while_loop_prim,
)
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.optimization.optimization_utils import fuse_rot_angles
from pennylane.transforms.optimization.single_qubit_fusion import (
    SingleQubitFusionInterpreter,
    single_qubit_plxpr_to_plxpr,
)

pytestmark = [pytest.mark.jax, pytest.mark.capture]


class TestSingleQubitFusionInterpreter:
    """Unit tests for the SingleQubitFusionInterpreter class"""

    def test_single_qubit_full_fusion(self):
        """Test that a sequence of single-qubit gates all fuse."""

        @SingleQubitFusionInterpreter()
        def circuit():
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.RX(0.1, wires=0)
            qml.SX(wires=0)
            qml.T(wires=0)
            qml.PauliX(wires=0)

        # This circuit should be transformed to a single Rot(-4.37,1.98,-0.96) gate

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 1

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        transformed_ops_check = [
            qml.Rot(-4.36933023474497, 1.9838145748474882, -0.9592113138119485, wires=[0])
        ]

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check, strict=True):
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize(
        "exclude_gates, expected_ops",
        [
            (
                None,
                [
                    qml.Rot(-6.183185307179587, np.pi / 2, 4.440892098500626e-16, wires=[0]),
                    qml.Rot(0.1, 0.0, 0.0, wires=[1]),
                    qml.CNOT(wires=[0, 1]),
                    qml.Rot(1.7707963267948965, np.pi, np.pi / 2, wires=[1]),
                    qml.Rot(3.241592653589793, np.pi / 2, 0, wires=[0]),
                ],
            ),
            (
                ["RZ"],
                [
                    qml.RZ(0.1, wires=[0]),
                    qml.RZ(0.1, wires=[1]),
                    qml.Rot(6.283185307179586, np.pi / 2, 0.0, wires=[0]),
                    qml.CNOT(wires=[0, 1]),
                    qml.RZ(0.2, wires=[1]),
                    qml.RZ(0.1, wires=[0]),
                    qml.H(0),
                    qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[1]),
                ],
            ),
            (
                ["Hadamard"],
                [
                    qml.RZ(0.1, wires=[0]),
                    qml.H(0),
                    qml.Rot(np.pi / 2, np.pi, -np.pi / 2, wires=[0]),
                    qml.Rot(0.1, 0.0, 0.0, wires=[1]),
                    qml.CNOT(wires=[0, 1]),
                    qml.RZ(0.1, wires=[0]),
                    qml.H(0),
                    qml.Rot(1.7707963267948965, np.pi, np.pi / 2, wires=[1]),
                ],
            ),
            (
                ["RZ", "Hadamard"],
                [
                    qml.RZ(0.1, wires=[0]),
                    qml.RZ(0.1, wires=[1]),
                    qml.H(0),
                    qml.Rot(np.pi / 2, np.pi, -np.pi / 2, wires=[0]),
                    qml.CNOT(wires=[0, 1]),
                    qml.RZ(0.2, wires=[1]),
                    qml.RZ(0.1, wires=[0]),
                    qml.H(0),
                    qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[1]),
                ],
            ),
        ],
    )
    def test_single_qubit_fusion_exclude_gates(self, exclude_gates, expected_ops):
        """Test that a sequence of single-qubit gates partially fuse when excluding certain gates."""

        @SingleQubitFusionInterpreter(exclude_gates=exclude_gates)
        def circuit():
            qml.RZ(0.1, wires=0)
            qml.RZ(0.1, wires=1)
            qml.Hadamard(wires=0)
            qml.PauliX(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.2, wires=1)
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=0)
            qml.PauliX(wires=1)
            qml.PauliZ(wires=1)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == len(expected_ops)

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qml.assert_equal(op1, op2)

    def test_single_qubit_fusion_no_gates_after(self):
        """Test that gates with nothing after are applied without modification."""

        @SingleQubitFusionInterpreter()
        def circuit():
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=1)

        # This circuit should remain unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 2

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        transformed_ops_check = [qml.RZ(0.1, wires=0), qml.Hadamard(wires=1)]
        for op1, op2 in zip(jaxpr_ops, transformed_ops_check, strict=True):
            qml.assert_equal(op1, op2)

    def test_single_qubit_cancelled_fusion(self):
        """Test if a sequence of single-qubit gates that all cancel yields no operations."""

        @SingleQubitFusionInterpreter()
        def circuit():
            qml.RZ(0.1, wires=0)
            qml.RX(0.2, wires=0)
            qml.RX(-0.2, wires=0)
            qml.RZ(-0.1, wires=0)

        # This circuit should be transformed to no operations as all gates cancel.

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 0

    def test_atol_kwarg(self):
        """Test that the atol kwarg is correctly passed to the fusion transformation."""

        @SingleQubitFusionInterpreter(atol=1e-5)
        def circuit():
            qml.RZ(0.1, wires=0)
            qml.RZ((-0.1 + 1e-4), wires=0)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 1

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        transformed_ops_check = [qml.Rot(0.0001, 0, 0, wires=[0])]

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check, strict=True):
            qml.assert_equal(op1, op2)

    def test_single_qubit_fusion_not_implemented(self):
        """Test that fusion is correctly skipped for single-qubit gates where the rotation angles are not specified."""

        @SingleQubitFusionInterpreter()
        def circuit():
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=0)
            # No rotation angles specified for PauliRot since it is a gate that
            # in principle acts on an arbitrary number of wires.
            qml.PauliRot(0.2, "X", wires=0)
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=0)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        transformed_ops_check = [
            qml.Rot(3.241592653589793, np.pi / 2, 0.0, wires=[0]),
            qml.PauliRot(0.2, "X", wires=[0]),
            qml.Rot(3.241592653589793, np.pi / 2, 0.0, wires=[0]),
        ]

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check, strict=True):
            qml.assert_equal(op1, op2)

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

        transformed_ops_check = [
            qml.Rot(0.1, 0.2, 0.3, wires=[1]),
            qml.Rot(0.5063034952270635, 0.5090696523044768, -0.18074939399704498, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.Rot(np.pi / 2, np.pi / 2, 0.3, wires=[1]),
        ]

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check, strict=True):
            qml.assert_equal(op1, op2)

    def test_returned_op_is_not_fused(self):
        """Test that ops that are returned by the function being transformed are not fused."""

        @SingleQubitFusionInterpreter()
        def circuit():
            qml.H(0)
            return qml.H(0)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 2

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        transformed_ops_check = [qml.H(0), qml.H(0)]

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check, strict=True):
            qml.assert_equal(op1, op2)

    def test_no_wire_ops_not_fused(self):
        """Test that inverse operations with no wires are not fused."""

        @SingleQubitFusionInterpreter()
        def circuit():
            qml.Identity()
            qml.PauliX(wires=0)
            qml.PauliY(wires=0)
            qml.Identity()

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        transformed_ops_check = [
            qml.Identity(),
            qml.Identity(),
            qml.Rot(-0.7853981633974485, 0.0, -2.356194490192345, wires=[0]),
        ]

        for op1, op2 in zip(jaxpr_ops, transformed_ops_check, strict=True):
            qml.assert_equal(op1, op2)

    def test_single_qubit_fusion_traced_wires(self):
        """Test that single-qubit gates with traced wires are fused correctly."""

        @SingleQubitFusionInterpreter()
        def circuit(w1, w2, w3, w4):

            qml.X(wires=w1)
            qml.Y(wires=w2)
            qml.Z(wires=w2)
            qml.CNOT(wires=[w1, w2])
            qml.Z(wires=w3)
            qml.Hadamard(wires=w4)
            qml.CNOT(wires=[w1, w3])

        wires = [0, 1, 2, 3]

        jaxpr = jax.make_jaxpr(circuit)(*wires)

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *wires)
        jaxpr_ops = collector.state["ops"]

        assert len(jaxpr_ops) == 6

        expected_ops = [
            qml.X(wires=[0]),  # X not fused
            qml.Rot(0, np.pi, np.pi, wires=[1]),  # Y and Z fused
            qml.CNOT(wires=[0, 1]),
            qml.Z(wires=[2]),  # Z not fused
            qml.Hadamard(wires=[3]),  # H not fused and not converted to Rot
            qml.CNOT(wires=[0, 2]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qml.assert_equal(op1, op2)

    def test_single_qubit_fusion_traced_consts_params(self):
        """Test that single-qubit gates with traced parameters and constants are fused correctly."""

        const_param = jax.numpy.array(0.3)

        @SingleQubitFusionInterpreter()
        def circuit(params):
            qml.Hadamard(wires=0)
            qml.RZ(params[0], wires=0)
            qml.PauliY(wires=1)
            qml.RZ(params[1], wires=0)
            qml.CNOT(wires=[1, 2])
            qml.CRY(params[2], wires=[1, 2])
            qml.PauliZ(wires=0)
            qml.CRY(params[3], wires=[1, 2])
            qml.Rot(params[0], params[1], const_param, wires=1)
            qml.Rot(const_param, params[3], params[0], wires=1)

        params = jax.numpy.array([0.1, 0.2, 0.3, 0.4])

        jaxpr = jax.make_jaxpr(circuit)(params)

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

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qml.assert_equal(op1, op2, check_interface=False)

    def test_dynamic_wires_between_static_wires(self):
        """Test that operations with dynamic wires between operations with static
        wires cause fusion to not happen."""

        @SingleQubitFusionInterpreter()
        def f(x, y, w):
            qml.RX(x, 0)
            qml.RY(y, w)
            qml.RX(x, 0)
            qml.RY(y, w)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(2.5, 3.5, 0)

        dyn_wire = 0
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 2.5, 3.5, dyn_wire)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]
        expected_meas = [qml.expval(qml.Z(0))]

        expected_ops = [qml.RX(2.5, 0), qml.RY(3.5, 0), qml.RX(2.5, 0), qml.RY(3.5, 0)]
        assert ops == expected_ops
        assert meas == expected_meas

        dyn_wire = 1
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 2.5, 3.5, dyn_wire)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qml.RX(2.5, 0), qml.RY(3.5, 1), qml.RX(2.5, 0), qml.RY(3.5, 1)]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_same_dyn_wires_fuse(self):
        """Test that ops on the same dynamic wires get fused."""

        @SingleQubitFusionInterpreter()
        def f(x, y, w):
            qml.RX(x, 0)
            qml.RY(y, w)
            qml.RY(y, w)
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(2.5, 3.5, 0)

        dyn_wire = 0
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 2.5, 3.5, dyn_wire)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        # We use jax.jit here to make `phi` abstract. Otherwise, a different pipeline
        # is used to compute the fused angles, which leads to the true and expected
        # angles to be different.
        @jax.jit
        def get_double_ry_rot_angles(angles):
            return fuse_rot_angles(angles, angles)

        angles = jax.numpy.array(qml.RY(3.5, 0).single_qubit_rot_angles())
        fused_angles = get_double_ry_rot_angles(angles)
        expected_ops = [qml.RX(2.5, 0), qml.Rot(*fused_angles, 0), qml.RX(2.5, 0)]
        expected_meas = [qml.expval(qml.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_different_dyn_wires_interleaved(self):
        """Test that ops on different dynamic wires interleaved with each other
        do not fuse."""

        @SingleQubitFusionInterpreter()
        def f(x, y, w1, w2):
            qml.RX(x, w1)
            qml.RY(y, w2)
            qml.RX(x, w1)
            qml.RY(y, w2)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(2.5, 3.5, 0, 0)

        dyn_wires = (0, 0)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 2.5, 3.5, *dyn_wires)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qml.RX(2.5, 0), qml.RY(3.5, 0), qml.RX(2.5, 0), qml.RY(3.5, 0)]
        expected_meas = [qml.expval(qml.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

        dyn_wires = (0, 1)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 2.5, 3.5, *dyn_wires)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qml.RX(2.5, 0), qml.RY(3.5, 1), qml.RX(2.5, 0), qml.RY(3.5, 1)]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_diff_dyn_wires_non_fusible_op(self):
        """Test that a non-fusible op with dynamic wires will cause all previous ops to be applied"""

        @SingleQubitFusionInterpreter()
        def f(w1, w2):
            qml.H(w1)
            qml.Z(0)
            qml.CNOT([0, w2])

        dyn_wires = (0, 1)
        jaxpr = jax.make_jaxpr(f)(*dyn_wires)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *dyn_wires)

        ops = collector.state["ops"]
        expected_ops = [
            qml.H(0),
            qml.Rot(np.pi, 0, 0, 0),  # Z not fused but changed to Rot
            qml.CNOT([0, 1]),
        ]
        assert ops == expected_ops


class TestSingleQubitFusionHigherOrderPrimitives:
    """Unit tests for the single-qubit fusion transformation with higher order primitives."""

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

        # This circuit should be transformed to a single Rot(-4.37,1.98,-0.96) gate

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 1
        circuit_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert len(circuit_jaxpr.eqns) == 3
        assert circuit_jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert circuit_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert circuit_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.literals)

        with qml.capture.pause():
            # pylint: disable=not-callable
            expected_result = circuit()

        assert qml.math.allclose(result, expected_result)

    @pytest.mark.parametrize("grad_fn", [qml.grad, qml.jacobian])
    def test_grad_and_jac_higher_order_primitives(self, grad_fn):
        """Test that grad and jacobian higher order primitives are transformed correctly."""

        @SingleQubitFusionInterpreter()
        @qml.qnode(qml.device("default.qubit", wires=3))
        def circuit(input):
            qml.Hadamard(wires=0)
            qml.RZ(input[0], wires=0)
            qml.PauliY(wires=1)
            qml.RZ(input[1], wires=0)
            qml.CNOT(wires=[1, 2])
            qml.CRY(input[2], wires=[1, 2])
            qml.PauliZ(wires=0)
            qml.CRY(input[3], wires=[1, 2])
            qml.Rot(input[0], input[1], input[2], wires=1)
            qml.Rot(input[2], input[3], input[0], wires=1)
            return qml.expval(qml.PauliX(0) @ qml.PauliX(2))

        input = jax.numpy.array([0.1, 0.2, 0.3, 0.4])

        jaxpr = jax.make_jaxpr(grad_fn(circuit))(input)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == grad_prim if grad_fn == qml.grad else jacobian_prim

        # pylint: disable=inconsistent-return-statements
        def _find_eq_with_name(jaxpr, name):
            for eq in jaxpr.eqns:
                if name in eq.params:
                    return eq.params[name]

        inner_jaxpr = _find_eq_with_name(jaxpr, "jaxpr")
        assert len(inner_jaxpr.eqns) == 1
        qfunc_jaxpr = _find_eq_with_name(inner_jaxpr, "qfunc_jaxpr")
        qfunc_jaxpr = qfunc_jaxpr.replace(
            eqns=[
                eqn
                for eqn in qfunc_jaxpr.eqns
                if getattr(eqn.primitive, "prim_type", "") == "operator"
            ]
        )
        assert len(qfunc_jaxpr.eqns) == 9

        assert qfunc_jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.CNOT._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.CRY._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.CRY._primitive
        assert qfunc_jaxpr.eqns[4].primitive == qml.Rot._primitive
        assert qfunc_jaxpr.eqns[5].primitive == qml.Rot._primitive
        assert qfunc_jaxpr.eqns[6].primitive == qml.PauliX._primitive
        assert qfunc_jaxpr.eqns[7].primitive == qml.PauliX._primitive
        assert qfunc_jaxpr.eqns[8].primitive == qml.ops.Prod._primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.literals, input)[0]
        result_check = [-0.19037952, -0.19037946, 0.73068166, 0.7306816]
        assert qml.math.allclose(result, result_check)

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

        @SingleQubitFusionInterpreter()
        def circuit(x):

            def true_branch(x):
                qml.RX(x, wires=0)
                qml.S(wires=0)

            # pylint: disable=unused-argument
            def false_branch(x):
                qml.H(0)
                qml.H(1)

            qml.CNOT(wires=[0, 1])
            qml.cond(x > 0.5, true_branch, false_branch)(x)
            qml.CNOT(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)(selector)

        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[1].primitive == jax.lax.gt_p
        assert jaxpr.eqns[2].primitive == cond_prim
        assert jaxpr.eqns[3].primitive == qml.CNOT._primitive

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, selector)
        jaxpr_ops = collector.state["ops"]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qml.assert_equal(op1, op2, check_interface=False)

    def test_for_loop(self):
        """Test that operators inside a for loop are correctly fused."""

        @SingleQubitFusionInterpreter()
        def circuit(x):

            qml.CNOT(wires=[0, 1])

            @qml.for_loop(0, 3)
            # pylint: disable=unused-argument
            def loop(i, x):
                qml.RX(x, wires=0)
                qml.RZ(x, wires=0)
                return x

            # pylint: disable=no-value-for-parameter
            loop(x)

            qml.CNOT(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)(np.pi)
        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[1].primitive == for_loop_prim
        assert jaxpr.eqns[2].primitive == qml.CNOT._primitive

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, np.pi)
        jaxpr_ops = collector.state["ops"]
        assert len(jaxpr_ops) == 5

        expected_ops = [
            qml.CNOT(wires=[0, 1]),
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.CNOT(wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qml.assert_equal(op1, op2, check_interface=False)

    def test_while_loop(self):
        """Test that while operators inside a while loop are correctly fused."""

        @SingleQubitFusionInterpreter()
        def circuit():

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

        jaxpr = jax.make_jaxpr(circuit)()

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[1].primitive == while_loop_prim
        assert jaxpr.eqns[2].primitive == qml.CNOT._primitive

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]
        assert len(jaxpr_ops) == 5

        expected_ops = [
            qml.CNOT(wires=[0, 1]),
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.Rot(np.pi / 2, np.pi, np.pi / 2, wires=[0]),  # RX and RZ fused
            qml.CNOT(wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops, strict=True):
            qml.assert_equal(op1, op2)

    def test_mid_circuit_measurement(self):
        """Test that mid-circuit measurements are correctly handled."""

        @SingleQubitFusionInterpreter()
        def circuit():
            qml.RX(0.1, wires=0)
            qml.S(wires=0)
            qml.measure(0)
            qml.RX(0.1, wires=0)
            qml.S(wires=0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 5

        # I test the jaxpr like this because `qml.assert_equal`
        # has issues with mid-circuit measurements
        # (Got <class 'pennylane.measurements.mid_measure.MidMeasureMP'>
        # and <class 'pennylane.measurements.mid_measure.MeasurementValue'>.)
        assert jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[2].primitive == qml.Rot._primitive
        assert jaxpr.eqns[3].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive


class TestSingleQubitFusionPLXPR:
    """Unit tests for the single-qubit fusion transformation on PLXPRs."""

    def test_single_qubit_fusion_plxpr_to_plxpr(self):
        """Test that the single-qubit fusion transformation works on a plxpr."""

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
        transformed_jaxpr = single_qubit_plxpr_to_plxpr(
            jaxpr.jaxpr, jaxpr.consts, [], {"atol": 1e-5, "exclude_gates": "RY"}
        )
        assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
        assert len(transformed_jaxpr.eqns) == 3
        assert transformed_jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_expand_plxpr_transform(self):
        """Test that the transform works with expand_plxpr_transform"""

        @qml.transforms.optimization.single_qubit_fusion
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
        assert jaxpr.eqns[0].primitive == qml.transforms.optimization.single_qubit_fusion._primitive

        transformed_qfunc = qml.capture.expand_plxpr_transforms(circuit)
        transformed_jaxpr = jax.make_jaxpr(transformed_qfunc)()

        assert len(transformed_jaxpr.eqns) == 3
        assert transformed_jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_applying_plxpr_decorator(self):
        """Test that the single-qubit fusion transformation works when applying the plxpr decorator."""

        @qml.capture.expand_plxpr_transforms
        @qml.transforms.optimization.single_qubit_fusion
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
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive
