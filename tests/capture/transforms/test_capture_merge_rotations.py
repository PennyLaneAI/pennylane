# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the `MergeRotationsInterpreter` class."""

# pylint:disable=protected-access, wrong-import-position

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pennylane as qml
from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    measure_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.optimization.merge_rotations import (
    MergeRotationsInterpreter,
    merge_rotations_plxpr_to_plxpr,
)
from pennylane.transforms.optimization.optimization_utils import fuse_rot_angles

pytestmark = [pytest.mark.jax, pytest.mark.capture]


class TestMergeRotationsInterpreter:
    """Test the MergeRotationsInterpreter class."""

    def test_traced_arguments(self):
        """Test that traced arguments work fine."""

        @MergeRotationsInterpreter()
        def f(a, b, c, wires):
            qml.RX(a, wires=wires)
            qml.RX(b, wires=wires)
            qml.RY(a, wires=2)
            qml.Rot(0, 0, c, wires=1)
            qml.Rot(0, 0, c, wires=1)
            return qml.expval(qml.PauliZ(0))

        a, b, c = 0.1, 0.2, 0.3
        args = (a, b, c, 0)
        jaxpr = jax.make_jaxpr(f)(*args)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)

        expected_ops = [
            qml.RX(jnp.array(a + b), wires=[0]),
            qml.RY(a, wires=[2]),
            # Two rotation gates merge to: RZ(c) RY(0) RZ(c)
            qml.Rot(jnp.array(c), jnp.array(0), jnp.array(c), wires=[1]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_rot_gate_traced_arguments(self):
        """Test that a qml.Rot gate is correctly merged when using traced arguments"""

        @MergeRotationsInterpreter()
        def circuit(angles1, angles2):
            qml.Rot(*angles1, wires=1)
            qml.Rot(*angles2, wires=1)
            return qml.expval(qml.PauliZ(0))

        angles1 = (1, 2, 3)
        angles2 = (4, 5, 6)
        jaxpr = jax.make_jaxpr(circuit)(angles1, angles2)
        args = (angles1, angles2)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *jax.tree_util.tree_leaves(args))

        expected_angles = fuse_rot_angles(angles1, angles2)
        expected_ops = [
            qml.Rot(
                jnp.array(expected_angles[0]),
                jnp.array(expected_angles[1]),
                jnp.array(expected_angles[2]),
                wires=[1],
            ),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_rot_gate(self):
        """Test that a qml.Rot gate is correctly merged if the angles are fixed."""

        @MergeRotationsInterpreter()
        def circuit():
            qml.Rot(1, 2, 3, wires=1)
            qml.Rot(4, 5, 6, wires=1)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_angles = fuse_rot_angles((1, 2, 3), (4, 5, 6))
        expected_ops = [
            qml.Rot(
                expected_angles[0],
                expected_angles[1],
                expected_angles[2],
                wires=[1],
            ),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    @pytest.mark.parametrize(("theta1, theta2"), [(0.1, 0.2), (0.1, -0.1)])
    def test_one_qubit_merge(self, theta1, theta2):
        """Test that a single qubit rotation is correctly merged."""

        @MergeRotationsInterpreter()
        def f(a, b):
            qml.RX(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        args = (theta1, theta2)
        jaxpr = jax.make_jaxpr(f)(*args)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)

        # Since arguments are traced, operator will still show up
        # even if the angle is 0.0.
        expected_ops = [
            qml.RX(jnp.array(theta1 + theta2), wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_two_qubit_rotation_no_merge(self):
        """Test that rotations on two different qubits are not merged."""

        @MergeRotationsInterpreter()
        def f():
            qml.RX(1, wires=1)
            qml.RX(2, wires=0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = [
            qml.RX(1, wires=[1]),
            qml.RX(2, wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_one_qubit_rotation_blocked(self):
        """Test that a single qubit rotation is not merged if it is blocked."""

        @MergeRotationsInterpreter()
        def f():
            qml.RX(1, wires=0)
            qml.Hadamard(wires=0)
            qml.RX(2, wires=0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = [
            qml.RX(1, wires=[0]),
            qml.Hadamard(wires=[0]),
            qml.RX(2, wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_one_qubit_blocked_by_two_qubit(self):
        """Test that a single qubit rotation is not merged if it is blocked by a two qubit operation."""

        @MergeRotationsInterpreter()
        def f():
            qml.RX(1, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(2, wires=0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = [
            qml.RX(1, wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.RX(2, wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_controlled_rotation_merge(self):
        """Test that a controlled rotation is correctly merged."""

        @MergeRotationsInterpreter()
        def f():
            qml.CRY(1, wires=[0, 1])
            qml.CRY(1, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = [
            qml.CRY(2, wires=[0, 1]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_controlled_rotation_no_merge(self):
        """Test that a controlled rotation is not merged if the control is different."""

        @MergeRotationsInterpreter()
        def f():
            qml.CRY(1, wires=[0, 1])
            qml.CRY(1, wires=[1, 0])
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = [
            qml.CRY(1, wires=[0, 1]),
            qml.CRY(1, wires=[1, 0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_merge_rotations_with_non_commuting_observables(self):
        """Test that rotations are merged if the observables are non-commuting."""

        @MergeRotationsInterpreter()
        def f():
            qml.RX(1, wires=0)
            qml.RX(1, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = [
            qml.RX(2, wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_returned_rotation_not_merged(self):
        """Test that a rotation is not merged if it is returned."""

        @MergeRotationsInterpreter()
        def f(a, b):
            qml.RX(a, wires=0)
            return qml.RX(b, wires=0)

        args = (1, 2)
        jaxpr = jax.make_jaxpr(f)(*args)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)

        expected_ops = [
            qml.RX(1, wires=[0]),
            qml.RX(2, wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

    def test_include_gates_kwarg(self):
        """Test that specifying a subset of operations to include works correctly."""

        @MergeRotationsInterpreter(include_gates=["RX", "CRX"])
        def f():
            qml.CRX(1, wires=[0, 1])
            qml.RY(1, wires=[3])
            qml.RX(1, wires=[2])
            qml.RY(1, wires=[3])
            qml.RX(1, wires=[2])
            qml.CRX(1, wires=[0, 1])
            qml.RZ(1, wires=[2])
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        # Order is slightly different, but remains correct.
        expected_ops = [
            qml.RY(1, wires=[3]),
            qml.RX(2, wires=[2]),
            qml.CRX(2, wires=[0, 1]),
            qml.RY(1, wires=[3]),
            qml.RZ(1, wires=[2]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_atol_kwarg(self):
        """Test that the atol keyword argument works correctly."""

        @MergeRotationsInterpreter(atol=1e-3)
        def f():
            qml.RX(1, wires=0)
            qml.RX(-(1 + 1e-4), wires=0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = []

        ops = collector.state["ops"]
        assert ops == expected_ops

    def test_dynamic_wires_between_static_wires(self):
        """Test that operations with dynamic wires between operations with static
        wires cause merging to not happen."""

        @MergeRotationsInterpreter()
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

    def test_same_dyn_wires_merge(self):
        """Test that ops on the same dynamic wires get merged."""

        @MergeRotationsInterpreter()
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

        expected_ops = [qml.RX(2.5, 0), qml.RY(jnp.array(7.0), 0), qml.RX(2.5, 0)]
        expected_meas = [qml.expval(qml.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_different_dyn_wires_interleaved(self):
        """Test that ops on different dynamic wires interleaved with each other
        do not merge."""

        @MergeRotationsInterpreter()
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


@pytest.mark.parametrize(("theta1, theta2"), [(0.1, 0.2), (0.1, -0.1)])
def test_merge_rotations_plxpr_to_plxpr_transform(theta1, theta2):
    """Test that the merge_rotations_plxpr_to_plxpr function works correctly."""

    @MergeRotationsInterpreter()
    def f(a, b):
        qml.RX(a, wires=0)
        qml.RX(b, wires=0)
        return qml.expval(qml.PauliZ(0))

    args = (theta1, theta2)
    jaxpr = jax.make_jaxpr(f)(*args)
    transformed_jaxpr = merge_rotations_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {}, *args)

    assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
    collector = CollectOpsandMeas()
    collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)

    # Order is slightly different, but remains correct.
    expected_ops = [
        qml.RX(jnp.array(theta1 + theta2), wires=[0]),
    ]

    ops = collector.state["ops"]
    assert ops == expected_ops

    expected_meas = [
        qml.expval(qml.PauliZ(0)),
    ]
    meas = collector.state["measurements"]
    assert meas == expected_meas


class TestHigherOrderPrimitiveIntegration:
    """Integration tests for the higher order primitives with the MergeRotationsInterpreter."""

    def test_ctrl_higher_order_primitive(self):
        """Test that evaluating a ctrl higher order primitive works correctly"""

        def ctrl_fn():
            qml.RX(1, 0)
            qml.RX(1, 0)

        @MergeRotationsInterpreter()
        def f():
            qml.RY(0, 1)
            qml.ctrl(ctrl_fn, [2])()
            qml.RZ(0, 1)

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.RY._primitive
        assert jaxpr.eqns[1].primitive == ctrl_transform_prim
        assert jaxpr.eqns[2].primitive == qml.RZ._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr"]
        assert len(inner_jaxpr.eqns) == 1
        assert inner_jaxpr.eqns[0].primitive == qml.RX._primitive

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_higher_order_primitive(self, lazy):
        """Test that the adjoint primitive is correctly interpreted"""

        @MergeRotationsInterpreter()
        def f():
            def g():
                qml.RX(1, 0)
                qml.RX(1, 0)

            qml.adjoint(g, lazy=lazy)()

        jaxpr = jax.make_jaxpr(f)()
        assert jaxpr.eqns[0].primitive == adjoint_transform_prim
        assert jaxpr.eqns[0].params["lazy"] == lazy

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        assert len(inner_jaxpr.eqns) == 1
        assert inner_jaxpr.eqns[0].primitive == qml.RX._primitive

    def test_for_loop_higher_order_primitive(self):
        """Test that the for_loop primitive is correctly interpreted"""

        @MergeRotationsInterpreter()
        def f(n):
            @qml.for_loop(n)
            def g(i):
                qml.RX(1, i)
                qml.RX(1, i)

            g()

            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(3)

        # Measurement
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        # For loop jaxpr
        assert jaxpr.eqns[0].primitive == for_loop_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 1
        assert inner_jaxpr.eqns[0].primitive == qml.RX._primitive

    def test_while_loop_higher_order_primitive(self):
        """Test that the while_loop primitive is correctly interpreted"""

        @MergeRotationsInterpreter()
        def f(n):
            @qml.while_loop(lambda i: i < n)
            def g(i):
                qml.RX(1, 0)
                qml.RX(1, 0)
                return i + 1

            g(0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(3)
        assert jaxpr.eqns[0].primitive == while_loop_prim
        # Measurement
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        # While loop jaxpr
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        # Last primitive is the i+1 in the while loop
        assert inner_jaxpr.eqns[-1].primitive == qml.RX._primitive

    def test_cond_higher_order_primitive(self):
        """Test that the cond primitive is correctly interpreted"""

        @MergeRotationsInterpreter()
        def f(n):
            @qml.cond(n > 0)
            def cond_f():
                qml.RZ(1, 0)
                qml.RZ(1, 0)
                return qml.expval(qml.Z(0))

            @cond_f.else_if(n > 1)
            def _():
                qml.RY(1, 0)
                qml.RY(1, 0)
                return qml.expval(qml.Y(0))

            @cond_f.otherwise
            def _():
                qml.RX(1, 0)
                qml.RX(1, 0)
                return qml.expval(qml.X(0))

            out = cond_f()
            return out

        jaxpr = jax.make_jaxpr(f)(3)
        # First 2 primitives are the conditions for the true and elif branches
        assert jaxpr.eqns[2].primitive == cond_prim

        # True branch
        branch_jaxpr = jaxpr.eqns[2].params["jaxpr_branches"][0]
        assert len(branch_jaxpr.eqns) == 3
        assert branch_jaxpr.eqns[-3].primitive == qml.RZ._primitive
        # Measurement
        assert branch_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert branch_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        # Elif branch
        branch_jaxpr = jaxpr.eqns[2].params["jaxpr_branches"][1]
        assert len(branch_jaxpr.eqns) == 3
        assert branch_jaxpr.eqns[-3].primitive == qml.RY._primitive
        # Measurement
        assert branch_jaxpr.eqns[-2].primitive == qml.PauliY._primitive
        assert branch_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        # Else branch
        branch_jaxpr = jaxpr.eqns[2].params["jaxpr_branches"][2]
        assert len(branch_jaxpr.eqns) == 3
        assert branch_jaxpr.eqns[-3].primitive == qml.RX._primitive
        # Measurement
        assert branch_jaxpr.eqns[-2].primitive == qml.PauliX._primitive
        assert branch_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_grad_higher_order_primitive(self):
        """Test that the grad primitive are correctly interpreted"""
        dev = qml.device("default.qubit", wires=1)

        @MergeRotationsInterpreter()
        def f(a, b):
            @qml.qnode(dev)
            def circuit(a, b):
                qml.RX(a, 0)
                qml.RX(b, 0)
                return qml.expval(qml.Z(0))

            return qml.grad(circuit)(a, b)

        args = (1.0, 2.0)
        jaxpr = jax.make_jaxpr(f)(*args)

        assert jaxpr.eqns[0].primitive == grad_prim

        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        collector = CollectOpsandMeas()
        collector.eval(qfunc_jaxpr, jaxpr.consts, *args)
        expected_ops = [
            qml.RX(jnp.array(3.0), wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_jac_higher_order_primitive(self):
        """Test that the jacobian primitive works correctly"""
        dev = qml.device("default.qubit", wires=1)

        @MergeRotationsInterpreter()
        def f(a, b):
            @qml.qnode(dev)
            def circuit(a, b):
                qml.RX(a, 0)
                qml.RX(b, 0)
                return qml.expval(qml.Z(0))

            return qml.jacobian(circuit)(a, b)

        args = (1.0, 2.0)
        jaxpr = jax.make_jaxpr(f)(*args)

        assert jaxpr.eqns[0].primitive == jacobian_prim

        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        collector = CollectOpsandMeas()
        collector.eval(qfunc_jaxpr, jaxpr.consts, *args)
        expected_ops = [
            qml.RX(jnp.array(3.0), wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_qnode_higher_order_primitive(self):
        """Test that the qnode primitive is correctly interpreted"""
        dev = qml.device("default.qubit", wires=1)

        @MergeRotationsInterpreter()
        @qml.qnode(dev)
        def f(a, b):
            qml.RX(a, 0)
            qml.RX(b, 0)
            return qml.expval(qml.Z(0))

        args = (1.0, 2.0)
        jaxpr = jax.make_jaxpr(f)(*args)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        collector = CollectOpsandMeas()
        collector.eval(qfunc_jaxpr, jaxpr.consts, *args)
        expected_ops = [
            qml.RX(jnp.array(3.0), wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_mid_circuit_measurement_prim(self):
        """Test that mid-circuit measurements are correctly handled."""

        @MergeRotationsInterpreter()
        def circuit():
            qml.RX(0.1, wires=0)
            qml.RX(0.1, wires=0)
            qml.measure(0)
            qml.RX(0.1, wires=0)
            qml.RX(0.1, wires=0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 5

        # I test the jaxpr like this because `qml.assert_equal`
        # has issues with mid-circuit measurements
        # (Got <class 'pennylane.measurements.mid_measure.MidMeasureMP'>
        # and <class 'pennylane.measurements.mid_measure.MeasurementValue'>.)
        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[2].primitive == qml.RX._primitive
        assert jaxpr.eqns[3].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive


class TestExpandPlxprTransformIntegration:
    """Test the expand_plxpr_transforms function with the MergeRotationsInterpreter."""

    def test_example(self):
        """Test that the transform works with expand_plxpr_transform"""

        @qml.transforms.optimization.merge_rotations
        def qfunc():
            qml.RX(1, wires=0)
            qml.RY(2, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(1, wires=0)
            qml.RY(2, wires=1)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()

        assert jaxpr.eqns[0].primitive == qml.transforms.optimization.merge_rotations._primitive

        transformed_qfunc = qml.capture.expand_plxpr_transforms(qfunc)
        transformed_jaxpr = jax.make_jaxpr(transformed_qfunc)()
        collector = CollectOpsandMeas()
        collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts)

        expected_ops = [
            qml.RY(2, wires=[1]),
            qml.CNOT(wires=[1, 2]),
            qml.RX(2, wires=[0]),
            qml.RY(2, wires=[1]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_decorator(self):
        """Test that the transform works with the decorator"""

        @qml.capture.expand_plxpr_transforms
        @qml.transforms.optimization.merge_rotations
        def qfunc():
            qml.RX(1, wires=0)
            qml.RY(2, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(1, wires=0)
            qml.RY(2, wires=1)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = [
            qml.RY(2, wires=[1]),
            qml.CNOT(wires=[1, 2]),
            qml.RX(2, wires=[0]),
            qml.RY(2, wires=[1]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qml.expval(qml.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas
