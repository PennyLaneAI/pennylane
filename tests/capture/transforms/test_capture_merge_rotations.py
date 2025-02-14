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

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.transforms.optimization.merge_rotations import (
    MergeRotationsInterpreter,
    merge_rotations_plxpr_to_plxpr,
)

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class TestMergeRotationsInterpreter:
    """Test the MergeRotationsInterpreter class."""

    def test_traced_arguments(self):
        """Test that traced arguments work fine."""

        @MergeRotationsInterpreter()
        def f(a, b, wires):
            qml.RX(a, wires=wires)
            qml.RY(a, wires=2)
            qml.RX(b, wires=wires)
            return qml.expval(qml.PauliZ(0))

        args = (0.1, 0.2, 0)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert jaxpr.eqns[-3].primitive == qml.RX._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

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
        assert jaxpr.eqns[-4].primitive != qml.RX._primitive
        assert jaxpr.eqns[-3].primitive == qml.RX._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_two_qubit_rotation_no_merge(self):
        """Test that rotations on two different qubits are not merged."""

        @MergeRotationsInterpreter()
        def f():
            qml.RX(1, wires=1)
            qml.RX(2, wires=0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[-4].primitive == qml.RX._primitive
        assert jaxpr.eqns[-3].primitive == qml.RX._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_one_qubit_rotation_blocked(self):
        """Test that a single qubit rotation is not merged if it is blocked."""

        @MergeRotationsInterpreter()
        def f():
            qml.RX(1, wires=0)
            qml.Hadamard(wires=0)
            qml.RX(2, wires=0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 5
        assert jaxpr.eqns[-5].primitive == qml.RX._primitive
        assert jaxpr.eqns[-4].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[-3].primitive == qml.RX._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_one_qubit_blocked_by_two_qubit(self):
        """Test that a single qubit rotation is not merged if it is blocked by a two qubit operation."""

        @MergeRotationsInterpreter()
        def f():
            qml.RX(1, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(2, wires=0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 5
        assert jaxpr.eqns[-5].primitive == qml.RX._primitive
        assert jaxpr.eqns[-4].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[-3].primitive == qml.RX._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_two_qubits_merge_with_gate_subset(self):
        """Test that specifying a subset of operations to include works correctly."""

        @MergeRotationsInterpreter(include_gates=["RX", "CRX"])
        def f():
            qml.CRX(1, wires=[0, 1])
            qml.CRX(1, wires=[0, 1])
            qml.RY(1, wires=[3])
            qml.RY(1, wires=[3])
            qml.RX(1, wires=[2])
            qml.RX(1, wires=[2])
            qml.RZ(1, wires=[2])
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 7
        assert jaxpr.eqns[-7].primitive == qml.RY._primitive
        assert jaxpr.eqns[-6].primitive == qml.RX._primitive
        assert jaxpr.eqns[-5].primitive == qml.RY._primitive
        assert jaxpr.eqns[-4].primitive == qml.CRX._primitive
        assert jaxpr.eqns[-3].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_controlled_rotation_merge(self):
        """Test that a controlled rotation is correctly merged."""

        @MergeRotationsInterpreter()
        def f():
            qml.CRY(1, wires=[0, 1])
            qml.CRY(1, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[-3].primitive == qml.CRY._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_controlled_rotation_no_merge(self):
        """Test that a controlled rotation is not merged if the control is different."""

        @MergeRotationsInterpreter()
        def f():
            qml.CRY(1, wires=[0, 1])
            qml.CRY(1, wires=[1, 0])
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[-4].primitive == qml.CRY._primitive
        assert jaxpr.eqns[-3].primitive == qml.CRY._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_merge_rotations_with_non_commuting_observables(self):

        @MergeRotationsInterpreter()
        def f():
            qml.RX(1, wires=0)
            qml.RX(1, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 5
        assert jaxpr.eqns[-5].primitive == qml.RX._primitive
        assert jaxpr.eqns[-4].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-3].primitive == qml.measurements.ExpectationMP._obs_primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliX._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive


@pytest.mark.parametrize(("theta1, theta2"), [(0.1, 0.2), (0.1, -0.1)])
def merge_rotations_plxpr_to_plxpr_transform(theta1, theta2):
    """Test that the merge_rotations_plxpr_to_plxpr function works correctly."""

    @MergeRotationsInterpreter()
    def f(a, b):
        qml.RX(a, wires=0)
        qml.RX(b, wires=0)
        return qml.expval(qml.PauliZ(0))

    args = (theta1, theta2)
    jaxpr = jax.make_jaxpr(f)(*args)
    transformed_jaxpr = merge_rotations_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {}, *args)

    assert isinstance(transformed_jaxpr, jax.core.ClosedJaxpr)

    assert jaxpr.eqns[-4].primitive != qml.RX._primitive
    assert jaxpr.eqns[-3].primitive == qml.RX._primitive
    assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
    assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive


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
            qml.ctrl(ctrl_fn, [])()
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

        jaxpr = jax.make_jaxpr(f)(1.0, 2.0)

        assert jaxpr.eqns[0].primitive == grad_prim

        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-4].primitive != qml.RX._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

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

        jaxpr = jax.make_jaxpr(f)(1.0, 2.0)

        assert jaxpr.eqns[0].primitive == jacobian_prim

        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-4].primitive != qml.RX._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_qnode_higher_order_primitive(self):
        """Test that the qnode primitive is correctly interpreted"""
        dev = qml.device("default.qubit", wires=1)

        @MergeRotationsInterpreter()
        @qml.qnode(dev)
        def f(a, b):
            qml.RX(a, 0)
            qml.RX(b, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.0, 2.0)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-4].primitive != qml.RX._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive


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
        assert len(transformed_jaxpr.eqns) == 6
        assert transformed_jaxpr.eqns[0].primitive == qml.RY._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.CNOT._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.RY._primitive
        assert transformed_jaxpr.eqns[3].primitive == qml.RX._primitive
        assert transformed_jaxpr.eqns[4].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[5].primitive == qml.measurements.ExpectationMP._obs_primitive

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
        assert len(jaxpr.eqns) == 6
        assert jaxpr.eqns[0].primitive == qml.RY._primitive
        assert jaxpr.eqns[1].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[2].primitive == qml.RY._primitive
        assert jaxpr.eqns[3].primitive == qml.RX._primitive
        assert jaxpr.eqns[4].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[5].primitive == qml.measurements.ExpectationMP._obs_primitive
