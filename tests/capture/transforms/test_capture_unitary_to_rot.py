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
"""Unit tests for the `UnitaryToRotInterpreter` class."""

# pylint:disable=protected-access, wrong-import-position

import pytest

import pennylane as qp

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    jacobian_prim,
    qnode_prim,
    transform_prim,
    while_loop_prim,
)
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.unitary_to_rot import (
    UnitaryToRotInterpreter,
    one_qubit_decomposition,
    two_qubit_decomposition,
    unitary_to_rot_plxpr_to_plxpr,
)

pytestmark = [pytest.mark.jax, pytest.mark.capture]


class TestUnitaryToRotInterpreter:
    """Unit tests for the UnitaryToRotInterpreter class for decomposing plxpr."""

    def test_one_qubit_conversion(self):
        """Test that a simple one qubit unitary can be decomposed correctly."""

        @UnitaryToRotInterpreter()
        def f(U):
            qp.QubitUnitary(U, 0)
            return qp.expval(qp.Z(0))

        U = qp.Rot(1.0, 2.0, 3.0, wires=0)
        jaxpr = jax.make_jaxpr(f)(U.matrix())

        # Qubit Unitary decomposition
        with qp.capture.pause():
            QU = qp.QubitUnitary(U.matrix(), 0)
            decomp = jax.jit(one_qubit_decomposition)(QU.parameters[0], QU.wires[0])
            assert len(decomp) > 1
        for i, eqn in enumerate(jaxpr.eqns[-len(decomp) : -2]):
            assert eqn.primitive == decomp[i]._primitive

        # Measurement
        assert jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

    # two_qubit_decomposition only supports decomps with
    # three CNOTs for abstract matrices
    def test_two_qubit_three_cnot_conversion(self):
        """Test that a two qubit unitary can be decomposed correctly."""
        U1 = qp.Rot(1.0, 2.0, 3.0, wires=0)
        U2 = qp.Rot(1.0, 2.0, 3.0, wires=1)
        U = qp.prod(U1, U2)

        @UnitaryToRotInterpreter()
        def f(U):
            qp.QubitUnitary(U, [0, 1])
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)(U.matrix())

        # Theoretical decomposition based on,
        # https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.two_qubit_decomposition.html

        with qp.capture.pause():
            QU = qp.QubitUnitary(U.matrix(), [0, 1])
            decomp = jax.jit(two_qubit_decomposition)(QU.parameters[0], QU.wires)
            assert len(decomp) > 1
        for i, eqn in enumerate(jaxpr.eqns[-len(decomp) - 2 : -2]):
            assert eqn.primitive == decomp[i]._primitive

        # Measurement
        assert jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_three_qubit_conversion(self):
        """Tests that no decomposition occurs since num_qubits > 2"""

        @UnitaryToRotInterpreter()
        def f(U):
            qp.QubitUnitary(U, [0, 1, 2])
            return qp.expval(qp.Z(0))

        U = qp.numpy.eye(8)
        jaxpr = jax.make_jaxpr(f)(U)

        assert jaxpr.eqns[0].primitive == qp.QubitUnitary._primitive

        # Measurement
        assert jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_traced_arguments(self):
        """Test that traced arguments are correctly handled."""

        @UnitaryToRotInterpreter()
        def f(U, wire):
            qp.QubitUnitary(U, wire)
            return qp.expval(qp.Z(0))

        U = qp.Rot(1.0, 2.0, 3.0, wires=0)
        args = (U.matrix(), 0)
        jaxpr = jax.make_jaxpr(f)(*args)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)

        expected_ops = [
            qp.RZ(jax.numpy.array(1.0), wires=[0]),
            qp.RY(jax.numpy.array(2.0), wires=[0]),
            qp.RZ(jax.numpy.array(3.0), wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qp.expval(qp.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas


def test_plxpr_to_plxpr():
    """Test that transforming plxpr works correctly."""

    def circuit(U):
        qp.QubitUnitary(U, 0)
        return qp.expval(qp.Z(0))

    U = qp.Rot(1.0, 2.0, 3.0, wires=0)
    args = (U.matrix(),)
    jaxpr = jax.make_jaxpr(circuit)(*args)
    transformed_jaxpr = unitary_to_rot_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {}, *args)

    assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)

    # Qubit Unitary decomposition
    with qp.capture.pause():
        QU = qp.QubitUnitary(U.matrix(), 0)
        decomp = jax.jit(one_qubit_decomposition)(QU.parameters[0], QU.wires[0])
        assert len(decomp) > 1

    for i, eqn in enumerate(transformed_jaxpr.eqns[-len(decomp) : -2]):
        assert eqn.primitive == decomp[i]._primitive

    # Measurement
    assert transformed_jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
    assert transformed_jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive


class TestHigherOrderPrimitiveIntegration:
    """Test that the transform works with higher order primitives."""

    def test_ctrl_higher_order_primitive(self):
        """Test that evaluating a ctrl higher order primitive works correctly"""

        def ctrl_fn(U):
            qp.QubitUnitary(U, 0)

        @UnitaryToRotInterpreter()
        def f(U):
            qp.RX(0, 1)
            qp.ctrl(ctrl_fn, [2, 3])(U)
            qp.RY(0, 1)

        jaxpr = jax.make_jaxpr(f)(qp.Rot(1.0, 2.0, 3.0, wires=0).matrix())
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == ctrl_transform_prim
        assert jaxpr.eqns[2].primitive == qp.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr"]
        assert inner_jaxpr.eqns[-3].primitive == qp.RZ._primitive
        assert inner_jaxpr.eqns[-2].primitive == qp.RY._primitive
        assert inner_jaxpr.eqns[-1].primitive == qp.RZ._primitive

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_higher_order_primitive(self, lazy):
        """Test that the adjoint primitive is correctly interpreted"""

        @UnitaryToRotInterpreter()
        def f(U):
            def g(matrix):
                qp.QubitUnitary(matrix, 0)

            qp.adjoint(g, lazy=lazy)(U)

        jaxpr = jax.make_jaxpr(f)(qp.Rot(1.0, 2.0, 3.0, wires=0).matrix())

        assert jaxpr.eqns[0].primitive == adjoint_transform_prim
        assert jaxpr.eqns[0].params["lazy"] == lazy

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        assert inner_jaxpr.eqns[-3].primitive == qp.RZ._primitive
        assert inner_jaxpr.eqns[-2].primitive == qp.RY._primitive
        assert inner_jaxpr.eqns[-1].primitive == qp.RZ._primitive

    def test_for_loop_higher_order_primitive(self):
        """Test that the for_loop primitive is correctly interpreted"""

        @UnitaryToRotInterpreter()
        def f(U, n):
            @qp.for_loop(n)
            def g(i):
                qp.QubitUnitary(U, i)

            g()

            return qp.expval(qp.Z(0))

        U = qp.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        args = (U, 3)
        jaxpr = jax.make_jaxpr(f)(*args)

        # Measurement
        assert jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

        # For loop jaxpr
        assert jaxpr.eqns[0].primitive == for_loop_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert inner_jaxpr.eqns[-3].primitive == qp.RZ._primitive
        assert inner_jaxpr.eqns[-2].primitive == qp.RY._primitive
        assert inner_jaxpr.eqns[-1].primitive == qp.RZ._primitive

    def test_while_loop_higher_order_primitive(self):
        """Test that the while_loop primitive is correctly interpreted"""

        @UnitaryToRotInterpreter()
        def f(U, n):
            @qp.while_loop(lambda i: i < n)
            def g(i):
                qp.QubitUnitary(U, i)
                return i + 1

            g(0)
            return qp.expval(qp.Z(0))

        U = qp.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        args = (U, 3)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == while_loop_prim
        # Measurement
        assert jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

        # While loop jaxpr
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        # Last primitive is the i+1 in the while loop
        assert inner_jaxpr.eqns[-4].primitive == qp.RZ._primitive
        assert inner_jaxpr.eqns[-3].primitive == qp.RY._primitive
        assert inner_jaxpr.eqns[-2].primitive == qp.RZ._primitive

    def test_cond_higher_order_primitive(self):
        """Test that the cond primitive is correctly interpreted"""

        @UnitaryToRotInterpreter()
        def f(U, n):
            @qp.cond(n > 0)
            def cond_f():
                qp.QubitUnitary(U, 0)
                return qp.expval(qp.Z(0))

            @cond_f.else_if(n > 1)
            def _else_if():
                qp.QubitUnitary(U, 1)
                return qp.expval(qp.Y(0))

            @cond_f.otherwise
            def _else():
                qp.QubitUnitary(U, 2)
                return qp.expval(qp.X(0))

            out = cond_f()
            return out

        U = qp.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        args = (U, 3)
        jaxpr = jax.make_jaxpr(f)(*args)
        # First 2 primitives are the conditions for the true and elif branches
        assert jaxpr.eqns[2].primitive == cond_prim

        # True branch
        branch_jaxpr = jaxpr.eqns[2].params["jaxpr_branches"][0]
        # Qubit unitary decomposition
        assert branch_jaxpr.eqns[-5].primitive == qp.RZ._primitive
        assert branch_jaxpr.eqns[-4].primitive == qp.RY._primitive
        assert branch_jaxpr.eqns[-3].primitive == qp.RZ._primitive
        # Make sure its on wire=0
        assert qp.math.allclose(branch_jaxpr.eqns[-3].invars[1].val, 0)
        # Measurement
        assert branch_jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert branch_jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

        # Elif branch
        branch_jaxpr = jaxpr.eqns[2].params["jaxpr_branches"][1]
        # Qubit unitary decomposition
        assert branch_jaxpr.eqns[-5].primitive == qp.RZ._primitive
        assert branch_jaxpr.eqns[-4].primitive == qp.RY._primitive
        assert branch_jaxpr.eqns[-3].primitive == qp.RZ._primitive
        # Make sure its on wire=1
        assert qp.math.allclose(branch_jaxpr.eqns[-3].invars[1].val, 1)
        # Measurement
        assert branch_jaxpr.eqns[-2].primitive == qp.PauliY._primitive
        assert branch_jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

        # Else branch
        branch_jaxpr = jaxpr.eqns[2].params["jaxpr_branches"][2]
        # Qubit unitary decomposition
        assert branch_jaxpr.eqns[-5].primitive == qp.RZ._primitive
        assert branch_jaxpr.eqns[-4].primitive == qp.RY._primitive
        assert branch_jaxpr.eqns[-3].primitive == qp.RZ._primitive
        # Make sure its on wire=2
        assert qp.math.allclose(branch_jaxpr.eqns[-3].invars[1].val, 2)
        # Measurement
        assert branch_jaxpr.eqns[-2].primitive == qp.PauliX._primitive
        assert branch_jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_grad_higher_order_primitive(self):
        """Test that the grad primitive are correctly interpreted"""
        dev = qp.device("default.qubit", wires=1)

        @UnitaryToRotInterpreter()
        def f(a, b, c):
            @qp.qnode(dev)
            def circuit(a, b, c):
                with qp.QueuingManager.stop_recording():
                    A = qp.Rot.compute_matrix(a, b, c)
                qp.QubitUnitary(A, 0)
                return qp.expval(qp.Z(0))

            return qp.grad(circuit)(a, b, c)

        jaxpr = jax.make_jaxpr(f)(1.0, 2.0, 3.0)

        assert jaxpr.eqns[0].primitive == jacobian_prim

        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-5].primitive == qp.RZ._primitive
        assert qfunc_jaxpr.eqns[-4].primitive == qp.RY._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qp.RZ._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_jac_higher_order_primitive(self):
        """Test that the jacobian primitive works correctly"""
        dev = qp.device("default.qubit", wires=1)

        @UnitaryToRotInterpreter()
        def f(a, b, c):
            @qp.qnode(dev)
            def circuit(a, b, c):
                with qp.QueuingManager.stop_recording():
                    A = qp.Rot.compute_matrix(a, b, c)
                qp.QubitUnitary(A, 0)
                return qp.expval(qp.Z(0))

            return qp.jacobian(circuit)(a, b, c)

        jaxpr = jax.make_jaxpr(f)(1.0, 2.0, 3.0)

        assert jaxpr.eqns[0].primitive == jacobian_prim
        assert not jaxpr.eqns[0].params["scalar_out"]

        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-5].primitive == qp.RZ._primitive
        assert qfunc_jaxpr.eqns[-4].primitive == qp.RY._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qp.RZ._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_qnode_higher_order_primitive(self):
        """Test that you can integrate the transform at the QNode level."""
        dev = qp.device("default.qubit", wires=1)

        @UnitaryToRotInterpreter()
        @qp.qnode(dev)
        def f(U):
            qp.QubitUnitary(U, 0)
            qp.X(0)
            return qp.expval(qp.Z(0))

        U = qp.Rot(jax.numpy.pi, 0, 0, wires=0)

        jaxpr = jax.make_jaxpr(f)(U.matrix())
        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        # Qubit Unitary decomposition
        with qp.capture.pause():
            QU = qp.QubitUnitary(U.matrix(), 0)
            decomp = jax.jit(one_qubit_decomposition)(QU.parameters[0], QU.wires[0])
            assert len(decomp) > 1
        for i, eqn in enumerate(qfunc_jaxpr.eqns[-len(decomp) - 3 : -3]):
            assert eqn.primitive == decomp[i]._primitive

        # X gate
        assert qfunc_jaxpr.eqns[-3].primitive == qp.PauliX._primitive

        # Measurement
        assert qfunc_jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, U.matrix())
        assert qp.math.allclose(res, -1.0)


class TestExpandPlxprTransformIntegration:
    """Test that the transform works with expand_plxpr_transform"""

    def test_example(self):
        """Test that the transform works with expand_plxpr_transform"""

        @qp.transforms.unitary_to_rot
        def f(U):
            qp.QubitUnitary(U, 0)
            return qp.expval(qp.Z(0))

        U = qp.Rot(1.0, 2.0, 3.0, wires=0)
        jaxpr = jax.make_jaxpr(f)(U.matrix())

        assert jaxpr.eqns[0].primitive == transform_prim
        assert jaxpr.eqns[0].params["transform"] == qp.transforms.unitary_to_rot

        transformed_f = qp.capture.expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(U.matrix())

        # Qubit Unitary decomposition
        with qp.capture.pause():
            QU = qp.QubitUnitary(U.matrix(), 0)
            decomp = jax.jit(one_qubit_decomposition)(QU.parameters[0], QU.wires[0])
            assert len(decomp) > 1
        for i, eqn in enumerate(transformed_jaxpr.eqns[-len(decomp) : -2]):
            assert eqn.primitive == decomp[i]._primitive

        # Measurement
        assert transformed_jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert transformed_jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_decorator(self):
        """Test that the transform works with the decorator"""

        @qp.capture.expand_plxpr_transforms
        @qp.transforms.unitary_to_rot
        def f(U):
            qp.QubitUnitary(U, 0)
            return qp.expval(qp.Z(0))

        U = qp.Rot(1.0, 2.0, 3.0, wires=0)
        jaxpr = jax.make_jaxpr(f)(U.matrix())

        # Qubit Unitary decomposition
        with qp.capture.pause():
            QU = qp.QubitUnitary(U.matrix(), 0)
            decomp = jax.jit(one_qubit_decomposition)(QU.parameters[0], QU.wires[0])
            assert len(decomp) > 1
        for i, eqn in enumerate(jaxpr.eqns[-len(decomp) : -2]):
            assert eqn.primitive == decomp[i]._primitive

        # Measurement
        assert jaxpr.eqns[-2].primitive == qp.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive
