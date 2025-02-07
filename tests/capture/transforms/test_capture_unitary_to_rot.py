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
from pennylane.transforms.unitary_to_rot import (
    UnitaryToRotInterpreter,
    unitary_to_rot_plxpr_to_plxpr,
)

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class TestUnitaryToRotInterpreter:
    """Unit tests for the UnitaryToRotInterpreter class for decomposing plxpr."""

    def test_one_qubit_conversion(self):
        """Test that a simple one qubit unitary can be decomposed correctly."""

        @UnitaryToRotInterpreter()
        def f(U):
            qml.QubitUnitary(U, 0)
            return qml.expval(qml.Z(0))

        U = qml.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        jaxpr = jax.make_jaxpr(f)(U)

        # Qubit Unitary decomposition
        assert jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert jaxpr.eqns[-3].primitive == qml.RZ._primitive

        # Measurement
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    # two_qubit_decomposition only supports decomps with
    # three CNOTs for abstract matrices
    def test_two_qubit_three_cnot_conversion(self):
        """Test that a two qubit unitary can be decomposed correctly."""
        U1 = qml.Rot(1.0, 2.0, 3.0, wires=0)
        U2 = qml.Rot(1.0, 2.0, 3.0, wires=1)
        U = qml.prod(U1, U2).matrix()

        @UnitaryToRotInterpreter()
        def f(U):
            qml.QubitUnitary(U, [0, 1])
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(U)

        # Theoretical decomposition based on,
        # https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.two_qubit_decomposition.html

        # C
        assert jaxpr.eqns[-20].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-19].primitive == qml.RY._primitive
        assert jaxpr.eqns[-18].primitive == qml.RZ._primitive

        # D
        assert jaxpr.eqns[-17].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-16].primitive == qml.RY._primitive
        assert jaxpr.eqns[-15].primitive == qml.RZ._primitive

        # CNOT 1
        assert jaxpr.eqns[-14].primitive == qml.CNOT._primitive

        # RZ RY
        assert jaxpr.eqns[-13].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-12].primitive == qml.RY._primitive

        # CNOT 2
        assert jaxpr.eqns[-11].primitive == qml.CNOT._primitive

        # RY
        assert jaxpr.eqns[-10].primitive == qml.RY._primitive

        # CNOT 3
        assert jaxpr.eqns[-9].primitive == qml.CNOT._primitive

        # A
        assert jaxpr.eqns[-8].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-7].primitive == qml.RY._primitive
        assert jaxpr.eqns[-6].primitive == qml.RZ._primitive

        # B
        assert jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert jaxpr.eqns[-3].primitive == qml.RZ._primitive

        # Measurement
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive


class TestQNodeIntegration:
    """Test that transform works at the QNode level."""

    def test_one_qubit_conversion_qnode(self):
        """Test that you can integrate the transform at the QNode level."""
        dev = qml.device("default.qubit", wires=1)

        @UnitaryToRotInterpreter()
        @qml.qnode(dev)
        def f(U):
            qml.QubitUnitary(U, 0)
            qml.X(0)
            return qml.expval(qml.Z(0))

        U = qml.Rot(jax.numpy.pi, 0, 0, wires=0).matrix()

        jaxpr = jax.make_jaxpr(f)(U)
        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        # Qubit unitary decomposition
        assert qfunc_jaxpr.eqns[-6].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-5].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-4].primitive == qml.RZ._primitive

        # X gate
        assert qfunc_jaxpr.eqns[-3].primitive == qml.PauliX._primitive

        # Measurement
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, U)
        assert qml.math.allclose(res, -1.0)

    # two_qubit_decomposition only supports decomps with
    # three CNOTs for abstract matrices
    def test_two_qubit_three_cnot_conversion_qnode(self):
        """Test that a two qubit unitary can be decomposed correctly."""
        dev = qml.device("default.qubit", wires=2)

        U1 = qml.Rot(jax.numpy.pi, 0, 0, wires=0)
        U2 = qml.Rot(jax.numpy.pi, 0, 0, wires=1)

        U = qml.prod(U1, U2).matrix()

        @UnitaryToRotInterpreter()
        @qml.qnode(dev)
        def f(U):
            qml.QubitUnitary(U, [0, 1])
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        jaxpr = jax.make_jaxpr(f)(U)
        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        # Theoretical decomposition based on,
        # https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.two_qubit_decomposition.html

        # C
        assert qfunc_jaxpr.eqns[-22].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-21].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-20].primitive == qml.RZ._primitive

        # D
        assert qfunc_jaxpr.eqns[-19].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-18].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-17].primitive == qml.RZ._primitive

        # CNOT 1
        assert qfunc_jaxpr.eqns[-16].primitive == qml.CNOT._primitive

        # RZ RY
        assert qfunc_jaxpr.eqns[-15].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-14].primitive == qml.RY._primitive

        # CNOT 2
        assert qfunc_jaxpr.eqns[-13].primitive == qml.CNOT._primitive

        # RY
        assert qfunc_jaxpr.eqns[-12].primitive == qml.RY._primitive

        # CNOT 3
        assert qfunc_jaxpr.eqns[-11].primitive == qml.CNOT._primitive

        # A
        assert qfunc_jaxpr.eqns[-10].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-9].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-8].primitive == qml.RZ._primitive

        # B
        assert qfunc_jaxpr.eqns[-7].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-6].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-5].primitive == qml.RZ._primitive

        # Measurement 1
        assert qfunc_jaxpr.eqns[-4].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.measurements.ExpectationMP._obs_primitive

        # Measurement 2
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, U)
        assert qml.math.allclose(res, (1.0, 1.0))


class TestUnitaryToRotPlxprTransform:
    """Tests that transforming plxpr works correctly."""

    def test_one_qubit_plxpr_transform(self):
        """Test that transforming plxpr works correctly."""

        def circuit(U):
            qml.QubitUnitary(U, 0)
            return qml.expval(qml.Z(0))

        U = qml.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        args = (U,)
        jaxpr = jax.make_jaxpr(circuit)(*args)
        transformed_jaxpr = unitary_to_rot_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {}, *args)

        assert isinstance(transformed_jaxpr, jax.core.ClosedJaxpr)

        # Qubit unitary decomposition
        assert transformed_jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert transformed_jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert transformed_jaxpr.eqns[-3].primitive == qml.RZ._primitive

        # Measurement
        assert transformed_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    # two_qubit_decomposition only supports decomps with
    # three CNOTs for abstract matrices
    def test_two_qubit_three_cnot_plxpr_transform(self):
        """Test that a two qubit unitary can be decomposed correctly."""

        def circuit(U):
            qml.QubitUnitary(U, [0, 1])
            return qml.expval(qml.Z(0))

        U1 = qml.Rot(1.0, 2.0, 3.0, wires=0)
        U2 = qml.Rot(1.0, 2.0, 3.0, wires=1)
        U = qml.prod(U1, U2).matrix()
        args = (U,)
        jaxpr = jax.make_jaxpr(circuit)(*args)
        transformed_jaxpr = unitary_to_rot_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {}, *args)

        # Theoretical decomposition based on,
        # https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.two_qubit_decomposition.html

        # C
        assert transformed_jaxpr.eqns[-20].primitive == qml.RZ._primitive
        assert transformed_jaxpr.eqns[-19].primitive == qml.RY._primitive
        assert transformed_jaxpr.eqns[-18].primitive == qml.RZ._primitive

        # D
        assert transformed_jaxpr.eqns[-17].primitive == qml.RZ._primitive
        assert transformed_jaxpr.eqns[-16].primitive == qml.RY._primitive
        assert transformed_jaxpr.eqns[-15].primitive == qml.RZ._primitive

        # CNOT 1
        assert transformed_jaxpr.eqns[-14].primitive == qml.CNOT._primitive

        # RZ RY
        assert transformed_jaxpr.eqns[-13].primitive == qml.RZ._primitive
        assert transformed_jaxpr.eqns[-12].primitive == qml.RY._primitive

        # CNOT 2
        assert transformed_jaxpr.eqns[-11].primitive == qml.CNOT._primitive

        # RY
        assert transformed_jaxpr.eqns[-10].primitive == qml.RY._primitive

        # CNOT 3
        assert transformed_jaxpr.eqns[-9].primitive == qml.CNOT._primitive

        # A
        assert transformed_jaxpr.eqns[-8].primitive == qml.RZ._primitive
        assert transformed_jaxpr.eqns[-7].primitive == qml.RY._primitive
        assert transformed_jaxpr.eqns[-6].primitive == qml.RZ._primitive

        # B
        assert transformed_jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert transformed_jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert transformed_jaxpr.eqns[-3].primitive == qml.RZ._primitive

        # Measurement
        assert transformed_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive


class TestHigherOrderPrimitiveIntegration:
    """Test that the transform works with higher order primitives."""

    def test_ctrl_higher_order_primitive(self):
        """Test that evaluating a ctrl higher order primitive works correctly"""

        def ctrl_fn(U):
            qml.QubitUnitary(U, 0)

        @UnitaryToRotInterpreter()
        def f(U):
            qml.RX(0, 1)
            qml.ctrl(ctrl_fn, [2, 3])(U)
            qml.RY(0, 1)

        jaxpr = jax.make_jaxpr(f)(qml.Rot(1.0, 2.0, 3.0, wires=0).matrix())
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == ctrl_transform_prim
        assert jaxpr.eqns[2].primitive == qml.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr"]
        assert inner_jaxpr.eqns[-3].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[-2].primitive == qml.RY._primitive
        assert inner_jaxpr.eqns[-1].primitive == qml.RZ._primitive

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_higher_order_primitive(self, lazy):
        """Test that the adjoint primitive is correctly interpreted"""

        @UnitaryToRotInterpreter()
        def f(U):
            def g(matrix):
                qml.QubitUnitary(matrix, 0)

            qml.adjoint(g, lazy=lazy)(U)

        jaxpr = jax.make_jaxpr(f)(qml.Rot(1.0, 2.0, 3.0, wires=0).matrix())

        assert jaxpr.eqns[0].primitive == adjoint_transform_prim
        assert jaxpr.eqns[0].params["lazy"] == lazy

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        assert inner_jaxpr.eqns[-3].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[-2].primitive == qml.RY._primitive
        assert inner_jaxpr.eqns[-1].primitive == qml.RZ._primitive

    def test_for_loop_higher_order_primitive(self):
        """Test that the for_loop primitive is correctly interpreted"""

        @UnitaryToRotInterpreter()
        def f(U, n):
            @qml.for_loop(n)
            def g(i):
                qml.QubitUnitary(U, i)

            g()

            return qml.expval(qml.Z(0))

        U = qml.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        args = (U, 3)
        jaxpr = jax.make_jaxpr(f)(*args)

        # Measurement
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        # For loop jaxpr
        assert jaxpr.eqns[0].primitive == for_loop_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert inner_jaxpr.eqns[-3].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[-2].primitive == qml.RY._primitive
        assert inner_jaxpr.eqns[-1].primitive == qml.RZ._primitive

    def test_while_loop_higher_order_primitive(self):
        """Test that the while_loop primitive is correctly interpreted"""

        @UnitaryToRotInterpreter()
        def f(U, n):
            @qml.while_loop(lambda i: i < n)
            def g(i):
                qml.QubitUnitary(U, i)
                return i + 1

            g(0)
            return qml.expval(qml.Z(0))

        U = qml.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        args = (U, 3)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == while_loop_prim
        # Measurement
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        # While loop jaxpr
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        # Last primitive is the i+1 in the while loop
        assert inner_jaxpr.eqns[-4].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[-3].primitive == qml.RY._primitive
        assert inner_jaxpr.eqns[-2].primitive == qml.RZ._primitive

    def test_cond_higher_order_primitive(self):
        """Test that the cond primitive is correctly interpreted"""

        @UnitaryToRotInterpreter()
        def f(U, n):
            @qml.cond(n > 0)
            def cond_f():
                qml.QubitUnitary(U, 0)
                return qml.expval(qml.Z(0))

            @cond_f.else_if(n > 1)
            def _():
                qml.QubitUnitary(U, 1)
                return qml.expval(qml.Y(0))

            @cond_f.otherwise
            def _():
                qml.QubitUnitary(U, 2)
                return qml.expval(qml.X(0))

            out = cond_f()
            return out

        U = qml.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        args = (U, 3)
        jaxpr = jax.make_jaxpr(f)(*args)
        # First 2 primitives are the conditions for the true and elif branches
        assert jaxpr.eqns[2].primitive == cond_prim

        # True branch
        branch_jaxpr = jaxpr.eqns[2].params["jaxpr_branches"][0]
        # Qubit unitary decomposition
        assert branch_jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert branch_jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert branch_jaxpr.eqns[-3].primitive == qml.RZ._primitive
        # Make sure its on wire=0
        assert qml.math.allclose(branch_jaxpr.eqns[-3].invars[1], 0)
        # Measurement
        assert branch_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert branch_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        # Elif branch
        branch_jaxpr = jaxpr.eqns[2].params["jaxpr_branches"][1]
        # Qubit unitary decomposition
        assert branch_jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert branch_jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert branch_jaxpr.eqns[-3].primitive == qml.RZ._primitive
        # Make sure its on wire=1
        assert qml.math.allclose(branch_jaxpr.eqns[-3].invars[1], 1)
        # Measurement
        assert branch_jaxpr.eqns[-2].primitive == qml.PauliY._primitive
        assert branch_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

        # Else branch
        branch_jaxpr = jaxpr.eqns[2].params["jaxpr_branches"][2]
        # Qubit unitary decomposition
        assert branch_jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert branch_jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert branch_jaxpr.eqns[-3].primitive == qml.RZ._primitive
        # Make sure its on wire=2
        assert qml.math.allclose(branch_jaxpr.eqns[-3].invars[1], 2)
        # Measurement
        assert branch_jaxpr.eqns[-2].primitive == qml.PauliX._primitive
        assert branch_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_grad_higher_order_primitive(self):
        """Test that the grad primitive are correctly interpreted"""
        dev = qml.device("default.qubit", wires=1)

        @UnitaryToRotInterpreter()
        def f(a, b, c):
            @qml.qnode(dev)
            def circuit(a, b, c):
                with qml.QueuingManager.stop_recording():
                    A = qml.Rot.compute_matrix(a, b, c)
                qml.QubitUnitary(A, 0)
                return qml.expval(qml.Z(0))

            return qml.grad(circuit)(a, b, c)

        jaxpr = jax.make_jaxpr(f)(1.0, 2.0, 3.0)

        assert jaxpr.eqns[0].primitive == grad_prim

        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_jac_higher_order_primitive(self):
        """Test that the jacobian primitive works correctly"""
        dev = qml.device("default.qubit", wires=1)

        @UnitaryToRotInterpreter()
        def f(a, b, c):
            @qml.qnode(dev)
            def circuit(a, b, c):
                with qml.QueuingManager.stop_recording():
                    A = qml.Rot.compute_matrix(a, b, c)
                qml.QubitUnitary(A, 0)
                return qml.expval(qml.Z(0))

            return qml.jacobian(circuit)(a, b, c)

        jaxpr = jax.make_jaxpr(f)(1.0, 2.0, 3.0)

        assert jaxpr.eqns[0].primitive == jacobian_prim

        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive
