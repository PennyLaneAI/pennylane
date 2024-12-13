# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ``DecomposeInterpreter`` class"""
# pylint:disable=protected-access,unused-argument, wrong-import-position
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.capture.transforms import DecomposeInterpreter

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class TestDecomposeInterpreter:
    """Unit tests for the DecomposeInterpreter class for decomposing plxpr."""

    @pytest.mark.parametrize(
        "gate_set", [["RX"], [qml.RX], lambda op: op.name == "RX", qml.RX, "RX"]
    )
    @pytest.mark.parametrize("max_expansion", [None, 4])
    def test_init(self, gate_set, max_expansion):
        """Test that DecomposeInterpreter is initialized correctly."""
        interpreter = DecomposeInterpreter(gate_set=gate_set, max_expansion=max_expansion)
        assert interpreter.max_expansion == max_expansion
        valid_op = qml.RX(1.5, 0)
        invalid_op = qml.RY(1.5, 0)
        assert interpreter.gate_set(valid_op)
        assert not interpreter.gate_set(invalid_op)

    @pytest.mark.parametrize("op", [qml.RX(1.5, 0), qml.RZ(1.5, 0)])
    def test_stopping_condition(self, op, recwarn):
        """Test that stopping_condition works correctly."""
        # pylint: disable=unnecessary-lambda-assignment
        gate_set = lambda op: op.name == "RX"
        interpreter = DecomposeInterpreter(gate_set=gate_set)

        if gate_set(op):
            assert interpreter.stopping_condition(op)
            assert len(recwarn) == 0

        else:
            if not op.has_decomposition:
                with pytest.warns(UserWarning, match="does not define a decomposition"):
                    assert interpreter.stopping_condition(op)
            else:
                assert not interpreter.stopping_condition(op)
                assert len(recwarn) == 0

    def test_decompose_simple(self):
        """Test that a simple function can be decomposed correctly."""
        gate_set = [qml.RX, qml.RY, qml.RZ]

        @DecomposeInterpreter(gate_set=gate_set)
        def f(x, y, z):
            qml.Rot(x, y, z, 0)
            return x

        jaxpr = jax.make_jaxpr(f)(1.2, 3.4, 5.6)
        assert jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert jaxpr.eqns[1].primitive == qml.RY._primitive
        assert jaxpr.eqns[2].primitive == qml.RZ._primitive

    def test_returned_op_not_decomposed(self):
        """Test that operators that are returned by the input function are not decomposed."""
        gate_set = [qml.RX, qml.RY, qml.RZ]

        @DecomposeInterpreter(gate_set=gate_set)
        def f(x, y, z):
            return qml.Rot(x, y, z, 0)

        jaxpr = jax.make_jaxpr(f)(1.2, 3.4, 5.6)
        assert jaxpr.eqns[0].primitive == qml.Rot._primitive
        assert len(jaxpr.jaxpr.outvars) == 1
        assert jaxpr.jaxpr.outvars[0] == jaxpr.eqns[0].outvars[0]

    def test_deep_decomposition(self):
        """Test that decomposing primitives that require multiple levels of decomposition
        is done correctly."""
        gate_set = [qml.RX, qml.RY, qml.RZ, qml.PhaseShift]

        @DecomposeInterpreter(gate_set=gate_set)
        def f(x, y, z):
            qml.U3(x, y, z, 0)
            return x

        jaxpr = jax.make_jaxpr(f)(1.2, 3.4, 5.6)
        assert jaxpr.eqns[0].primitive.name == "neg"
        assert jaxpr.eqns[1].primitive == qml.RZ._primitive
        assert jaxpr.eqns[2].primitive == qml.RY._primitive
        assert jaxpr.eqns[3].primitive == qml.RZ._primitive
        assert jaxpr.eqns[4].primitive == qml.PhaseShift._primitive
        assert jaxpr.eqns[5].primitive == qml.PhaseShift._primitive

    def test_max_expansion(self):
        """Test that giving a max_expansion to the interpreter results in early stoppage in
        decomposition."""
        gate_set = [qml.RX, qml.RY, qml.RZ, qml.PhaseShift]

        @DecomposeInterpreter(gate_set=gate_set, max_expansion=1)
        def f(x, y, z):
            qml.U3(x, y, z, 0)
            return x

        jaxpr = jax.make_jaxpr(f)(1.2, 3.4, 5.6)
        assert jaxpr.eqns[0].primitive.name == "neg"
        assert jaxpr.eqns[1].primitive == qml.Rot._primitive
        assert jaxpr.eqns[2].primitive == qml.PhaseShift._primitive
        assert jaxpr.eqns[3].primitive == qml.PhaseShift._primitive

    @pytest.mark.parametrize("decompose", [True, False])
    def test_decompose_sum(self, decompose, recwarn):
        """Test that a function containing `Sum` can be decomposed correctly."""
        gate_set = [qml.PauliX, qml.PauliY, qml.PauliZ]
        if not decompose:
            gate_set.append(qml.ops.Sum)
        interpreter = DecomposeInterpreter(gate_set=gate_set)

        def f(x):
            qml.sum(qml.X(0), qml.Y(0), qml.Z(0))

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[-4].primitive == qml.PauliX._primitive
        assert jaxpr.eqns[-3].primitive == qml.PauliY._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.ops.Sum._primitive

        transformed_f = interpreter(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)

        if decompose:
            assert len(recwarn) == 1
            w = recwarn.pop()
            assert w.category == UserWarning
            assert str(w.message).startswith("Operator Sum does not define a decomposition")
        else:
            assert len(recwarn) == 0

        assert transformed_jaxpr.eqns[-4].primitive == qml.PauliX._primitive
        assert transformed_jaxpr.eqns[-3].primitive == qml.PauliY._primitive
        assert transformed_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[-1].primitive == qml.ops.Sum._primitive

    @pytest.mark.parametrize("decompose", [True, False])
    def test_decompose_sprod(self, decompose, recwarn):
        """Test that a function containing `SProd` can be decomposed correctly."""
        gate_set = [qml.PauliX, qml.PauliY, qml.PauliZ]
        if not decompose:
            gate_set.append(qml.ops.SProd)
        interpreter = DecomposeInterpreter(gate_set=gate_set)

        def f(x):
            qml.s_prod(x, qml.Z(0))

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.ops.SProd._primitive

        transformed_f = interpreter(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)

        if decompose:
            assert len(recwarn) == 1
            w = recwarn.pop()
            assert w.category == UserWarning
            assert str(w.message).startswith("Operator SProd does not define a decomposition")
        else:
            assert len(recwarn) == 0

        assert transformed_jaxpr.eqns[-2].primitive == qml.ops.PauliZ._primitive
        assert transformed_jaxpr.eqns[-1].primitive == qml.ops.SProd._primitive

    @pytest.mark.parametrize("decompose", [True, False])
    def test_decompose_prod(self, decompose):
        """Test that a function containing `Prod` can be decomposed correctly."""
        gate_set = [qml.PauliX, qml.PauliY, qml.PauliZ]
        if not decompose:
            gate_set.append(qml.ops.Prod)
        interpreter = DecomposeInterpreter(gate_set=gate_set)

        def f(x):
            qml.prod(qml.X(0), qml.Y(0), qml.Z(0))

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[-4].primitive == qml.PauliX._primitive
        assert jaxpr.eqns[-3].primitive == qml.PauliY._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.ops.Prod._primitive

        transformed_f = interpreter(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        if decompose:
            assert transformed_jaxpr.eqns[-3].primitive == qml.PauliZ._primitive
            assert transformed_jaxpr.eqns[-2].primitive == qml.PauliY._primitive
            assert transformed_jaxpr.eqns[-1].primitive == qml.PauliX._primitive
        else:
            for orig_eqn, transformed_eqn in zip(jaxpr.eqns, transformed_jaxpr.eqns):
                assert orig_eqn.primitive == transformed_eqn.primitive

    @pytest.mark.parametrize("decompose", [True, False])
    def test_decompose_ctrl(self, decompose):
        """Test that a function containing `Controlled` can be decomposed correctly."""
        gate_set = [qml.RX, qml.RY, qml.RZ, qml.CNOT]
        if not decompose:
            gate_set.append(qml.ops.Controlled)
        interpreter = DecomposeInterpreter(gate_set=gate_set)

        def f(x):
            qml.ctrl(qml.RX(x, 0), 1)

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[-2].primitive == qml.RX._primitive
        assert jaxpr.eqns[-1].primitive == qml.ops.Controlled._primitive

        transformed_f = interpreter(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        if decompose:
            op_prims = [
                eqn.primitive
                for eqn in transformed_jaxpr.eqns
                if eqn.outvars[0].aval == qml.capture.AbstractOperator()
            ]
            expected_prims = [op._primitive for op in qml.ctrl(qml.RX(*args, 0), 1).decomposition()]
            assert all(
                prim == exp_prim for prim, exp_prim in zip(op_prims, expected_prims, strict=True)
            )
        else:
            for orig_eqn, transformed_eqn in zip(jaxpr.eqns, transformed_jaxpr.eqns):
                assert orig_eqn.primitive == transformed_eqn.primitive

    @pytest.mark.parametrize("decompose", [True, False])
    def test_decompose_adjoint(self, decompose):
        """Test that a function containing `Adjoint` can be decomposed correctly."""
        gate_set = [qml.RX, qml.RY, qml.RZ]
        if not decompose:
            gate_set.append(qml.ops.Adjoint)
        interpreter = DecomposeInterpreter(gate_set=gate_set)

        def f(x):
            qml.adjoint(qml.RX(x, 0))

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[-2].primitive == qml.RX._primitive
        assert jaxpr.eqns[-1].primitive == qml.ops.Adjoint._primitive

        transformed_f = interpreter(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        if decompose:
            assert transformed_jaxpr.eqns[-1].primitive == qml.RX._primitive
        else:
            for orig_eqn, transformed_eqn in zip(jaxpr.eqns, transformed_jaxpr.eqns):
                assert orig_eqn.primitive == transformed_eqn.primitive

    def test_ctrl_higher_order_primitive_not_implemented(self):
        """Test that evaluating a ctrl higher order primitive raises a NotImplementedError"""

        def inner_f(x):
            qml.X(0)
            qml.RX(x, 0)

        def f(x):
            qml.ctrl(inner_f, control=[1])(x)

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        interpreter = DecomposeInterpreter()

        with pytest.raises(NotImplementedError):
            interpreter.eval(jaxpr.jaxpr, jaxpr.consts, *args)

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_higher_order_primitive_not_implemented(self, lazy):
        """Test that evaluating a ctrl higher order primitive raises a NotImplementedError"""
        gate_set = [qml.RX, qml.RY, qml.RZ]

        @DecomposeInterpreter(gate_set=gate_set)
        def f(x, y, z):
            def g(a, b, c):
                qml.Rot(x, y, z, 0)

            qml.adjoint(g, lazy=lazy)(x, y, z)

        jaxpr = jax.make_jaxpr(f)(1.2, 3.4, 5.6)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == adjoint_transform_prim
        assert jaxpr.eqns[0].params["lazy"] == lazy

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        assert len(inner_jaxpr.eqns) == 3
        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[1].primitive == qml.RY._primitive
        assert inner_jaxpr.eqns[2].primitive == qml.RZ._primitive

    def test_cond_higher_order_primitive(self):
        """Test that the cond primitive is correctly interpreted"""
        gate_set = [qml.RX, qml.RY, qml.RZ, qml.PhaseShift]

        @DecomposeInterpreter(gate_set=gate_set)
        def f(x):
            @qml.cond(x > 2)
            def cond_f():
                qml.X(0)
                return qml.expval(qml.Z(0))

            @cond_f.else_if(x > 1)
            def _():
                qml.Y(0)
                return qml.expval(qml.Y(0))

            @cond_f.otherwise
            def _():
                qml.Z(0)
                return qml.expval(qml.X(0))

            out = cond_f()
            return out

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        # First 2 primitives are the conditions for the true and elif branches
        assert jaxpr.eqns[2].primitive == cond_prim

        # True branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][0]
        expected_primitives = [
            qml.PhaseShift._primitive,
            qml.RX._primitive,
            qml.PhaseShift._primitive,
            qml.Z._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim for eqn, exp_prim in zip(branch.eqns, expected_primitives)
        )

        # Elif branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][1]
        expected_primitives = [
            qml.PhaseShift._primitive,
            qml.RY._primitive,
            qml.PhaseShift._primitive,
            qml.Y._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim for eqn, exp_prim in zip(branch.eqns, expected_primitives)
        )

        # Else branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][2]
        expected_primitives = [
            qml.PhaseShift._primitive,
            qml.X._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim for eqn, exp_prim in zip(branch.eqns, expected_primitives)
        )

    def test_for_loop_higher_order_primitive(self):
        """Test that the for_loop primitive is correctly interpreted"""
        gate_set = [qml.RX, qml.RY, qml.RZ]

        @DecomposeInterpreter(gate_set=gate_set)
        def f(x, y, z, n):
            @qml.for_loop(n)
            def g(i):
                qml.Rot(x, y, z, i)

            g()

        args = (1.5, 2.5, 3.5, 5)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == for_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 3
        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[1].primitive == qml.RY._primitive
        assert inner_jaxpr.eqns[2].primitive == qml.RZ._primitive

    def test_while_loop_higher_order_primitive(self):
        """Test that the while_loop primitive is correctly interpreted"""
        gate_set = [qml.RX, qml.RY, qml.RZ]

        @DecomposeInterpreter(gate_set=gate_set)
        def f(x, y, z, n):
            @qml.while_loop(lambda i: i < 2 * n)
            def g(i):
                qml.Rot(x, y, z, i)
                return i + 1

            g(0)

        args = (1.5, 2.5, 3.5, 5)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == while_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 4
        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[1].primitive == qml.RY._primitive
        assert inner_jaxpr.eqns[2].primitive == qml.RZ._primitive

    def test_qnode_higher_order_primitive(self):
        """Test that the qnode primitive is correctly interpreted"""
        dev = qml.device("default.qubit", wires=2)
        gate_set = [qml.RX, qml.RY, qml.RZ]

        @DecomposeInterpreter(gate_set=gate_set)
        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.Rot(a, b, c, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(0.5, 1.5, 2.5)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive

    @pytest.mark.parametrize("grad_fn", [qml.grad, qml.jacobian])
    def test_grad_and_jac_higher_order_primitive(self, grad_fn):
        """Test that the grad and jacobian primitives are correctly interpreted"""
        dev = qml.device("default.qubit", wires=2)
        gate_set = [qml.RX, qml.RY, qml.RZ]

        @DecomposeInterpreter(gate_set=gate_set)
        def f(x, y, z):
            @qml.qnode(dev)
            def circuit(a, b, c):
                qml.Rot(a, b, c, 0)
                return qml.expval(qml.Z(0))

            return grad_fn(circuit)(x, y, z)

        jaxpr = jax.make_jaxpr(f)(0.5, 1.5, 2.5)

        if grad_fn == qml.grad:
            assert jaxpr.eqns[0].primitive == grad_prim
        else:
            assert jaxpr.eqns[0].primitive == jacobian_prim
        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive
