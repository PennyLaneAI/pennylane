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

import numpy as np
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
from pennylane.measurements import MidMeasureMP
from pennylane.operation import Operation
from pennylane.ops import Conditional
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.decompose import DecomposeInterpreter, decompose_plxpr_to_plxpr

pytestmark = [pytest.mark.jax, pytest.mark.capture]


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
        assert interpreter.stopping_condition(valid_op)
        assert not interpreter.stopping_condition(invalid_op)

    @pytest.mark.unit
    def test_fixed_alt_decomps_not_available_capture(self):
        """Test that a TypeError is raised when graph is disabled and
        fixed_decomps or alt_decomps is used."""

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(*_, **__):
            raise NotImplementedError

        with pytest.raises(TypeError, match="The keyword arguments fixed_decomps and alt_decomps"):
            DecomposeInterpreter(fixed_decomps={qml.CNOT: my_cnot})

        with pytest.raises(TypeError, match="The keyword arguments fixed_decomps and alt_decomps"):
            DecomposeInterpreter(alt_decomps={qml.CNOT: [my_cnot]})

    @pytest.mark.parametrize("op", [qml.RX(1.5, 0), qml.RZ(1.5, 0)])
    def test_stopping_condition(self, op):
        """Test that stopping_condition works correctly."""
        # pylint: disable=unnecessary-lambda-assignment
        gate_set = lambda op: op.name == "RX"
        interpreter = DecomposeInterpreter(gate_set=gate_set)
        assert interpreter.stopping_condition(op) == gate_set(op)

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

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_higher_order_primitive(self, lazy):
        """Test that adjoint higher order primitives are correctly interpreted."""
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
        gate_set = [qml.RX, qml.RY, qml.RZ, qml.GlobalPhase, qml.PhaseShift]

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
            qml.RX._primitive,
            qml.GlobalPhase._primitive,
            qml.Z._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim for eqn, exp_prim in zip(branch.eqns, expected_primitives)
        ), f"Expected: {expected_primitives}, got: {[eqn.primitive for eqn in branch.eqns]}"

        # Elif branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][1]
        expected_primitives = [
            qml.RY._primitive,
            qml.GlobalPhase._primitive,
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

    def test_decompose_conditionals(self):
        """Tests decomposing a classically controlled operator"""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self) -> dict:
                return {}

            def compute_qfunc_decomposition(self, *wires, **_):
                qml.H(wires[0])
                m0 = qml.measure(wires[0])
                qml.cond(m0, qml.H)(wires[1])

        @DecomposeInterpreter(gate_set={qml.RX, qml.RZ})
        def circuit():
            CustomOp(wires=[1, 0])
            m0 = qml.measure(0)
            qml.cond(m0, qml.X)(wires=0)

        jaxpr = jax.make_jaxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        ops = collector.state["ops"]

        def equivalent_circuit():
            qml.RZ(np.pi / 2, wires=1)
            qml.RX(np.pi / 2, wires=1)
            qml.RZ(np.pi / 2, wires=1)
            m0 = qml.measure(1)
            qml.cond(m0, qml.RZ)(np.pi / 2, wires=0)
            qml.cond(m0, qml.RX)(np.pi / 2, wires=0)
            qml.cond(m0, qml.RZ)(np.pi / 2, wires=0)
            m1 = qml.measure(0)
            qml.cond(m1, qml.RX)(np.pi, wires=0)

        with qml.queuing.AnnotatedQueue() as q:
            equivalent_circuit()

        qml.assert_equal(ops[0], q.queue[0])
        qml.assert_equal(ops[1], q.queue[1])
        qml.assert_equal(ops[2], q.queue[2])
        assert isinstance(ops[4], Conditional)
        assert isinstance(ops[5], Conditional)
        assert isinstance(ops[6], Conditional)
        assert isinstance(ops[8], Conditional)
        qml.assert_equal(ops[4].base, q.queue[4].base)
        qml.assert_equal(ops[5].base, q.queue[5].base)
        qml.assert_equal(ops[6].base, q.queue[6].base)
        qml.assert_equal(ops[8].base, q.queue[8].base)
        assert isinstance(ops[3], MidMeasureMP)
        assert isinstance(ops[7], MidMeasureMP)


class TestControlledDecompositions:
    """Unit tests for decomposing ctrl_transform primitives."""

    def test_ctrl_simple(self):
        """Test that ctrl higher order primitives are correctly interpreted."""

        @DecomposeInterpreter(gate_set=[qml.CRX, qml.CRY, qml.CRZ])
        def inner_f(x):
            qml.Rot(x, 1.0, 2.0, 0)

        def f(x):
            qml.ctrl(inner_f, control=[1])(x)

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)
        assert collector.state["ops"] == [
            qml.CRZ(1.5, [1, 0]),
            qml.CRY(1.0, [1, 0]),
            qml.CRZ(2.0, [1, 0]),
        ]

    def test_ctrl_no_decomposition(self):
        """Test that ctrl_transform that does not need to be decomposed gets changed into
        individually controlled ops"""

        def inner_f(x):
            qml.RX(x, 0)
            qml.IsingXX(x, [0, 1])

        # C(IsingXX) is not in the default gate set
        @DecomposeInterpreter(gate_set="C(IsingXX)")
        def f(x):
            qml.ctrl(inner_f, control=[2, 3])(x)

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert not any(eqn.primitive == ctrl_transform_prim for eqn in jaxpr.eqns)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)
        assert collector.state["ops"] == [
            qml.ctrl(qml.RX(1.5, 0), [2, 3]),
            qml.ctrl(qml.IsingXX(1.5, [0, 1]), [2, 3]),
        ]

    def test_ctrl_for_loop(self):
        """Test that a for_loop inside a ctrl_transform is not unrolled."""

        def inner_f(x, n):
            @qml.for_loop(n)
            def g(i):
                qml.RX(x, i)

            g()

        @DecomposeInterpreter()
        def f(x, n):
            qml.ctrl(inner_f, control=[4, 5])(x, n)

        args = (1.5, 3)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == for_loop_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert inner_jaxpr.eqns[-2].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[-1].primitive == qml.ops.Controlled._primitive

    def test_ctrl_while_loop(self):
        """Test that a while_loop inside a ctrl_transform is not unrolled."""

        def inner_f(x, n):
            @qml.while_loop(lambda i: i < 2 * n)
            def g(i):
                qml.RX(x, i)
                return i + 1

            g(0)

        @DecomposeInterpreter()
        def f(x, n):
            qml.ctrl(inner_f, control=[4, 5])(x, n)

        args = (1.5, 3)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[0].primitive == while_loop_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert inner_jaxpr.eqns[-3].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[-2].primitive == qml.ops.Controlled._primitive
        # final primitive is the increment
        assert inner_jaxpr.eqns[-1].primitive.name == "add"

    def test_ctrl_cond(self):
        """Test that a cond inside a ctrl_transform is not unrolled."""

        def inner_f(x):
            @qml.cond(x > 1)
            def cond_f():
                qml.RX(x, 0)

            cond_f()

        @DecomposeInterpreter()
        def f(x):
            qml.ctrl(inner_f, control=[4, 5])(x)

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        # condition is the first primitive
        assert jaxpr.eqns[1].primitive == cond_prim

        # True branch
        branch_jaxpr = jaxpr.eqns[1].params["jaxpr_branches"][0]
        assert branch_jaxpr.eqns[-2].primitive == qml.RX._primitive
        assert branch_jaxpr.eqns[-1].primitive == qml.ops.Controlled._primitive


def test_decompose_plxpr_to_plxpr():
    """Test that transforming plxpr works."""
    gate_set = [qml.RX, qml.RY, qml.RZ, qml.PhaseShift]

    def circuit(x, y, z):
        qml.Rot(x, y, z, 0)
        return qml.expval(qml.Z(0))

    args = (1.2, 3.4, 5.6)
    jaxpr = jax.make_jaxpr(circuit)(*args)
    transformed_jaxpr = decompose_plxpr_to_plxpr(
        jaxpr.jaxpr, jaxpr.consts, [], {"gate_set": gate_set}, *args
    )
    assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
    assert len(transformed_jaxpr.eqns) == 5
    assert transformed_jaxpr.eqns[0].primitive == qml.RZ._primitive
    assert transformed_jaxpr.eqns[1].primitive == qml.RY._primitive
    assert transformed_jaxpr.eqns[2].primitive == qml.RZ._primitive
    assert transformed_jaxpr.eqns[3].primitive == qml.PauliZ._primitive
    assert transformed_jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive
