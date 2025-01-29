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
"""Unit tests for the ``DeferMeasurementsInterpreter`` class"""

import numpy as np

# pylint:disable=wrong-import-position, protected-access
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pennylane.capture.primitives import grad_prim, jacobian_prim, qnode_prim
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.defer_measurements import (
    DeferMeasurementsInterpreter,
    defer_measurements_plxpr_to_plxpr,
)
from pennylane.wires import Wires

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class TestDeferMeasurementsInterpreter:
    """Unit tests for DeferMeasurementsInterpreter."""

    @pytest.mark.parametrize("aux_wires", [1, (), [1, 2, 3], Wires([1, 2, 3])])
    def test_init(self, aux_wires):
        """Test that the interpreter is initialized correctly."""
        interpreter = DeferMeasurementsInterpreter(aux_wires)
        assert interpreter._aux_wires == Wires(aux_wires)

    @pytest.mark.parametrize("reset", [True, False])
    def test_single_mcm(self, reset):
        """Test that a function with a single MCM is transformed correctly."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f():
            qml.measure(0, reset=reset)

        jaxpr = jax.make_jaxpr(f)()
        expected_len = 2 if reset else 1  # CNOT for measure, CNOT for reset
        assert len(jaxpr.eqns) == expected_len
        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        invals = [invar.val for invar in jaxpr.eqns[0].invars]
        assert invals == [0, 5]
        if reset:
            assert jaxpr.eqns[1].primitive == qml.CNOT._primitive
            invals = [invar.val for invar in jaxpr.eqns[1].invars]
            assert invals == [5, 0]

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [0, 1])
    def test_single_mcm_postselect(self, reset, postselect):
        """Test that a function with a single MCM with postselection is transformed correctly."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f():
            qml.measure(0, reset=reset, postselect=postselect)

        jaxpr = jax.make_jaxpr(f)()
        expected_len = 3 if reset and postselect == 1 else 2  # CNOT + Projector + optional(X)
        assert len(jaxpr.consts) == 1  # Input to Projector for postselect

        assert len(jaxpr.eqns) == expected_len
        # Projector for postselect
        assert jaxpr.eqns[0].primitive == qml.Projector._primitive
        assert jaxpr.eqns[0].invars[0] == jaxpr.jaxpr.constvars[0]

        # CNOT for measure
        assert jaxpr.eqns[1].primitive == qml.CNOT._primitive
        invals = [invar.val for invar in jaxpr.eqns[1].invars]
        assert invals == [0, 5]

        if reset:
            if postselect == 1:
                # PauliX since we know the state is |1>
                assert jaxpr.eqns[2].primitive == qml.PauliX._primitive
                assert jaxpr.eqns[2].invars[0].val == 0

    def test_multiple_mcms(self):
        """Test that applying multiple MCMs is transformed correctly."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(w):
            qml.measure(w)
            qml.measure(w + 1)
            qml.measure(w + 2)

        jaxpr = jax.make_jaxpr(f)(0)  # wires will be 0, 1, 2
        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[0].invars[1].val == 5
        assert jaxpr.eqns[2].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[2].invars[1].val == 6
        assert jaxpr.eqns[4].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[4].invars[1].val == 7

    def test_too_many_mcms(self):
        """Test that an error is raised if more MCMs are present than the number of aux_wires."""

        @DeferMeasurementsInterpreter(aux_wires=[5])
        def f(w):
            qml.measure(w)
            qml.measure(w)

        with pytest.raises(ValueError, match="Not enough auxiliary wires"):
            f(0)

    def test_mcms_as_gate_parameters(self):
        """Test that MCMs can be used as gate parameters."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x):
            m = qml.measure(0)
            qml.RX(x * m, 0)

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [
            qml.CNOT([0, 5]),
            qml.ops.Controlled(qml.RX(jnp.array(0), 0), 5, control_values=[0]),
            qml.ops.Controlled(qml.RX(jnp.array(x), 0), 5, control_values=[1]),
        ]
        assert ops == expected_ops

    def test_mcms_as_nested_gate_parameters(self):
        """Test that MCMs can be used as gate parameters."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x, y):
            m = qml.measure(0)
            qml.s_prod(y, qml.RX(x * m, 0))

        args = (1.5, 2.5)
        jaxpr = jax.make_jaxpr(f)(*args)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)

        ops = collector.state["ops"]
        expected_ops = [
            qml.CNOT([0, 5]),
            qml.ops.Controlled(qml.s_prod(args[1], qml.RX(jnp.array(0), 0)), 5, control_values=[0]),
            qml.ops.Controlled(
                qml.s_prod(args[1], qml.RX(jnp.array(args[0]), 0)), 5, control_values=[1]
            ),
        ]
        assert ops == expected_ops

    def test_simple_cond(self):
        """Test that a qml.cond using a single MCM predicate is transformed correctly."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x):
            m = qml.measure(0)

            @qml.cond(m)
            def true_fn(phi):
                qml.RX(phi, 0)

            true_fn(x)

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [qml.CNOT([0, 5]), qml.CRX(x, [5, 0])]
        assert ops == expected_ops

    def test_non_trivial_cond_predicate(self):
        """Test that a qml.cond using processed MCMs as a predicate is transformed correctly."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x):
            m0 = qml.measure(0)
            m1 = qml.measure(0)

            @qml.cond(2 * m0 + m1)
            def true_fn(phi):
                qml.RX(phi, 0)

            true_fn(x)

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [
            qml.CNOT([0, 5]),
            qml.CNOT([0, 6]),
            qml.ctrl(qml.RX(x, 0), [5, 6], [0, 1]),
            qml.ctrl(qml.RX(x, 0), [5, 6], [1, 0]),
            qml.ctrl(qml.RX(x, 0), [5, 6]),
        ]
        assert ops == expected_ops

    def test_cond_elif_false_fn(self):
        """Test that a qml.cond with elif and false branches is transformed correctly."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x):
            m0 = qml.measure(0)
            m1 = qml.measure(0)

            @qml.cond(m0)
            def cond_fn(phi):
                qml.RX(phi, 0)

            @cond_fn.else_if(m1)
            def _(phi):
                qml.RY(phi, 0)

            @cond_fn.otherwise
            def _(phi):
                qml.RZ(phi, 0)

            cond_fn(x)

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [
            qml.CNOT([0, 5]),
            qml.CNOT([0, 6]),
            qml.CRX(x, [5, 0]),
            qml.ctrl(qml.RY(x, 0), [5, 6], [0, 1]),
            qml.ctrl(qml.RZ(x, 0), [5, 6], [0, 0]),
        ]
        assert ops == expected_ops

    def test_cond_non_mcm(self):
        """Test that a qml.cond that does not use MCM predicates is transformed correctly."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x):
            qml.measure(0)

            @qml.cond(x > 2.0)
            def cond_fn(phi):
                return qml.RX(phi, 0)

            @cond_fn.otherwise
            def _(phi):
                return qml.RZ(phi, 0)

            return cond_fn(x)

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)
        ops = collector.state["ops"]
        expected_ops = [qml.CNOT([0, 5]), qml.RZ(x, 0)]
        assert ops == expected_ops

        x = 2.5
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)
        ops = collector.state["ops"]
        expected_ops = [qml.CNOT([0, 5]), qml.RX(x, 0)]
        assert ops == expected_ops

    @pytest.mark.parametrize(
        "mp_fn, mp_class",
        [
            (qml.expval, qml.measurements.ExpectationMP),
            (qml.var, qml.measurements.VarianceMP),
            (qml.sample, qml.measurements.SampleMP),
            (qml.probs, qml.measurements.ProbabilityMP),
            (qml.counts, qml.measurements.CountsMP),
        ],
    )
    def test_mcm_statistics(self, mp_fn, mp_class):
        """Test that collecting statistics on MCMs is handled correctly."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f():
            m0 = qml.measure(0)
            m1 = qml.measure(0)
            m2 = qml.measure(0)

            outs = (mp_fn(op=m0), mp_fn(op=2.5 * m1 - m2))
            if mp_fn not in (qml.expval, qml.var):
                outs += (mp_fn(op=[m0, m1, m2]),)

            if mp_fn == qml.counts:
                outs += (mp_fn(m0, all_outcomes=True),)

            return outs

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        ops = collector.state["ops"]
        expected_ops = [qml.CNOT([0, 5]), qml.CNOT([0, 6]), qml.CNOT([0, 7])]
        assert ops == expected_ops

        measurements = collector.state["measurements"]
        expected_measurements = [
            mp_class(wires=Wires([5]), eigvals=jnp.arange(0, 2)),
            mp_class(wires=Wires([6, 7]), eigvals=jnp.array([0.0, -1.0, 2.5, 1.5])),
        ]
        if mp_fn not in (qml.expval, qml.var):
            expected_measurements.append(
                mp_class(wires=Wires([5, 6, 7]), eigvals=qml.math.arange(0, 2**3))
            )
        if mp_fn == qml.counts:
            expected_measurements.append(
                mp_class(wires=Wires([5]), eigvals=jnp.arange(0, 2), all_outcomes=True)
            )
        assert measurements == expected_measurements

    def test_arbitrary_mcm_processing(self):
        """Test that arbitrary classical processing can be used with MCMs."""

        def processing_fn(*ms):
            a = jnp.sin(0.5 * jnp.pi * ms[0])
            b = a - (ms[1] + 1) ** 4
            c = jnp.sinh(b) ** (-ms[2] + 2)
            return c

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f():
            m0 = qml.measure(0)
            m1 = qml.measure(0)
            m2 = qml.measure(0)

            inval = processing_fn(m0, m1, m2)
            return qml.expval(inval)

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        mp = collector.state["measurements"][0]
        assert isinstance(mp, qml.measurements.ExpectationMP)
        assert mp.wires == Wires([5, 6, 7])

        expected_eigvals = []
        n_mcms = 3
        # Iterate through all 3-bit binary numbers
        for i in range(2**n_mcms):
            branch = tuple(int(b) for b in f"{i:0{n_mcms}b}")
            expected_eigvals.append(processing_fn(*branch))

        expected_eigvals = jnp.array(expected_eigvals)
        assert jnp.allclose(mp.eigvals(), expected_eigvals)


@pytest.mark.parametrize("postselect", [None, 0, 1])
class TestDeferMeasurementsHigherOrderPrimitives:
    """Unit tests for transforming higher-order primitives with DeferMeasurementsInterpreter."""

    def test_for_loop(self, postselect):
        """Test that a for_loop primitive is transformed correctly."""
        n = jnp.array(3, dtype=int)

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x):
            qml.measure(0, postselect=postselect)

            @qml.for_loop(n)
            def loop_fn(i):
                qml.RX(x, i)

            loop_fn()

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [
            qml.CNOT([0, 5]),
            qml.RX(x, [0]),
            qml.RX(x, [1]),
            qml.RX(x, [2]),
        ]
        if postselect is not None:
            expected_ops.insert(0, qml.Projector(np.array([postselect]), 0))
        assert ops == expected_ops

    def test_while_loop(self, postselect):
        """Test that a while_loop primitive is transformed correctly."""
        n = jnp.array(3, dtype=int)

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x):
            qml.measure(0, postselect=postselect)

            @qml.while_loop(lambda a: a < n)
            def loop_fn(i):
                qml.RX(x, i)
                return i + 1

            loop_fn(0)

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [
            qml.CNOT([0, 5]),
            qml.RX(x, [0]),
            qml.RX(x, [1]),
            qml.RX(x, [2]),
        ]
        if postselect is not None:
            expected_ops.insert(0, qml.Projector(np.array([postselect]), 0))
        assert ops == expected_ops

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint(self, lazy, postselect):
        """Test that the adjoint_transform primitive is transformed correctly."""

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x):
            qml.measure(0, postselect=postselect)

            def adjoint_fn(phi):
                qml.RX(phi, 0)
                qml.RY(phi, 0)

            qml.adjoint(adjoint_fn, lazy=lazy)(x)

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [
            qml.CNOT([0, 5]),
            qml.ops.Adjoint(qml.RY(x, 0)) if lazy else qml.RY(-x, 0),
            qml.ops.Adjoint(qml.RX(x, 0)) if lazy else qml.RX(-x, 0),
        ]
        if postselect is not None:
            expected_ops.insert(0, qml.Projector(np.array([postselect]), 0))
        assert ops == expected_ops

    def test_control(self, postselect):
        """Test that the ctrl_transform primitive is transformed correctly."""
        ctrl_wires = [1, 2]
        ctrl_vals = [True, False]

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        def f(x):
            qml.measure(0, postselect=postselect)

            def ctrl_fn(phi):
                qml.RX(phi, 0)
                qml.RY(phi, 0)

            qml.ctrl(ctrl_fn, ctrl_wires, control_values=ctrl_vals)(x)

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [
            qml.CNOT([0, 5]),
            qml.ops.Controlled(qml.RX(x, 0), ctrl_wires, control_values=ctrl_vals),
            qml.ops.Controlled(qml.RY(x, 0), ctrl_wires, control_values=ctrl_vals),
        ]
        if postselect is not None:
            expected_ops.insert(0, qml.Projector(np.array([postselect]), 0))
        assert ops == expected_ops

    def test_qnode(self, postselect):
        """Test that a qnode primitive is transformed correctly."""
        dev = qml.device("default.qubit", wires=10, shots=10)

        @DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))
        @qml.qnode(dev, diff_method="parameter-shift")
        def f(x):
            m0 = qml.measure(0, postselect=postselect)
            m1 = qml.measure(
                0, postselect=postselect if postselect is None else int(not postselect)
            )

            @qml.cond(2 * m0 + m1)
            def true_fn(phi):
                qml.RX(phi, 0)

            true_fn(x)

            return (
                qml.expval(qml.Z(0)),
                qml.probs(wires=[0, 1, 2]),
                qml.sample(op=[m0, m1]),
                qml.sample(op=m0 - 4.0 * m1),
            )

        x = 1.5
        jaxpr = jax.make_jaxpr(f)(x)
        assert jaxpr.eqns[0].primitive == qnode_prim
        assert jaxpr.eqns[0].params["device"] == dev
        assert jaxpr.eqns[0].params["qnode_kwargs"]["diff_method"] == "parameter-shift"

        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        collector = CollectOpsandMeas()
        collector.eval(inner_jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [
            qml.CNOT([0, 5]),
            qml.CNOT([0, 6]),
            qml.ctrl(qml.RX(x, 0), [5, 6], [0, 1]),
            qml.ctrl(qml.RX(x, 0), [5, 6], [1, 0]),
            qml.ctrl(qml.RX(x, 0), [5, 6]),
        ]
        if postselect is not None:
            expected_ops.insert(0, qml.Projector(np.array([postselect]), 0))
            expected_ops.insert(2, qml.Projector(np.array([int(not postselect)]), 0))
        assert ops == expected_ops

        measurements = collector.state["measurements"]
        expected_measurements = [
            qml.expval(qml.Z(0)),
            qml.probs(wires=[0, 1, 2]),
            qml.measurements.SampleMP(wires=[5, 6], eigvals=jnp.arange(0, 2**2)),
            qml.measurements.SampleMP(wires=[5, 6], eigvals=qml.math.array([0.0, -4.0, 1.0, -3.0])),
        ]
        assert measurements == expected_measurements

    @pytest.mark.parametrize(
        "diff_fn, diff_prim", [(qml.grad, grad_prim), (qml.jacobian, jacobian_prim)]
    )
    def test_grad_jac(self, diff_fn, diff_prim, postselect):
        """Test that differentiation primitives are transformed correctly."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            m0 = qml.measure(0, postselect=postselect)

            @qml.cond(m0)
            def true_fn(phi):
                qml.RX(phi, 0)

            true_fn(x)
            return qml.expval(qml.PauliZ(0))

        x = 1.5
        transformed_fn = DeferMeasurementsInterpreter(aux_wires=list(range(5, 10)))(
            diff_fn(circuit)
        )
        jaxpr = jax.make_jaxpr(transformed_fn)(x)
        assert jaxpr.eqns[0].primitive == diff_prim
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        assert inner_jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = inner_jaxpr.eqns[0].params["qfunc_jaxpr"]

        collector = CollectOpsandMeas()
        collector.eval(qfunc_jaxpr, jaxpr.consts, x)

        ops = collector.state["ops"]
        expected_ops = [qml.RX(x, 0), qml.CNOT([0, 5]), qml.CRX(x, [5, 0])]
        if postselect is not None:
            expected_ops.insert(1, qml.Projector(np.array([postselect]), 0))
        assert ops == expected_ops

        measurements = collector.state["measurements"]
        expected_measurements = [qml.expval(qml.Z(0))]
        assert measurements == expected_measurements


def test_defer_measurements_plxpr_to_plxpr():
    """Test that transforming plxpr works."""

    def f(x):
        m = qml.measure(0)

        @qml.cond(m)
        def true_fn(phi):
            qml.RX(phi, 0)

        true_fn(x)

    args = (1.5,)
    targs = ()
    tkwargs = {"aux_wires": list(range(5, 10))}
    jaxpr = jax.make_jaxpr(f)(*args)
    transformed_jaxpr = defer_measurements_plxpr_to_plxpr(
        jaxpr.jaxpr, jaxpr.consts, targs, tkwargs, *args
    )
    collector = CollectOpsandMeas()
    collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts, *args)

    ops = collector.state["ops"]
    expected_ops = [qml.CNOT([0, 5]), qml.CRX(args[0], [5, 0])]
    assert ops == expected_ops


def test_defer_measurements_plxpr_to_plxpr_no_aux_wires_error():
    """Test that an error is raised if ``aux_wires`` are not provided."""

    def f(x):
        m = qml.measure(0)

        @qml.cond(m)
        def true_fn(phi):
            qml.RX(phi, 0)

        true_fn(x)

    args = (1.5,)
    targs = ()
    tkwargs = {}
    jaxpr = jax.make_jaxpr(f)(*args)

    with pytest.raises(ValueError, match="'aux_wires' argument for qml.defer_measurements must be"):
        defer_measurements_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, targs, tkwargs, *args)


def test_defer_measurements_plxpr_to_plxpr_reduce_postselected_warning():
    """Test that a warning is raised if ``reduce_postselected=True``."""

    def f(x):
        m = qml.measure(0)

        @qml.cond(m)
        def true_fn(phi):
            qml.RX(phi, 0)

        true_fn(x)

    args = (1.5,)
    targs = ()
    tkwargs = {"aux_wires": list(range(5, 10)), "reduce_postselected": True}
    jaxpr = jax.make_jaxpr(f)(*args)

    with pytest.warns(UserWarning, match="Cannot set 'reduce_postselected=True'"):
        defer_measurements_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, targs, tkwargs, *args)
