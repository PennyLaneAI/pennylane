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
"""Unit tests for the ``CancelInversesInterpreter`` class"""

# pylint:disable=wrong-import-position,protected-access
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    cond_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    measure_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.optimization.cancel_inverses import (
    CancelInversesInterpreter,
    cancel_inverses_plxpr_to_plxpr,
)

pytestmark = [pytest.mark.jax, pytest.mark.capture]


# pylint: disable=too-many-public-methods
class TestCancelInversesInterpreter:
    """Unit tests for the CancelInversesInterpreter for canceling adjacent inverse
    operations in plxpr."""

    def test_cancel_inverses_simple(self):
        """Test that inverse ops in a simple circuit are cancelled."""

        @CancelInversesInterpreter()
        def f():
            qml.X(0)
            qml.X(0)
            qml.S(1)
            qml.adjoint(qml.S(1))
            qml.adjoint(qml.T(2))
            qml.Hadamard(1)  # Applied
            qml.T(2)
            qml.Z(0)  # Applied
            qml.IsingXX(1.5, [2, 3])  # Applied
            qml.IsingXX(2.5, [0, 1])  # Applied
            qml.SWAP([2, 0])
            qml.SWAP([0, 2])
            qml.CNOT([2, 0])  # Applied
            qml.Z(1)  # Applied

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 6

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        ops = collector.state["ops"]

        expected_ops = [
            qml.Z(0),
            qml.Hadamard(1),
            qml.IsingXX(1.5, [2, 3]),
            qml.IsingXX(2.5, [0, 1]),
            qml.CNOT([2, 0]),
            qml.Z(1),
        ]

        assert ops == expected_ops

    def test_cancel_inverses_true_inverses(self):
        """Test that operations that are inverses with the same wires are cancelled."""

        @CancelInversesInterpreter()
        def f():
            qml.CRX(1.5, [0, 1])
            qml.adjoint(qml.CRX(1.5, [0, 1]))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 0

    def test_cancel_inverses_symmetric_wires(self):
        """Test that operations that are inverses regardless of wire order are cancelled."""

        @CancelInversesInterpreter()
        def f():
            qml.CCZ([0, 1, 2])
            qml.CCZ([2, 0, 1])

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 0

    def test_cancel_inverses_symmetric_control_wires(self):
        """Test that operations that are inverses regardless of control_wire order are cancelled."""

        @CancelInversesInterpreter()
        def f():
            qml.Toffoli([0, 1, 2])
            qml.Toffoli([1, 0, 2])

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 0

    def test_cancel_inverese_nested_ops_on_same_wires(self):
        """Test that only the innermost adjacent adjoint ops are cancelled when multiple
        cancellable operators are present."""

        @CancelInversesInterpreter()
        def f():
            qml.S(0)  # Applied
            qml.adjoint(qml.T(0))
            qml.T(0)
            qml.adjoint(qml.S(0))  # Applied

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        ops = collector.state["ops"]

        expected_ops = [
            qml.S(0),
            qml.adjoint(qml.S(0)),
        ]

        assert ops == expected_ops

    def test_returned_op_is_not_cancelled(self):
        """Test that ops that are returned by the function being transformed are not cancelled."""

        @CancelInversesInterpreter()
        def f():
            qml.PauliX(0)
            return qml.PauliX(0)

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.PauliX._primitive
        assert jaxpr.eqns[1].primitive == qml.PauliX._primitive
        assert jaxpr.jaxpr.outvars[0] == jaxpr.eqns[1].outvars[0]

    def test_no_wire_ops_not_cancelled(self):
        """Test that inverse operations with no wires do not get cancelled."""

        @CancelInversesInterpreter()
        def f():
            qml.Identity()
            qml.Identity()
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 4

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qml.Identity(), qml.Identity()]
        assert ops == expected_ops

        expected_meas = [qml.expval(qml.Z(0))]
        assert meas == expected_meas

    def test_dynamic_wires_between_static_wires(self):
        """Test that operations with dynamic wires between operations with static
        wires cause cancellation to not happen."""

        @CancelInversesInterpreter()
        def f(w):
            qml.H(0)
            qml.T(w)
            qml.H(0)
            qml.adjoint(qml.T(w))
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0)
        assert len(jaxpr.eqns) == 7

        dyn_wire = 0
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_wire)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]
        expected_meas = [qml.expval(qml.Z(0))]

        expected_ops = [qml.H(0), qml.T(0), qml.H(0), qml.adjoint(qml.T(0))]
        assert ops == expected_ops
        assert meas == expected_meas

        dyn_wire = 1
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_wire)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qml.H(0), qml.T(1), qml.H(0), qml.adjoint(qml.T(1))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_same_dyn_wires_cancel(self):
        """Test that ops on the same dynamic wires get cancelled."""

        @CancelInversesInterpreter()
        def f(w):
            qml.H(0)
            qml.T(w)
            qml.adjoint(qml.T(w))
            qml.H(0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0)
        assert len(jaxpr.eqns) == 4

        dyn_wire = 0
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_wire)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qml.H(0), qml.H(0)]
        expected_meas = [qml.expval(qml.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_same_dyn_param_cancel(self):
        """Test that ops on the same wires with the same dynamic parameter get cancelled."""

        @CancelInversesInterpreter()
        def f(x):
            qml.H(1)
            qml.RX(x, 1)
            qml.adjoint(qml.RX(x, 1))
            qml.H(0)
            return qml.expval(qml.Z(0))

        dyn_param = jax.numpy.array(0.1)
        jaxpr = jax.make_jaxpr(f)(dyn_param)
        assert len(jaxpr.eqns) == 4

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_param)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qml.H(1), qml.H(0)]
        expected_meas = [qml.expval(qml.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_different_dyn_params_not_cancel(self):
        """Test that ops on the same wires with different dynamic parameters do not get cancelled."""

        @CancelInversesInterpreter()
        def f(x, y):
            qml.H(1)
            qml.RX(x, 1)
            qml.adjoint(qml.RX(y, 1))
            qml.H(0)
            return qml.expval(qml.Z(0))

        dyn_param1, dyn_param2 = jax.numpy.array(0.1), jax.numpy.array(0.2)
        jaxpr = jax.make_jaxpr(f)(dyn_param1, dyn_param2)

        assert len(jaxpr.eqns) == 7
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_param1, dyn_param2)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [
            qml.H(1),
            qml.RX(dyn_param1, 1),
            qml.adjoint(qml.RX(dyn_param2, 1)),
            qml.H(0),
        ]
        expected_meas = [qml.expval(qml.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_different_dyn_wires_interleaved(self):
        """Test that ops on different dynamic wires interleaved with each other
        do not cancel."""

        @CancelInversesInterpreter()
        def f(w1, w2):
            qml.H(w1)
            qml.X(w2)
            qml.H(w1)
            qml.X(w2)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0, 0)
        assert len(jaxpr.eqns) == 6

        dyn_wires = (0, 0)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *dyn_wires)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qml.H(0), qml.X(0), qml.H(0), qml.X(0)]
        expected_meas = [qml.expval(qml.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

        dyn_wires = (0, 1)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *dyn_wires)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qml.H(0), qml.X(1), qml.H(0), qml.X(1)]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_ctrl_higher_order_primitive(self):
        """Test that ctrl higher order primitives are transformed correctly."""

        def ctrl_fn(y):
            qml.S(0)
            qml.Hadamard(1)
            qml.Hadamard(1)
            qml.adjoint(qml.S(0))
            qml.RX(y, 0)  # Applied

        @CancelInversesInterpreter()
        def f(x):
            qml.RX(x, 0)
            qml.ctrl(ctrl_fn, [2, 3])(x)
            qml.RY(x, 1)

        arg = 1.5
        jaxpr = jax.make_jaxpr(f)(arg)
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, arg)
        ops = collector.state["ops"]

        excepted_ops = [
            qml.RX(arg, 0),
            qml.ctrl(qml.RX(arg, 0), [2, 3]),
            qml.RY(arg, 1),
        ]

        assert ops == excepted_ops

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_higher_order_primitive(self, lazy):
        """Test that adjoint higher order primitives are transformed correctly."""

        def adjoint_fn(y):
            qml.S(0)
            qml.Hadamard(1)
            qml.Hadamard(1)
            qml.adjoint(qml.S(0))
            qml.RX(y, 0)

        @CancelInversesInterpreter()
        def f(x):
            qml.RX(x, 0)
            qml.adjoint(adjoint_fn, lazy=lazy)(x)
            qml.RY(x, 1)

        arg = 1.5
        jaxpr = jax.make_jaxpr(f)(arg)
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, arg)
        ops = collector.state["ops"]

        expected_ops = [
            qml.RX(arg, 0),
            qml.adjoint(qml.RX(arg, 0), lazy=lazy),
            qml.RY(arg, 1),
        ]
        assert ops == expected_ops

    def test_cond_higher_order_primitive(self):
        """Test that cond higher order primitives are transformed correctly."""

        @CancelInversesInterpreter()
        def f(x):
            qml.RX(x, 0)

            @qml.cond(x > 2)
            def cond_fn():
                qml.Hadamard(0)
                qml.Hadamard(0)
                return qml.S(0)

            @cond_fn.else_if(x > 1)
            def _():
                qml.S(0)
                qml.adjoint(qml.S(0))
                return qml.T(0)

            @cond_fn.otherwise
            def _():
                qml.adjoint(qml.T(0))
                qml.T(0)
                return qml.Hadamard(0)

            return cond_fn()

        jaxpr = jax.make_jaxpr(f)(1.5)
        # 2 primitives for true and elif branch conditions of the conditional
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[2].primitive == qml.RX._primitive
        assert jaxpr.eqns[3].primitive == cond_prim

        # true branch
        branch = jaxpr.eqns[3].params["jaxpr_branches"][0]
        assert len(branch.eqns) == 1
        assert branch.eqns[0].primitive == qml.S._primitive
        assert branch.outvars[0] == branch.eqns[0].outvars[0]

        # elif branch
        branch = jaxpr.eqns[3].params["jaxpr_branches"][1]
        assert len(branch.eqns) == 1
        assert branch.eqns[0].primitive == qml.T._primitive
        assert branch.outvars[0] == branch.eqns[0].outvars[0]

        # true branch
        branch = jaxpr.eqns[3].params["jaxpr_branches"][2]
        assert len(branch.eqns) == 1
        assert branch.eqns[0].primitive == qml.Hadamard._primitive
        assert branch.outvars[0] == branch.eqns[0].outvars[0]

    def test_for_loop_higher_order_primitive(self):
        """Test that for_loop higher order primitives are transformed correctly."""

        @CancelInversesInterpreter()
        def f(x, n):
            qml.RX(x, 0)

            @qml.for_loop(n)
            def loop_fn(i):  # pylint: disable=unused-argument
                qml.S(0)
                qml.Hadamard(1)
                qml.Hadamard(1)
                qml.adjoint(qml.S(0))
                qml.RX(x, 0)

            loop_fn()
            qml.RY(x, 1)

        jaxpr = jax.make_jaxpr(f)(1.5, 4)
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == for_loop_prim
        assert jaxpr.eqns[2].primitive == qml.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 1
        assert inner_jaxpr.eqns[0].primitive == qml.RX._primitive

    def test_while_loop_higher_order_primitive(self):
        """Test that while_loop higher order primitives are transformed correctly."""

        @CancelInversesInterpreter()
        def f(x, n):
            qml.RX(x, 0)

            @qml.while_loop(lambda i: i < 2 * n)
            def loop_fn(i):
                qml.S(0)
                qml.Hadamard(1)
                qml.Hadamard(1)
                qml.adjoint(qml.S(0))
                qml.RX(x, 0)
                return i + 1

            loop_fn(x)
            qml.RY(x, 1)

        jaxpr = jax.make_jaxpr(f)(1.5, 4)
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == while_loop_prim
        assert jaxpr.eqns[2].primitive == qml.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 2
        # The i + 1 primitive and the RX may get reordered, but the outcome will not be impacted
        assert any(eqn.primitive == qml.RX._primitive for eqn in inner_jaxpr.eqns)

        # Check that the output of the i + 1 is returned
        if inner_jaxpr.eqns[0].primitive == qml.RX._primitive:
            add_eqn = inner_jaxpr.eqns[1]
        else:
            add_eqn = inner_jaxpr.eqns[0]
        assert add_eqn.primitive.name == "add"
        assert inner_jaxpr.outvars[0] == add_eqn.outvars[0]

    def test_qnode_higher_order_primitive(self):
        """Test that qnode higher order primitives are transformed correctly."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(y):
            qml.S(0)
            qml.Hadamard(1)
            qml.Hadamard(1)
            qml.adjoint(qml.S(0))
            qml.RX(y, 0)
            return qml.expval(qml.PauliZ(0))

        @CancelInversesInterpreter()
        def f(x):
            qml.RX(x, 0)
            circuit(x)
            qml.RY(x, 1)

        jaxpr = jax.make_jaxpr(f)(1.5)
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == qnode_prim
        assert jaxpr.eqns[2].primitive == qml.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["qfunc_jaxpr"]
        assert len(inner_jaxpr.eqns) == 3
        assert inner_jaxpr.eqns[0].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert inner_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

    @pytest.mark.parametrize("grad_fn", [qml.grad, qml.jacobian])
    def test_grad_and_jac_higher_order_primitives(self, grad_fn):
        """Test that grad and jacobian higher order primitives are transformed correctly."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(y):
            qml.S(0)
            qml.Hadamard(1)
            qml.Hadamard(1)
            qml.adjoint(qml.S(0))
            qml.RX(y, 0)
            return qml.expval(qml.PauliZ(0))

        @CancelInversesInterpreter()
        def f(x):
            qml.RX(x, 0)
            out = grad_fn(circuit)(x)
            qml.RY(x, 1)
            return out

        jaxpr = jax.make_jaxpr(f)(1.5)
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == grad_prim if grad_fn == qml.grad else jacobian_prim
        assert jaxpr.eqns[2].primitive == qml.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr"]
        assert len(inner_jaxpr.eqns) == 1
        qfunc_jaxpr = inner_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_mid_circuit_measurement(self):
        """Test that mid-circuit measurements are correctly handled."""

        @CancelInversesInterpreter()
        def circuit():
            qml.S(0)
            qml.measure(0)
            qml.adjoint(qml.S(0))
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 6

        assert jaxpr.eqns[0].primitive == qml.S._primitive
        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[2].primitive == qml.S._primitive
        assert jaxpr.eqns[3].primitive == qml.ops.Adjoint._primitive
        assert jaxpr.eqns[4].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[5].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_mid_circuit_measurement_not_blocked(self):
        """Test that mid-circuit measurements do not block the cancellation of adjacent inverses."""

        @CancelInversesInterpreter()
        def circuit():
            qml.S(0)
            qml.adjoint(qml.S(0))
            qml.measure(0)
            qml.H(1)
            qml.adjoint(qml.H(1))
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == measure_prim
        assert jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive


def test_cancel_inverses_plxpr_to_plxpr():
    """Test that transforming plxpr works."""

    def circuit():
        qml.X(0)
        qml.S(1)
        qml.X(0)
        qml.adjoint(qml.S(1))
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(circuit)()
    transformed_jaxpr = cancel_inverses_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {})
    assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
    assert len(transformed_jaxpr.eqns) == 2
    assert transformed_jaxpr.eqns[0].primitive == qml.PauliZ._primitive
    assert transformed_jaxpr.eqns[1].primitive == qml.measurements.ExpectationMP._obs_primitive
