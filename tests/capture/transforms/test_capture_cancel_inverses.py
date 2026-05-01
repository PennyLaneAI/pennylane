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

import pennylane as qp

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    cond_prim,
    for_loop_prim,
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
            qp.X(0)
            qp.X(0)
            qp.S(1)
            qp.adjoint(qp.S(1))
            qp.adjoint(qp.T(2))
            qp.Hadamard(1)  # Applied
            qp.T(2)
            qp.Z(0)  # Applied
            qp.IsingXX(1.5, [2, 3])  # Applied
            qp.IsingXX(2.5, [0, 1])  # Applied
            qp.SWAP([2, 0])
            qp.SWAP([0, 2])
            qp.CNOT([2, 0])  # Applied
            qp.Z(1)  # Applied

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 6

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        ops = collector.state["ops"]

        expected_ops = [
            qp.Z(0),
            qp.Hadamard(1),
            qp.IsingXX(1.5, [2, 3]),
            qp.IsingXX(2.5, [0, 1]),
            qp.CNOT([2, 0]),
            qp.Z(1),
        ]

        assert ops == expected_ops

    def test_cancel_inverses_true_inverses(self):
        """Test that operations that are inverses with the same wires are cancelled."""

        @CancelInversesInterpreter()
        def f():
            qp.CRX(1.5, [0, 1])
            qp.adjoint(qp.CRX(1.5, [0, 1]))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 0

    def test_cancel_inverses_symmetric_wires(self):
        """Test that operations that are inverses regardless of wire order are cancelled."""

        @CancelInversesInterpreter()
        def f():
            qp.CCZ([0, 1, 2])
            qp.CCZ([2, 0, 1])

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 0

    def test_cancel_inverses_symmetric_control_wires(self):
        """Test that operations that are inverses regardless of control_wire order are cancelled."""

        @CancelInversesInterpreter()
        def f():
            qp.Toffoli([0, 1, 2])
            qp.Toffoli([1, 0, 2])

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 0

    def test_cancel_inverese_nested_ops_on_same_wires(self):
        """Test that only the innermost adjacent adjoint ops are cancelled when multiple
        cancellable operators are present."""

        @CancelInversesInterpreter()
        def f():
            qp.S(0)  # Applied
            qp.adjoint(qp.T(0))
            qp.T(0)
            qp.adjoint(qp.S(0))  # Applied

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        ops = collector.state["ops"]

        expected_ops = [
            qp.S(0),
            qp.adjoint(qp.S(0)),
        ]

        assert ops == expected_ops

    def test_returned_op_is_not_cancelled(self):
        """Test that ops that are returned by the function being transformed are not cancelled."""

        @CancelInversesInterpreter()
        def f():
            qp.PauliX(0)
            return qp.PauliX(0)

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qp.PauliX._primitive
        assert jaxpr.eqns[1].primitive == qp.PauliX._primitive
        assert jaxpr.jaxpr.outvars[0] == jaxpr.eqns[1].outvars[0]

    def test_no_wire_ops_not_cancelled(self):
        """Test that inverse operations with no wires do not get cancelled."""

        @CancelInversesInterpreter()
        def f():
            qp.Identity()
            qp.Identity()
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 4

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qp.Identity(), qp.Identity()]
        assert ops == expected_ops

        expected_meas = [qp.expval(qp.Z(0))]
        assert meas == expected_meas

    def test_dynamic_wires_between_static_wires(self):
        """Test that operations with dynamic wires between operations with static
        wires cause cancellation to not happen."""

        @CancelInversesInterpreter()
        def f(w):
            qp.H(0)
            qp.T(w)
            qp.H(0)
            qp.adjoint(qp.T(w))
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)(0)
        assert len(jaxpr.eqns) == 7

        dyn_wire = 0
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_wire)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]
        expected_meas = [qp.expval(qp.Z(0))]

        expected_ops = [qp.H(0), qp.T(0), qp.H(0), qp.adjoint(qp.T(0))]
        assert ops == expected_ops
        assert meas == expected_meas

        dyn_wire = 1
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_wire)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qp.H(0), qp.T(1), qp.H(0), qp.adjoint(qp.T(1))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_same_dyn_wires_cancel(self):
        """Test that ops on the same dynamic wires get cancelled."""

        @CancelInversesInterpreter()
        def f(w):
            qp.H(0)
            qp.T(w)
            qp.adjoint(qp.T(w))
            qp.H(0)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)(0)
        assert len(jaxpr.eqns) == 4

        dyn_wire = 0
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_wire)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qp.H(0), qp.H(0)]
        expected_meas = [qp.expval(qp.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_same_dyn_param_cancel(self):
        """Test that ops on the same wires with the same dynamic parameter get cancelled."""

        @CancelInversesInterpreter()
        def f(x):
            qp.H(1)
            qp.RX(x, 1)
            qp.adjoint(qp.RX(x, 1))
            qp.H(0)
            return qp.expval(qp.Z(0))

        dyn_param = jax.numpy.array(0.1)
        jaxpr = jax.make_jaxpr(f)(dyn_param)
        assert len(jaxpr.eqns) == 4

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_param)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qp.H(1), qp.H(0)]
        expected_meas = [qp.expval(qp.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_different_dyn_params_not_cancel(self):
        """Test that ops on the same wires with different dynamic parameters do not get cancelled."""

        @CancelInversesInterpreter()
        def f(x, y):
            qp.H(1)
            qp.RX(x, 1)
            qp.adjoint(qp.RX(y, 1))
            qp.H(0)
            return qp.expval(qp.Z(0))

        dyn_param1, dyn_param2 = jax.numpy.array(0.1), jax.numpy.array(0.2)
        jaxpr = jax.make_jaxpr(f)(dyn_param1, dyn_param2)

        assert len(jaxpr.eqns) == 7
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, dyn_param1, dyn_param2)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [
            qp.H(1),
            qp.RX(dyn_param1, 1),
            qp.adjoint(qp.RX(dyn_param2, 1)),
            qp.H(0),
        ]
        expected_meas = [qp.expval(qp.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_different_dyn_wires_interleaved(self):
        """Test that ops on different dynamic wires interleaved with each other
        do not cancel."""

        @CancelInversesInterpreter()
        def f(w1, w2):
            qp.H(w1)
            qp.X(w2)
            qp.H(w1)
            qp.X(w2)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)(0, 0)
        assert len(jaxpr.eqns) == 6

        dyn_wires = (0, 0)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *dyn_wires)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qp.H(0), qp.X(0), qp.H(0), qp.X(0)]
        expected_meas = [qp.expval(qp.Z(0))]
        assert ops == expected_ops
        assert meas == expected_meas

        dyn_wires = (0, 1)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *dyn_wires)
        ops = collector.state["ops"]
        meas = collector.state["measurements"]

        expected_ops = [qp.H(0), qp.X(1), qp.H(0), qp.X(1)]
        assert ops == expected_ops
        assert meas == expected_meas

    def test_ctrl_higher_order_primitive(self):
        """Test that ctrl higher order primitives are transformed correctly."""

        def ctrl_fn(y):
            qp.S(0)
            qp.Hadamard(1)
            qp.Hadamard(1)
            qp.adjoint(qp.S(0))
            qp.RX(y, 0)  # Applied

        @CancelInversesInterpreter()
        def f(x):
            qp.RX(x, 0)
            qp.ctrl(ctrl_fn, [2, 3])(x)
            qp.RY(x, 1)

        arg = 1.5
        jaxpr = jax.make_jaxpr(f)(arg)
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, arg)
        ops = collector.state["ops"]

        excepted_ops = [
            qp.RX(arg, 0),
            qp.ctrl(qp.RX(arg, 0), [2, 3]),
            qp.RY(arg, 1),
        ]

        assert ops == excepted_ops

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_higher_order_primitive(self, lazy):
        """Test that adjoint higher order primitives are transformed correctly."""

        def adjoint_fn(y):
            qp.S(0)
            qp.Hadamard(1)
            qp.Hadamard(1)
            qp.adjoint(qp.S(0))
            qp.RX(y, 0)

        @CancelInversesInterpreter()
        def f(x):
            qp.RX(x, 0)
            qp.adjoint(adjoint_fn, lazy=lazy)(x)
            qp.RY(x, 1)

        arg = 1.5
        jaxpr = jax.make_jaxpr(f)(arg)
        assert len(jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, arg)
        ops = collector.state["ops"]

        expected_ops = [
            qp.RX(arg, 0),
            qp.adjoint(qp.RX(arg, 0), lazy=lazy),
            qp.RY(arg, 1),
        ]
        assert ops == expected_ops

    def test_cond_higher_order_primitive(self):
        """Test that cond higher order primitives are transformed correctly."""

        @CancelInversesInterpreter()
        def f(x):
            qp.RX(x, 0)

            @qp.cond(x > 2)
            def cond_fn():
                qp.Hadamard(0)
                qp.Hadamard(0)
                return qp.S(0)

            @cond_fn.else_if(x > 1)
            def _else_if():
                qp.S(0)
                qp.adjoint(qp.S(0))
                return qp.T(0)

            @cond_fn.otherwise
            def _else():
                qp.adjoint(qp.T(0))
                qp.T(0)
                return qp.Hadamard(0)

            return cond_fn()

        jaxpr = jax.make_jaxpr(f)(1.5)
        # 2 primitives for true and elif branch conditions of the conditional
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[2].primitive == qp.RX._primitive
        assert jaxpr.eqns[3].primitive == cond_prim

        # true branch
        branch = jaxpr.eqns[3].params["jaxpr_branches"][0]
        assert len(branch.eqns) == 1
        assert branch.eqns[0].primitive == qp.S._primitive
        assert branch.outvars[0] == branch.eqns[0].outvars[0]

        # elif branch
        branch = jaxpr.eqns[3].params["jaxpr_branches"][1]
        assert len(branch.eqns) == 1
        assert branch.eqns[0].primitive == qp.T._primitive
        assert branch.outvars[0] == branch.eqns[0].outvars[0]

        # true branch
        branch = jaxpr.eqns[3].params["jaxpr_branches"][2]
        assert len(branch.eqns) == 1
        assert branch.eqns[0].primitive == qp.Hadamard._primitive
        assert branch.outvars[0] == branch.eqns[0].outvars[0]

    def test_for_loop_higher_order_primitive(self):
        """Test that for_loop higher order primitives are transformed correctly."""

        @CancelInversesInterpreter()
        def f(x, n):
            qp.RX(x, 0)

            @qp.for_loop(n)
            def loop_fn(i):  # pylint: disable=unused-argument
                qp.S(0)
                qp.Hadamard(1)
                qp.Hadamard(1)
                qp.adjoint(qp.S(0))
                qp.RX(x, 0)

            loop_fn()
            qp.RY(x, 1)

        jaxpr = jax.make_jaxpr(f)(1.5, 4)
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == for_loop_prim
        assert jaxpr.eqns[2].primitive == qp.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 1
        assert inner_jaxpr.eqns[0].primitive == qp.RX._primitive

    def test_while_loop_higher_order_primitive(self):
        """Test that while_loop higher order primitives are transformed correctly."""

        @CancelInversesInterpreter()
        def f(x, n):
            qp.RX(x, 0)

            @qp.while_loop(lambda i: i < 2 * n)
            def loop_fn(i):
                qp.S(0)
                qp.Hadamard(1)
                qp.Hadamard(1)
                qp.adjoint(qp.S(0))
                qp.RX(x, 0)
                return i + 1

            loop_fn(x)
            qp.RY(x, 1)

        jaxpr = jax.make_jaxpr(f)(1.5, 4)
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == while_loop_prim
        assert jaxpr.eqns[2].primitive == qp.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 2
        # The i + 1 primitive and the RX may get reordered, but the outcome will not be impacted
        assert any(eqn.primitive == qp.RX._primitive for eqn in inner_jaxpr.eqns)

        # Check that the output of the i + 1 is returned
        if inner_jaxpr.eqns[0].primitive == qp.RX._primitive:
            add_eqn = inner_jaxpr.eqns[1]
        else:
            add_eqn = inner_jaxpr.eqns[0]
        assert add_eqn.primitive.name == "add"
        assert inner_jaxpr.outvars[0] == add_eqn.outvars[0]

    def test_qnode_higher_order_primitive(self):
        """Test that qnode higher order primitives are transformed correctly."""
        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev)
        def circuit(y):
            qp.S(0)
            qp.Hadamard(1)
            qp.Hadamard(1)
            qp.adjoint(qp.S(0))
            qp.RX(y, 0)
            return qp.expval(qp.PauliZ(0))

        @CancelInversesInterpreter()
        def f(x):
            qp.RX(x, 0)
            circuit(x)
            qp.RY(x, 1)

        jaxpr = jax.make_jaxpr(f)(1.5)
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == qnode_prim
        assert jaxpr.eqns[2].primitive == qp.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["qfunc_jaxpr"]
        assert len(inner_jaxpr.eqns) == 3
        assert inner_jaxpr.eqns[0].primitive == qp.RX._primitive
        assert inner_jaxpr.eqns[1].primitive == qp.PauliZ._primitive
        assert inner_jaxpr.eqns[2].primitive == qp.measurements.ExpectationMP._obs_primitive

    @pytest.mark.parametrize("grad_fn", [qp.grad, qp.jacobian])
    def test_grad_and_jac_higher_order_primitives(self, grad_fn):
        """Test that grad and jacobian higher order primitives are transformed correctly."""
        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev)
        def circuit(y):
            qp.S(0)
            qp.Hadamard(1)
            qp.Hadamard(1)
            qp.adjoint(qp.S(0))
            qp.RX(y, 0)
            return qp.expval(qp.PauliZ(0))

        @CancelInversesInterpreter()
        def f(x):
            qp.RX(x, 0)
            out = grad_fn(circuit)(x)
            qp.RY(x, 1)
            return out

        jaxpr = jax.make_jaxpr(f)(1.5)
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == jacobian_prim
        assert jaxpr.eqns[2].primitive == qp.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr"]
        assert len(inner_jaxpr.eqns) == 1
        qfunc_jaxpr = inner_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qp.RX._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qp.PauliZ._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_mid_circuit_measurement(self):
        """Test that mid-circuit measurements are correctly handled."""

        @CancelInversesInterpreter()
        def circuit():
            qp.S(0)
            qp.measure(0)
            qp.adjoint(qp.S(0))
            return qp.expval(qp.PauliZ(0))

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 6

        assert jaxpr.eqns[0].primitive == qp.S._primitive
        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[2].primitive == qp.S._primitive
        assert jaxpr.eqns[3].primitive == qp.ops.Adjoint._primitive
        assert jaxpr.eqns[4].primitive == qp.PauliZ._primitive
        assert jaxpr.eqns[5].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_mid_circuit_measurement_not_blocked(self):
        """Test that mid-circuit measurements do not block the cancellation of adjacent inverses."""

        @CancelInversesInterpreter()
        def circuit():
            qp.S(0)
            qp.adjoint(qp.S(0))
            qp.measure(0)
            qp.H(1)
            qp.adjoint(qp.H(1))
            return qp.expval(qp.PauliZ(0))

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == measure_prim
        assert jaxpr.eqns[1].primitive == qp.PauliZ._primitive
        assert jaxpr.eqns[2].primitive == qp.measurements.ExpectationMP._obs_primitive


def test_cancel_inverses_plxpr_to_plxpr():
    """Test that transforming plxpr works."""

    def circuit():
        qp.X(0)
        qp.S(1)
        qp.X(0)
        qp.adjoint(qp.S(1))
        return qp.expval(qp.Z(0))

    jaxpr = jax.make_jaxpr(circuit)()
    transformed_jaxpr = cancel_inverses_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {})
    assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
    assert len(transformed_jaxpr.eqns) == 2
    assert transformed_jaxpr.eqns[0].primitive == qp.PauliZ._primitive
    assert transformed_jaxpr.eqns[1].primitive == qp.measurements.ExpectationMP._obs_primitive
