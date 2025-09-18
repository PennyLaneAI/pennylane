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
"""Unit tests for the ``MapWiresInterpreter`` class"""

from functools import partial

# pylint:disable=wrong-import-position, unused-argument
import numpy as np
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]

from pennylane.ops.functions.map_wires import MapWiresInterpreter, map_wires_plxpr_to_plxpr


# pylint: disable=protected-access, expression-not-assigned
class TestMapWiresInterpreter:
    """Unit tests for the MapWiresInterpreter class."""

    def test_single_operation(self):
        """Test that a single operation is transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1})
        def transformed_circuit(x):
            return qml.RZ(x, wires=0)

        jaxpr = jax.make_jaxpr(transformed_circuit)(0.1)

        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert jaxpr.eqns[0].invars[-1].val == 1

    def test_quantum_function_simple(self):
        """Test that a simple circuit is transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1, 1: 2})
        def transformed_circuit(x):
            qml.RZ(x, wires=0)
            qml.PhaseShift(0.1, wires=1)
            return qml.expval(qml.PauliZ(2))

        jaxpr = jax.make_jaxpr(transformed_circuit)(0.1)

        assert len(jaxpr.eqns) == 4

        assert jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert jaxpr.eqns[0].invars[-1].val == 1

        assert jaxpr.eqns[1].primitive == qml.PhaseShift._primitive
        assert jaxpr.eqns[1].invars[-1].val == 2

        assert jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[2].invars[-1].val == 2

    def test_nested_ops(self):
        """Test that a circuit with nested operations is transformed correctly."""

        const = jax.numpy.array(0.1)

        @MapWiresInterpreter(wire_map={0: 1, 1: 2})
        def transformed_circuit(x):
            qml.RZ(const, 0) + qml.RX(x, 0)
            qml.PhaseShift(const, 1) @ qml.RX(x, 0) @ qml.X(0)
            qml.Hadamard(2) @ qml.RX(x, 0) + 4 * qml.RX(const, 0)
            return qml.state()

        jaxpr = jax.make_jaxpr(transformed_circuit)(0.1)

        assert len(jaxpr.eqns) == 15

        assert jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert jaxpr.eqns[0].invars[-1].val == 1

        assert jaxpr.eqns[1].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].invars[-1].val == 1

        assert jaxpr.eqns[2].primitive == qml.ops.Sum._primitive

        assert jaxpr.eqns[3].primitive == qml.PhaseShift._primitive
        assert jaxpr.eqns[3].invars[-1].val == 2

        assert jaxpr.eqns[4].primitive == qml.RX._primitive
        assert jaxpr.eqns[4].invars[-1].val == 1

        assert jaxpr.eqns[5].primitive == qml.ops.Prod._primitive

        assert jaxpr.eqns[6].primitive == qml.X._primitive
        assert jaxpr.eqns[6].invars[-1].val == 1

        assert jaxpr.eqns[7].primitive == qml.ops.Prod._primitive

        assert jaxpr.eqns[8].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[8].invars[-1].val == 2

        assert jaxpr.eqns[9].primitive == qml.RX._primitive
        assert jaxpr.eqns[9].invars[-1].val == 1

        assert jaxpr.eqns[10].primitive == qml.ops.Prod._primitive

        assert jaxpr.eqns[11].primitive == qml.RX._primitive
        assert jaxpr.eqns[11].invars[-1].val == 1

        assert jaxpr.eqns[12].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[13].primitive == qml.ops.Sum._primitive

    def test_controlled_ops(self):
        """Test that a circuit with controlled operations is transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1, 1: 2, 2: 3, 3: 2})
        def f(x):
            qml.CNOT(wires=[0, 1])
            qml.CY(wires=[1, 2])
            qml.CZ(wires=[2, 3])
            qml.CRX(x, wires=[0, 1])
            qml.CRY(x, wires=[1, 2])
            qml.CRZ(x, wires=[2, 3])
            qml.ctrl(qml.RX, (1, 2, 3), control_values=(0, 1, 0))(x, wires=0)
            return qml.var(
                qml.QubitUnitary(
                    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), wires=[0, 1]
                )
            )

        jaxpr = jax.make_jaxpr(f)(0.1)

        assert len(jaxpr.eqns) == 9

        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[0].invars[0].val == 1
        assert jaxpr.eqns[0].invars[1].val == 2

        assert jaxpr.eqns[1].primitive == qml.CY._primitive
        assert jaxpr.eqns[1].invars[0].val == 2
        assert jaxpr.eqns[1].invars[1].val == 3

        assert jaxpr.eqns[2].primitive == qml.CZ._primitive
        assert jaxpr.eqns[2].invars[0].val == 3
        assert jaxpr.eqns[2].invars[1].val == 2

        assert jaxpr.eqns[3].primitive == qml.CRX._primitive
        assert jaxpr.eqns[3].invars[1].val == 1
        assert jaxpr.eqns[3].invars[2].val == 2

        assert jaxpr.eqns[4].primitive == qml.CRY._primitive
        assert jaxpr.eqns[4].invars[1].val == 2
        assert jaxpr.eqns[4].invars[2].val == 3

        assert jaxpr.eqns[5].primitive == qml.CRZ._primitive
        assert jaxpr.eqns[5].invars[1].val == 3
        assert jaxpr.eqns[5].invars[2].val == 2

        assert jaxpr.eqns[6].primitive == qml.capture.primitives.ctrl_transform_prim
        assert jaxpr.eqns[6].params["jaxpr"].eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[6].params["jaxpr"].eqns[0].invars[-1].val == 1

        assert jaxpr.eqns[7].primitive == qml.QubitUnitary._primitive
        assert jaxpr.eqns[7].invars[1].val == 1
        assert jaxpr.eqns[7].invars[2].val == 2

    def test_adjoint(self):
        """Test that adjoint operations are transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1})
        def f(x):
            qml.adjoint(qml.RX)(x, wires=0)
            return qml.expval(qml.PauliZ(3))

        jaxpr = jax.make_jaxpr(f)(0.1)

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.capture.primitives.adjoint_transform_prim
        assert jaxpr.eqns[0].params["jaxpr"].eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[0].params["jaxpr"].eqns[0].invars[-1].val == 1

    def test_qnode_simple(self):
        """Test that a qnode is transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1, 1: 2, 2: 3, 3: 2})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            qml.RZ(x, 0)
            qml.Hadamard(2)
            return qml.expval(qml.PauliZ(3)), qml.probs(wires=[1])

        jaxpr = jax.make_jaxpr(f)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        assert len(inner_jaxpr.eqns) == 5

        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[0].invars[-1].val == 1

        assert inner_jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert inner_jaxpr.eqns[1].invars[-1].val == 3

        assert inner_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert inner_jaxpr.eqns[2].invars[-1].val == 2

        assert inner_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

        assert inner_jaxpr.eqns[4].primitive == qml.measurements.ProbabilityMP._wires_primitive
        assert inner_jaxpr.eqns[4].invars[-1].val == 2

    def test_qnode_batched_parameters(self):
        """Test qnode transformation with batched parameters."""

        @MapWiresInterpreter(wire_map={0: 1, 1: 2})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            qml.RZ(x, 0)
            qml.PhaseShift(0.1, 1)
            return qml.expval(qml.PauliZ(2))

        x = jax.numpy.array([0.1, 0.2, 0.3])
        jaxpr = jax.make_jaxpr(f)(x)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        assert len(inner_jaxpr.eqns) == 4

        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[0].invars[-1].val == 1

        assert inner_jaxpr.eqns[1].primitive == qml.PhaseShift._primitive
        assert inner_jaxpr.eqns[1].invars[-1].val == 2

        assert inner_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert inner_jaxpr.eqns[2].invars[-1].val == 2

    def test_qnode_for_loop(self):
        """Test that a qnode with a for loop is transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            @qml.for_loop(3)
            def g(i):
                qml.RZ(x, 0)

            g()
            return qml.expval(qml.PauliZ(3))

        jaxpr = jax.make_jaxpr(f)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        loop_jaxpr = inner_jaxpr.eqns[0].params["jaxpr_body_fn"]

        assert len(loop_jaxpr.eqns) == 1

        assert loop_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert loop_jaxpr.eqns[0].invars[-1].val == 1

    def test_qnode_while_loop(self):
        """Test that a qnode with a while loop is transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            @qml.while_loop(lambda i: i < 3)
            def g(i):
                qml.RZ(x, 0)
                return i + 1

            g(0)
            return qml.expval(qml.PauliZ(3))

        jaxpr = jax.make_jaxpr(f)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        loop_jaxpr = inner_jaxpr.eqns[0].params["jaxpr_body_fn"]

        assert len(loop_jaxpr.eqns) == 2

        assert loop_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert loop_jaxpr.eqns[0].invars[-1].val == 1

    def test_qnode_conditional(self):
        """Test that a qnode with a conditional is transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            @qml.cond(x > 0.5)
            def g():
                qml.RZ(x, 0)

            g()
            return qml.expval(qml.PauliZ(3))

        jaxpr = jax.make_jaxpr(f)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        cond_jaxpr = inner_jaxpr.eqns[1].params["jaxpr_branches"][0]

        assert len(cond_jaxpr.eqns) == 1

        assert cond_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert cond_jaxpr.eqns[0].invars[-1].val == 1

    def test_qnode_grad(self):
        """Test that the gradient of a qnode is transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            qml.RZ(x, 0)
            return qml.expval(qml.PauliZ(3))

        grad = qml.grad(f)
        jaxpr = jax.make_jaxpr(grad)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"].eqns[0].params["qfunc_jaxpr"]

        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[0].invars[-1].val == 1

    def test_qnode_jacobian(self):
        """Test that the jacobian of a qnode is transformed correctly."""

        @MapWiresInterpreter(wire_map={0: 1})
        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(x):
            qml.RZ(x, 0)
            return qml.expval(qml.PauliZ(3))

        jac = qml.jacobian(f)
        jaxpr = jax.make_jaxpr(jac)(0.1)
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"].eqns[0].params["qfunc_jaxpr"]

        assert inner_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert inner_jaxpr.eqns[0].invars[-1].val == 1

    def test_invalid_wire_map_keys(self):
        """Test that invalid wire mappings raise an error."""
        with pytest.raises(ValueError, match="Wire map keys must be constant positive integers"):
            MapWiresInterpreter(wire_map={"a": 1, 1: 2})

    def test_invalid_wire_map_values(self):
        """Test that invalid wire mappings raise an error."""
        with pytest.raises(ValueError, match="Wire map values must be constant positive integers"):
            MapWiresInterpreter(wire_map={0: "a", 1: 2})


def test_map_wires_plxpr_to_plxpr():
    """Test that transforming plxpr works."""

    def circuit():
        qml.X(0)
        qml.CRX(0, [0, 1])
        qml.CNOT([1, 2])
        return qml.expval(qml.Z(0))

    wire_map = {0: 5, 1: 6, 2: 7}
    targs = (wire_map,)
    tkwargs = {}
    jaxpr = jax.make_jaxpr(circuit)()
    transformed_jaxpr = map_wires_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, targs, tkwargs)
    assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
    assert len(transformed_jaxpr.eqns) == 5

    def assert_wires(orig_eqn, transformed_eqn):
        orig_wires = [orig_eqn.invars[-i].val for i in range(orig_eqn.params["n_wires"], 0, -1)]
        expected_wires = [wire_map[w] for w in orig_wires]
        transformed_wires = [
            transformed_eqn.invars[-i].val for i in range(transformed_eqn.params["n_wires"], 0, -1)
        ]
        assert transformed_wires == expected_wires

    assert transformed_jaxpr.eqns[0].primitive == qml.PauliX._primitive
    assert_wires(jaxpr.eqns[0], transformed_jaxpr.eqns[0])

    assert transformed_jaxpr.eqns[1].primitive == qml.CRX._primitive
    assert_wires(jaxpr.eqns[1], transformed_jaxpr.eqns[1])

    assert transformed_jaxpr.eqns[2].primitive == qml.CNOT._primitive
    assert_wires(jaxpr.eqns[2], transformed_jaxpr.eqns[2])

    assert transformed_jaxpr.eqns[3].primitive == qml.PauliZ._primitive
    assert_wires(jaxpr.eqns[3], transformed_jaxpr.eqns[3])

    assert transformed_jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive


def test_map_wires_plxpr_to_plxpr_queue_warning():
    """Test that a warning is raised if ``queue=True``."""

    def f():
        qml.Hadamard(0)
        return qml.expval(qml.PauliZ(0))

    wire_map = {0: 1}
    targs = (wire_map,)
    tkwargs = {"queue": True}
    jaxpr = jax.make_jaxpr(f)()

    with pytest.warns(UserWarning, match="Cannot set 'queue=True'"):
        map_wires_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, targs, tkwargs)


def test_map_wires_plxpr_to_plxpr_replace_warning():
    """Test that a warning is raised if ``replace=True``."""

    def f():
        qml.Hadamard(0)
        return qml.expval(qml.PauliZ(0))

    wire_map = {0: 1}
    targs = (wire_map,)
    tkwargs = {"replace": True}
    jaxpr = jax.make_jaxpr(f)()

    with pytest.warns(UserWarning, match="Cannot set 'replace=True'"):
        map_wires_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, targs, tkwargs)


def test_map_wire_decorator():
    """Test that the map wires transforms works when applying the plxpr decorator."""

    @qml.capture.expand_plxpr_transforms
    @partial(qml.map_wires, wire_map={0: 1})
    def circuit(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(circuit)(0.1)
    assert len(jaxpr.eqns) == 3
    assert jaxpr.eqns[0].primitive == qml.RX._primitive
    assert jaxpr.eqns[0].invars[-1].val == 1
    assert jaxpr.eqns[1].primitive == qml.PauliZ._primitive
    assert jaxpr.eqns[1].invars[-1].val == 1
    assert jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive
