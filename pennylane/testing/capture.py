# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for testing programs captured with :mod:`~pennylane.capture`.
"""

from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Primitive

from pennylane.capture import pause
from pennylane.capture.primitives import operator_p
from pennylane.core import Operator2
from pennylane.core.qscript import QuantumScript
from pennylane.ops.mid_measure import MidMeasure, PauliMeasure
from pennylane.tape.plxpr_conversion import CollectOpsandMeas


def plxpr_to_tape(plxpr: Jaxpr, consts, *args) -> QuantumScript:
    """Convert a plxpr into a tape.

    Args:
        plxpr (jax.extend.core.Jaxpr): the jaxpr to extract a program from
        consts (list): the consts for the jaxpr
        *args : the arguments to execute the plxpr with

    Returns:
        QuantumScript: a single quantum script containing the quantum operations and measurements

    .. code-block:: python

        @qp.for_loop(3)
        def loop(i):
            qp.X(i)

        def f(x):
            loop()
            qp.adjoint(qp.S)(0)
            m0 = qp.measure(0)
            qp.RX(2*x, 0)
            return qp.probs(wires=0), qp.expval(qp.Z(1))

        qp.capture.enable()

        plxpr = jax.make_jaxpr(f)(0.5)
        tape = qp.testing.plxpr_to_tape(plxpr.jaxpr, plxpr.consts, 1.2)
        print(qp.drawer.tape_text(tape, decimals=2))

    .. code-block::

        0: ──X──S†──┤↗├──RX(2.40)─┤  Probs
        1: ──X────────────────────┤  <Z>
        2: ──X────────────────────┤

    """

    collector = CollectOpsandMeas()
    collector.eval(plxpr, consts, *args)
    assert collector.state
    wire_map = collector.state["dynamic_wire_map"]
    mcm_map = {}
    with pause():
        ops = [_map_op_wires(op, wire_map, mcm_map) for op in collector.state["ops"]]
        meas = [_map_meas_wires(m, wire_map, mcm_map) for m in collector.state["measurements"]]
    return QuantumScript(ops, meas)


def _map_op_wires(op, wire_map, mcm_map):
    new_op = op.map_wires(wire_map)
    if isinstance(op, (MidMeasure, PauliMeasure)):
        mcm_map[op] = new_op
    return new_op


def _map_meas_wires(m, wire_map, mcm_map):
    new_meas = m.map_wires(wire_map)
    if m.mv is None:
        return new_meas
    for i, meas in enumerate(m.mv.measurements):
        if meas in mcm_map:
            new_meas.mv.measurements[i] = mcm_map[meas]
    return new_meas


def extract_all_primitives(jaxpr: Jaxpr | ClosedJaxpr) -> set:
    """Collect the primitives of every equation in a jaxpr, including the equations of
    jaxprs nested inside higher-order primitives such as ``qnode``, ``cond`` and ``for_loop``.

    Args:
        jaxpr (jax.extend.core.Jaxpr | jax.extend.core.ClosedJaxpr): the jaxpr to search

    Returns:
        set[jax.extend.core.Primitive]: all primitives found

    **Example**

    >>> qp.capture.enable()
    >>> def f(x):
    ...     @qp.for_loop(3)
    ...     def loop(i):
    ...         qp.RX(x, i)
    ...     loop()
    >>> plxpr = jax.make_jaxpr(f)(0.5)
    >>> sorted(p.name for p in qp.testing.extract_all_primitives(plxpr.jaxpr))
    ['for_loop', 'operator']

    """

    if isinstance(jaxpr, ClosedJaxpr):
        return extract_all_primitives(jaxpr.jaxpr)

    primitives = set()
    for eqn in jaxpr.eqns:

        # add the primitive itself
        primitives.add(eqn.primitive)

        # Search all params rather than specific keys (like 'qfunc_jaxpr') so that new
        # higher-order primitives are covered without changes here.
        for val in eqn.params.values():
            if isinstance(val, (Jaxpr, ClosedJaxpr)):
                primitives.update(extract_all_primitives(val))
            elif isinstance(val, (list, tuple)):
                _jaxprs = (item for item in val if isinstance(item, (Jaxpr, ClosedJaxpr)))
                for _jaxpr in _jaxprs:
                    primitives.update(extract_all_primitives(_jaxpr))

    return primitives


def assert_eqn_matches_op(eqn, expected_op: type) -> None:
    """Assert that a jaxpr equation creates an operator of the expected type.

    Subclasses of :class:`~.core.Operator2` are all captured with the shared ``operator``
    primitive, so the operator class is read from the equation's parameters. Other
    operators are captured with a primitive of their own.

    Args:
        eqn (jax.extend.core.JaxprEqn): the equation to check
        expected_op (type): the expected operator class

    Raises:
        AssertionError: if the equation does not create an ``expected_op``

    **Example**

    >>> qp.capture.enable()
    >>> plxpr = jax.make_jaxpr(lambda x: qp.RX(x, 0))(0.5)
    >>> qp.testing.assert_eqn_matches_op(plxpr.eqns[0], qp.RX)

    """
    if issubclass(expected_op, Operator2):
        assert eqn.primitive == operator_p
        assert eqn.params["op_cls"] == expected_op
    else:
        assert eqn.primitive == expected_op._primitive  # pylint: disable=protected-access


def find_eqns(jaxpr: Jaxpr | ClosedJaxpr, primitive: Primitive) -> list[JaxprEqn]:
    """Find the equations in a jaxpr that use a given primitive.

    Only the top-level equations are searched, not those of jaxprs nested inside
    higher-order primitives.

    Args:
        jaxpr (jax.extend.core.Jaxpr | jax.extend.core.ClosedJaxpr): the jaxpr to search
        primitive (jax.extend.core.Primitive): the primitive to look for

    Returns:
        list[jax.extend.core.JaxprEqn]: the matching equations, in program order

    **Example**

    Unpacking the result checks that there is exactly one matching equation:

    >>> from pennylane.capture.primitives import operator_p
    >>> qp.capture.enable()
    >>> plxpr = jax.make_jaxpr(lambda x: qp.RX(x, 0))(0.5)
    >>> [eqn] = qp.testing.find_eqns(plxpr, operator_p)
    >>> eqn.params["op_cls"]
    <class 'pennylane.ops.qubit.parametric_ops_single_qubit.RX'>

    """
    return [eqn for eqn in jaxpr.eqns if eqn.primitive == primitive]
