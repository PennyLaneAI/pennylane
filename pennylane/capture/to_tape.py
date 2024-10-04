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
"""
This submodule contains a conversion function from plxpr to a tape.
"""

import pennylane as qml

from .base_interpreter import FlattenedHigherOrderPrimitives, PlxprInterpreter
from .primitives import adjoint_transform_prim, cond_prim, ctrl_transform_prim, measure_prim


class CollectOpsandMeas(PlxprInterpreter):
    """Collect the dropped operations and measurements in a jaxpr. Used by ``convert_to_tape``.

    .. code-block:: python

        @qml.for_loop(3)
        def loop(i):
            qml.X(i)

        def f(x):
            loop()
            qml.adjoint(qml.S)(0)
            m0 = qml.measure(0)
            qml.RX(2*x, 0)
            return qml.probs(wires=0), qml.expval(qml.Z(1))

    >>> jaxpr = jax.make_jaxpr(f)(0.5)
    >>> collector = CollectOpsandMeas()
    >>> collector.eval(jaxpr.jaxpr, jaxpr.consts, 1.2)
    [probs(wires=[0]), expval(Z(1))]
    >>> collector.state
    {'ops': [X(0),
    X(1),
    X(2),
    Adjoint(S(wires=[0])),
    measure(wires=[0]),
    RX(Array(2.4, dtype=float32, weak_type=True), wires=[0])],
    'measurements': [probs(wires=[0]), expval(Z(1))]}

    After an execution, the collected operations and measurements are available in the ``state``
    property.

    Note that if the same instance is used again, the new operations will be appended onto the
    same state.

    >>> collector = CollectOpsandMeas()
    >>> collector(qml.T)(0)
    >>> collector.state['ops']
    [T(wires=[0])]
    >>> collector(qml.S)(0)
    >>> collector.state['ops']
    [T(wires=[0]), S(wires=[0])

    """

    def __init__(self, state=None):
        self.state = state
        super().__init__()

    def setup(self):
        if self.state is None:
            self.state = {"ops": [], "measurements": []}

    def interpret_operation(self, op: "pennylane.operation.Operator"):
        self.state["ops"].append(op)

    def interpret_measurement_eqn(self, primitive, *invals, **params):
        mp = primitive.impl(*invals, **params)
        self.state["measurements"].append(mp)
        return mp


# pylint: disable=protected-access
CollectOpsandMeas._primitive_registrations.update(FlattenedHigherOrderPrimitives)


@CollectOpsandMeas.register_primitive(adjoint_transform_prim)
def _(self, *invals, jaxpr, lazy, n_consts):
    """Handle a adjoint transform primitive by collecting the operations in the jaxpr, and
    then applying their adjoint in reverse order."""
    consts = invals[:n_consts]
    args = invals[n_consts:]
    child = CollectOpsandMeas()
    child.eval(jaxpr, consts, *args)
    assert child.state

    for op in reversed(child.state["ops"]):
        self.state["ops"].append(qml.adjoint(op, lazy=lazy))

    return []


@CollectOpsandMeas.register_primitive(ctrl_transform_prim)
def _(self, *invals, n_control, jaxpr, n_consts, **params):
    """Handle a control transform primitive by collecting the operations in the jaxpr, and their applying their controlled
    versions.
    """
    consts = invals[:n_consts]
    args = invals[n_consts:-n_control]
    control = invals[-n_control:]

    child = CollectOpsandMeas()
    child.eval(jaxpr, consts, *args)
    assert child.state

    for op in child.state["ops"]:
        self.state["ops"].append(qml.ctrl(op, control=control, **params))

    return []


@CollectOpsandMeas.register_primitive(cond_prim)
def _(*all_args, jaxpr_branches, n_consts_per_branch, n_args):
    """
    Placeholder for custom logic for hadnling the controlled primitive.
    """
    raise NotImplementedError


@CollectOpsandMeas.register_primitive(measure_prim)
def _(self, wires, reset, postselect):
    m0 = qml.measure(wires, reset=reset, postselect=postselect)
    self.state["ops"].extend(m0.measurements)
    return m0


def convert_to_tape(jaxpr: "jax.core.Jaxpr", consts, *args, shots=None):
    """Convert a jaxpr into a tape.

    Args:
        jaxpr (jax.core.Jaxpr): a pennylane variant jaxpr
        consts (list): the consts for the jaxpr
        *args : the arguments to execute the jaxpr with

    Keyword Args:
        shots (None, int, Sequence[int], Shots): the shots for the tape.

    .. code-block:: python

        @qml.for_loop(3)
        def loop(i):
            qml.X(i)

        def f(x):
            loop()
            qml.adjoint(qml.S)(0)
            m0 = qml.measure(0)
            qml.RX(2*x, 0)
            return qml.probs(wires=0), qml.expval(qml.Z(1))

        jaxpr = jax.make_jaxpr(f)(0.5)
        tape = qml.capture.convert_to_tape(jaxpr.jaxpr, jaxpr.consts, 1.2)
        print(qml.drawer.tape_text(tape, decimals=2))

    .. code-block::

        0: ──X──S†──┤↗├──RX(2.40)─┤  Probs
        1: ──X────────────────────┤  <Z>
        2: ──X────────────────────┤

    """

    collector = CollectOpsandMeas()
    collector.eval(jaxpr, consts, *args)
    assert collector.state
    return qml.tape.QuantumScript(
        collector.state["ops"], collector.state["measurements"], shots=shots
    )
