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
Defines a function for converting plxpr to a tape.
"""
from copy import copy

import pennylane as qml
from pennylane.capture.base_interpreter import FlattenedInterpreter
from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    grad_prim,
    jacobian_prim,
    measure_prim,
    qnode_prim,
)
from pennylane.measurements import MeasurementValue, get_mcm_predicates

from .qscript import QuantumScript


class CollectOpsandMeas(FlattenedInterpreter):
    """Collect the dropped operations and measurements in a plxpr. Used by ``convert_to_tape``.

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

    >>> from pennylane.tape.plxpr_conversion import CollectOpsandMeas
    >>> from jax import make_jaxpr
    >>> qml.capture.enable()
    >>> plxpr = make_jaxpr(f)(0.5)
    >>> collector = CollectOpsandMeas()
    >>> collector.eval(plxpr.jaxpr, plxpr.consts, 1.2)
    [probs(wires=[0]), expval(Z(1))]
    >>> collector.state
    {'ops': [X(0),
    X(1),
    X(2),
    Adjoint(S(0)),
    measure(wires=[0]),
    RX(Array(2.4, dtype=float32, weak_type=True), wires=[0])],
    'measurements': [probs(wires=[0]), expval(Z(1))]}

    After execution, the collected operations and measurements are available in the ``state``
    property.

    Note that if the same instance is used again, the new operations will be appended to the
    same state.

    >>> collector = CollectOpsandMeas()
    >>> collector(qml.T)(0)
    >>> collector.state['ops']
    [T(0)]
    >>> collector(qml.S)(0)
    >>> collector.state['ops']
    [T(0), S(0)]

    """

    def __init__(self, state=None):
        self.state = state
        super().__init__()

    def setup(self):
        if self.state is None:
            self.state = {"ops": [], "measurements": []}

    def interpret_operation(self, op: "pennylane.operation.Operator"):
        self.state["ops"].append(op)

    def interpret_measurement(self, measurement):
        self.state["measurements"].append(measurement)
        return measurement


@CollectOpsandMeas.register_primitive(adjoint_transform_prim)
def _(self, *invals, jaxpr, lazy, n_consts):
    """Handle an adjoint transform primitive by collecting the operations in the jaxpr, and
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
    """Handle a control transform primitive by collecting the operations in the jaxpr,
    and then applying their controlled versions.
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
def _(self, *all_args, jaxpr_branches, consts_slices, args_slice):
    n_branches = len(jaxpr_branches)
    conditions = all_args[:n_branches]
    args = all_args[args_slice]

    # Find predicates that use mid-circuit measurements. We don't check the last
    # condition as that is always `True`.
    mcm_conditions = tuple(pred for pred in conditions[:-1] if isinstance(pred, MeasurementValue))
    if mcm_conditions:
        if len(mcm_conditions) != len(conditions) - 1:
            raise ValueError(
                "Cannot use qml.cond with a combination of mid-circuit measurements "
                "and other classical conditions as predicates."
            )
        conditions = get_mcm_predicates(mcm_conditions)

    for pred, jaxpr, const_slice in zip(conditions, jaxpr_branches, consts_slices):
        consts = all_args[const_slice]
        if jaxpr is None:
            continue
        if isinstance(pred, qml.measurements.MeasurementValue):
            if jaxpr.outvars:
                outvals = [v.aval for v in jaxpr.outvars]
                raise ValueError(
                    (
                        "Conditional branches of mid circuit measurements are not allowed to"
                        f" return anything with plxpr_to_tape and CollectOpsandMeas. Branch returns {outvals}"
                    )
                )
            child = CollectOpsandMeas()
            child.eval(jaxpr, consts, *args)
            assert child.state
            self.state["ops"].extend(qml.ops.Conditional(pred, op) for op in child.state["ops"])
        elif pred:
            return copy(self).eval(jaxpr, consts, *args)
    return ()


@CollectOpsandMeas.register_primitive(measure_prim)
def _(self, wires, reset, postselect):
    m0 = qml.measure(wires, reset=reset, postselect=postselect)
    self.state["ops"].extend(m0.measurements)
    return m0


# pylint: disable=unused-argument
@CollectOpsandMeas.register_primitive(grad_prim)
def _(self, *invals, jaxpr, n_consts, **params):
    raise NotImplementedError("CollectOpsandMeas cannot handle the grad primitive")


# pylint: disable=unused-argument
@CollectOpsandMeas.register_primitive(jacobian_prim)
def _(self, *invals, jaxpr, n_consts, **params):
    raise NotImplementedError("CollectOpsandMeas cannot handle the jacobian primitive")


@CollectOpsandMeas.register_primitive(qnode_prim)
def _(
    self, *invals, shots, qnode, device, execution_config, qfunc_jaxpr, n_consts
):  # pylint: disable=too-many-arguments,unused-argument
    consts = invals[:n_consts]
    args = invals[n_consts:]

    child = CollectOpsandMeas()
    out = child.eval(qfunc_jaxpr, consts, *args)
    assert child.state
    self.state["ops"].extend(child.state["ops"])
    self.state["measurements"].extend(child.state["measurements"])
    return out


def plxpr_to_tape(plxpr: "jax.core.Jaxpr", consts, *args, shots=None) -> QuantumScript:
    """Convert a plxpr into a tape.

    Args:
        plxpr (jax.core.Jaxpr): a pennylane variant jaxpr
        consts (list): the consts for the jaxpr
        *args : the arguments to execute the plxpr with

    Keyword Args:
        shots (None, int, Sequence[int], Shots): the shots for the tape.

    Returns:
        QuantumScript: a single quantum script containing the quantum operations and measurements

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

        qml.capture.enable()

        plxpr = jax.make_jaxpr(f)(0.5)
        tape = qml.tape.plxpr_to_tape(plxpr.jaxpr, plxpr.consts, 1.2)
        print(qml.drawer.tape_text(tape, decimals=2))

    .. code-block::

        0: ──X──S†──┤↗├──RX(2.40)─┤  Probs
        1: ──X────────────────────┤  <Z>
        2: ──X────────────────────┤

    """

    collector = CollectOpsandMeas()
    collector.eval(plxpr, consts, *args)
    assert collector.state
    return QuantumScript(collector.state["ops"], collector.state["measurements"], shots=shots)
