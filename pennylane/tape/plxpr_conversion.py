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

import numpy as np

from pennylane import ops
from pennylane.allocation import Allocate, Deallocate, allocate_prim, deallocate_prim
from pennylane.capture import pause
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
from pennylane.measurements import MeasurementValue, get_mcm_predicates, measure
from pennylane.measurements.mid_measure import MidMeasureMP
from pennylane.operation import Operator
from pennylane.wires import DynamicWire

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
    {'ops': [X(0), X(1), X(2), Adjoint(S(0)), MidMeasureMP(wires=[0]), RX(Array(2.4, dtype=float..., weak_type=True), wires=[0])], 'measurements': [probs(wires=[0]), expval(Z(1))], 'dynamic_wire_map': {}}

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
            self.state = {"ops": [], "measurements": [], "dynamic_wire_map": {}}

    def interpret_operation(self, op: Operator):
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
        self.state["ops"].append(ops.adjoint(op, lazy=lazy))

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
        self.state["ops"].append(ops.ctrl(op, control=control, **params))

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
        if isinstance(pred, MeasurementValue):
            if jaxpr.outvars:
                outvals = [v.aval for v in jaxpr.outvars]
                raise ValueError(
                    "Conditional branches of mid circuit measurements are not allowed to"
                    f" return anything with plxpr_to_tape and CollectOpsandMeas. Branch returns {outvals}"
                )
            child = CollectOpsandMeas()
            child.eval(jaxpr, consts, *args)
            assert child.state
            self.state["ops"].extend(ops.Conditional(pred, op) for op in child.state["ops"])
        elif pred:
            return copy(self).eval(jaxpr, consts, *args)
    return ()


@CollectOpsandMeas.register_primitive(measure_prim)
def _(self, wires, reset, postselect):
    m0 = measure(wires, reset=reset, postselect=postselect)
    self.state["ops"].extend(m0.measurements)
    return m0


@CollectOpsandMeas.register_primitive(grad_prim)
def _(self, *invals, jaxpr, n_consts, **params):
    raise NotImplementedError("CollectOpsandMeas cannot handle the grad primitive")


# pylint: disable=unused-argument
@CollectOpsandMeas.register_primitive(jacobian_prim)
def _(self, *invals, jaxpr, n_consts, **params):
    raise NotImplementedError("CollectOpsandMeas cannot handle the jacobian primitive")


@CollectOpsandMeas.register_primitive(qnode_prim)
def _(
    self, *invals, shots_len, qnode, device, execution_config, qfunc_jaxpr, n_consts
):  # pylint: disable=too-many-arguments
    consts = invals[shots_len : shots_len + n_consts]
    args = invals[shots_len + n_consts :]

    child = CollectOpsandMeas()
    out = child.eval(qfunc_jaxpr, consts, *args)
    assert child.state
    self.state["ops"].extend(child.state["ops"])
    self.state["measurements"].extend(child.state["measurements"])
    return out


@CollectOpsandMeas.register_primitive(allocate_prim)
def _(self, *, num_wires, state, restored):
    wires = [DynamicWire() for _ in range(num_wires)]
    num_dynamic_wires = len(self.state["dynamic_wire_map"])
    int_wires = [np.iinfo(np.int32).max - i - num_dynamic_wires for i in range(num_wires)]
    self.state["dynamic_wire_map"].update(dict(zip(int_wires, wires, strict=True)))
    self.state["ops"].append(Allocate(int_wires, state=state, restored=restored))
    return int_wires


@CollectOpsandMeas.register_primitive(deallocate_prim)
def _(self, *wires):
    self.state["ops"].append(Deallocate(wires))
    return []


def plxpr_to_tape(plxpr: "jax.extend.core.Jaxpr", consts, *args, shots=None) -> QuantumScript:
    """Convert a plxpr into a tape.

    Args:
        plxpr (jax.extend.core.Jaxpr): a pennylane variant jaxpr
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
    wire_map = collector.state["dynamic_wire_map"]
    mcm_map = {}
    with pause():
        operations = [_map_op_wires(op, wire_map, mcm_map) for op in collector.state["ops"]]
        measurements = [
            _map_meas_wires(m, wire_map, mcm_map) for m in collector.state["measurements"]
        ]
    return QuantumScript(operations, measurements, shots=shots)


def _map_op_wires(op, wire_map, mcm_map):
    new_op = op.map_wires(wire_map)
    if isinstance(op, MidMeasureMP):
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
