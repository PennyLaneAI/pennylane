# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Defines transforms that can be applied to jaxprs.
"""
from jax.core import Literal, DropVar
from jax._src.util import safe_map
from pennylane.operation import Operator
from pennylane.ops.qubit.non_parametric_ops import X
from pennylane.ops.op_math.controlled import Controlled
from pennylane.ops.op_math.adjoint import Adjoint
from .meta_type import get_abstract_type

_abstract_operator_type = get_abstract_type(Operator)


def _is_operator_primitive(primitive):
    return issubclass(getattr(primitive, "_class", object), Operator)


def _make_jaxpr_env():
    env = {}

    def read(var):
        if type(var) is Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    return read, write


def ctrl_jaxpr(
    jaxpr, consts, *args, control_wires=None, control_values=None, work_wires=None, **kwargs
):
    read, write = _make_jaxpr_env()

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)
    abs_op_type = get_abstract_type(Operator)
    flip_controls_on_zero = control_values is not None  # Slightly different logic than PL
    if flip_controls_on_zero:
        [
            write(DropVar, X._primitive.bind(wires=w))
            for w, val in zip(control_wires, control_values)
            if not val
        ]
    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)

        if _is_operator_primitive(eqn.primitive):
            outvals = [
                Controlled._primitive.bind(
                    eqn.primitive.bind(*invals, **eqn.params),
                    control_wires=control_wires,
                    control_values=None,
                    work_wires=work_wires,
                )
            ]
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]

        safe_map(write, eqn.outvars, outvals)

    if flip_controls_on_zero:
        [
            write(DropVar, X._primitive.bind(wires=w))
            for w, val in zip(control_wires, control_values)
            if not val
        ]

    return safe_map(read, jaxpr.outvars)


def adjoint_jaxpr(jaxpr, consts, *args, **kwargs):
    env = {}

    def read(var):
        if type(var) is Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)
    abs_op_type = get_abstract_type(Operator)
    op_eqns = []
    for eqn in jaxpr.eqns:

        if _is_operator_primitive(eqn.primitive):
            # Store until after all classical processing is completed
            op_eqns.append(eqn)
            continue
        invals = safe_map(read, eqn.invars)
        outvals = eqn.primitive.bind(*invals, **eqn.params)

        if not eqn.primitive.multiple_results:
            outvals = [outvals]

        safe_map(write, eqn.outvars, outvals)

    for eqn in op_eqns[::-1]:
        invals = safe_map(read, eqn.invars)
        outval = Adjoint._primitive.bind(eqn.primitive.bind(*invals, **eqn.params))
        write(eqn.outvars[0], outval)

    return safe_map(read, jaxpr.outvars)
