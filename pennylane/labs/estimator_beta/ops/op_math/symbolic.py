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
r"""Resource operators for symbolic operations."""

from functools import wraps
from inspect import signature
from typing import Callable

import pennylane.labs.estimator_beta as qre
from pennylane.labs.estimator_beta import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
)
from pennylane.operation import Operator
from pennylane.queuing import AnnotatedQueue, QueuingManager
from pennylane.wires import Wires


def _generate_name(func_name, func_sig, include_params, *args, **kwargs):
    if include_params is None:
        return func_name

    param_strs = []
    for param_name in include_params:

        param_kind = str(func_sig.parameters[param_name].kind)
        param_index = list(func_sig.parameters.keys()).index(param_name)
        if (param_kind == "KEYWORD_ONLY") or (param_kind == "POSITIONAL_OR_KEYWORD"):
            if param_name in kwargs:
                param_value = kwargs[param_name]
            else:
                param_value = args[param_index]

        if param_kind == "VAR_POSITIONAL":
            param_value = args[param_index:]

        if param_kind == "VAR_KEYWORD":
            param_value = {k: v for k, v in kwargs.items() if k not in func_sig.parameters}

        param_str = param_name + " = " + str(param_value)
        param_strs.append(param_str)

    name = func_name + "(" + ", ".join(param_strs) + ")"
    return name


def prod(qfunc: Callable, include_params=None):

    @wraps(qfunc)
    def wrapper(*args, **kwargs):
        func_name = qfunc.__name__
        func_sig = signature(qfunc)

        name = _generate_name(func_name, func_sig, include_params, *args, **kwargs)
        return ResourceQfunc(name, qfunc, *args, **kwargs)

    return wrapper


class ResourceQfunc(ResourceOperator):
    """A class which generates ResourceOperator instances from quantum functions"""

    resource_keys = {"name", "num_wires", "cmpr_ops"}

    def __init__(self, name, resource_decomp_fn, *resource_args, **resource_kwargs):
        self.name = name

        with QueuingManager.stop_recording():
            with AnnotatedQueue() as q:
                resource_decomp_fn(*resource_args, **resource_kwargs)

            decomp = []
            for op in q.queue:  # Filter the queue and only pull the operators
                if isinstance(op, ResourceOperator):
                    decomp.append(op)
                elif isinstance(op, Operator):
                    decomp.append(qre._map_to_resource_op(op))
                else:
                    continue

        self.cmpr_ops = tuple(op.resource_rep_from_op() for op in decomp)

        ops_wires = Wires.all_wires([op.wires for op in decomp if op.wires is not None])
        num_unique_wires_required = max(op.num_wires for op in self.cmpr_ops)

        if (
            len(ops_wires) < num_unique_wires_required
        ):  # If factors didn't provide enough wire labels
            self.wires = None  # we assume they all act on the same set
            self.num_wires = num_unique_wires_required

        else:  # If there are more wire labels, use that as the operator wires
            self.wires = ops_wires
            self.num_wires = len(self.wires)

        self.queue()

    @property
    def resource_params(self):
        return {
            "name": self.name,
            "num_wires": self.num_wires,
            "cmpr_ops": self.cmpr_ops,
        }

    @classmethod
    def resource_rep(cls, name, num_wires, cmpr_ops):
        params = {
            "name": name,
            "num_wires": num_wires,
            "cmpr_ops": cmpr_ops,
        }
        return CompressedResourceOp(cls, num_wires, params, name=name)

    @classmethod
    def resource_decomp(cls, name, num_wires, cmpr_ops):
        return [GateCount(cmpr_op) for cmpr_op in cmpr_ops]

    @staticmethod
    def tracking_name(name, num_wires, cmpr_ops) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return name
