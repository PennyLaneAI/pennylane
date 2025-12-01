# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the abstractions for subroutines.
"""
from copy import deepcopy
from functools import update_wrapper
from inspect import signature

import jax
import numpy as np

from pennylane import capture, queuing
from pennylane.capture import subroutine as capture_subroutine
from pennylane.operation import Operator
from pennylane.pytrees import flatten
from pennylane.wires import Wires


def default_setup_inputs(*args, **kwargs):
    return args, kwargs


array_types = (jax.numpy.ndarray, np.ndarray)
iterable_wires_types = (
    list,
    tuple,
    Wires,
    range,
    capture.autograph.ag_primitives.PRange,
    set,
    *array_types,
)


def _setup_wires(wires):
    if isinstance(wires, array_types) and wires.shape == ():
        return (wires,)
    elif isinstance(wires, iterable_wires_types):
        return tuple(wires)
    return (wires,)


class SubroutineOp(Operator):
    """This class combines a subroutine definition together with what it was called
    with and it's decomposition to be backward compatible with operator.

    """

    _primitive = None

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        raise ValueError(
            "SubroutineOp's should never be directly captured. That should occur in Subroutine instead."
        )

    grad_method = None

    def _flatten(self):
        dynamic_args = {
            arg: self._bound_args.arguments[arg] for arg in self.subroutine.dynamic_argnames
        }
        static_args = {
            arg: self._bound_args.arguments[arg] for arg in self.subroutine.static_argnames
        }
        for wire_argname in self.subroutine.wire_argnames:
            static_args[wire_argname] = self._bound_args.arguments[wire_argname]
        static_args = tuple((key, value) for key, value in static_args.items())
        return (dynamic_args,), (self.subroutine, static_args)

    @classmethod
    def _unflatten(cls, data, metadata):
        subroutine = metadata[0]
        return subroutine.construct_op(**data[0], **dict(metadata[1]))

    def __init__(self, subroutine: "Subroutine", bound_args, decomposition, output):
        self._subroutine = subroutine
        self._bound_args = bound_args
        self._decomp = decomposition
        self.output = output

        wires = []
        for wire_argname in self._subroutine.wire_argnames:
            wires.extend(_setup_wires(self._bound_args.arguments[wire_argname]))

        super().__init__(wires=wires)
        self.name = subroutine.definition.__name__

        dynamic_args = [self._bound_args.arguments[arg] for arg in self.subroutine.dynamic_argnames]
        self.data = tuple(flatten(dynamic_args)[0])

    @property
    def subroutine(self):
        return self._subroutine

    def map_wires(self, wire_map):
        new_args = deepcopy(self._bound_args)
        for wire_argname in self._subroutine.wire_argnames:
            new_wires = tuple(wire_map.get(w, w) for w in self._bound_args.arguments[wire_argname])
            new_args.arguments[wire_argname] = new_wires
        return self.subroutine.construct_op(*new_args.args, **new_args.kwargs)

    def decomposition(self):
        if queuing.QueuingManager.recording():
            _ = [queuing.apply(op) for op in self._decomp]
        return self._decomp

    def label(
        self, decimals: int | None = None, base_label: str | None = None, cache: dict | None = None
    ) -> str:
        return super().label(decimals, base_label=self.name, cache=cache)


class Subroutine:
    """The definition of a Subroutine, compatible both with program capture and backward
    compatible with operators."""

    def __repr__(self):
        return f"<Subroutine: {self.definition.__name__}>"

    def __instancecheck__(self, instance) -> bool:
        return isinstance(instance, SubroutineOp) and instance.subroutine is self

    def __init__(
        self,
        definition,
        setup_inputs=default_setup_inputs,
        static_argnames: set | frozenset = frozenset(),
        wire_argnames: set | frozenset = frozenset({"wires"}),
    ):
        self.definition = definition
        self.setup_inputs = setup_inputs
        self.static_argnames = static_argnames
        self.subroutine = capture_subroutine(definition, static_argnames=static_argnames)
        self.wire_argnames = wire_argnames
        self._signature = signature(definition)
        update_wrapper(self, definition)

    @property
    def signature(self):
        return self._signature

    @property
    def dynamic_argnames(self):
        def is_static(name):
            return name in self.static_argnames or name in self.wire_argnames

        return frozenset(name for name in self._signature.parameters if not is_static(name))

    def construct_op(self, *args, **kwargs):
        args, kwargs = self.setup_inputs(*args, **kwargs)
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for wire_argname in self.wire_argnames:
            register = bound_args.arguments[wire_argname]
            bound_args.arguments[wire_argname] = Wires(register)

        with queuing.AnnotatedQueue() as decomposition:
            output = self.definition(*bound_args.args, **bound_args.kwargs)
        return SubroutineOp(self, bound_args, decomposition.queue, output)

    def __call__(self, *args, **kwargs):
        if capture.enabled():
            args, kwargs = self.setup_inputs(*args, **kwargs)
            bound_args = self._signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for wire_argname in self.wire_argnames:
                register = bound_args.arguments[wire_argname]
                if isinstance(register, int):
                    register = [register]
                if len(register) > 0:
                    bound_args.arguments[wire_argname] = jax.numpy.stack(register)
            return self.subroutine(*bound_args.args, **bound_args.kwargs)

        op = self.construct_op(*args, **kwargs)
        return op if op.output is None else op.output
