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

from functools import update_wrapper
from inspect import signature

import jax
import numpy as np

from pennylane import capture, queuing
from pennylane.capture import subroutine as capture_subroutine
from pennylane.operation import Operator
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

    def __init__(self, subroutine, bound_args, decomposition, output):
        self._subroutine = subroutine
        self._bound_args = bound_args
        self._decomp = decomposition
        self.output = output

        wires = []
        for wire_argname in self._subroutine.wire_argnames:
            wires.extend(_setup_wires(self._bound_args.arguments[wire_argname]))

        super().__init__(wires=wires)
        self.name = subroutine.definition.__name__

    @property
    def subroutine(self):
        return self._subroutine

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
        static_argnames: set | frozenset | None = None,
        wire_argnames: set | frozenset = frozenset({"wires"}),
    ):
        self.definition = staticmethod(definition)
        self.setup_inputs = staticmethod(setup_inputs)
        self.static_argnames = static_argnames
        self.subroutine = capture_subroutine(definition, static_argnames=static_argnames)
        self.wire_argnames = wire_argnames
        self._signature = signature(definition)
        update_wrapper(self, definition)

    def __call__(self, *args, **kwargs):
        args, kwargs = self.setup_inputs(*args, **kwargs)
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        if capture.enabled():
            for wire_argname in self.wire_argnames:
                register = bound_args.arguments[wire_argname]
                if isinstance(register, int):
                    register = [register]
                if len(register) > 0:
                    bound_args.arguments[wire_argname] = jax.numpy.stack(register)
            return self.subroutine(*bound_args.args, **bound_args.kwargs)

        for wire_argname in self.wire_argnames:
            register = bound_args.arguments[wire_argname]
            bound_args.arguments[wire_argname] = Wires(register)

        with queuing.AnnotatedQueue() as decomposition:
            output = self.definition(*bound_args.args, **bound_args.kwargs)
        SubroutineOp(self, bound_args, decomposition.queue, output)
        return output
