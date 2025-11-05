from functools import update_wrapper
from inspect import signature

import jax

from pennylane.capture import subroutine as capture_subroutine


def default_setup_inputs(*args, **kwargs):
    return args, kwargs


class Subroutine:

    def __init__(
        self,
        definition,
        setup_inputs=default_setup_inputs,
        static_argnames=None,
        wire_argnames=frozenset({"wires"}),
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
        for wire_argname in self.wire_argnames:
            bound_args.arguments[wire_argname] = jax.numpy.stack(bound_args.arguments[wire_argname])
        return self.subroutine(*bound_args.args, **bound_args.kwargs)
