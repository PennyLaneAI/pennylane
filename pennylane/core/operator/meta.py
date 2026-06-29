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
Defines a metaclass for automatic integration of any ``Operator`` with plxpr program capture.

See ``explanations.md`` for technical explanations of how this works.
"""

from abc import ABCMeta
from enum import Enum, auto
from inspect import Signature, signature
from numbers import Number

from pennylane import math
from pennylane.capture import enabled
from pennylane.core.wires import Wires
from pennylane.pytrees import flatten
from pennylane.typing import AbstractArray, AbstractWires

from .utils import abstractify


class _ArgType(Enum):
    """Enum to keep track of an arguments type."""

    WIRES = auto()
    DYN = auto()
    HYBRID = auto()


def _resolve_arg_kind(cls, name: str) -> _ArgType:
    """Resolves an arguments name to what kind of argument type it is."""
    # Check hybrid first: hybrid args can also appear in wire_argnames
    # and must be treated as hybrid.
    if name in cls.hybrid_argnames:
        return _ArgType.HYBRID
    if name in cls.wire_argnames:
        return _ArgType.WIRES
    return _ArgType.DYN


def _stop_autograph(f):
    """Stop the autograph interpretation of operators by making it so that ``f`` always
    belongs to the pennylane namespace.

    Autograph only transforms functions belonging to non-pennylane namespaces. So, custom
    operators created outside the pennylane namespace would be transformed by autograph
    without this decorator.
    """

    def new_f(*args, **kwargs):
        return f(*args, **kwargs)

    return new_f


def _contains_abstract_type(val):
    """Check if pytree contains any abstract types."""
    leaves, _ = flatten(val)

    for leaf in leaves:
        if isinstance(leaf, (AbstractArray, AbstractWires)):
            return True

        if isinstance(val, type) and issubclass(val, Number):
            return True

    return False


# NOTE: This pylint disable will be removed in the PR that adds abstractify
# pylint: disable=too-many-branches
def _canonicalize_abstract_type(val, kind: _ArgType):
    """Canonicalizes the input into its abstract equivalent.

    Args:
        val (Any): The input value.
        kind (_ArgType): The argument's classification.
            - WIRES: Coerce the value to be an AbstractWires instance.
            - DYN: Flatten into a single, unified AbstractArray
            - HYBRID: Preserve the PyTree structure, mapping internal leaves
                to either AbstractWires or AbstractArray.
    """

    if isinstance(val, (AbstractArray, AbstractWires)):
        return val
    if isinstance(val, type) and issubclass(val, Number):
        return AbstractArray((), val)

    match kind:
        case _ArgType.WIRES:
            # abstractify expects a Wires object for wire-routing, so we sanitize it first
            return abstractify(Wires(val))

        case _ArgType.DYN:
            # A sequence of types is not supported (i.e., [float, float, float])
            # for dynamic args. Ambiguous how to canonicalize it generally.
            if isinstance(val, (list, tuple)) and any(
                isinstance(x, AbstractArray) or (isinstance(x, type) and issubclass(x, Number))
                for x in val
            ):
                raise NotImplementedError(
                    "A sequence of types for a dynamic argument is not "
                    "currently supported. Instead, please use the type "
                    "specifiers found in pennylane.typing."
                )
            # Ensure it behaves like a clean array/scalar leaf before abstractifying
            return abstractify(math.asarray(val))

        case _ArgType.HYBRID:
            # Since abstractify natively handles PyTree recursion and leaves,
            # we can pass the entire structure straight through
            return abstractify(val)

        case _:  # pragma: no cover
            raise ValueError(f"Unknown kind: '{kind}'")


class OperatorMeta(ABCMeta):
    """A metatype that overrides class construction for operators for program capture
    and graph-based decompositions integration.
    TODO: [sc-120453] Fill docstring
    """

    @property
    def __signature__(cls):
        # __signature__ must be overridden because using custom metaclasses causes
        # signature(cls) to return ``self`` as the first argument, which is inconsistent
        # with the behaviour of regular classes.
        sig = signature(cls.__init__)
        without_self = tuple(sig.parameters.values())[1:]
        return Signature(without_self)

    @_stop_autograph
    def __call__(cls, *args, **kwargs):
        bound = cls._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments: dict = bound.arguments

        target_args = cls.dynamic_argnames + cls.hybrid_argnames + cls.wire_argnames

        if any(_contains_abstract_type(arguments[name]) for name in target_args):
            for name in target_args:
                kind = _resolve_arg_kind(cls, name)
                arguments[name] = _canonicalize_abstract_type(arguments[name], kind)

            obj = cls.__new__(cls)  # pylint: disable=no-value-for-parameter
            from .operator2 import Operator2  # pylint: disable=import-outside-toplevel

            Operator2.__init__(obj, *bound.args, **bound.kwargs)
            return obj

        # This method is called everytime we want to create an instance of the class.
        # default behavior uses __new__ then __init__
        op = type.__call__(cls, *args, **kwargs)
        op.queue()

        if enabled():
            # When tracing is enabled, we want to use bind to construct the class
            # if we want class construction to add it to the jaxpr
            op._bind_primitive()

        return op
