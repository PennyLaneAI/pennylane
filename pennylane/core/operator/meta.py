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

from pennylane import math
from pennylane.capture import enabled
from pennylane.pytrees import flatten, unflatten
from pennylane.typing import AbstractArray
from pennylane.wires import AbstractWires, Wires


class _ArgType(Enum):
    """Enum to keep track of an arguments type."""

    WIRES = auto()
    DYN = auto()
    HYBRID = auto()


def _stop_autograph(f):
    """Stop the autograph interpretation of operators by making it so that ``f`` always
    belongs to the pennylane namespace."""

    def new_f(*args, **kwargs):
        return f(*args, **kwargs)

    return new_f


def _contains_abstract_type(val):
    """Check if pytree contains any abstract types."""
    leaves = flatten(val)[0]
    return any(isinstance(leaf, (AbstractArray, AbstractWires)) for leaf in leaves)


def _canonicalize_wire_leaf(leaf) -> AbstractWires:
    """Abstractifies a leaf that represents a wires object."""
    if isinstance(leaf, Wires):
        return AbstractWires(len(leaf))

    if isinstance(leaf, (int, str)):
        return AbstractWires(1)

    return AbstractWires(len(list(leaf)))


def _canonicalize_abstract_type(val, kind: _ArgType):
    """Check if pytree contains any abstract types."""

    if isinstance(val, (AbstractArray, AbstractWires)):
        return val

    match kind:
        case _ArgType.WIRES:
            return _canonicalize_wire_leaf(val)

        case _ArgType.DYN:
            canonical_arr = math.asarray(val)
            return AbstractArray(canonical_arr.shape, canonical_arr.dtype)

        case _ArgType.HYBRID:
            leaves, structure = flatten(val, is_leaf=lambda x: isinstance(x, Wires))
            new_leaves = []
            for leaf in leaves:
                if isinstance(leaf, (AbstractArray, AbstractWires)):
                    new_leaves.append(leaf)
                elif isinstance(leaf, Wires):
                    new_leaves.append(_canonicalize_wire_leaf(leaf))
                # Process arrays
                elif hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                    new_leaves.append(AbstractArray(leaf.shape, leaf.dtype))
                # Process scalars
                else:
                    new_leaves.append(AbstractArray((), type(leaf)))
            return unflatten(new_leaves, structure)

        case _:  # pragma: no cover
            raise ValueError(f"Unknown kind: '{kind}'")


class OperatorMeta(type):
    """
    A metatype that overrides class construction for operators for program capture
    and graph-based decompositions integration.
    TODO: [sc-120453] Fill docstring
    """

    @property
    def __signature__(cls):
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
                kind = _ArgType.DYN

                # NOTE: Check hybrid first as hybrid args can
                # appear in both hybrid and wires args; these arguments
                # must be treated as hybrid.
                if name in cls.hybrid_argnames:
                    kind = _ArgType.HYBRID
                elif name in cls.wire_argnames:
                    kind = _ArgType.WIRES

                arguments[name] = _canonicalize_abstract_type(arguments[name], kind)

            obj = cls.__new__(cls)
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


# pylint: disable=abstract-method
class ABCOperatorMeta(OperatorMeta, ABCMeta):
    """A combination of the operator metaclass and ABCMeta."""
