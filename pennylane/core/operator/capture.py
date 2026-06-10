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
from inspect import Signature, signature

from pennylane.capture import enabled
from pennylane.pytrees import flatten, unflatten
from pennylane.typing import AbstractArray
from pennylane.wires import AbstractWires, Wires


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
    if isinstance(leaf, Wires):
        return AbstractWires(len(leaf))

    if isinstance(leaf, (int, str)):
        return AbstractWires(1)

    return AbstractWires(len(list(leaf)))


def _canonicalize_abstract_type(val, is_wires: bool = False):
    """Check if pytree contains any abstkact types."""

    if is_wires:
        if isinstance(val, AbstractWires):  # already abstract, return as is
            return val
        return _canonicalize_wire_leaf(val)

    leaves, structure = flatten(val)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, (AbstractArray, AbstractWires)):
            new_leaves.append(leaf)
        # Process arrays
        elif hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            new_leaves.append(AbstractArray(leaf.shape, leaf.dtype))
        # Process scalars
        else:
            new_leaves.append(AbstractArray((), type(leaf)))
    return unflatten(new_leaves, structure)


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

        arguments_that_can_be_abstract = (
            cls.dynamic_argnames + cls.hybrid_argnames + cls.wire_argnames
        )
        has_abstract_arguments = any(
            _contains_abstract_type(arguments[name]) for name in arguments_that_can_be_abstract
        )
        if has_abstract_arguments:
            # "Canonicalize" abstract operators
            # NOTE: Because at least one of the arguments is abstract
            # all of them need to be "promoted" to abstract
            for name in arguments_that_can_be_abstract:
                is_wires_arg = name in cls.wire_argnames
                arguments[name] = _canonicalize_abstract_type(
                    arguments[name], is_wires=is_wires_arg
                )

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
