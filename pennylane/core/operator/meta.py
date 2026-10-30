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

import warnings
from abc import ABCMeta
from inspect import Signature, signature
from typing import ClassVar

from pennylane import capture, math
from pennylane.pytrees import flatten
from pennylane.typing import AbstractArray, AbstractWires


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


class OperatorRegistry:
    """A Singleton class that keeps track of all the Operator2 operators (including user-defined operators) defined."""

    instance = None
    registry = {}

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __getitem__(self, index):
        return self.registry[index]

    def __setitem__(self, index, value):
        if index not in self.registry:
            self.registry[index] = value
        elif self.registry[index] is not value:
            warnings.warn(
                f"More than one operator class by the same name ({index}) has been defined."
            )


def get_op_class_by_name(name):
    """
    Helper to get an operator class by name using a function call.
    To use indexing syntax instead, use `OperatorRegistry` directly.

    Args:
        name (str): the name of the class to retrieve.

    Returns:
        The class with the provided name.

    **Example**

    .. code-block:: python
        from pennylane import Operator2

        class MyNamedOp(Operator2):
            static_argnames = ("label",)

            def __init__(self, label, wires):
                super().__init__(label, wires=wires)

    >>> from pennylane.core import get_op_class_by_name
    >>> get_op_class_by_name("MyNamedOp")("my-label", [0, 1])
    MyNamedOp(label=my-label, wires=[0, 1])

    >>> from pennylane.core import OperatorRegistry
    >>> OperatorRegistry["MyNamedOp"]("my-second-label", [0, 1])
    MyNamedOp(label=my-second-label, wires=[0, 1])
    """
    return OperatorRegistry()[name]


class OperatorMeta(ABCMeta):
    """A metatype that overrides class construction for operators for program capture
    and graph-based decompositions integration.
    TODO: [sc-120453] Fill docstring
    """

    _sig: ClassVar[Signature]
    """The signature of the operator. Internal use only."""

    @property
    def __signature__(cls):
        # __signature__ must be overridden because using custom metaclasses causes
        # signature(cls) to return ``self`` as the first argument, which is inconsistent
        # with the behaviour of regular classes.
        sig = signature(cls.__init__)
        without_self = tuple(sig.parameters.values())[1:]
        return Signature(without_self)

    def __new__(mcls, name, *args, **kwargs):
        cls = super().__new__(mcls, name, *args, **kwargs)
        OperatorRegistry()[name] = cls
        return cls

    @_stop_autograph
    def __call__(cls, *args, **kwargs):

        bound = cls._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments: dict = bound.arguments

        _validate_compile_time_args(cls, arguments)

        # default behaviour calls __new__ and then __init__
        op = super().__call__(*args, **kwargs)

        # adds the operator to the program
        op.queue()
        if capture.enabled():
            # binding the operator primitive adds it to the jaxpr
            op._bind_primitive()

        return op


def _validate_compile_time_args(cls, arguments):
    """Validate abstract compilable inputs and traced static/compilable inputs."""
    for arg_name in cls.static_argnames + cls.compilable_argnames:
        leaves, _ = flatten(arguments[arg_name])
        if any(
            (
                arg_name in cls.compilable_argnames
                and isinstance(value, (AbstractArray, AbstractWires))
            )
            or math.is_tracer(value)
            for value in leaves
        ):
            raise ValueError(
                f"Argument '{arg_name}' of operator '{cls.__name__}' must be a concrete, "
                "compile-time constant. An abstract input or dynamic/traced variable was provided instead."
            )
