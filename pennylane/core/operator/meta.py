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
from typing import ClassVar

from pennylane import capture, math
from pennylane.pytrees import flatten


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

    @_stop_autograph
    def __call__(cls, *args, **kwargs):

        bound = cls._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments: dict = bound.arguments

        # NOTE: Detect if static / compilable argument received a tracer
        # indicating it is incorrectly being used as a dynamic argument.
        if capture.enabled():
            _verify_no_traced_static_args(cls, arguments)

        # default behaviour calls __new__ and then __init__
        op = super().__call__(*args, **kwargs)

        # adds the operator to the program
        op.queue()
        if capture.enabled():
            # binding the operator primitive adds it to the jaxpr
            op._bind_primitive()

        return op


def _verify_no_traced_static_args(cls, arguments):
    """Verify that no static/compilable arguments received a tracer."""
    for arg_name in cls.static_argnames + cls.compilable_argnames:
        leaves, _ = flatten(arguments[arg_name])
        if any(math.is_abstract(l) for l in leaves):
            raise ValueError(
                f"Argument '{arg_name}' of operator '{cls.__name__}' must be a concrete, "
                f"compile-time constant. A dynamic/traced variable was provided instead. "
            )
