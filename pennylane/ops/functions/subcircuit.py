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
r"""
This module contains ``qp.subscircuit``, a factory function to create operators
using quantum functions.
"""

import inspect

from pennylane.core import Operator2
from pennylane.decomposition import DecompositionRule, add_decomps


def _subcircuit(qfunc: DecompositionRule, **cls_attrs):
    # pylint: disable=protected-access

    # 'self' shouldn't be in signature(OpClass), but it should be in
    # signature(OpClass.__init__)
    s = inspect.signature(qfunc._impl)
    self_p = inspect.Parameter("self", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    init_sig = inspect.Signature((self_p, *s.parameters.values()))

    def _init(self, *args, **kwargs):
        Operator2.__init__(self, *args, **kwargs)

    _init.__signature__ = init_sig

    # sets up various class properties
    attrs = dict(cls_attrs)
    attrs["__init__"] = _init
    attrs["__doc__"] = qfunc._impl.__doc__
    attrs["__module__"] = qfunc._impl.__module__

    # creates a new class
    new_operator = type(qfunc.name, (Operator2,), attrs)

    # registers the decomposition
    add_decomps(new_operator, qfunc)
    return new_operator


def subcircuit(
    qfunc: DecompositionRule | None = None,
    dynamic_argnames=(),
    wire_argnames=("wires",),
    compilable_argnames=(),
    hybrid_argnames=(),
    static_argnames=(),
    arg_specs=None,
    **additional_attrs,
):  # pylint: disable=too-many-arguments
    """Decorator to create an operator using a quantum function."""
    if qfunc is not None:
        return _subcircuit(
            qfunc,
            dynamic_argnames=dynamic_argnames,
            wire_argnames=wire_argnames,
            compilable_argnames=compilable_argnames,
            hybrid_argnames=hybrid_argnames,
            static_argnames=static_argnames,
            arg_specs=arg_specs,
            **additional_attrs,
        )

    def wrapper(qfunc_):
        return _subcircuit(
            qfunc_,
            dynamic_argnames=dynamic_argnames,
            wire_argnames=wire_argnames,
            compilable_argnames=compilable_argnames,
            hybrid_argnames=hybrid_argnames,
            static_argnames=static_argnames,
            arg_specs=arg_specs,
            **additional_attrs,
        )

    return wrapper
