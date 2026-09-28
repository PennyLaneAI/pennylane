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
This module contains ``qp.subcircuit``, a factory function to create operators
using quantum functions.
"""

from collections.abc import Callable
from inspect import Parameter, Signature, signature

from pennylane.core import Operator2
from pennylane.decomposition import DecompositionRule, add_decomps


def _subcircuit(qfunc: DecompositionRule, **cls_attrs) -> type[Operator2]:
    """Implementation of ``subcircuit``."""
    # pylint: disable=protected-access

    if not isinstance(qfunc, DecompositionRule):
        raise TypeError(
            "The provided quantum function must register resources using 'qp.register_resources'."
        )

    # 'self' shouldn't be in signature(OpClass), but it should be in
    # signature(OpClass.__init__)
    s = signature(qfunc._impl)
    self_p = Parameter("self", kind=Parameter.POSITIONAL_OR_KEYWORD)
    init_sig = Signature((self_p, *s.parameters.values()))

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
    decomp_name = qfunc.name + "_decomp"
    decomp_rule = DecompositionRule(
        qfunc._impl,
        resources=qfunc._compute_resources,
        work_wires=qfunc._work_wire_spec,
        exact_resources=qfunc.exact_resources,
        name=decomp_name,
    )
    add_decomps(new_operator, decomp_rule)
    return new_operator


def subcircuit(  # pylint: disable=too-many-arguments
    qfunc: DecompositionRule | None = None,
    dynamic_argnames=(),
    wire_argnames=("wires",),
    compilable_argnames=(),
    hybrid_argnames=(),
    static_argnames=(),
    **cls_attrs,
) -> Callable | type[Operator2]:
    r"""A decorator for turning a quantum function with registered resources into an
    :class:`~.Operator2` subclass.

    The quantum function decorated with `subcircuit` can be any valid quantum
    function, provided that nothing is returned from it (e.g., terminal measurements are
    not allowed) and that it has resources registered to it via :func:`~.register_resources`.

    The decorated quantum function will automatically be registered as an :class:`~.Operator2`
    subclass with a single decomposition rule that corresponds to the quantum function body.


    Args:
        qfunc (DecompositionRule): a quantum function whose resources have been registered with
            :func:`~pennylane.register_resources`

    Keyword Args:
        dynamic_argnames (Sequence[str]): names of the arguments that are dynamic data of the
            operator. For more details, see :attr:`~.Operator2.dynamic_argnames`
        wire_argnames (Sequence[str]): names of the arguments that are wires. Defaults to
            ``("wires",)``. For more details, see :attr:`~.Operator2.wire_argnames`
        compilable_argnames (Sequence[str]): names of arguments that are static **and compilable**
            data of the operator. For more details, see :attr:`~.Operator2.compilable_argnames`
        hybrid_argnames (Sequence[str]): names of arguments that are dynamic data wrapped in static
            data structures (pytrees). For more details, see :attr:`~.Operator2.hybrid_argnames`
        static_argnames (Sequence[str]): names of arguments that are static but **not compilable**.
            For more details, see :attr:`~.Operator2.static_argnames`
        **cls_attrs: additional class attributes to set on the created operator class

    Returns:
        type[~.Operator2]: a new operator class named after the quantum function

    .. note::

        Every parameter of the quantum function must be classified into one of ``dynamic_argnames``,
        ``wire_argnames``, ``compilable_argnames``, ``hybrid_argnames`` or ``static_argnames``.
        See :class:`~.Operator2` for details on how arguments are classified.

    .. seealso:: :class:`~.Operator2`, :func:`~pennylane.register_resources`

    **Example**

    An operator can be created by decorating a quantum function that has registered resources. Its
    parameters are classified using the ``*_argnames`` keyword arguments; here
    ``phi`` is dynamic data and ``wires`` are (by default) the wires:

    .. code-block:: python

        @qp.subcircuit(dynamic_argnames=("phi",))
        @qp.register_resources({qp.H: 1, qp.RZ: 1})
        def MyOp(phi, wires):
            qp.H(wires)
            qp.RZ(phi, wires)

    The returned object is an :class:`~.Operator2` subclass named after the quantum function, and
    is instantiated and used like any other operator:

    >>> op = MyOp(0.5, wires=0)
    >>> op
    MyOp(0.5, wires=[0])
    >>> isinstance(op, qp.core.Operator2)
    True

    The body of the quantum function is registered as its decomposition rule:

    >>> qp.inspect_decomps(op)
    Decomposition 0 (name: MyOp_decomp)
    0: ──H──RZ(0.50)─┤
    Gate Count: {Hadamard: 1, RZ: 1}
    """
    if qfunc is not None:
        return _subcircuit(
            qfunc,
            dynamic_argnames=dynamic_argnames,
            wire_argnames=wire_argnames,
            compilable_argnames=compilable_argnames,
            hybrid_argnames=hybrid_argnames,
            static_argnames=static_argnames,
            **cls_attrs,
        )

    def wrapper(qfunc_):
        return _subcircuit(
            qfunc_,
            dynamic_argnames=dynamic_argnames,
            wire_argnames=wire_argnames,
            compilable_argnames=compilable_argnames,
            hybrid_argnames=hybrid_argnames,
            static_argnames=static_argnames,
            **cls_attrs,
        )

    return wrapper
