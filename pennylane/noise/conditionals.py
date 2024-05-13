# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains utility functions for building boolean conditionals for noise models"""

from functools import partial
from inspect import isclass, signature

import pennylane as qml
from pennylane.boolean_fn import BooleanFn
from pennylane.wires import Wires

# pylint: disable = unnecessary-lambda, too-few-public-methods


class NoiseConditional(BooleanFn):
    """Wrapper for callables with boolean output that help implement noise models
    and can be manipulated via bit-wise operations.

    Args:
        fn (callable): Function to be wrapped. It can accept any number
            of arguments, and must return a boolean.
        repr (str): String representation to be used by ``repr`` dunder method.
            Default is to use function's name as ``NoiseConditional(<name>)``.

    .. note:: This is a *developer-facing* class for implementing conditionals for building
        noise models. Users are encouraged to rather use :func:`~.wires_in`, :func:`~.wires_eq`,
        :func:`~.op_in`, and :func:`~.op_eq` for this purpose.
    """

    def __init__(self, fn, repr=None):
        super().__init__(fn)
        self.repr = repr if repr else f"NoiseConditional({fn.__name__})"

    def __and__(self, other):
        return AndConditional(self, other)

    def __or__(self, other):
        return OrConditional(self, other)

    def __repr__(self):
        return self.repr


# pylint: disable = bad-super-call
class AndConditional(NoiseConditional):
    """Developer facing class for implemeting bit-wise ``AND`` for
    :class:`NoiseConditional <pennylane.noise.NoiseConditional>` callables with boolean output.

    Args:
        left (~.NoiseConditional): Left operand in the bit-wise expression.
        right (~.NoiseConditional): Right operand in the bit-wise expression.

    .. note::

        This is a *developer-facing* class for implementing bit-wise expression
        for the noise model's conditionals built via
        :class:`NoiseConditional <pennylane.noise.NoiseConditional>`.
    """

    def __init__(self, left, right):
        self.l_op = left
        self.r_op = right
        self.func = super(NoiseConditional, left).__and__(right)
        super(NoiseConditional, self).__init__(self.func)
        self.repr = f"And({left.repr}, {right.repr})"

    def __str__(self):
        return f"And({self.l_op}, {self.r_op})"


# pylint: disable = bad-super-call
class OrConditional(NoiseConditional):
    """Developer facing class for implemeting bit-wise ``OR`` for
    :class:`NoiseConditional <pennylane.noise.NoiseConditional>` callables with boolean output.

    Args:
        left (~.NoiseConditional): Left operand in the bit-wise expression.
        right (~.NoiseConditional): Right operand in the bit-wise expression.

    .. note:: This is a *developer-facing* class for implementing bit-wise expression
        for the noise model's conditionals built via
        :class:`NoiseConditional <pennylane.noise.NoiseConditional>`.
    """

    def __init__(self, left, right):
        self.l_op = left
        self.r_op = right
        self.func = super(NoiseConditional, left).__or__(right)
        super(NoiseConditional, self).__init__(self.func)
        self.repr = f"Or({left.repr}, {right.repr})"

    def __str__(self):
        return f"Or({self.l_op}, {self.r_op})"


def _get_wires(val):
    """Extract wires as a set from an integer, string, Iterable, Wires or Operation instance.

    Args:
        val (Union[int, str, Iterable, ~.wires.Wires, ~.operation.Operation]): object to be used
            for building the wire set.

    Returns:
        set[Union[int, str]]: computed wire set

    Raises:
        ValueError: if the wire set cannot be computed for ``val``.

    """
    iters = val if isinstance(val, (list, tuple, set, Wires)) else getattr(val, "wires", [val])
    try:
        wires = [[w] if isinstance(w, (int, str)) else getattr(w, "wires").tolist() for w in iters]
    except TypeError:
        raise ValueError(f"Wires cannot be computed for {val}") from None
    return set(w for wire in wires for w in wire)


def wire_in(wires):
    """BooleanFn for checking if a wire exist in a set of specified wires"""
    return NoiseConditional(
        lambda x: _get_wires(x).issubset(_get_wires(wires)), f"WiresIn({wires})"
    )


def wire_eq(wires):
    """BooleanFn for checking if a wire is equal to the set of specified wires"""
    return NoiseConditional(
        lambda x: _get_wires(x).issubset(_get_wires(wires)), f"WiresEq({wires})"
    )


def _get_ops(val):
    """Help deal with arithmetic ops"""
    vals = val if isinstance(val, (list, tuple, set, qml.wires.Wires)) else [val]
    return tuple(
        (
            getattr(qml.ops, val)
            if isinstance(val, str)
            else (val if isclass(val) else getattr(val, "__class__"))
        )
        for val in vals
    )


def op_eq(ops):
    """BooleanFn for checking if an op is equal to the set of specified ops"""
    op_cls = _get_ops(ops)
    op_repr = str([getattr(op, "__name__") for op in op_cls])[1:-1]

    return NoiseConditional(lambda x: _get_ops(x) == op_cls, f"OpEq({op_repr})")


def op_in(ops):
    """BooleanFn for checking if an op exist in a set of specified ops"""
    op_cls = _get_ops(ops)
    op_repr = list(getattr(op, "__name__") for op in op_cls)
    return NoiseConditional(
        lambda x: x in op_cls if isclass(x) else isinstance(x, op_cls), f"OpIn({op_repr})"
    )


def partial_wires(operation, *args, **kwargs):
    """Wrapper for calling operation with all arguments except the wires"""

    def _partial_op(x):
        """Wrapper function for partial_wires"""
        wires = getattr(x, "wires", None) or ([x] if isinstance(x, (int, str)) else list(x))
        return op(wires=wires)

    if not callable(operation):
        if args:
            raise ValueError(
                "Args cannot be provided when operation is an instance, "
                f"got operation = {operation} and args = {args}."
            )
        args, metadata = getattr(operation, "_flatten")()
        if len(metadata) > 1:
            kwargs = {**dict(metadata[1]), **kwargs}
        operation = type(operation)

    fsignature = signature(getattr(operation, "__init__", operation)).parameters
    parameters = list(fsignature)[int("self" in fsignature) :]
    arg_params = {**dict(zip(parameters, args)), **kwargs}

    if "wires" in arg_params:  # Ensure we don't include wires arg
        arg_params.pop("wires")

    op = partial(operation, **{**arg_params, **kwargs})

    return _partial_op
