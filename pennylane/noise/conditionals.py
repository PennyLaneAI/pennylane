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
    """Developer facing class for implemeting bit-wise ``AND`` for callables
    wrapped up with :class:`NoiseConditional <pennylane.noise.NoiseConditional>`.

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
    """Developer facing class for implemeting bit-wise ``OR`` for callables
    wrapped up with :class:`NoiseConditional <pennylane.noise.NoiseConditional>`.

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


def wires_in(wires):
    """Builds a ``Conditional`` as a boolean function for evaluating
    if a given wire exist in a specified set of wires.

    Args:
        wires (Union(list[int, str], Wires, Operation)): object to be used
            for building the wire set.

    Returns:
        :class:`NoiseConditional <pennylane.noise.NoiseConditional>`: a boolean function
        represented as ``WiresIn``, which evaluates to ``True`` if a given wire exist in a
        specified set of wires.

    Raises:
        ValueError: if the wire set cannot be computed from ``wires``.

    **Example**

    One may use ``wires_in`` with a given sequence of wires which are used as a wire set:

    >>> cond_func = qml.noise.wires_in([0, 1])
    >>> cond_func(0)
    True
    >>> cond_func(3)
    False

    Additionally, if an :class:`Operation <pennylane.operation.Operation>` is provided,
    its ``wires`` are extracted and used to build the wire set:

    >>> cond_func = qml.noise.wires_in(qml.CNOT(["alice", "bob"]))
    >>> cond_func("alice")
    True
    >>> cond_func("eve")
    False
    """
    return NoiseConditional(
        lambda x: _get_wires(x).issubset(_get_wires(wires)), f"WiresIn({wires})"
    )


def wires_eq(wires):
    """Builds a ``Conditional`` as a boolean function for evaluating
    if a given wire is equal to specified set of wires.

    Args:
        wires (Union(list[int, str], Wires, Operation)): object to be used
            for building the wire set.

    Returns:
        :class:`NoiseConditional <pennylane.noise.NoiseConditional>`: a boolean function
        represented as ``WiresEq``, which evaluates to ``True`` if a given wire is equal to
        specified set of wires.

    Raises:
        ValueError: if the wire set cannot be computed from ``wires``.

    **Example**

    One may use ``wires_eq`` with a given sequence of wires which are used as a wire set:

    >>> cond_func = qml.noise.wires_eq(0)
    >>> cond_func(0)
    True
    >>> cond_func(qml.RY(1.23, wires=[3]))
    False

    Additionally, if an :class:`Operation <pennylane.operation.Operation>` is provided,
    its ``wires`` are extracted and used to build the wire set:

    >>> cond_func = qml.noise.wires_in(qml.RX(1.0, "dino"))
    >>> cond_func(qml.RZ(1.23, wires="dino"))
    True
    >>> cond_func("eve")
    False
    """
    return NoiseConditional(lambda x: _get_wires(x) == _get_wires(wires), f"WiresEq({wires})")


def _get_ops(val):
    """Computes the class for a given argument from its string name, instance,
    or a sequence of them.

    Args:
        val (Union[str, Operation, Iterable]): object to be used
            for building the wire set.

    Returns:
        tuple[class]: tuple of :class:`Operation <pennylane.operation.Operation>`
        classes corresponding to val.
    """
    vals = val if isinstance(val, (list, tuple, set, qml.wires.Wires)) else [val]
    return tuple(
        (
            getattr(qml.ops, val)
            if isinstance(val, str)
            else (val if isclass(val) else getattr(val, "__class__"))
        )
        for val in vals
    )


def _check_with_lc_op(op1, op2):
    """Helper method for comparing two arithmetic operators using their LinearCombination"""
    # pylint: disable = unnecessary-lambda-assignment
    lc_cop = lambda op: qml.ops.LinearCombination(*qml.simplify(op).terms())

    coeffs, op_terms = lc_cop(op1).terms()
    sprods = [_get_ops(getattr(op_term, "operands", op_term)) for op_term in op_terms]

    def _lc_op(x):
        coeffs2, op_terms2 = lc_cop(x).terms()
        sprods2 = [_get_ops(getattr(op_term, "operands", op_term)) for op_term in op_terms2]

        present = True
        for coeff, sprod in zip(coeffs2, sprods2):
            present = sprod in sprods
            if not present:
                break
            if not qml.math.equal(coeff, coeffs[sprods.index(sprod)]):
                present = False
                break
            sprods.remove(sprod)

        return present

    return _lc_op(op2)


def op_eq(ops):
    """Builds a ``Conditional`` as a boolean function for evaluating
    if a given operation is equal to the specified operation.

    Args:
        ops (str, Operation, Union(list[str, Operation])): string
            representation or instance of the operation.

    Returns:
        :class:`NoiseConditional <pennylane.noise.NoiseConditional>`: a boolean function
        represented as ``OpEq``, which evaluates to ``True`` if a given operation is
        equal to the specified set of operation(s), irrespective of wires they act on.

    **Example**

    One may use ``op_eq`` with a string representation of the name of the operation:

    >>> cond_func = qml.noise.op_eq("RX")
    >>> cond_func(qml.RX(1.23, wires=[0]))
    True
    >>> cond_func(qml.RZ(1.23, wires=[3]))
    False
    >>> cond_func("CNOT")
    False

    Additionally, an instance of :class:`Operation <pennylane.operation.Operation>`
    can also be provided:

    >>> cond_func = qml.noise.op_eq(qml.RX(1.0, "dino"))
    >>> cond_func(qml.RX(1.23, wires=["eve"]))
    True
    >>> cond_func(qml.RY(1.23, wires=["dino"]))
    False
    """
    op_cls = _get_ops(ops)
    op_repr = str([getattr(op, "__name__") for op in op_cls])[1:-1]

    if (len(op_cls) == 1 and (isclass(ops) or not getattr(ops, "arithmetic_depth", 0))) or (
        len(op_cls) > 1
        and not any(not isclass(ops) and getattr(op, "arithmetic_depth", 0) for op in ops)
    ):
        return NoiseConditional(lambda x: _get_ops(x) == op_cls, f"OpEq({op_repr})")

    try:
        return NoiseConditional(
            lambda x: _get_ops(x) == op_cls
            and (
                _check_with_lc_op(ops, x)
                if len(op_cls) == 1
                else all(
                    _check_with_lc_op(ops, _x)
                    for _x in x
                    if not isclass(_x) and getattr(_x, "arithmetic_depth", 0)
                )
            ),
            f"OpEq({op_repr})",
        )
    except:  # pylint: disable = bare-except
        raise ValueError(
            "OpEq does not operations with artihmetic operations "
            "that cannot be converted to a linear combination"
        ) from None


def op_in(ops):
    """Builds a ``Conditional`` as a boolean function for evaluating
    if a given operation exist in a specified set of operation.

    Args:
        ops (str, Operation, Union(list[str, Operation])): string
            representation or instance of the operation.

    Returns:
        :class:`NoiseConditional <pennylane.noise.NoiseConditional>`: a boolean function
        represented as ``OpIn``, which evaluates to ``True`` if a given operation exist
        in a specified set of operation(s), irrespective of wires they act on.

    **Example**

    One may use ``op_in`` with a string representation of the name of the operation:

    >>> cond_func = qml.noise.op_in(["RX", "RY"])
    >>> cond_func(qml.RX(1.23, wires=[0]))
    True
    >>> cond_func(qml.RZ(1.23, wires=[3]))
    False
    >>> cond_func([qml.RX(1.23, wires=[1]), qml.RY(4.56, wires=[2])])
    True

    Additionally, an instance of :class:`Operation <pennylane.operation.Operation>`
    can also be provided:

    >>> cond_func = qml.noise.op_in([qml.RX(1.0, "dino"), qml.RY(2.0, "rhino")])
    >>> cond_func(qml.RX(1.23, wires=["eve"]))
    True
    >>> cond_func(qml.RY(1.23, wires=["dino"]))
    True
    >>> cond_func([qml.RX(1.23, wires=[1]), qml.RZ(4.56, wires=[2])])
    False
    """
    ops = [ops] if not isinstance(ops, (list, tuple, set)) else ops
    op_cls = _get_ops(ops)
    op_repr = list(getattr(op, "__name__") for op in op_cls)

    def _check_in_ops(x):
        x = [x] if not isinstance(x, (list, tuple, set)) else x

        return all(
            (
                _x in op_cls
                if isclass(_x)
                else (
                    isinstance(_x, op_cls)
                    if not getattr(_x, "arithmetic_depth", 0)
                    else any(
                        _check_with_lc_op(_x, ops)
                        for op in ops
                        if not isclass(op) and getattr(op, "arithmetic_depth", 0)
                    )
                )
            )
            for _x in x
        )

    return NoiseConditional(_check_in_ops, f"OpIn({op_repr})")


def partial_wires(operation, *args, **kwargs):
    """Builds a partial function based on the given operation with
    all argument frozen except ``wires``.

    Args:
        operation (Operation, class): instance of the operation or the class
            corresponding to operation.
        args: Positional arguments provided in the case where the keyword argument
            ``operation`` is a class for building the partially evaluated instance.
        kwargs: Keyword arguments for the building the partially evaluated instance.
            These will override any arguments present in the operation instance or ``args``.

    Returns:
        callable: a wrapper function that accepts a sequence of wires as an argument or
        any object with ``wires`` property.

    Raises:
        ValueError: if ``args`` are provided when the given ``operation`` is an instance.

    **Example**

    One may give an instance of :class:`Operation <pennylane.operation.Operation>`
    for the ``operation`` argument:

    >>> func = qml.noise.partial_wires(qml.RX(1.2, [12]))
    >>> func(2)
    qml.RX(1.2, wires=[2])
    >>> func(qml.RY(1.0, ["wires"]))
    qml.RX(1.2, wires=["wires"])

    Additionally, class of :class:`Operation <pennylane.operation.Operation>` can
    also be provided, while provided required positional arguments via ``args``:

    >>> func = qml.noise.partial_wires(qml.RX, 3.2, [20])
    >>> func(qml.RY(1.0, [0]))
    qml.RX(3.2, wires=[0])

    Finally, one can also use ``kwargs`` instead of positional arguments:

    >>> func = qml.noise.partial_wires(qml.RX, phi=1.2)
    >>> func(qml.RY(1.0, [2]))
    qml.RX(1.2, wires=[2])
    >>> rfunc = qml.noise.partial_wires(qml.RX(1.2, [12]), phi=2.3)
    >>> rfunc(qml.RY(1.0, ["light"]))
    qml.RX(2.3, wires=["light"])
    """

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
