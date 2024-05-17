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


class WiresIn(BooleanFn):
    """A ``Conditional`` for evaluating if a given wire exist in a specified set of wires

    Args:
        wires (Union[list[int, str], Wires]): sequence of wires for building the wire set.

    .. seealso:: Users are advised to use :func:`~.wires_in` for a functional construction.
    """

    def __init__(self, wires):
        self._cond = set(wires)
        self.condition = self._cond
        super().__init__(lambda x: _get_wires(x).issubset(self._cond), f"WiresIn({list(wires)})")


class WiresEq(BooleanFn):
    """A ``Conditional`` for evaluating if a given wire is equal to a specified set of wires

    Args:
        wires (Union[list[int, str], Wires]): sequence of wires for building the wire set.

    .. seealso:: Users are advised to use :func:`~.wires_eq` for a functional construction.
    """

    def __init__(self, wires):
        self._cond = set(wires)
        self.condition = self._cond
        super().__init__(
            lambda x: _get_wires(x) == self._cond,
            f"WiresEq({list(wires) if len(wires) > 1 else list(wires)[0]})",
        )


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
        :class:`WiresIn <pennylane.noise.WiresIn>`: a boolean function
        which evaluates to ``True`` if a given wire exist in a specified set of wires.

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
    return WiresIn(_get_wires(wires))


def wires_eq(wires):
    """Builds a ``Conditional`` as a boolean function for evaluating
    if a given wire is equal to specified set of wires.

    Args:
        wires (Union(list[int, str], Wires, Operation)): object to be used
            for building the wire set.

    Returns:
        :class:`WiresEq <pennylane.noise.WiresEq>`: a boolean function
        which evaluates to ``True`` if a given wire is equal to specified set of wires.

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
    return WiresEq(_get_wires(wires))


class OpIn(BooleanFn):
    """A ``Conditional`` for evaluating if a given operation exist in a specified set of operation

    Args:
        wires (Union[str, Operation, list[str, Operation]]): sequence of operations to build the operation set.

    .. seealso:: Users are advised to use :func:`~.op_in` for a functional construction.
    """

    def __init__(self, ops):
        self._cond = ops
        self._cops = _get_ops(ops)
        self.condition = self._cops
        super().__init__(
            self._check_in_ops, f"OpIn({[getattr(op, '__name__') for op in self._cops]})"
        )

    def _check_in_ops(self, x):
        x = [x] if not isinstance(x, (list, tuple, set)) else x

        try:
            return all(
                (
                    _x in self._cops
                    if isclass(_x)
                    else (
                        isinstance(_x, self._cops)
                        if not getattr(_x, "arithmetic_depth", 0)
                        else any(
                            _check_with_lc_op(op, _x)
                            for op in self._cond
                            if not isclass(op) and getattr(op, "arithmetic_depth", 0)
                        )
                    )
                )
                for _x in x
            )
        except:  # pylint: disable = bare-except # pragma: no cover
            raise ValueError(
                "OpEq does not support arithmetic operations "
                "that cannot be converted to a linear combination"
            ) from None


class OpEq(BooleanFn):
    """A ``Conditional`` for evaluating if a given operation is equal to the specified operation

    Args:
        wires (Union[str, Operation, list[str, Operation]]): sequence of operations to build the operation set.

    .. seealso:: Users are advised to use :func:`~.op_eq` for a functional construction.
    """

    def __init__(self, ops):
        self._cond = ops
        self._cops = _get_ops(ops)
        self.condition = self._cops
        cops_names = list(getattr(op, "__name__") for op in self._cops)
        super().__init__(
            self._check_eq_ops,
            f"OpEq({cops_names if len(cops_names) > 1 else cops_names[0]})",
        )

    def _check_eq_ops(self, x):
        if not any(not isclass(op) and getattr(op, "arithmetic_depth", 0) for op in self._cond):
            return _get_ops(x) == self._cops

        try:
            return _get_ops(x) == self._cops and (
                _check_with_lc_op(self._cond[0], x)
                if len(self._cops) == 1
                else len(x) == len(self._cond)
                and all(
                    _check_with_lc_op(_op, _x)
                    for (_x, _op) in zip(x, self._cond)
                    if not isclass(_x) and getattr(_x, "arithmetic_depth", 0)
                )
            )
        except:  # pylint: disable = bare-except # pragma: no cover
            raise ValueError(
                "OpEq does not support arithmetic operations "
                "that cannot be converted to a linear combination"
            ) from None


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
            p_index = sprods.index(sprod)
            if not qml.math.equal(coeff, coeffs[p_index]):
                present = False
                break
            coeffs.pop(p_index)
            sprods.pop(p_index)

        return present

    return _lc_op(op2)


def op_in(ops):
    """Builds a ``Conditional`` as a boolean function for evaluating
    if a given operation exist in a specified set of operation.

    Args:
        ops (str, Operation, Union(list[str, Operation])): string
            representation or instance of the operation.

    Returns:
        :class:`OpIn <pennylane.noise.OpIn>`: a boolean function that evaluates to ``True``, if a
        given operation exist in a specified set of operation(s), irrespective of their wires.

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
    return OpIn(ops)


def op_eq(ops):
    """Builds a ``Conditional`` as a boolean function for evaluating
    if a given operation is equal to the specified operation.

    Args:
        ops (str, Operation, Union(list[str, Operation])): string
            representation or instance of the operation.

    Returns:
        :class:`OpEq <pennylane.noise.OpEq>`: a boolean function that evaluates to ``True``, if a
        given operation is equal to the specified set of operation(s), irrespective of their wires.

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
    ops = [ops] if not isinstance(ops, (list, tuple, set)) else ops
    return OpEq(ops)


def _rename(newname):
    """Decorator function for renaming ``_partial_op`` function used in partial_wires"""

    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


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

    op_name = f"{operation.__name__}("
    for key, val in op.keywords.items():
        op_name += f"{key}={val}"
    op_name += ")"

    @_rename(op_name)
    def _partial_op(x, **model_kwargs):  # pylint: disable = unused-argument
        """Wrapper function for partial_wires"""
        wires = getattr(x, "wires", None) or ([x] if isinstance(x, (int, str)) else list(x))
        return op(wires=wires)

    return _partial_op
