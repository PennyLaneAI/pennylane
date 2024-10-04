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
"""Contains utility functions for building boolean conditionals for noise models

Developer note: Conditionals inherit from BooleanFn and store the condition they
utilize in the ``condition`` attribute.
"""

from inspect import isclass, signature

import pennylane as qml
from pennylane.boolean_fn import BooleanFn
from pennylane.measurements import MeasurementProcess, MeasurementValue, MidMeasureMP
from pennylane.ops import Adjoint, Controlled
from pennylane.templates import ControlledSequence
from pennylane.wires import WireError, Wires

# pylint: disable = unnecessary-lambda, too-few-public-methods, too-many-statements, too-many-branches


class WiresIn(BooleanFn):
    """A conditional for evaluating if the wires of an operation exist in a specified set of wires.

    Args:
        wires (Union[Iterable[int, str], Wires]): Sequence of wires for building the wire set.

    .. seealso:: Users are advised to use :func:`~.wires_in` for a functional construction.
    """

    def __init__(self, wires):
        self._cond = set(wires)
        self.condition = self._cond
        super().__init__(
            lambda wire: _get_wires(wire).issubset(self._cond), f"WiresIn({list(wires)})"
        )


class WiresEq(BooleanFn):
    """A conditional for evaluating if a given wire is equal to a specified set of wires.

    Args:
        wires (Union[Iterable[int, str], Wires]): Sequence of wires for building the wire set.

    .. seealso:: Users are advised to use :func:`~.wires_eq` for a functional construction.
    """

    def __init__(self, wires):
        self._cond = set(wires)
        self.condition = self._cond
        super().__init__(
            lambda wire: _get_wires(wire) == self._cond,
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
        wires = set().union(*((getattr(w, "wires", None) or Wires(w)).tolist() for w in iters))
    except (TypeError, WireError):
        raise ValueError(f"Wires cannot be computed for {val}") from None
    return wires


def wires_in(wires):
    """Builds a conditional as a :class:`~.BooleanFn` for evaluating
    if the wires of an input operation are within the specified set of wires.

    Args:
        wires (Union(Iterable[int, str], Wires, Operation, int, str)): Object to be used
            for building the wire set.

    Returns:
        :class:`WiresIn <pennylane.noise.conditionals.WiresIn>`: A callable object with
        signature ``Union(Iterable[int, str], Wires, Operation, int, str)``. It evaluates
        to ``True`` if the wire set constructed from the input to the callable is a
        subset of the one built from the specified ``wires`` set.

    Raises:
        ValueError: If the wire set cannot be computed from ``wires``.

    **Example**

    One may use ``wires_in`` with a given sequence of wires which are used as a wire set:

    >>> cond_func = qml.noise.wires_in([0, 1])
    >>> cond_func(qml.X(0))
    True

    >>> cond_func(qml.X(3))
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
    """Builds a conditional as a :class:`~.BooleanFn` for evaluating
    if a given wire is equal to specified set of wires.

    Args:
        wires (Union(Iterable[int, str], Wires, Operation, int, str)): Object to be used
            for building the wire set.

    Returns:
        :class:`WiresEq <pennylane.noise.conditionals.WiresEq>`: A callable object with
        signature ``Union(Iterable[int, str], Wires, Operation, int, str)``. It evaluates
        to ``True`` if the wire set constructed from the input to the callable is equal
        to the one built from the specified ``wires`` set.

    Raises:
        ValueError: If the wire set cannot be computed from ``wires``.

    **Example**

    One may use ``wires_eq`` with a given sequence of wires which are used as a wire set:

    >>> cond_func = qml.noise.wires_eq(0)
    >>> cond_func(qml.X(0))
    True

    >>> cond_func(qml.RY(1.23, wires=[3]))
    False

    Additionally, if an :class:`Operation <pennylane.operation.Operation>` is provided,
    its ``wires`` are extracted and used to build the wire set:

    >>> cond_func = qml.noise.wires_eq(qml.RX(1.0, "dino"))
    >>> cond_func(qml.RZ(1.23, wires="dino"))
    True

    >>> cond_func("eve")
    False
    """
    return WiresEq(_get_wires(wires))


class OpIn(BooleanFn):
    """A conditional for evaluating if a given operation exist in a specified set of operations.

    Args:
        ops (Union[str, class, Operation, list[str, class, Operation]]): Sequence of operation
            instances, string representations or classes to build the operation set.

    .. seealso:: Users are advised to use :func:`~.op_in` for a functional construction.
    """

    def __init__(self, ops):
        ops_ = [ops] if not isinstance(ops, (list, tuple, set)) else ops
        self._cond = [
            (
                op
                if not isinstance(op, MeasurementProcess)
                else (getattr(op, "obs", None) or getattr(op, "H", None))
            )
            for op in ops_
        ]
        self._cops = _get_ops(ops)
        self.condition = self._cops
        super().__init__(
            self._check_in_ops, f"OpIn({[getattr(op, '__name__', op) for op in self._cops]})"
        )

    def _check_in_ops(self, operation):
        xs = operation if isinstance(operation, (list, tuple, set)) else [operation]
        xs = [
            (
                op
                if not isinstance(op, MeasurementProcess)
                else (getattr(op, "obs", None) or getattr(op, "H", None))
            )
            for op in xs
        ]
        cs = _get_ops(xs)

        try:
            return all(
                (
                    c in self._cops
                    if isclass(x) or not getattr(x, "arithmetic_depth", 0)
                    else any(
                        (
                            _check_arithmetic_ops(op, x)
                            if isinstance(op, cp) and getattr(op, "arithmetic_depth", 0)
                            else cp == _get_ops(x)[0]
                        )
                        for op, cp in zip(self._cond, self._cops)
                    )
                )
                for x, c in zip(xs, cs)
            )
        except:  # pylint: disable = bare-except # pragma: no cover
            raise ValueError(
                "OpIn does not support arithmetic operations "
                "that cannot be converted to a linear combination"
            ) from None


class OpEq(BooleanFn):
    """A conditional for evaluating if a given operation is equal to the specified operation.

    Args:
        ops (Union[str, class, Operation]): An operation instance, string representation or
            class to build the operation set.

    .. seealso:: Users are advised to use :func:`~.op_eq` for a functional construction.
    """

    def __init__(self, ops):
        ops_ = [ops] if not isinstance(ops, (list, tuple, set)) else ops
        self._cond = [
            (
                op
                if not isinstance(op, MeasurementProcess)
                else (getattr(op, "obs", None) or getattr(op, "H", None))
            )
            for op in ops_
        ]
        self._cops = _get_ops(ops)
        self.condition = self._cops
        cops_names = list(getattr(op, "__name__", op) for op in self._cops)
        super().__init__(
            self._check_eq_ops,
            f"OpEq({cops_names if len(cops_names) > 1 else cops_names[0]})",
        )

    def _check_eq_ops(self, operation):
        if all(isclass(op) or not getattr(op, "arithmetic_depth", 0) for op in self._cond):
            return _get_ops(operation) == self._cops

        try:
            xs = operation if isinstance(operation, (list, tuple, set)) else [operation]
            xs = [
                (
                    op
                    if not isinstance(op, MeasurementProcess)
                    else (getattr(op, "obs", None) or getattr(op, "H", None))
                )
                for op in xs
            ]

            return (
                len(xs) == len(self._cond)
                and _get_ops(xs) == self._cops
                and all(
                    _check_arithmetic_ops(op, x)
                    for (op, x) in zip(self._cond, xs)
                    if not isclass(x) and getattr(x, "arithmetic_depth", 0)
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
            for building the operation set.

    Returns:
        tuple[class]: tuple of :class:`Operation <pennylane.operation.Operation>`
        classes corresponding to val.
    """
    vals = val if isinstance(val, (list, tuple, set)) else [val]
    op_names = []
    for _val in vals:
        if isinstance(_val, str):
            op_names.append(getattr(qml.ops, _val, None) or getattr(qml, _val))
        elif isclass(_val) and not issubclass(_val, MeasurementProcess):
            op_names.append(_val)
        elif isinstance(_val, (MeasurementValue, MidMeasureMP)):
            mid_measure = _val if isinstance(_val, MidMeasureMP) else _val.measurements[0]
            op_names.append(["MidMeasure", "Reset"][getattr(mid_measure, "reset", 0)])
        elif isinstance(_val, MeasurementProcess):
            obs_name = _get_ops(getattr(_val, "obs", None) or getattr(_val, "H", None))
            if len(obs_name) == 1:
                obs_name = obs_name[0]
            op_names.append(obs_name)
        else:
            op_names.append(getattr(_val, "__class__"))
    return tuple(op_names)


def _check_arithmetic_ops(op1, op2):
    """Helper method for comparing two arithmetic operators based on type check of the bases"""
    # pylint: disable = unnecessary-lambda-assignment

    if isinstance(op1, (Adjoint, Controlled, ControlledSequence)) or isinstance(
        op2, (Adjoint, Controlled, ControlledSequence)
    ):
        return (
            isinstance(op1, type(op2))
            and op1.arithmetic_depth == op2.arithmetic_depth
            and _get_ops(op1.base) == _get_ops(op2.base)
        )

    lc_cop = lambda op: qml.ops.LinearCombination(*op.terms())

    if isinstance(op1, qml.ops.Exp) or isinstance(op2, qml.ops.Exp):
        if (
            not isinstance(op1, type(op2))
            or (op1.base.arithmetic_depth != op2.base.arithmetic_depth)
            or not qml.math.allclose(op1.coeff, op2.coeff)
            or (op1.num_steps != op2.num_steps)
        ):
            return False
        if op1.base.arithmetic_depth:
            return _check_arithmetic_ops(op1.base, op2.base)
        return _get_ops(op1.base) == _get_ops(op2.base)

    op1, op2 = qml.simplify(op1), qml.simplify(op2)
    if op1.arithmetic_depth != op2.arithmetic_depth:
        return False

    coeffs, op_terms = lc_cop(op1).terms()
    sprods = [_get_ops(getattr(op_term, "operands", op_term)) for op_term in op_terms]

    def _lc_op(x):
        coeffs2, op_terms2 = lc_cop(x).terms()
        sprods2 = [_get_ops(getattr(op_term, "operands", op_term)) for op_term in op_terms2]
        for coeff, sprod in zip(coeffs2, sprods2):
            present, p_index = False, -1
            while sprod in sprods[p_index + 1 :]:
                p_index = sprods[p_index + 1 :].index(sprod) + (p_index + 1)
                if qml.math.allclose(coeff, coeffs[p_index]):
                    coeffs.pop(p_index)
                    sprods.pop(p_index)
                    present = True
                    break

            if not present:
                break

        return present

    return _lc_op(op2)


def op_in(ops):
    """Builds a conditional as a :class:`~.BooleanFn` for evaluating
    if a given operation exist in a specified set of operations.

    Args:
        ops (str, class, Operation, list(Union[str, class, Operation])): Sequence of string
            representations, instances, or classes of the operation(s).

    Returns:
        :class:`OpIn <pennylane.noise.conditionals.OpIn>`: A callable object that accepts
        an :class:`~.Operation` and returns a boolean output. It accepts any input from:
        ``Union[str, class, Operation, list(Union[str, class, Operation])]`` and evaluates
        to ``True`` if the input operation(s) exists in the set of operation(s) specified by
        ``ops``. Comparison is based on the operation's type, irrespective of wires.

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
    """Builds a conditional as a :class:`~.BooleanFn` for evaluating
    if a given operation is equal to the specified operation.

    Args:
        ops (str, class, Operation): String representation, an instance or class of the operation.

    Returns:
        :class:`OpEq <pennylane.noise.conditionals.OpEq>`: A callable object that accepts
        an :class:`~.Operation` and returns a boolean output. It accepts any input from:
        ``Union[str, class, Operation]`` and evaluates to ``True`` if the input operation(s)
        is equal to the set of operation(s) specified by ``ops``. Comparison is based on
        the operation's type, irrespective of wires.

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
    return OpEq(ops)


class MeasEq(qml.BooleanFn):
    """A conditional for evaluating if a given measurement process is equal to the specified measurement process.

    Args:
        mp(Union[Iterable[MeasurementProcess], MeasurementProcess]): Sequence of measurement process.

    .. seealso:: Users are advised to use :func:`~.meas_eq` for a functional construction.
    """

    def __init__(self, mps):
        """ """
        self._cond = [mps] if not isinstance(mps, (list, tuple, set)) else mps
        self.condition, self._cmps = [], []
        for mp in self._cond:
            if (callable(mp) and (mp := _MEAS_FUNC_MAP.get(mp, None)) is None) or (
                isclass(mp) and not issubclass(mp, MeasurementProcess)
            ):
                raise ValueError(
                    f"MeasEq should be initialized with a MeasurementProcess, got {mp}"
                )
            self.condition.append(mp)
            self._cmps.append(mp if isclass(mp) else mp.__class__)

        mp_ops = list(getattr(op, "return_type", op.__class__.__name__) for op in self.condition)
        mp_names = [
            repr(op) if not isinstance(op, property) else self.condition[idx].__name__
            for idx, op in enumerate(mp_ops)
        ]
        super().__init__(
            self._check_meas, f"MeasEq({mp_names if len(mp_names) > 1 else mp_names[0]})"
        )

    def _check_meas(self, mp):

        if isclass(mp) and not issubclass(mp, MeasurementProcess):
            return False

        if callable(mp) and (mp := _MEAS_FUNC_MAP.get(mp, None)) is None:
            return False

        cmps = [
            m_ if isclass(m_) else m_.__class__
            for m_ in ([mp] if not isinstance(mp, (list, tuple, set)) else mp)
        ]
        if len(cmps) != len(self._cond):
            return False

        return all(mp1 == mp2 for mp1, mp2 in zip(cmps, self._cmps))


def meas_eq(mp):
    """Builds a conditional as a :class:`~.BooleanFn` for evaluating
    if a given operation is equal to the specified operation.

    Args:
        mp (MeasurementProcess): An instance of any class that inherits from :class:`~.MeasurementProcess`.
    """
    return MeasEq(mp)


def _rename(newname):
    """Decorator function for renaming ``_partial_op`` function used in partial_wires"""

    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


_MEAS_FUNC_MAP = {
    qml.expval: qml.measurements.ExpectationMP,
    qml.var: qml.measurements.VarianceMP,
    qml.state: qml.measurements.StateMP,
    qml.density_matrix: qml.measurements.DensityMatrixMP,
    qml.counts: qml.measurements.CountsMP,
    qml.sample: qml.measurements.SampleMP,
    qml.probs: qml.measurements.ProbabilityMP,
    qml.vn_entropy: qml.measurements.VnEntropyMP,
    qml.mutual_info: qml.measurements.MutualInfoMP,
    qml.purity: qml.measurements.PurityMP,
    qml.classical_shadow: qml.measurements.ClassicalShadowMP,
    qml.shadow_expval: qml.measurements.ShadowExpvalMP,
    qml.measure: qml.measurements.MidMeasureMP,
}


def partial_wires(operation, *args, **kwargs):
    """Builds a partial function based on the given operation with
    all argument frozen except ``wires``.

    Args:
        operation (Operation, class): Instance of an operation or the class
            corresponding to the operation.
        *args: Positional arguments provided in the case where the keyword argument
            ``operation`` is a class for building the partially evaluated instance.
        **kwargs: Keyword arguments for the building the partially evaluated instance.
            These will override any arguments present in the operation instance or ``args``.

    Returns:
        callable: A wrapper function that accepts a sequence of wires as an argument or
        any object with a ``wires`` property.

    Raises:
        ValueError: If ``args`` are provided when the given ``operation`` is an instance.

    **Example**

    One may give an instance of :class:`Operation <pennylane.operation.Operation>`
    for the ``operation`` argument:

    >>> func = qml.noise.partial_wires(qml.RX(1.2, [12]))
    >>> func(2)
    qml.RX(1.2, wires=[2])
    >>> func(qml.RY(1.0, ["wires"]))
    qml.RX(1.2, wires=["wires"])

    Additionally, an :class:`Operation <pennylane.operation.Operation>` class can
    also be provided, while providing required positional arguments via ``args``:

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
    is_meas_class = isinstance(operation, qml.measurements.MeasurementProcess)
    is_meta_class = isinstance(operation, (qml.ops.Adjoint, qml.ops.Controlled))

    if not callable(operation):
        if args:
            raise ValueError(
                "Args cannot be provided when operation is an instance, "
                f"got operation = {operation} and args = {args}."
            )
        args, metadata = getattr(operation, "_flatten")()
        is_metadata_flat = is_meas_class or isinstance(operation, (qml.ops.Controlled))
        if len(metadata) > 1:
            kwargs = {**dict(metadata[1] if not is_metadata_flat else metadata), **kwargs}
        operation = type(operation)
    elif operation in _MEAS_FUNC_MAP:
        is_meas_class = True
        operation = _MEAS_FUNC_MAP[operation]
    elif operation in (
        qml.adjoint,
        qml.ctrl,
    ):
        is_meta_class = True
        operation = {qml.adjoint: qml.ops.Adjoint, qml.ctrl: qml.ops.Controlled}[operation]

    fsignature = signature(getattr(operation, "__init__", operation)).parameters
    parameters = list(fsignature)[int("self" in fsignature) :]
    arg_params = {**dict(zip(parameters, args)), **kwargs}

    arg_wires = arg_params.pop("wires", None)  # Ensure we don't include wires in name
    if is_meas_class and "op" in arg_params:
        arg_params["obs"] = arg_params.pop("op")

    if is_meta_class and "op" in arg_params:
        arg_params["base"] = arg_params.pop("op")

    if operation == qml.ops.Controlled and "control" in arg_params:
        arg_params["control_wires"] = arg_params.pop("control")

    op_name = f"{operation.__name__}("
    for key, val in arg_params.copy().items():
        if key in parameters:
            op_name += f"{key}={val}, "
        else:  # pragma: no cover
            del arg_params[key]
    op_name = op_name[:-2] + ")" if len(arg_params) else op_name[:-1]

    @_rename(op_name)
    def _partial(wires=None, **partial_kwargs):
        """Wrapper function for partial_wires"""
        op_args = arg_params
        op_args["wires"] = wires or arg_wires
        if wires is not None:
            op_args["wires"] = getattr(wires, "wires", None) or (
                [wires] if isinstance(wires, (int, str)) else list(wires)
            )

        if is_meas_class:
            if "obs" not in parameters:
                _ = partial_kwargs.pop("op", None)

            if not op_args.get("obs", None) and (obs := partial_kwargs.get("op", None)):
                op_args["obs"] = obs
                if op_args["wires"] is None:
                    op_args["wires"] = obs.wires

            if (obs := op_args.get("obs", None)) is not None and op_args["wires"] is not None:
                op_args["obs"] = obs.map_wires(dict(zip(obs.wires, op_args.pop("wires"))))

            if "H" in parameters and operation == qml.measurements.ShadowExpvalMP:
                mp_ham = op_args["H"]
                op_args["H"] = mp_ham.map_wires(dict(zip(mp_ham.wires, op_args.pop("wires"))))

        if is_meta_class:
            if not op_args.get("base", None) and (obs := partial_kwargs.get("op", None)):
                op_args["base"] = obs
                if op_args["wires"] is None:
                    op_args["wires"] = obs.wires

            if (base := op_args.get("base", None)) is not None and op_args["wires"] is not None:
                op_args["base"] = base.map_wires(dict(zip(base.wires, op_args.pop("wires"))))

        for key, val in partial_kwargs.items():
            if key in parameters:
                op_args[key] = val

        if "wires" not in parameters:
            _ = op_args.pop("wires", None)

        return operation(**op_args)

    return _partial
