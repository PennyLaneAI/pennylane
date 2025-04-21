# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Contains the condition transform.
"""
import functools
from functools import wraps
from typing import Callable, Optional, Sequence, Type, Union

import pennylane as qml
from pennylane import QueuingManager
from pennylane.capture import FlatFn
from pennylane.compiler import compiler
from pennylane.measurements import MeasurementValue, MidMeasureMP, get_mcm_predicates
from pennylane.operation import AnyWires, Operation, Operator
from pennylane.ops.op_math.symbolicop import SymbolicOp


def _add_abstract_shapes(f):
    """Add the shapes of all the returned variables before the returned variables.

    Dynamic shape support currently has a lot of dragons. This function is subject to change
    at any moment. Use duplicate code till reliable abstractions are found.

    >>> @qml.capture.FlatFn
    ... def f(x):
    ...     return x + 1
    >>> jax.make_jaxpr(f, abstracted_axes={0:"a"})(jnp.zeros(4))
    { lambda ; a:i32[] b:f32[a]. let
        c:f32[a] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 a
        d:f32[a] = add b c
    in (d,) }
    >>> jax.make_jaxpr(_add_abstract_shapes(f), abstracted_axes={0:"a"})(jnp.zeros(4))
    { lambda ; a:i32[] b:f32[a]. let
        c:f32[a] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 a
        d:f32[a] = add b c
    in (a, d) }

    Now both the dimension of the array and the array are getting returned, rather than
    just the array.

    Note that we assume that ``f`` returns a sequence of tensorlikes, like ``FlatFn`` would.

    """

    def new_f(*args, **kwargs):
        out = f(*args, **kwargs)
        shapes = []
        for x in out:
            shapes.extend(s for s in getattr(x, "shape", ()) if qml.math.is_abstract(s))
        return *shapes, *out

    return new_f


class ConditionalTransformError(ValueError):
    """Error for using qml.cond incorrectly"""


class Conditional(SymbolicOp, Operation):
    """A Conditional Operation.

    Unless you are a Pennylane plugin developer, **you should NOT directly use this class**,
    instead, use the :func:`qml.cond <.cond>` function.

    The ``Conditional`` class is a container class that defines an operation
    that should be applied relative to a single measurement value.

    Support for executing ``Conditional`` operations is device-dependent. If a
    device doesn't support mid-circuit measurements natively, then the QNode
    will apply the :func:`defer_measurements` transform.

    Args:
        expr (MeasurementValue): the measurement outcome value to consider
        then_op (Operation): the PennyLane operation to apply conditionally
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    num_wires = AnyWires

    def __init__(self, expr, then_op: Type[Operation], id=None):
        self.hyperparameters["meas_val"] = expr
        self._name = f"Conditional({then_op.name})"
        super().__init__(then_op, id=id)
        if self.grad_recipe is None:
            self.grad_recipe = [None] * self.num_params

    def label(self, decimals=None, base_label=None, cache=None):
        return self.base.label(decimals=decimals, base_label=base_label, cache=cache)

    @property
    def meas_val(self):
        """the measurement outcome value to consider from `expr` argument"""
        return self.hyperparameters["meas_val"]

    @property
    def num_params(self):
        return self.base.num_params

    @property
    def ndim_params(self):
        return self.base.ndim_params

    def map_wires(self, wire_map):
        meas_val = self.meas_val.map_wires(wire_map)
        then_op = self.base.map_wires(wire_map)
        return Conditional(meas_val, then_op=then_op)

    def matrix(self, wire_order=None):
        return self.base.matrix(wire_order=wire_order)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def eigvals(self):
        return self.base.eigvals()

    @property
    def has_adjoint(self):
        return self.base.has_adjoint

    def adjoint(self):
        return Conditional(self.meas_val, self.base.adjoint())


class CondCallable:  # pylint:disable=too-few-public-methods
    """Base class to represent a conditional function with boolean predicates.

    Args:
        condition (bool): a conditional expression
        true_fn (callable): The function to apply if ``condition`` is ``True``
        false_fn (callable): The function to apply if ``condition`` is ``False``
        elifs (List(Tuple(bool, callable))): A list of (bool, elif_fn) clauses.

    Passing ``false_fn`` and ``elifs`` on initialization
    is optional; these functions can be registered post-initialization
    via decorators:

    .. code-block:: python

        def f(x):
            @qml.cond(x > 0)
            def conditional(y):
                return y ** 2

            @conditional.else_if(x < -2)
            def conditional(y):
                return y

            @conditional.otherwise
            def conditional_false_fn(y):
                return -y

            return conditional(x + 1)

    >>> [f(0.5), f(-3), f(-0.5)]
    [2.25, -2, -0.5]
    """

    def __init__(self, condition, true_fn, false_fn=None, elifs=()):
        self.preds = [condition]
        self.branch_fns = [true_fn]
        self.otherwise_fn = false_fn

        # when working with `qml.capture.enabled()`,
        # it's easier to store the original `elifs` argument
        self.orig_elifs = elifs

        if false_fn is None and not qml.capture.enabled():
            self.otherwise_fn = lambda *args, **kwargs: None

        if elifs and not qml.capture.enabled():
            elif_preds, elif_fns = list(zip(*elifs))
            self.preds.extend(elif_preds)
            self.branch_fns.extend(elif_fns)

    def else_if(self, pred):
        """Decorator that allows else-if functions to be registered with a corresponding
        boolean predicate.

        Args:
            pred (bool): The predicate that will determine if this branch is executed.

        Returns:
            callable: decorator that is applied to the else-if function
        """

        def decorator(branch_fn):
            self.preds.append(pred)
            self.branch_fns.append(branch_fn)
            self.orig_elifs += ((pred, branch_fn),)
            return self

        return decorator

    def otherwise(self, otherwise_fn):
        """Decorator that registers the function to be run if all
        conditional predicates (including optional) evaluates to ``False``.

        Args:
            otherwise_fn (callable): the function to apply if all ``self.preds`` evaluate to ``False``
        """
        self.otherwise_fn = otherwise_fn
        return self

    @property
    def false_fn(self):
        """callable: the function to apply if all ``self.preds`` evaluate to ``False``.

        Alias for ``otherwise_fn``.
        """
        return self.otherwise_fn

    @property
    def true_fn(self):
        """callable: the function to apply if all ``self.condition`` evaluate to ``True``"""
        return self.branch_fns[0]

    @property
    def condition(self):
        """bool: the condition that determines if ``self.true_fn`` is applied"""
        return self.preds[0]

    @property
    def elifs(self):
        """(List(Tuple(bool, callable))): a list of (bool, elif_fn) clauses"""
        return list(zip(self.preds[1:], self.branch_fns[1:]))

    def __call_capture_disabled(self, *args, **kwargs):
        # python fallback
        for pred, branch_fn in zip(self.preds, self.branch_fns):
            if pred:
                return branch_fn(*args, **kwargs)

        return self.false_fn(*args, **kwargs)  # pylint: disable=not-callable

    def __call_capture_enabled(self, *args, **kwargs):
        import jax  # pylint: disable=import-outside-toplevel

        cond_prim = _get_cond_qfunc_prim()

        elifs = (
            [self.orig_elifs]
            if len(self.orig_elifs) > 0 and not isinstance(self.orig_elifs[0], tuple)
            else list(self.orig_elifs)
        )
        flat_true_fn = FlatFn(self.true_fn)
        branches = [(self.preds[0], flat_true_fn), *elifs, (True, self.otherwise_fn)]

        end_const_ind = len(
            branches
        )  # consts go after the len(branches) conditions, first const at len(branches)
        conditions = []
        jaxpr_branches = []
        consts = []
        consts_slices = []

        abstracted_axes, abstract_shapes = qml.capture.determine_abstracted_axes(args)

        for pred, fn in branches:
            conditions.append(pred)
            if fn is None:
                jaxpr_branches.append(None)
                consts_slices.append(slice(0, 0))
            else:
                f = FlatFn(functools.partial(fn, **kwargs))
                if jax.config.jax_dynamic_shapes:
                    f = _add_abstract_shapes(f)
                jaxpr = jax.make_jaxpr(f, abstracted_axes=abstracted_axes)(*args)
                jaxpr_branches.append(jaxpr.jaxpr)
                consts_slices.append(slice(end_const_ind, end_const_ind + len(jaxpr.consts)))
                consts += jaxpr.consts
                end_const_ind += len(jaxpr.consts)

        _validate_jaxpr_returns(jaxpr_branches)
        flat_args, _ = jax.tree_util.tree_flatten(args)
        results = cond_prim.bind(
            *conditions,
            *consts,
            *abstract_shapes,
            *flat_args,
            jaxpr_branches=jaxpr_branches,
            consts_slices=consts_slices,
            args_slice=slice(end_const_ind, None),
        )
        assert flat_true_fn.out_tree is not None, "out_tree of flat_true_fn should exist"
        results = results[-flat_true_fn.out_tree.num_leaves :]
        return jax.tree_util.tree_unflatten(flat_true_fn.out_tree, results)

    def __call__(self, *args, **kwargs):

        if qml.capture.enabled():
            return self.__call_capture_enabled(*args, **kwargs)

        return self.__call_capture_disabled(*args, **kwargs)


def cond(
    condition: Union[MeasurementValue, bool],
    true_fn: Optional[Callable] = None,
    false_fn: Optional[Callable] = None,
    elifs: Sequence = (),
):
    """Quantum-compatible if-else conditionals --- condition quantum operations
    on parameters such as the results of mid-circuit qubit measurements.

    This method is restricted to simply branching on mid-circuit measurement
    results when it is not used with the :func:`~.qjit` decorator.

    When used with the :func:`~.qjit` decorator, this function allows for general
    if-elif-else constructs. All ``true_fn``, ``false_fn`` and ``elifs`` branches
    will be captured by Catalyst, the just-in-time (JIT) compiler, with the executed
    branch determined at runtime. For more details, please see :func:`catalyst.cond`.

    .. note::

        With the Python interpreter, support for :func:`~.cond`
        is device-dependent. If a device doesn't
        support mid-circuit measurements natively, then the QNode will
        apply the :func:`defer_measurements` transform.

    .. note::

        When used with :func:`~.qjit`, this function only supports
        the Catalyst compiler. See :func:`catalyst.cond` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`.

    .. note::

        When used with :func:`.pennylane.capture.enabled`, this function allows for general
        if-elif-else constructs. As with the JIT mode, all branches are captured,
        with the executed branch determined at runtime.

        Each branch can receive arguments, but the arguments must be JAX-compatible.
        If a branch returns one or more variables, every other branch must return the same abstract values.

    Args:
        condition (Union[.MeasurementValue, bool]): a conditional expression that may involve a mid-circuit
           measurement value (see :func:`.pennylane.measure`).
        true_fn (callable): The quantum function or PennyLane operation to
            apply if ``condition`` is ``True``
        false_fn (callable): The quantum function or PennyLane operation to
            apply if ``condition`` is ``False``
        elifs (Sequence(Tuple(bool, callable))): A sequence of (bool, elif_fn) clauses. Can only
            be used when decorated by :func:`~.qjit` or if the condition is not
            a mid-circuit measurement.

    Returns:
        function: A new function that applies the conditional equivalent of ``true_fn``. The returned
        function takes the same input arguments as ``true_fn``.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode(x, y):
            qml.Hadamard(0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)

            qml.Hadamard(2)
            qml.RY(-np.pi/2, wires=[2])
            m_1 = qml.measure(2)
            qml.cond(m_1 == 0, qml.RX)(y, wires=1)
            return qml.expval(qml.Z(1))

    .. code-block :: pycon

        >>> first_par = np.array(0.3)
        >>> sec_par = np.array(1.23)
        >>> qnode(first_par, sec_par)
        tensor(0.32677361, requires_grad=True)

    .. note::

        If the first argument of ``cond`` is a measurement value (e.g., ``m_0``
        in ``qml.cond(m_0, qml.RY)``), then ``m_0 == 1`` is considered
        internally.

    .. warning::

        Expressions with boolean logic flow using operators like ``and``,
        ``or`` and ``not`` are not supported as the ``condition`` argument.

        While such statements may not result in errors, they may result in
        incorrect behaviour.

    In just-in-time (JIT) mode using the :func:`~.qjit` decorator,

    .. code-block:: python3

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x: float):
            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            def ansatz_false():
                qml.RY(x, wires=0)

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

    >>> circuit(1.4)
    Array(0.16996714, dtype=float64)
    >>> circuit(1.6)
    Array(0., dtype=float64)

    Additional 'else-if' clauses can also be included via the ``elif`` argument:

    .. code-block:: python3

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x):

            def true_fn():
                qml.RX(x, wires=0)

            def elif_fn():
                qml.RY(x, wires=0)

            def false_fn():
                qml.RX(x ** 2, wires=0)

            qml.cond(x > 2.7, true_fn, false_fn, ((x > 1.4, elif_fn),))()
            return qml.expval(qml.Z(0))

    >>> circuit(1.2)
    Array(0.13042371, dtype=float64)

    .. note::

        If the above syntax is used with a ``QNode`` that is not decorated with
        :func:`~pennylane.qjit` and none of the predicates contain mid-circuit measurements,
        ``qml.cond`` will fall back to using native Python ``if``-``elif``-``else`` blocks.

    .. details::
        :title: Usage Details

        **Conditional quantum functions**

        The ``cond`` transform allows conditioning quantum functions too:

        .. code-block:: python3

            dev = qml.device("default.qubit")

            def qfunc(par, wires):
                qml.Hadamard(wires[0])
                qml.RY(par, wires[0])

            @qml.qnode(dev)
            def qnode(x):
                qml.Hadamard(0)
                m_0 = qml.measure(0)
                qml.cond(m_0, qfunc)(x, wires=[1])
                return qml.expval(qml.Z(1))

        .. code-block :: pycon

            >>> par = np.array(0.3)
            >>> qnode(par)
            tensor(0.3522399, requires_grad=True)

        **Postprocessing multiple measurements into a condition**

        The Boolean condition for ``cond`` may consist of arithmetic expressions
        of one or multiple mid-circuit measurements:

        .. code-block:: python3

            def cond_fn(mcms):
                first_term = np.prod(mcms)
                second_term = (2 ** np.arange(len(mcms))) @ mcms
                return (1 - first_term) * (second_term > 3)

            @qml.qnode(dev)
            def qnode(x):
                ...
                mcms = [qml.measure(w) for w in range(4)]
                qml.cond(cond_fn(mcms), qml.RX)(x, wires=4)
                ...
                return qml.expval(qml.Z(1))

        **Passing two quantum functions**

        In the qubit model, single-qubit measurements may result in one of two
        outcomes. Such measurement outcomes may then be used to create
        conditional expressions.

        According to the truth value of the conditional expression passed to
        ``cond``, the transform can apply a quantum function in both the
        ``True`` and ``False`` case:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            def qfunc1(x, wires):
                qml.Hadamard(wires[0])
                qml.RY(x, wires[0])

            def qfunc2(x, wires):
                qml.Hadamard(wires[0])
                qml.RZ(x, wires[0])

            @qml.qnode(dev)
            def qnode1(x):
                qml.Hadamard(0)
                m_0 = qml.measure(0)
                qml.cond(m_0, qfunc1, qfunc2)(x, wires=[1])
                return qml.expval(qml.Z(1))

        .. code-block :: pycon

            >>> par = np.array(0.3)
            >>> qnode1(par)
            tensor(-0.1477601, requires_grad=True)

        The previous QNode is equivalent to using ``cond`` twice, inverting the
        conditional expression in the second case using the ``~`` unary
        operator:

        .. code-block:: python3

            @qml.qnode(dev)
            def qnode2(x):
                qml.Hadamard(0)
                m_0 = qml.measure(0)
                qml.cond(m_0, qfunc1)(x, wires=[1])
                qml.cond(~m_0, qfunc2)(x, wires=[1])
                return qml.expval(qml.Z(1))

        .. code-block :: pycon

            >>> qnode2(par)
            tensor(-0.1477601, requires_grad=True)

        **Quantum functions with different signatures**

        It may be that the two quantum functions passed to ``qml.cond`` have
        different signatures. In such a case, ``lambda`` functions taking no
        arguments can be used with Python closure:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            def qfunc1(x, wire):
                qml.Hadamard(wire)
                qml.RY(x, wire)

            def qfunc2(x, y, z, wire):
                qml.Hadamard(wire)
                qml.Rot(x, y, z, wire)

            @qml.qnode(dev)
            def qnode(a, x, y, z):
                qml.Hadamard(0)
                m_0 = qml.measure(0)
                qml.cond(m_0, lambda: qfunc1(a, wire=1), lambda: qfunc2(x, y, z, wire=1))()
                return qml.expval(qml.Z(1))

        .. code-block :: pycon

            >>> par = np.array(0.3)
            >>> x = np.array(1.2)
            >>> y = np.array(1.1)
            >>> z = np.array(0.3)
            >>> qnode(par, x, y, z)
            tensor(-0.30922805, requires_grad=True)
    """

    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()

        if true_fn is None:
            return ops_loader.cond(condition)

        cond_func = ops_loader.cond(condition)(true_fn)

        # Optional 'elif' branches
        for cond_val, elif_fn in elifs:
            cond_func.else_if(cond_val)(elif_fn)

        # Optional 'else' branch
        if false_fn:
            cond_func.otherwise(false_fn)

        return cond_func

    if not isinstance(condition, MeasurementValue):
        # The condition is not a mid-circuit measurement. This will also work
        # when the condition is a mid-circuit measurement but qml.capture.enabled()
        if true_fn is None:
            return lambda fn: CondCallable(condition, fn)

        return CondCallable(condition, true_fn, false_fn, elifs)

    if true_fn is None:
        raise TypeError(
            "cond missing 1 required positional argument: 'true_fn'.\n"
            "Note that if the conditional includes a mid-circuit measurement, "
            "qml.cond cannot be used as a decorator.\n"
            "Instead, please use the form qml.cond(condition, true_fn, false_fn)."
        )

    if elifs:
        raise ConditionalTransformError(
            "'elif' branches are not supported when not using @qjit and with qml.capture.disabled()\n"
            "if the conditional includes mid-circuit measurements."
        )

    if callable(true_fn):
        # We assume that the callable is an operation or a quantum function
        with_meas_err = (
            "Only quantum functions that contain no measurements can be applied conditionally."
        )

        @wraps(true_fn)
        def wrapper(*args, **kwargs):
            # We assume that the callable is a quantum function

            recorded_ops = [a for a in args if isinstance(a, Operator)] + [
                k for k in kwargs.values() if isinstance(k, Operator)
            ]

            # This will dequeue all operators passed in as arguments to the qfunc that is
            # being conditioned. These are queued incorrectly due to be fully constructed
            # before the wrapper function is called.
            if recorded_ops and QueuingManager.recording():
                for op in recorded_ops:
                    QueuingManager.remove(op)

            # 1. Apply true_fn conditionally
            qscript = qml.tape.make_qscript(true_fn)(*args, **kwargs)

            if qscript.measurements:
                raise ConditionalTransformError(with_meas_err)

            for op in qscript.operations:
                if isinstance(op, MidMeasureMP):
                    raise ConditionalTransformError(with_meas_err)
                Conditional(condition, op)

            if false_fn is not None:
                # 2. Apply false_fn conditionally
                else_qscript = qml.tape.make_qscript(false_fn)(*args, **kwargs)

                if else_qscript.measurements:
                    raise ConditionalTransformError(with_meas_err)

                inverted_condition = ~condition

                for op in else_qscript.operations:
                    if isinstance(op, MidMeasureMP):
                        raise ConditionalTransformError(with_meas_err)
                    Conditional(inverted_condition, op)

    else:
        raise ConditionalTransformError(
            "Only operations and quantum functions with no measurements can be applied conditionally."
        )

    return wrapper


def _aval_mismatch_error(branch_type, branch_index, i, outval, expected_outval):
    raise ValueError(
        f"Mismatch in output abstract values in {branch_type} branch "
        f"#{branch_index} at position {i}: "
        f"{outval} vs {expected_outval}."
    )


def _validate_abstract_values(
    outvals: list, expected_outvals: list, branch_type: str, branch_index: int
) -> None:
    """Ensure the collected abstract values match the expected ones."""
    import jax  # pylint: disable=import-outside-toplevel

    if len(outvals) != len(expected_outvals):
        msg = (
            f"Mismatch in number of output variables in {branch_type} branch"
            f" #{branch_index}: {len(outvals)} vs {len(expected_outvals)} "
            f" for {outvals} and {expected_outvals}"
        )
        if jax.config.jax_dynamic_shapes:
            msg += "\n This may be due to different sized shapes when dynamic shapes are enabled."
        raise ValueError(msg)

    for i, (outval, expected_outval) in enumerate(zip(outvals, expected_outvals)):
        if jax.config.jax_dynamic_shapes:
            # we need to be a bit more manual with the comparison.
            if type(outval) != type(expected_outval):  # pylint: disable=unidiomatic-typecheck
                _aval_mismatch_error(branch_type, branch_index, i, outval, expected_outval)
            if getattr(outval, "dtype", None) != getattr(expected_outval, "dtype", None):
                _aval_mismatch_error(branch_type, branch_index, i, outval, expected_outval)

            shape1 = getattr(outval, "shape", ())
            shape2 = getattr(expected_outval, "shape", ())
            for s1, s2 in zip(shape1, shape2, strict=True):
                if isinstance(s1, jax.core.Var) != isinstance(s2, jax.core.Var):
                    _aval_mismatch_error(branch_type, branch_index, i, outval, expected_outval)
                elif isinstance(s1, int) and s1 != s2:
                    _aval_mismatch_error(branch_type, branch_index, i, outval, expected_outval)

        elif outval != expected_outval:
            _aval_mismatch_error(branch_type, branch_index, i, outval, expected_outval)


def _validate_jaxpr_returns(jaxpr_branches):
    out_avals_true = [out.aval for out in jaxpr_branches[0].outvars]
    for idx, jaxpr_branch in enumerate(jaxpr_branches):

        if idx == 0:
            continue

        if jaxpr_branch is None:
            if out_avals_true:
                raise ValueError(
                    "The false branch must be provided if the true branch returns any variables"
                )
            # this is tested, but coverage does not pick it up
            continue  # pragma: no cover

        out_avals_branch = [out.aval for out in jaxpr_branch.outvars]
        branch_type = "elif" if idx < len(jaxpr_branches) - 1 else "false"
        _validate_abstract_values(out_avals_branch, out_avals_true, branch_type, idx - 1)


@functools.lru_cache
def _get_cond_qfunc_prim():
    """Get the cond primitive for quantum functions."""

    # pylint: disable=import-outside-toplevel
    from pennylane.capture.custom_primitives import NonInterpPrimitive

    cond_prim = NonInterpPrimitive("cond")
    cond_prim.multiple_results = True
    cond_prim.prim_type = "higher_order"
    qml.capture.register_custom_staging_rule(
        cond_prim, lambda params: params["jaxpr_branches"][0].outvars
    )

    @cond_prim.def_abstract_eval
    def _(*_, jaxpr_branches, **__):
        return [out.aval for out in jaxpr_branches[0].outvars]

    @cond_prim.def_impl
    def _(*all_args, jaxpr_branches, consts_slices, args_slice):
        n_branches = len(jaxpr_branches)
        conditions = all_args[:n_branches]
        args = all_args[args_slice]

        # Find predicates that use mid-circuit measurements. We don't check the last
        # condition as that is always `True`.
        mcm_conditions = tuple(
            pred for pred in conditions[:-1] if isinstance(pred, MeasurementValue)
        )
        if len(mcm_conditions) != 0:
            if len(mcm_conditions) != len(conditions) - 1:
                raise ConditionalTransformError(
                    "Cannot use qml.cond with a combination of mid-circuit measurements "
                    "and other classical conditions as predicates."
                )
            conditions = get_mcm_predicates(mcm_conditions)

        for pred, jaxpr, const_slice in zip(conditions, jaxpr_branches, consts_slices):
            consts = all_args[const_slice]
            if jaxpr is None:
                continue
            if isinstance(pred, qml.measurements.MeasurementValue):

                with qml.queuing.AnnotatedQueue() as q:
                    out = qml.capture.eval_jaxpr(jaxpr, consts, *args)
                if len(out) != 0:
                    raise ConditionalTransformError(
                        "Only quantum functions without return values can be applied "
                        "conditionally with mid-circuit measurement predicates."
                    )
                for wrapped_op in q:
                    Conditional(pred, wrapped_op.obj)
            elif pred:
                return qml.capture.eval_jaxpr(jaxpr, consts, *args)

        return ()

    return cond_prim
