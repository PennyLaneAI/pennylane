# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule defines the symbolic operation that indicates the adjoint of an operator.
"""
from collections.abc import Callable
from functools import lru_cache, partial
from typing import overload

import pennylane as qml
from pennylane.capture.autograph import wraps
from pennylane.compiler import compiler
from pennylane.math import conj, moveaxis, transpose
from pennylane.operation import Operation, Operator
from pennylane.queuing import QueuingManager

from .symbolicop import SymbolicOp


@overload
def adjoint(fn: Operator, lazy: bool = True) -> Operator: ...
@overload
def adjoint(fn: Callable, lazy: bool = True) -> Callable: ...
def adjoint(fn, lazy=True):
    """Create the adjoint of an Operator or a function that applies the adjoint of the provided function.
    :func:`~.qjit` compatible.

    Args:
        fn (function or :class:`~.operation.Operator`): A single operator or a quantum function that
            applies quantum operations.

    Keyword Args:
        lazy=True (bool): If the transform is behaving lazily, all operations are wrapped in a ``Adjoint`` class
            and handled later. If ``lazy=False``, operation-specific adjoint decompositions are first attempted.
            Setting ``lazy=False`` is not supported when used with :func:`~.qjit`.

    Returns:
        (function or :class:`~.operation.Operator`): If an Operator is provided, returns an Operator that is the adjoint.
        If a function is provided, returns a function with the same call signature that returns the Adjoint of the
        provided function.

    .. note::

        The adjoint and inverse are identical for unitary gates, but not in general. For example, quantum channels and
        observables may have different adjoint and inverse operators.

    .. note::

        When used with :func:`~.qjit`, this function only supports the Catalyst compiler.
        See :func:`catalyst.adjoint` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    .. note::

        This function supports a batched operator:

        >>> op = qml.adjoint(qml.RX([1, 2, 3], wires=0))
        >>> qml.matrix(op).shape
        (3, 2, 2)

        But it doesn't support batching of operators:

        >>> op = qml.adjoint([qml.RX(1, wires=0), qml.RX(2, wires=0)])
        Traceback (most recent call last):
            ...
        ValueError: The object [RX(1, wires=[0]), RX(2, wires=[0])] of type <class 'list'> is not callable.
        This error might occur if you apply adjoint to a list of operations instead of a function or template.

    .. seealso:: :class:`~.ops.op_math.Adjoint` and :meth:`.Operator.adjoint`

    **Example**

    The adjoint transform can accept a single operator.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit2(y):
    ...     qml.adjoint(qml.RY(y, wires=0))
    ...     return qml.expval(qml.Z(0))
    >>> print(qml.draw(circuit2)("y"))
    0: ──RY(y)†─┤  <Z>
    >>> print(qml.draw(circuit2, level="device")(0.1))
    0: ──RY(0.10)†─┤  <Z>

    The adjoint transforms can also be used to apply the adjoint of
    any quantum function.  In this case, ``adjoint`` accepts a single function and returns
    a function with the same call signature.

    We can create a QNode that applies the ``my_ops`` function followed by its adjoint:

    .. code-block:: python

        def my_ops(a, wire):
            qml.RX(a, wires=wire)
            qml.SX(wire)

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(a):
            my_ops(a, wire=0)
            qml.adjoint(my_ops)(a, wire=0)
            return qml.expval(qml.Z(0))

    Printing this out, we can see that the inverse quantum
    function has indeed been applied:

    >>> print(qml.draw(circuit)(0.2))
    0: ──RX(0.20)──SX──SX†──RX(0.20)†─┤  <Z>

    **Example with compiler**

    The adjoint used in a compilation context can be applied on control flow.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def workflow(theta, n, wires):
            def func():
                @qml.for_loop(0, n, 1)
                def loop_fn(i):
                    qml.RX(theta, wires=wires)

                loop_fn()
            qml.adjoint(func)()
            return qml.probs()

    >>> import jax.numpy as jnp
    >>> workflow(jnp.pi/2, 3, 0)
    Array([0.5, 0.5], dtype=float64)

    .. warning::

        The Catalyst adjoint function does not support performing the adjoint
        of quantum functions that contain mid-circuit measurements.

    .. details::
        :title: Lazy Evaluation

        When ``lazy=False``, the function first attempts operation-specific decomposition of the
        adjoint via the :meth:`.Operator.adjoint` method. Only if an Operator doesn't have
        an :meth:`.Operator.adjoint` method is the object wrapped with the :class:`~.ops.op_math.Adjoint`
        wrapper class.

        >>> qml.adjoint(qml.Z(0), lazy=False)
        Z(0)
        >>> qml.adjoint(qml.RX, lazy=False)(1.0, wires=0)
        RX(-1.0, wires=[0])
        >>> qml.adjoint(qml.S, lazy=False)(0)
        Adjoint(S(0))

    """
    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.adjoint(fn, lazy=lazy)
    return create_adjoint_op(fn, lazy)


def create_adjoint_op(fn, lazy):
    """Main logic for qml.adjoint, but allows bypassing the compiler dispatch if needed."""
    if qml.math.is_abstract(fn):
        return Adjoint(fn)
    if isinstance(fn, Operator):
        return Adjoint(fn) if lazy else _single_op_eager(fn, update_queue=True)
    if callable(fn):
        if qml.capture.enabled():
            return _capture_adjoint_transform(fn, lazy=lazy)
        return _adjoint_transform(fn, lazy=lazy)
    raise ValueError(
        f"The object {fn} of type {type(fn)} is not callable. "
        "This error might occur if you apply adjoint to a list "
        "of operations instead of a function or template."
    )


@lru_cache  # only create the first time requested
def _get_adjoint_qfunc_prim():
    """See capture/explanations.md : Higher Order primitives for more information on this code."""
    # if capture is enabled, jax should be installed
    # pylint: disable=import-outside-toplevel
    from pennylane.capture.custom_primitives import QmlPrimitive

    adjoint_prim = QmlPrimitive("adjoint_transform")
    adjoint_prim.multiple_results = True
    adjoint_prim.prim_type = "higher_order"

    @adjoint_prim.def_impl
    def _(*args, jaxpr, lazy, n_consts):
        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        consts = args[:n_consts]
        args = args[n_consts:]
        collector = CollectOpsandMeas()
        collector.eval(jaxpr, consts, *args)
        for op in reversed(collector.state["ops"]):
            adjoint(op, lazy=lazy)
        return []

    @adjoint_prim.def_abstract_eval
    def _(*_, **__):
        return []

    return adjoint_prim


def _capture_adjoint_transform(qfunc: Callable, lazy=True) -> Callable:
    """Capture compatible way of performing an adjoint transform."""
    # note that this logic is tested in `tests/capture/test_nested_plxpr.py`
    import jax  # pylint: disable=import-outside-toplevel

    adjoint_prim = _get_adjoint_qfunc_prim()

    @wraps(qfunc)
    def new_qfunc(*args, **kwargs):
        abstracted_axes, abstract_shapes = qml.capture.determine_abstracted_axes(args)
        jaxpr = jax.make_jaxpr(partial(qfunc, **kwargs), abstracted_axes=abstracted_axes)(*args)
        flat_args = jax.tree_util.tree_leaves(args)
        adjoint_prim.bind(
            *jaxpr.consts,
            *abstract_shapes,
            *flat_args,
            jaxpr=jaxpr.jaxpr,
            lazy=lazy,
            n_consts=len(jaxpr.consts),
        )

    return new_qfunc


def _adjoint_transform(qfunc: Callable, lazy=True) -> Callable:
    # default adjoint transform when capture is not enabled.
    @wraps(qfunc)
    def wrapper(*args, **kwargs):
        qscript = qml.tape.make_qscript(qfunc)(*args, **kwargs)

        leaves, _ = qml.pytrees.flatten((args, kwargs), lambda obj: isinstance(obj, Operator))
        _ = [qml.QueuingManager.remove(l) for l in leaves if isinstance(l, Operator)]

        if lazy:
            adjoint_ops = [Adjoint(op) for op in reversed(qscript.operations)]
        else:
            adjoint_ops = [_single_op_eager(op) for op in reversed(qscript.operations)]

        return adjoint_ops[0] if len(adjoint_ops) == 1 else adjoint_ops

    return wrapper


def _single_op_eager(op: Operator, update_queue: bool = False) -> Operator:
    if op.has_adjoint:
        adj = op.adjoint()
        if update_queue:
            QueuingManager.remove(op)
            QueuingManager.append(adj)
        return adj
    return Adjoint(op)


class Adjoint(SymbolicOp):
    """
    The Adjoint of an operator.

    Args:
        base (~.operation.Operator): The operator that is adjointed.

    .. seealso:: :func:`~.adjoint`, :meth:`~.operation.Operator.adjoint`

    This is a *developer*-facing class, and the :func:`~.adjoint` transform should be used to
    construct instances
    of this class.

    **Example**

    >>> op = Adjoint(qml.S(0))
    >>> op.name
    'Adjoint(S)'
    >>> qml.matrix(op)
    array([[1.-0.j, 0.-0.j],
       [0.-0.j, 0.-1.j]])
    >>> qml.generator(Adjoint(qml.RX(1.0, wires=0)))
    (X(0), np.float64(0.5))
    >>> Adjoint(qml.RX(1.234, wires=0)).data
    (1.234,)

    .. details::
        :title: Developer Details

        This class mixes in parent classes based on the inheritance tree of the provided ``Operator``.
        For example, when provided an ``Operation``, the instance will inherit from ``Operation`` and
        the ``AdjointOperation`` mixin.

        >>> op = Adjoint(qml.RX(1.234, wires=0))
        >>> isinstance(op, qml.operation.Operation)
        True
        >>> isinstance(op, AdjointOperation)
        True
        >>> op.grad_method
        'A'

    """

    resource_keys = {"base_class", "base_params"}

    def _flatten(self):
        return (self.base,), tuple()

    @classmethod
    def _unflatten(cls, data, _):
        return cls(data[0])

    def __new__(cls, base=None, id=None):
        """Returns an uninitialized type with the necessary mixins.

        If the ``base`` is an ``Operation``, this will return an instance of ``AdjointOperation``.

        """

        if isinstance(base, Operation):
            # not an observable
            return object.__new__(AdjointOperation)

        return object.__new__(Adjoint)

    def __init__(self, base=None, id=None):
        self._name = f"Adjoint({base.name})"
        super().__init__(base, id=id)
        if self.base.pauli_rep:
            pr = {pw: qml.math.conjugate(coeff) for pw, coeff in self.base.pauli_rep.items()}
            self._pauli_rep = qml.pauli.PauliSentence(pr)
        else:
            self._pauli_rep = None

    def __repr__(self):
        return f"Adjoint({self.base})"

    @property
    def resource_params(self) -> dict:
        return {"base_class": type(self.base), "base_params": self.base.resource_params}

    @property
    def ndim_params(self):
        return self.base.ndim_params

    def label(self, decimals=None, base_label=None, cache=None):
        base_label = self.base.label(decimals, base_label, cache=cache)
        return (
            f"({base_label})†"
            if self.base.arithmetic_depth > 0 and len(base_label) > 1
            else f"{base_label}†"
        )

    def matrix(self, wire_order=None):
        base_matrix = self.base.matrix(wire_order=wire_order)
        return moveaxis(conj(base_matrix), -2, -1)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_sparse_matrix(self) -> bool:
        return self.base.has_sparse_matrix

    def sparse_matrix(self, wire_order=None, format="csr"):
        base_matrix = self.base.sparse_matrix(wire_order=wire_order)
        return transpose(conj(base_matrix)).asformat(format=format)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_decomposition(self):
        return self.base.has_adjoint or self.base.has_decomposition

    def decomposition(self):
        if self.base.has_adjoint:
            return [self.base.adjoint()]
        base_decomp = self.base.decomposition()
        return [Adjoint(op) for op in reversed(base_decomp)]

    def eigvals(self):
        # Cannot define ``compute_eigvals`` because Hermitian only defines ``eigvals``
        return conj(self.base.eigvals())

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_adjoint(self):
        return True

    def adjoint(self):
        return self.base.queue()

    def simplify(self):
        base = self.base if qml.capture.enabled() else self.base.simplify()
        if base.has_adjoint:
            return base.adjoint() if qml.capture.enabled() else base.adjoint().simplify()
        return Adjoint(base=base)


class AdjointOperation(Adjoint, Operation):
    """This mixin class is dynamically added to an ``Adjoint`` instance if the provided base class
    is an ``Operation``.

    .. warning::
        This mixin class should never be initialized independent of ``Adjoint``.

    Overriding the dunder method ``__new__`` in ``Adjoint`` allows us to customize the creation of
    an instance and dynamically add in parent classes.

    .. note:: Once the ``Operation`` class does not contain any unique logic any more, this mixin
    class can be removed.
    """

    def __new__(cls, *_, **__):
        return object.__new__(cls)

    @property
    def name(self):
        return self._name

    @property
    def basis(self):
        return self.base.basis

    @property
    def control_wires(self):
        return self.base.control_wires

    def single_qubit_rot_angles(self):
        omega, theta, phi = self.base.single_qubit_rot_angles()
        return [-phi, -theta, -omega]

    @property
    def grad_method(self):
        return self.base.grad_method

    # pylint: disable=missing-function-docstring
    @property
    def grad_recipe(self):
        return self.base.grad_recipe

    @property
    def parameter_frequencies(self):
        return self.base.parameter_frequencies

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_generator(self):
        return self.base.has_generator

    def generator(self):
        return -1 * self.base.generator()


AdjointOperation._primitive = Adjoint._primitive  # pylint: disable=protected-access
