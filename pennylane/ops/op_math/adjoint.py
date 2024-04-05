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
from functools import wraps

import pennylane as qml
from pennylane.math import conj, moveaxis, transpose
from pennylane.operation import Observable, Operation, Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import make_qscript
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError

from .symbolicop import SymbolicOp


# pylint: disable=no-member
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
    >>> print(qml.draw(circuit2, expansion_strategy="device")(0.1))
    0: ──RY(-0.10)─┤  <Z>

    The adjoint transforms can also be used to apply the adjoint of
    any quantum function.  In this case, ``adjoint`` accepts a single function and returns
    a function with the same call signature.

    We can create a QNode that applies the ``my_ops`` function followed by its adjoint:

    .. code-block:: python3

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

    >>> workflow(jnp.pi/2, 3, 0)
    array([0.5, 0.5])

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
        Adjoint(S)(wires=[0])

    """
    if active_jit := compiler.active_compiler():
        if lazy is False:
            raise CompileError("Setting lazy=False is not supported with qjit.")
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.adjoint(fn)
    if isinstance(fn, Operator):
        return Adjoint(fn) if lazy else _single_op_eager(fn, update_queue=True)
    if not callable(fn):
        raise ValueError(
            f"The object {fn} of type {type(fn)} is not callable. "
            "This error might occur if you apply adjoint to a list "
            "of operations instead of a function or template."
        )

    @wraps(fn)
    def wrapper(*args, **kwargs):
        qscript = make_qscript(fn)(*args, **kwargs)
        if lazy:
            adjoint_ops = [Adjoint(op) for op in reversed(qscript.operations)]
        else:
            adjoint_ops = [_single_op_eager(op) for op in reversed(qscript.operations)]

        return adjoint_ops[0] if len(adjoint_ops) == 1 else adjoint_ops

    return wrapper


def _single_op_eager(op, update_queue=False):
    if op.has_adjoint:
        adj = op.adjoint()
        if update_queue:
            QueuingManager.remove(op)
            QueuingManager.append(adj)
        return adj
    return Adjoint(op)


# pylint: disable=too-many-public-methods
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
    (X(0), 0.5)
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

        If the base class is an ``Observable`` instead, the ``Adjoint`` will be an ``Observable`` as
        well.

        >>> op = Adjoint(1.0 * qml.X(0))
        >>> isinstance(op, qml.operation.Observable)
        True
        >>> isinstance(op, qml.operation.Operation)
        False
        >>> Adjoint(qml.X(0)) @ qml.Y(1)
        (Adjoint(X(0))) @ Y(1)

    """

    def _flatten(self):
        return (self.base,), tuple()

    @classmethod
    def _unflatten(cls, data, _):
        return cls(data[0])

    # pylint: disable=unused-argument
    def __new__(cls, base=None, id=None):
        """Returns an uninitialized type with the necessary mixins.

        If the ``base`` is an ``Operation``, this will return an instance of ``AdjointOperation``.
        If ``Observable`` but not ``Operation``, it will be ``AdjointObs``.
        And if both, it will be an instance of ``AdjointOpObs``.

        """

        if isinstance(base, Operation):
            if isinstance(base, Observable):
                return object.__new__(AdjointOpObs)

            # not an observable
            return object.__new__(AdjointOperation)

        if isinstance(base, Observable):
            return object.__new__(AdjointObs)

        return object.__new__(Adjoint)

    def __init__(self, base=None, id=None):
        self._name = f"Adjoint({base.name})"
        super().__init__(base, id=id)

    def __repr__(self):
        return f"Adjoint({self.base})"

    @property
    def ndim_params(self):
        return self.base.ndim_params

    def label(self, decimals=None, base_label=None, cache=None):
        base_label = self.base.label(decimals, base_label, cache=cache)
        return f"({base_label})†" if self.base.arithmetic_depth > 0 else f"{base_label}†"

    def matrix(self, wire_order=None):
        if isinstance(self.base, qml.ops.Hamiltonian):
            base_matrix = qml.matrix(self.base, wire_order=wire_order)
        else:
            base_matrix = self.base.matrix(wire_order=wire_order)

        return moveaxis(conj(base_matrix), -2, -1)

    # pylint: disable=arguments-differ
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
        base = self.base.simplify()
        if self.base.has_adjoint:
            return base.adjoint().simplify()
        return Adjoint(base=base.simplify())


# pylint: disable=no-member
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

    # pylint: disable=missing-function-docstring
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


class AdjointObs(Adjoint, Observable):
    """A child of :class:`~.Adjoint` that also inherits from :class:`~.Observable`."""

    def __new__(cls, *_, **__):
        return object.__new__(cls)


# pylint: disable=too-many-ancestors
class AdjointOpObs(AdjointOperation, Observable):
    """A child of :class:`~.AdjointOperation` that also inherits from :class:`~.Observable."""

    def __new__(cls, *_, **__):
        return object.__new__(cls)
