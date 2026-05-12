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
This module contains the abstract base classes for defining PennyLane
operations and observables.

.. warning::

    Unless you are a PennyLane or plugin developer, you likely do not need
    to use these classes directly.

    See the :doc:`main operations page <../introduction/operations>` for
    details on available operations and observables.

Description
-----------

Qubit Operations
~~~~~~~~~~~~~~~~
The :class:`Operator2` class serves as a base class for operators,
and is inherited by the
:class:`Operation2` class. These classes are subclassed to implement quantum operations
and measure observables in PennyLane.

* Each :class:`~.Operator2` subclass represents a general type of
  map between physical states. Each instance of these subclasses
  represents either

  - an application of the operator or
  - an instruction to measure and return the respective result.

  Operators act on a sequence of wires (subsystems) using given parameter values.

* Each :class:`~.Operation2` subclass represents a type of quantum operation,
  for example a unitary quantum gate. Each instance of these subclasses
  represents an application of the operation with given parameter values to
  a given sequence of wires (subsystems).


Differentiation
^^^^^^^^^^^^^^^

In general, an :class:`Operation2` is differentiable (at least using the finite-difference
method) with respect to a parameter iff

* the domain of that parameter is continuous.

For an :class:`Operation2` to be differentiable with respect to a parameter using the
analytic method of differentiation, it must satisfy an additional constraint:

* the parameter domain must be real.

.. note::

    These conditions are *not* sufficient for analytic differentiation. For example,
    CV gates must also define a matrix representing their Heisenberg linear
    transformation on the quadrature operators.

Contents
--------

.. currentmodule:: pennylane.operation2

Operator Types
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.operation2

.. autosummary::
    :toctree: api

    ~Operator2
    ~Operation2
    ~StatePrepBase2

.. currentmodule:: pennylane.operation2

.. inheritance-diagram:: Operator2 Operation2 StatePrepBase2
    :parts: 1
"""

import warnings
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from enum import Enum, auto
from functools import partial
from inspect import BoundArguments, Signature, signature
from typing import Any, Callable, ClassVar, Hashable, Iterable, Literal

import numpy as np
from scipy.sparse import spmatrix

import pennylane as qp
from pennylane import math
from pennylane.capture import ABCCaptureMeta
from pennylane.exceptions import (
    AdjointUndefinedError,
    DecompositionUndefinedError,
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    GeneratorUndefinedError,
    MatrixUndefinedError,
    PowUndefinedError,
    SparseMatrixUndefinedError,
    TermsUndefinedError,
)
from pennylane.operation import FlatPytree, classproperty, create_operator_primitive
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, QueuingManager
from pennylane.typing import TensorLike, WiresLike
from pennylane.wires import Wires


class Operator2(ABC, metaclass=ABCCaptureMeta):
    r"""Base class representing quantum operators.

    Operators are uniquely defined by their name, the wires they act on, their dynamic
    inputs, and their static inputs. The dynamic inputs can be tensors of any supported
    auto-differentiation framework.

    An operator can define any of the following representations:

    * Representation as a **matrix** (:meth:`.Operator2.matrix`), as specified by a
      global wire order that tells us where the wires are found on a register.

    * Representation as a **sparse matrix** (:meth:`.Operator2.sparse_matrix`). Currently, this
      is a SciPy CSR matrix format.

    * Representation via the **eigenvalue decomposition** specified by eigenvalues
      (:meth:`.Operator2.eigvals`) and diagonalizing gates (:meth:`.Operator2.diagonalizing_gates`).

    * Representation as a **product of operators** (:meth:`.Operator2.decomposition`).

    * Representation as a **linear combination of operators** (:meth:`.Operator2.terms`).

    * Representation by a **generator** via :math:`e^{G}` (:meth:`.Operator2.generator`).

    Each representation method comes with a static method prefixed by ``compute_``, which takes
    the same signature as the operator itself.

    Args:
        *args (tuple[Any]): Positional arguments
        *kwargs (dict[str, Any]): Key-word arguments

    **Example**

    A custom operator can be created by inheriting from :class:`~.Operator2` or one of its
    subclasses.

    The following is an example for a custom gate that inherits from the :class:`~.Operation2`
    subclass. It acts by potentially flipping a qubit and rotating another qubit. The custom
    operator defines a decomposition, which the devices can use (since it is unlikely that a
    device knows a native implementation for ``FlipAndRotate``). It also defines an adjoint
    operator.

    # FIXME: Update example if necessary
    .. code-block:: python

        import pennylane as qp

        class FlipAndRotate(qp.operation2.Operation2):

            # This attribute tells PennyLane what differentiation method to use. Here
            # we request parameter-shift (or "analytic") differentiation.
            grad_method = "A"

            # This attribute tells PennyLane which arguments correspond to wires. If
            # not specified, it will be assumed that the operator only has one argument
            # corresponding to wires, aptly called "wires".
            wire_argnames = ("wire_rot", "wire_flip")

            # This attrubute tells PennyLane which arguments correspond to static inputs.
            # An input is static if its value is guaranteed to be known at compile-time.
            static_argnames = ("do_flip",)

            # This attribute tells PennyLane which arguments correspond to dynamic inputs.
            # An input is dynamic if its value may not be known at compile-time. This attribute
            # can be inferred automatically if ``static_argnames`` and ``wire_argnames`` are
            # defined, but can also be specified explicitly.
            dynamic_argnames = ("angle",)

            # The number of dynamic arguments. This must match ``len(dynamic_argnames)``.
            num_params = 1

            # The number of allowed wires. If an arbitrary number of wires is allowed, use ``None``.
            # This must be specified for each wire argument.
            num_wires = (1, 1)

            # The number of dimensions expected for each dynamic argument. In this example,
            # ``angle`` is expected to be a scalar.
            ndim_params = (0,)

            def __init__(self, angle, wire_rot, wire_flip=None, do_flip=False):

                # checking the inputs --------------

                if do_flip and wire_flip is None:
                    raise ValueError("Expected a wire to flip; got None.")

                if wire_flip is None:
                    wire_flip = ()

                #------------------------------------

                # do_flip is not dynamic but influences the action of the operator,
                # which is why we define it to be a static input

                # The parent class expects to receive the canonicalized arguments.
                super().__init__(angle, wire_rot, wire_flip, do_flip)

            @staticmethod
            def compute_decomposition(angle, wire_rot, wire_flip=None, do_flip=False):
                # Overwriting this method defines the decomposition of the new gate, as it is
                # called by Operator2.decomposition().
                # The signature of this function must match the operator's signature.
                op_list = []
                if do_flip:
                    op_list.append(qp.X(wire_flip))
                op_list.append(qp.RX(angle, wires=wire_rot))
                return op_list

            def adjoint(self):
                # The adjoint operator of this gate simply negates the angle.
                # Additionally, note that for convenience, all arguments of the operator are
                # automatically available as read-only properties.
                return FlipAndRotate(-self.angle, self.wire_rot, self.wire_flip, do_flip=self.do_flip)

    We can use the operation as follows:

    # FIXME: Update example if necessary
    .. code-block:: python

        from pennylane import numpy as np

        dev = qp.device("default.qubit", wires=["q1", "q2", "q3"])

        @qp.qnode(dev)
        def circuit(angle):
            FlipAndRotate(angle, wire_rot="q1", wire_flip="q1")
            return qp.expval(qp.Z("q1"))

    >>> a = np.array(3.14)
    >>> circuit(a)
    tensor(-0.99999873, requires_grad=True)

    .. details::
        :title: Serialization and Pytree format
        :href: serialization

        PennyLane operations are automatically registered as `Pytrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_ .
        See the ``Operator2._flatten`` and ``Operator2._unflatten`` methods for more information.

        # FIXME: Update pytree example if necessary
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> op._flatten()
        ((1.2,), (Wires([0, 1]), (('pauli_word', 'XY'),)))
        >>> qp.PauliRot._unflatten(*op._flatten())
        PauliRot(1.2, XY, wires=[0, 1])

    # TODO: ##################### FIX BELOW THIS POINT #####################
    .. details::
        :title: Parameter broadcasting
        :href: parameter-broadcasting

        Many quantum functions are executed repeatedly at different parameters, which
        can be done with parameter broadcasting. For usage details and examples see the
        :class:`~.pennylane.QNode` documentation.

        In order to support parameter broadcasting with an operator class,
        the following steps are necessary:

        #. Define the class attribute ``ndim_params``, a tuple that indicates
           the expected number of dimensions for each dynamic argument
           *without broadcasting*. For example, ``FlipAndRotate``
           above has ``ndim_params = (0,)`` for a single scalar argument.
           An operator taking a matrix argument and a scalar would have ``ndim_params = (2, 0)``.
           Note that ``ndim_params`` does *not require the size* of the axes.
        #. Make the representations of the operator broadcasting-compatible. Typically, one or
           multiple of the methods ``compute_matrix``, ``compute_eigvals`` and
           ``compute_decomposition`` are defined by an operator, and these need to work with
           the original input and output as well as with broadcasted inputs and outputs
           that have an additional, leading axis. See below for an example.
        #. Make sure that validation within the above representation methods and
           ``__init__``---if it is overwritten by the operator class---allow
           for broadcasted inputs. For custom operators this usually is a minor
           step or not necessary at all.
        #. For proper registration, add the name of the operator to
           :obj:`~.pennylane.ops.qubit.attributes.supports_broadcasting` in the file
           ``pennylane/ops/qubit/attributes.py``.
        #. Make sure that the operator's ``_check_batching`` method is called in all
           places required. This is typically done automatically but needs to be assured.
           See further below for details.

        **Examples**

        Consider an operator with the same matrix as ``qp.RX``. A basic variant of
        ``compute_matrix`` (which will not be compatible with all autodifferentiation
        frameworks or backpropagation) is

        .. code-block:: python

            @staticmethod
            def compute_matrix(theta):
                '''Broadcasting axis ends up in the wrong position.'''
                c = qp.math.cos(theta / 2)
                s = qp.math.sin(theta / 2)
                return qp.math.array([[c, -1j * s], [-1j * s, c]])

        If we passed a broadcasted argument ``theta`` of shape ``(batch_size,)`` to this method,
        which would have one instead of zero dimensions, ``cos`` and ``sin`` would correctly
        be applied elementwise.
        We would also obtain the correct matrix with shape ``(2, 2, batch_size)``.
        However, the broadcasting axis needs to be the *first* axis by convention, so that we need
        to move the broadcasting axis--if it exists--to the front before returning the matrix:

        .. code-block:: python

            @staticmethod
            def compute_matrix(theta):
                '''Broadcasting axis ends up in the correct leading position.'''
                c = qp.math.cos(theta / 2)
                s = qp.math.sin(theta / 2)
                mat = qp.math.array([[c, -1j * s], [-1j * s, c]])
                # Check whether the input has a broadcasting axis
                if qp.math.ndim(theta)==1:
                    # Move the broadcasting axis to the first position
                    return qp.math.moveaxis(mat, 2, 0)
                return mat

        Adapting ``compute_eigvals`` to broadcasting looks similar.

        Usually no major changes are required for ``compute_decomposition``, but we need
        to take care of the correct mapping of input arguments to the operators in the
        decomposition. As an example, consider the operator that represents a layer of
        ``RX`` rotations with individual angles for each rotation. Without broadcasting,
        it takes one onedimensional array, i.e. ``ndim_params=(1,)``.
        Its decomposition, which is a convenient way to support this custom operation
        on all devices that implement ``RX``, might look like this:

        .. code-block:: python

            @staticmethod
            def compute_decomposition(theta, wires):
                '''Iterate over the first axis of theta.'''
                decomp_ops = [qp.RX(x, wires=w) for x, w in zip(theta, wires)]
                return decomp_ops

        If ``theta`` is a broadcasted argument, its first axis is the broadcasting
        axis and we would like to iterate over the *second* axis within the ``for``
        loop instead. This is easily achieved by adding a transposition of ``theta``
        that switches the axes in this case. Conveniently this does not have any
        effect in the non-broadcasted case, so that we do not need to handle two
        cases separately.

        .. code-block:: python

            @staticmethod
            def compute_decomposition(theta, wires):
                '''Iterate over the last axis of theta, which is also the first axis
                or the second axis without and with broadcasting, respectively.'''
                decomp_ops = [qp.RX(x, wires=w) for x, w in zip(qp.math.T(theta), wires)]
                return decomp_ops

        **The ``_check_batching`` method**

        Each operator determines whether it is used with a batch of parameters within
        the ``_check_batching`` method, by comparing the shape of the input data to
        the expected shape. Therefore, it is necessary to call ``_check_batching`` on
        any new input parameters passed to the operator. By default, any class inheriting
        from :class:`~.operation.Operator2` will do so the first time its
        ``batch_size`` property is accessed.

        ``_check_batching`` modifies the following instance attributes:

        - ``_ndim_params``: The number of dimensions of the parameters passed to
          ``_check_batching``. For an ``Operator2`` that does _not_ set the ``ndim_params``
          attribute, ``_ndim_params`` is used as a surrogate, interpreting any parameters
          as "not broadcasted". This attribute should be understood as temporary and likely
          should not be used in other contexts.

        - ``_batch_size``: If the ``Operator2`` is broadcasted: The batch size/size of the
          broadcasting axis. If it is not broadcasted: ``None``. An ``Operator2`` that does
          not support broadcasting will report to not be broadcasted independently of the
          input.

        These two properties are defined lazily, and accessing the public version of either
        one of them (in other words, without the leading underscore) for the first time will
        trigger a call to ``_check_batching``, which validates and sets these properties.
    """

    # pylint: disable=too-many-public-methods, too-many-instance-attributes

    # ------------ Class variables set manually --------------------

    wire_argnames: ClassVar[tuple[str, ...]] = ("wires",)
    """The names of arguments corresponding to wires."""

    dyn_argnames: ClassVar[tuple[str, ...]]
    """The names of arguments corresponding to dynamic arguments. Dynamic arguments
    are those whose concrete values may not be known at compile-time."""

    static_argnames: ClassVar[tuple[str, ...]]
    """The names of arguments corresponding to static arguments. Static arguments
    are those whose concrete values must be known at compile-time."""

    ndim_params: ClassVar[tuple[int, ...]]
    """The number of dimensions expected for each dynamic argument. This must be specified
    for parameter broadcasting support."""

    # ------------ Class variables set automatically ---------------

    _sig: ClassVar[Signature]
    """The signature of the operator."""

    resource_keys: ClassVar[set]
    """The set of parameters that affects the resource requirement of the operator."""

    # ------------ Instance variables set automatically ------------

    _bound_args: BoundArguments
    """BoundArguments mapping arguments names to their values."""

    # ------------------ Initialization ------------------

    # this allows scalar multiplication from left with numpy arrays np.array(0.5) * ps1
    # taken from [stackexchange](https://stackoverflow.com/questions/40694380/forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp/44634634#44634634)
    __array_priority__ = 1000

    _primitive: "jax.extend.core.Primitive" | None = None
    """Optional[jax.extend.core.Primitive]"""

    def __init__(self, *args, **kwargs):
        self._name = type(self).__name__

        # Union[PauliSentence, None]: Representation of the operator as a
        # pauli sentence, if applicable
        self._pauli_rep: qp.pauli.PauliSentence | None = None

        # FIXME: Complete
        self._bound_args = self._sig.bind(*args, **kwargs)
        self._bound_args.apply_defaults()

    def __init_subclass__(cls: type["Operator2"]) -> None:
        # turn has_decomposition into a class property if possible

        # Some operators will overwrite `decomposition` instead of `compute_decomposition`
        # Currently, those are mostly classes from the operator arithmetic module.
        # if class overrides has_decomposition property, we do not want to
        # override it here

        if (
            cls.compute_decomposition != Operator2.compute_decomposition
            or cls.decomposition != Operator2.decomposition
        ) and (cls.has_decomposition == Operator2.has_decomposition):
            cls.has_decomposition = True

        register_pytree(cls, cls._flatten, cls._unflatten)
        cls._primitive = create_operator_primitive(cls)

        cls._sig = signature(cls)
        _add_dynamic_properties(cls)

        cls.resource_keys = set(cls._sig.parameters.keys())

    @classmethod
    def _primitive_bind_call(cls: type["Operator2"], *args, **kwargs) -> None:
        # FIXME:
        return

    def _flatten(self) -> FlatPytree:
        """Serialize the operation into dynamic and static components.

        Returns:
            data, metadata: The dynamic and static components.

        See ``Operator2._unflatten``.

        The dynamic component can be recursive and include other operators.

        The metadata **must** be hashable. If the static data contains a non-hashable component, then this
        method and ``Operator2._unflatten`` should be overridden to provide a hashable version of the static data.

        **Example:**

        # FIXME: Update example
        >>> op = qp.Rot(1.2, 2.3, 3.4, wires=0)
        >>> qp.Rot._unflatten(*op._flatten())
        Rot(1.2, 2.3, 3.4, wires=[0])
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> qp.PauliRot._unflatten(*op._flatten())
        PauliRot(1.2, XY, wires=[0, 1])

        Operators that have trainable components that differ from their ``Operator2.data`` must implement their own
        ``_flatten`` methods.

        >>> op = qp.ctrl(qp.U2(3.4, 4.5, wires="a"), ("b", "c") )
        >>> op._flatten()
        ((U2(3.4, 4.5, wires=['a']),), (Wires(['b', 'c']), (True, True), Wires([]), 'borrowed'))
        """
        dyn_data = []
        hashable_data = []

        for k, v in self.arguments.items():
            if k in (self.dyn_argnames + self.wire_argnames):
                dyn_data.append(v)
                hashable_data.append((k, _DYNARG_MARKER))
            else:
                hashable_data.append((k, v))

        return tuple(dyn_data), tuple(hashable_data)

    @classmethod
    def _unflatten(cls, data: Iterable[Any], metadata: Hashable):
        """Recreate an operation from its serialized format.

        Args:
            data: the dynamic component of the operation
            metadata: the static component of the operation.

        The output of ``Operator2._flatten`` and the class type must be sufficient to reconstruct the original
        operation with ``Operator2._unflatten``.

        **Example:**

        # FIXME: Update example
        >>> op = qp.Rot(1.2, 2.3, 3.4, wires=0)
        >>> op._flatten()
        ((1.2, 2.3, 3.4), (Wires([0]), ()))
        >>> qp.Rot._unflatten(*op._flatten())
        Rot(1.2, 2.3, 3.4, wires=[0])
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> op._flatten()
        ((1.2,), (Wires([0, 1]), (('pauli_word', 'XY'),)))
        >>> op = qp.ctrl(qp.U2(3.4, 4.5, wires="a"), ("b", "c") )
        >>> type(op)._unflatten(*op._flatten())
        Controlled(U2(3.4, 4.5, wires=['a']), control_wires=['b', 'c'])
        """
        args = {}
        dyn_idx = 0

        for k, v in metadata:
            if v is _DYNARG_MARKER:
                args[k] = data[dyn_idx]
                dyn_idx += 1
            else:
                args[k] = v

        return cls(**args)

    # ------------------ General properties ------------------

    @property
    def arguments(self) -> dict[str, Any]:
        """Dictionary mapping argument names to their values."""
        return self._bound_args.arguments

    @property
    def dyn_args(self) -> dict[str, Any]:
        """Dictionary mapping dynamic argument names to their values."""
        return {name: self.arguments[name] for name in self.dyn_argnames}

    @property
    def static_args(self) -> dict[str, Any]:
        """Dictionary mapping static argument names to their values."""
        return {name: self.arguments[name] for name in self.static_argnames}

    @property
    def wire_args(self) -> dict[str, Any]:
        """Dictionary mapping wire argument names to their values."""
        return {name: self.arguments[name] for name in self.wire_argnames}

    @property
    def name(self) -> str:
        """Operator name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Operator name setter."""
        self._name = value

    @property
    def wires(self) -> Wires:
        """Wires that the operator acts on.

        Returns:
            Wires: wires
        """
        return Wires.all_wires([self.arguments[w] for w in self.wire_argnames])

    @property
    def arithmetic_depth(self) -> int:
        """Arithmetic depth of the operator."""
        return 0

    @property
    def is_verified_hermitian(self) -> bool:
        """This property determines if an operator is verified to be Hermitian.

        .. note::

            This property provides a fast, non-exhaustive check used for internal
            optimizations. It relies on quick, provable shortcuts (e.g., operator
            properties) rather than a full, computationally expensive check.

            For a definitive check, use the :func:`pennylane.is_hermitian` function.
            Please note that this comes with increased computational cost.

        Returns:
            bool: The property will return ``True`` if the operator is guaranteed to be Hermitian and
            ``False`` if the check is inconclusive and the operator may or may not be Hermitian.

        Consider this operator,

        >>> op = (qp.X(0) @ qp.Y(0) - qp.X(0) @ qp.Z(0)) * 1j

        In this case, Hermicity cannot be verified and leads to an inconclusive result:

        >>> op.is_verified_hermitian # inconclusive
        False

        However, using :func:`pennylane.is_hermitian` will give the correct answer:

        >>> qp.is_hermitian(op) # definitive
        True

        """
        return False

    @property
    def pauli_rep(self) -> "qp.pauli.PauliSentence" | None:
        """A :class:`~.PauliSentence` representation of the Operator, or ``None`` if it doesn't have one."""
        return self._pauli_rep

    # ------------------ Operator actions ------------------

    def pow(self, z: float) -> list["Operator2"]:
        """A list of new operators equal to this one raised to the given power. This method is used to simplify
        :class:`~.Pow` instances created by :func:`~.pow` or ``op ** power``.

        ``Operator2.pow`` can be optionally defined by Operator developers, while :func:`~.pow` or ``op ** power``
        are the entry point for constructing generic powers to exponents.

        Args:
            z (float): exponent for the operator

        Returns:
            list[:class:`~.operation2.Operator2`]

        >>> class MyClass(qp.operation2.Operator2):
        ...
        ...     def pow(self, z):
        ...         return [MyClass(self.data[0]*z, self.wires)]
        ...
        >>> op = MyClass(0.5, 0) ** 2
        >>> op
        MyClass(0.5, wires=[0])**2
        >>> op.decomposition()
        [MyClass(1.0, wires=[0])]
        >>> op.simplify()
        MyClass(1.0, wires=[0])

        """
        # Child methods may call super().pow(z%period) where op**period = I
        # For example, PauliX**2 = I, SX**4 = I, TShift**3 = I (for qutrit)
        # Hence we define the non-negative integer cases here as a repeated list
        if z == 0:
            return []
        if isinstance(z, int) and z > 0:
            if QueuingManager.recording():
                return [qp.apply(self) for _ in range(z)]
            return [copy.copy(self) for _ in range(z)]
        raise PowUndefinedError

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self

    @property
    def _queue_category(self) -> Literal["_ops", "_measurements", None]:
        """Used for sorting objects into their respective lists in ``QuantumScript`` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumScript._process_queue``.

        Options are:
            * ``"_ops"``
            * ``"_measurements"``
            * ``None`` (deprecated)
        """
        return "_ops"

    # pylint: disable=no-self-argument
    @classproperty
    def has_adjoint(cls) -> bool:
        r"""Bool: Whether or not the Operator can compute its own adjoint.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.adjoint != Operator2.adjoint

    def adjoint(self) -> "Operator2":  # pylint:disable=no-self-use
        """Create an operation that is the adjoint of this one. Used to simplify
        :class:`~.Adjoint` operators constructed by :func:`~.adjoint`.

        Adjointed operations are the conjugated and transposed version of the
        original operation. Adjointed ops are equivalent to the inverted operation for unitary
        gates.

        ``Operator2.adjoint`` can be optionally defined by Operator developers, while :func:`~.adjoint`
        is the entry point for constructing generic adjoint representations.

        Returns:
            The adjointed operation.

        >>> class MyClass(qp.operation2.Operator2):
        ...
        ...     def adjoint(self):
        ...         return self
        ...
        >>> op = qp.adjoint(MyClass(wires=0))
        >>> op
        Adjoint(MyClass(wires=[0]))
        >>> op.decomposition()
        [MyClass(wires=[0])]
        >>> op.simplify()
        MyClass(wires=[0])


        """
        raise AdjointUndefinedError

    def map_wires(self, wire_map: dict[Hashable, Hashable]) -> "Operator2":
        """Returns a copy of the current operator with its wires changed according to the given
        wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .Operator2: new operator
        """
        new_op = copy.copy(self)
        for n, wires in self.wire_args.items():
            new_op._bound_args.arguments[n] = Wires([wire_map.get(w, w) for w in wires])
        if (p_rep := self.pauli_rep) is not None:
            new_op._pauli_rep = p_rep.map_wires(wire_map)
        return new_op

    def simplify(self) -> "Operator2":
        """Reduce the depth of nested operators to the minimum.

        Returns:
            .Operator2: simplified operator
        """
        return self

    # ------------------ Operator representations ------------------
    # pylint: disable=unused-argument,no-self-argument,comparison-with-callable,no-self-use

    @staticmethod
    def compute_matrix(*args, **kwargs) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`.Operator2.matrix` and :func:`qp.matrix() <pennylane.matrix>`

        Args:
            *args (tuple): Positional arguments of the operator
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            tensor_like: matrix representation
        """
        raise MatrixUndefinedError

    @classproperty
    def has_matrix(cls) -> bool:
        r"""Bool: Whether or not the Operator returns a defined matrix.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.compute_matrix != Operator2.compute_matrix or cls.matrix != Operator2.matrix

    def matrix(self, wire_order: WiresLike | None = None) -> TensorLike:
        r"""Representation of the operator as a matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        If the matrix depends on trainable parameters, the result
        will be cast in the same autodifferentiation framework as the parameters.

        A ``MatrixUndefinedError`` is raised if the matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator2.compute_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires

        Returns:
            tensor_like: matrix representation
        """
        canonical_matrix = self.compute_matrix(**self.arguments)

        if (
            wire_order is None
            or self.wires == Wires(wire_order)
            or (
                self.name in qp.ops.qubit.attributes.symmetric_over_all_wires
                and set(self.wires) == set(wire_order)
            )
        ):
            return canonical_matrix

        return math.expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    @staticmethod
    def compute_sparse_matrix(*args, format: str = "csr", **kwargs) -> spmatrix:
        r"""Representation of the operator as a sparse matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Operator2.sparse_matrix`

        Args:
            *args (tuple): Positional arguments of the operator
            format (str): format of the returned scipy sparse matrix, for example 'csr'
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            scipy.sparse._csr.csr_matrix: sparse matrix representation
        """
        raise SparseMatrixUndefinedError

    @classproperty
    def has_sparse_matrix(cls) -> bool:
        r"""Bool: Whether the Operator returns a defined sparse matrix.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return (
            cls.compute_sparse_matrix != Operator2.compute_sparse_matrix
            or cls.sparse_matrix != Operator2.sparse_matrix
        )

    def sparse_matrix(self, wire_order: WiresLike | None = None, format="csr") -> spmatrix:
        r"""Representation of the operator as a sparse matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        A ``SparseMatrixUndefinedError`` is raised if the sparse matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator2.compute_sparse_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires
            format (str): format of the returned scipy sparse matrix, for example 'csr'

        Returns:
            scipy.sparse._csr.csr_matrix: sparse matrix representation

        """
        canonical_sparse_matrix = self.compute_sparse_matrix(**self.arguments, format="csr")
        return math.expand_matrix(
            canonical_sparse_matrix, wires=self.wires, wire_order=wire_order
        ).asformat(format)

    @staticmethod
    def compute_decomposition(*args, **kwargs) -> list["Operator2"]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator2.decomposition`.

        Args:
            *args (tuple): Positional arguments of the operator
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            list[Operator2]: decomposition of the operator
        """
        raise DecompositionUndefinedError

    @property
    def has_decomposition(self) -> bool:
        r"""Bool: Whether or not the Operator returns a defined decomposition."""
        # if compute_decomposition or decomposition overwritten and property
        # not overwritten, set as class property during __init_subclass__
        return any(rule.is_applicable(**self.resource_params) for rule in qp.list_decomps(self))

    def decomposition(self) -> list["Operator2"]:
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n

        A ``DecompositionUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator2.compute_decomposition`.

        Returns:
            list[Operator2]: decomposition of the operator
        """
        if type(self).compute_decomposition != Operator2.compute_decomposition:
            return self.compute_decomposition(**self.arguments)

        for decomp in qp.list_decomps(self):
            if decomp.is_applicable(**self.resource_params):
                with AnnotatedQueue() as q:
                    decomp(**self.arguments)
                if QueuingManager.recording():
                    # no need for copies if we just use queue method
                    _ = [op.queue() for op in q.queue]
                return q.queue

        raise DecompositionUndefinedError

    @staticmethod
    def compute_eigvals(*args, **kwargs) -> TensorLike:
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`Operator2.eigvals() <.eigvals>` and :func:`qp.eigvals() <pennylane.eigvals>`

        Args:
            *args (tuple): Positional arguments of the operator
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            tensor_like: eigenvalues
        """
        raise EigvalsUndefinedError

    def eigvals(self) -> TensorLike:
        r"""Eigenvalues of the operator in the computational basis.

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`, the operator
        can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. note::
            When eigenvalues are not explicitly defined, they are computed automatically from the matrix representation.
            Currently, this computation is *not* differentiable.

        A ``EigvalsUndefinedError`` is raised if the eigenvalues have not been defined and cannot be
        inferred from the matrix representation.

        .. seealso:: :meth:`~.Operator2.compute_eigvals` and :func:`qp.eigvals() <pennylane.eigvals>`

        Returns:
            tensor_like: eigenvalues
        """

        try:
            return self.compute_eigvals(**self.arguments)
        except EigvalsUndefinedError as e:
            # By default, compute the eigenvalues from the matrix representation if one is defined.
            if self.has_matrix:  # pylint: disable=using-constant-test
                return qp.math.linalg.eigvals(self.matrix())
            raise EigvalsUndefinedError from e

    @staticmethod
    def compute_diagonalizing_gates(*args, **kwargs) -> list["Operator2"]:
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Operator2.diagonalizing_gates`.

        Args:
            *args (tuple): Positional arguments of the operator
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            list[.Operator2]: list of diagonalizing gates
        """
        raise DiagGatesUndefinedError

    @classproperty
    def has_diagonalizing_gates(cls) -> bool:
        r"""Bool: Whether or not the Operator returns defined diagonalizing gates.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        # Operators may overwrite `diagonalizing_gates` instead of `compute_diagonalizing_gates`
        # Currently, those are mostly classes from the operator arithmetic module.
        return (
            cls.compute_diagonalizing_gates != Operator2.compute_diagonalizing_gates
            or cls.diagonalizing_gates != Operator2.diagonalizing_gates
        )

    def diagonalizing_gates(self) -> list["Operator2"]:
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator2.compute_diagonalizing_gates`.

        Returns:
            list[.Operator2] or None: a list of operators
        """
        return self.compute_diagonalizing_gates(**self.arguments)

    def terms(self) -> tuple[list[TensorLike], list["Operator2"]]:
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        Returns:
            tuple[list[tensor_like or float], list[.Operator2]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`
        """
        raise TermsUndefinedError

    @classproperty
    def has_generator(cls) -> bool:
        r"""Bool: Whether or not the Operator returns a defined generator.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.generator != Operator2.generator

    def generator(self) -> "Operator2":
        r"""Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U.generator() # doctest: +SKIP
        0.5 * Y(0) + Z(0) @ X(1)

        The generator may also be provided in the form of a dense or sparse Hamiltonian
        (using :class:`.LinearCombination` and :class:`.SparseHamiltonian` respectively).

        """
        raise GeneratorUndefinedError(f"Operation {self.name} does not have a generator")

    # ------------------ General dunder methods ------------------

    def __repr__(self) -> str:
        """Constructor-call-like representation."""
        if self.dyn_argnames:
            params = ", ".join([self.arguments[d] for d in self.dyn_argnames])
            return f"{self.name}({params}, wires={self.wires.tolist()})"
        return f"{self.name}(wires={self.wires.tolist()})"

    def __hash__(self) -> int:
        serialized_dyn = tuple(
            (n, str(id(d) if math.is_abstract(d) else d)) for n, d in self.dyn_args.items()
        )
        serialized_wires = tuple((n, tuple(w)) for n, w in self.wire_args.items())
        serialized_static = tuple((n, str(s)) for n, s in self.static_args.items())
        return hash((self.name, serialized_dyn, serialized_wires, serialized_static))

    def __eq__(self, other) -> bool:
        return qp.equal(self, other)

    @property
    def hash(self):
        """Hash."""
        # TODO: This is an artifact and should be removed if possible
        return hash(self)

    def __copy__(self) -> "Operator2":
        # FIXME: probably needs to be updated
        cls = type(self)
        copied_op = cls.__new__(cls)
        for attr, value in vars(self).items():
            setattr(copied_op, attr, value)

        return copied_op

    def __deepcopy__(self, memo) -> "Operator2":
        # FIXME: probably needs to be updated
        copied_op = object.__new__(type(self))

        # The memo dict maps object ID to object, and is required by
        # the deepcopy function to keep track of objects it has already
        # deep copied.
        memo[id(self)] = copied_op

        for attr, value in vars(self).items():
            setattr(copied_op, attr, deepcopy(value, memo))
        return copied_op

    # ------------------ Arithmetic dunder methods ------------------

    def __add__(self, other: "Operator2" | TensorLike) -> "Operator2":
        """The addition operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, Operator2):
            return qp.sum(self, other, lazy=False)
        if isinstance(other, TensorLike):
            if qp.math.allequal(other, 0):
                return self
            return qp.sum(
                self,
                qp.s_prod(scalar=other, operator=qp.Identity(self.wires), lazy=False),
                lazy=False,
            )
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other: Callable | TensorLike) -> "Operator2":
        """The scalar multiplication between scalars and Operators."""
        if callable(other):
            return qp.pulse.ParametrizedHamiltonian([other], [self])
        if isinstance(other, TensorLike):
            return qp.s_prod(scalar=other, operator=self, lazy=False)
        return NotImplemented

    def __truediv__(self, other: TensorLike):
        """The division between an Operator and a number."""
        if isinstance(other, TensorLike):
            return self.__mul__(1 / other)
        return NotImplemented

    __rmul__ = __mul__

    def __matmul__(self, other: "Operator2") -> "Operator2":
        """The product operation between Operator objects."""
        return qp.prod(self, other, lazy=False) if isinstance(other, Operator2) else NotImplemented

    def __sub__(self, other: "Operator2" | TensorLike) -> "Operator2":
        """The subtraction operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, Operator2):
            return self + qp.s_prod(-1, other, lazy=False)
        if isinstance(other, TensorLike):
            return self + (qp.math.multiply(-1, other))
        return NotImplemented

    def __rsub__(self, other: "Operator2" | TensorLike):
        """The reverse subtraction operation of Operator-Operator objects and Operator-scalar."""
        return -self + other

    def __neg__(self):
        """The negation operation of an Operator object."""
        return qp.s_prod(scalar=-1, operator=self, lazy=False)

    def __pow__(self, other: TensorLike) -> "Operator2":
        r"""The power operation of an Operator object."""
        if isinstance(other, TensorLike):
            return qp.pow(self, z=other)
        return NotImplemented


def _add_dynamic_properties(cls: type[Operator2]) -> None:
    """Create dynamic properties for an operator using its signature."""

    for name in cls._sig.parameters.keys():
        if name not in vars(cls):
            dyn_property = partial(_dynamic_property, name=name)
            cls.__dict__[name] = property(dyn_property)


def _dynamic_property(self: Operator2, name: str) -> Any:
    """Dynamic property for an argument called ``name``."""
    if "_bound_args" in vars(self) and name in self._bound_args.arguments:
        return self._bound_args.arguments[name]

    return object.__getattribute__(self, name)


class _DYNARG_MARKER:
    """Marker class to mark dynamic arguments for Pytree registration."""
