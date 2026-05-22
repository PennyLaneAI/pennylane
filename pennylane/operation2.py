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

from abc import ABC
from collections.abc import Hashable, Iterable
from copy import deepcopy
from functools import partial
from inspect import BoundArguments, Signature, signature
from typing import Any, ClassVar, Literal

import pennylane as qp
from pennylane import math
from pennylane.operation import _UNSET_BATCH_SIZE, FlatPytree
from pennylane.pytrees import flatten, register_pytree
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires


class Operator2(ABC):
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

    # FIXME: ##################### FIX BELOW THIS POINT #####################
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

    dynamic_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments corresponding to dynamic arguments. Dynamic arguments
    are those whose concrete values may not be known at compile-time."""

    static_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments corresponding to static arguments. Static arguments
    are those whose concrete values must be known at compile-time."""

    hybrid_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments which correspond to dynamic data wrapped in static
    structures (known as Pytrees). This feature is opt-in, but is required for cases
    where arrays, operators, and wires are supplied within a collection."""

    compilable_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments which correspond to compilable static operator data.
    This feature is opt-in, but can be useful PauliString arguments and the like."""

    # ------------ Class variables set automatically ---------------

    _sig: ClassVar[Signature]
    """The signature of the operator."""

    # ------------ Instance variables set automatically ------------

    _bound_args: BoundArguments
    """BoundArguments mapping arguments names to their values."""

    # ------------------ Initialization ------------------

    # this allows scalar multiplication from left with numpy arrays np.array(0.5) * ps1
    # taken from [stackexchange](https://stackoverflow.com/questions/40694380/forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp/44634634#44634634)
    __array_priority__ = 1000

    def __init__(self, *args, **kwargs):
        # Union[PauliSentence, None]: Representation of the operator as a
        # pauli sentence, if applicable
        self._pauli_rep: qp.pauli.PauliSentence | None = None

        self._bound_args = self._sig.bind(*args, **kwargs)
        self._bound_args.apply_defaults()

        # Initialize wires:
        #   1. Union of all wire_argnames into _wires
        #   2. Flatten pytree arguments and look for operators
        #   3. Append operator argument wires to _wires
        # pylint: disable=unnecessary-lambda-assignment
        is_wires = lambda v: isinstance(v, Wires)
        is_op = lambda v: isinstance(v, Operator2)
        all_wires = []

        for w in self.wire_argnames:
            leaves, _ = flatten(self._bound_args.arguments[w], is_leaf=is_wires)
            all_wires.extend(leaves)

        for h in self.hybrid_argnames:
            leaves, _ = flatten(self._bound_args.arguments[h], is_leaf=is_op)
            ops = filter(is_op, leaves)
            all_wires.extend(op.wires for op in ops)

        self._wires = Wires.all_wires(all_wires)

        # Broadcasting-related initialization
        self._batch_size: int | None = _UNSET_BATCH_SIZE
        self._ndim_params: tuple[int] = _UNSET_BATCH_SIZE

        self.queue()

    def __init_subclass__(cls: type["Operator2"]) -> None:
        # TODO: [sc-120429] Add processing for overriding has_decomposition

        cls._sig = signature(cls)
        _add_dynamic_properties(cls)
        register_pytree(cls, cls._flatten, cls._unflatten)

        # Argnames setup
        for attr in (
            "dynamic_argnames",
            "wire_argnames",
            "static_argnames",
            "hybrid_argnames",
            "compilable_argnames",
        ):
            if isinstance(v := getattr(cls, attr), str):
                setattr(cls, attr, (v,))

        if cls.static_argnames and cls.compilable_argnames:
            raise TypeError(
                "Operators can only contain 'static_argnames' or 'compilable_argnames', not both."
            )

        # dynamic/wire/static/compilable argnames must be disjoint.
        seen: dict[str, str] = {}
        for group_name in (
            "dynamic_argnames",
            "wire_argnames",
            "static_argnames",
            "compilable_argnames",
        ):
            for name in getattr(cls, group_name):
                if other := seen.get(name):
                    raise TypeError(
                        f"Argument '{name}' appears in both '{other}' and '{group_name}'; "
                        "dynamic, wire, static, and compilable argnames must not overlap."
                    )
                seen[name] = group_name

        # hybrid_argnames may overlap with wire_argnames, but not with the others.
        hybrid = set(cls.hybrid_argnames)
        non_wire = {n for n, g in seen.items() if g != "wire_argnames"}
        if bad := hybrid & non_wire:
            raise TypeError(
                f"hybrid_argnames {bad} overlap with dynamic, static, or "
                "compilable argnames; hybrid_argnames may only overlap with wire_argnames."
            )

        # Every named signature parameter must appear in at least one *_argnames.
        sig_params = set(cls._sig.parameters.keys())
        if unclassified := sig_params - seen.keys() - hybrid:
            raise TypeError(
                f"The following parameters of '{cls.__name__}' are not classified in "
                f"any argnames tuples: {unclassified}."
            )

    def _flatten(self) -> FlatPytree:
        """Serialize the operation into dynamic and static components.

        Returns:
            data, metadata: The dynamic and static components.

        See ``Operator2._unflatten``.

        The dynamic component can be recursive and include other operators.

        The metadata **must** be hashable. If the static data contains a non-hashable component, then this
        method and ``Operator2._unflatten`` should be overridden to provide a hashable version of the static data.

        **Example:**

        >>> op = qp.Rot(1.2, 2.3, 3.4, wires=0)
        >>> qp.Rot._unflatten(*op._flatten())
        Rot(1.2, 2.3, 3.4, wires=[0])
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> qp.PauliRot._unflatten(*op._flatten())
        PauliRot(1.2, XY, wires=[0, 1])
        """
        # Sort dynamic data as dynamic_args, wire_args, hybrid_args
        dyn_data = []

        dyn_data.extend(self._bound_args.arguments[d] for d in self.dynamic_argnames)
        dyn_data.extend(self._bound_args.arguments[w] for w in self.wire_argnames)
        dyn_data.extend(
            self._bound_args.arguments[h]
            for h in self.hybrid_argnames
            if h not in self.wire_argnames
        )

        # Put static/compilable args in hashable_data
        hashable_data = []
        hashable_data.extend(self._bound_args.arguments[s] for s in self.static_argnames)
        hashable_data.extend(self._bound_args.arguments[c] for c in self.compilable_argnames)

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

        >>> op = qp.Rot(1.2, 2.3, 3.4, wires=0)
        >>> op._flatten()
        ((1.2, 2.3, 3.4, Wires([0])), ())
        >>> qp.Rot._unflatten(*op._flatten())
        Rot(1.2, 2.3, 3.4, wires=[0])
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> op._flatten()
        ((1.2, Wires([0, 1])), ('XY',))
        """
        args = {}

        # Process dynamic data
        i = 0
        for n in cls.dynamic_argnames + cls.wire_argnames:
            args[n] = data[i]
            i += 1
        for n in cls.hybrid_argnames:
            if n not in cls.wire_argnames:
                args[n] = data[i]
                i += 1

        # Process static data. The length of metadata should be the same as the length
        # of static_argnames + compilable_argnames. Additionally, only one of them will
        # ever have values inside, so adding them like below is safe.
        for i, n in enumerate(cls.static_argnames + cls.compilable_argnames):
            args[n] = metadata[i]

        return cls(**args)

    def _check_batching(self):
        """Check if the expected numbers of dimensions of parameters coincides with the
        ones received and sets the ``_batch_size`` attribute.

        The check always passes and sets the ``_batch_size`` to ``None`` for the default
        ``Operator.ndim_params`` property but subclasses may overwrite it to define fixed
        expected numbers of dimensions, allowing to infer a batch size.
        """
        self._batch_size = None
        dynamic_args = tuple(self.dynamic_args.values())

        try:
            ndims = tuple(qp.math.ndim(arg) for arg in dynamic_args)
        except (
            ValueError
        ) as e:  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            # When using tf.function with an input_signature that contains
            # an unknown-shaped input, ndim() will not be able to determine the number of
            # dimensions because they are not specified yet. Failing example: Let `fun` be
            # a single-parameter QNode.
            # `tf.function(fun, input_signature=(tf.TensorSpec(shape=None, dtype=tf.float32),))`
            # There might be a way to support batching nonetheless, which remains to be
            # investigated. For now, the batch_size is left to be `None` when instantiating
            # an operation with abstract parameters that make `qp.math.ndim` fail.
            if any(math.is_abstract(p) for p in dynamic_args):
                self._batch_size = None
                self._ndim_params = (0,) * len(dynamic_args)
                return
            raise e  # pragma: no cover

        if any(
            len(qp.math.shape(arg)) >= 1 and qp.math.shape(arg)[0] is None for arg in dynamic_args
        ):
            # if the batch dimension is unknown, then skip the validation
            # this happens when a tensor with a partially known shape is passed, e.g. (None, 12),
            # typically during compilation of a function decorated with jax.jit or tf.function
            return

        self._ndim_params = ndims
        if ndims != self.ndim_params:
            ndims_matches = [
                (ndim == exp_ndim, ndim == exp_ndim + 1)
                for ndim, exp_ndim in zip(ndims, self.ndim_params, strict=True)
            ]
            if not all(correct or batched for correct, batched in ndims_matches):
                raise ValueError(
                    f"{self.name}: wrong number(s) of dimensions in parameters. "
                    f"Parameters with ndims {ndims} passed, {self.ndim_params} expected."
                )

            first_dims = [
                qp.math.shape(arg)[0]
                for (_, batched), arg in zip(ndims_matches, dynamic_args, strict=True)
                if batched
            ]
            if not qp.math.allclose(first_dims, first_dims[0]):
                raise ValueError(
                    "Broadcasting was attempted but the broadcasted dimensions "
                    f"do not match: {first_dims}."
                )

            self._batch_size = first_dims[0]

    # ------------------ Public properties ------------------

    @property
    def arguments(self) -> dict[str, Any]:
        """Dictionary mapping argument names to their values."""
        return self._bound_args.arguments

    @property
    def dynamic_args(self) -> dict[str, Any]:
        """Dictionary mapping dynamic argument names to their values."""
        return {name: self.arguments[name] for name in self.dynamic_argnames}

    @property
    def static_args(self) -> dict[str, Any]:
        """Dictionary mapping static argument names to their values."""
        return {name: self.arguments[name] for name in self.static_argnames}

    @property
    def wire_args(self) -> dict[str, Any]:
        """Dictionary mapping wire argument names to their values."""
        return {name: self.arguments[name] for name in self.wire_argnames}

    @property
    def hybrid_args(self) -> dict[str, Any]:
        """Dictionary mapping hybrid argument names to their values."""
        return {name: self.arguments[name] for name in self.hybrid_argnames}

    @property
    def compilable_args(self) -> dict[str, Any]:
        """Dictionary mapping compilable argument names to their values."""
        return {name: self.arguments[name] for name in self.compilable_argnames}

    @property
    def name(self) -> str:
        """Operator name."""
        return self.__class__.__name__

    @property
    def wires(self) -> Wires:
        """Wires that the operator acts on.

        Returns:
            Wires: wires
        """
        return self._wires

    @property
    def batch_size(self) -> int | None:
        """Batch size of the operator if it is used with broadcasted parameters.

        The ``batch_size`` is determined based on ``ndim_params`` and the provided parameters
        for the operator. If (some of) the latter have an additional dimension, and this
        dimension has the same size for all parameters, its size is the batch size of the
        operator. If no parameter has an additional dimension, the batch size is ``None``.

        Returns:
            int or None: Size of the parameter broadcasting dimension if present, else ``None``.
        """
        if self._batch_size is _UNSET_BATCH_SIZE:
            self._check_batching()
        return self._batch_size

    @property
    def ndim_params(self) -> tuple[int]:
        """Number of dimensions per trainable parameter of the operator.

        By default, this property returns the numbers of dimensions of the parameters used
        for the operator creation. If the parameter sizes for an operator subclass are fixed,
        this property can be overwritten to return the fixed value.

        Returns:
            tuple: Number of dimensions for each trainable parameter.
        """
        if self._batch_size is _UNSET_BATCH_SIZE:
            self._check_batching()
        return self._ndim_params

    @property
    def pauli_rep(self) -> qp.pauli.PauliSentence | None:
        """A :class:`~.PauliSentence` representation of the Operator, or ``None`` if it doesn't have one."""
        return self._pauli_rep

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self  # so pre-constructed Observable instances can be queued and returned in a single statement

    @property
    def _queue_category(self) -> Literal["_ops", "_measurements"]:
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Options are:
            * `"_ops"`
            * `"_measurements"`
        """
        return "_ops"

    # ------------------ General dunder methods ------------------

    def __repr__(self) -> str:
        """Constructor-call-like representation."""
        if self.dynamic_argnames:
            params = ", ".join([repr(self.arguments[d]) for d in self.dynamic_argnames])
            return f"{self.name}({params}, wires={self.wires.tolist()})"
        return f"{self.name}(wires={self.wires.tolist()})"

    # TODO: [sc-120431] Implement __hash__ and __eq__ after qp.equal supports `Operator2`
    # def __hash__(self) -> int:
    #     serialized_dyn = tuple(
    #         (n, str(id(d) if math.is_abstract(d) else d)) for n, d in self.dynamic_args.items()
    #     )
    #     serialized_wires = tuple((n, tuple(w)) for n, w in self.wire_args.items())
    #     serialized_static = tuple((n, str(s)) for n, s in self.static_args.items())
    #     return hash((self.name, serialized_dyn, serialized_wires, serialized_static))

    # def __eq__(self, other) -> bool:
    #     return qp.equal(self, other)

    def __copy__(self) -> "Operator2":
        cls = type(self)
        copied_op = cls.__new__(cls)
        for attr, value in vars(self).items():
            setattr(copied_op, attr, value)
        return copied_op

    def __deepcopy__(self, memo) -> "Operator2":
        cls = type(self)
        copied_op = cls.__new__(cls)

        # The memo dict maps object ID to object, and is required by
        # the deepcopy function to keep track of objects it has already
        # deep copied.
        memo[id(self)] = copied_op

        for attr, value in vars(self).items():
            setattr(copied_op, attr, deepcopy(value, memo))
        return copied_op


def _add_dynamic_properties(cls: type[Operator2]) -> None:
    """Create dynamic properties for an operator using its signature."""
    # pylint: disable=protected-access
    for name in cls._sig.parameters.keys():
        if name not in vars(cls):
            dyn_property = partial(_dynamic_property, name=name)
            setattr(cls, name, property(dyn_property))


def _dynamic_property(self: Operator2, name: str) -> Any:
    """Dynamic property for an argument called ``name``."""
    # pylint: disable=protected-access
    if "_bound_args" in vars(self) and name in self._bound_args.arguments:
        return self._bound_args.arguments[name]

    return object.__getattribute__(self, name)
