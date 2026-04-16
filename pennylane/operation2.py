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
# pylint: disable=protected-access
r"""
This module contains the abstract base classes for defining PennyLane
operations and observables.
"""

# pylint: disable=access-member-before-definition
import abc
import copy
import warnings
from collections.abc import Callable, Hashable, Iterable
from inspect import BoundArguments, Signature, signature
from typing import Any, ClassVar, Literal, Optional, Union

import numpy as np
from scipy.sparse import spmatrix

import pennylane as qml
from pennylane import capture, math
from pennylane.exceptions import (
    AdjointUndefinedError,
    DecompositionUndefinedError,
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    GeneratorUndefinedError,
    MatrixUndefinedError,
    ParameterFrequenciesUndefinedError,
    PowUndefinedError,
    SparseMatrixUndefinedError,
    TermsUndefinedError,
)
from pennylane.operation import _get_abstract_operator, classproperty
from pennylane.pytrees import register_pytree
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

has_jax = True
try:
    import jax

except ImportError:
    has_jax = False


# =============================================================================
# Capture operators infrastructure
# =============================================================================


def create_operator_primitive(
    operator_type: type["Operator2"],
) -> Optional["jax.extend.core.Primitive"]:
    """Create a primitive corresponding to an operator type."""
    if not has_jax:
        return None

    primitive = capture.QmlPrimitive(operator_type.__name__)
    primitive.prim_type = "operator2"

    @primitive.def_impl
    def _impl(*args, dyn_argnames, wire_argnames, wire_lengths, **kwargs):
        args_dict = {**kwargs}

        cur_idx = 0
        for d in dyn_argnames:
            args_dict[d] = args[cur_idx]
            cur_idx += 1

        for i, w in enumerate(wire_argnames):
            cur_slice = slice(cur_idx, wire_lengths[i], 1)
            args_dict[w] = tuple(
                wire if math.is_abstract(wire) else int(wire) for wire in args[cur_slice]
            )
            cur_idx += wire_lengths[i]

        return type.__call__(operator_type, **args_dict)

    abstract_type = _get_abstract_operator()

    @primitive.def_abstract_eval
    def _abstract_eval(*_, **__):
        return abstract_type()

    return primitive


# =============================================================================
# Base Operator class
# =============================================================================


FlatPytree = tuple[Iterable[Any], Hashable]


class Operator2(abc.ABC, metaclass=capture.ABCCaptureMeta):
    r"""Base class representing quantum operators."""

    # pylint: disable=too-many-public-methods, too-many-instance-attributes

    __array_priority__ = 1000
    _primitive: Optional["jax.extend.core.Primitive"] = None
    _name: ClassVar[str]
    _batch_size = None
    batch_size = None

    # NOTE: These should all be defined as ClassVars
    num_wires: ClassVar[int | None] = None
    num_params: ClassVar[int] = 0
    ndim_params: ClassVar[tuple] = ()

    # Will be defined by the instance lazily
    _parameters: tuple | None = None

    # NOTE: new, will be set automatically; must not be manually overriden
    dyn_argnames: ClassVar[tuple[str, ...]]

    # NOTE: new
    wire_argnames: ClassVar[tuple[str, ...]] = ("wires",)

    # NOTE: new
    static_argnames: ClassVar[tuple[str, ...]] = ()

    # NOTE: new
    _sig: ClassVar[Signature]

    # NOTE: new
    _bound_args: BoundArguments

    def __init__(self, *args, **kwargs):
        self._pauli_rep: qml.pauli.PauliSentence | None = (
            None  # Union[PauliSentence, None]: Representation of the operator as a pauli sentence, if applicable
        )
        self._bound_args = self._sig.bind(*args, **kwargs)
        self._bound_args.apply_defaults()

        for n in self.wire_argnames:
            self._bound_args.arguments[n] = Wires(self._bound_args.arguments[n])

        self.queue()

    def __init_subclass__(cls, **_):
        register_pytree(cls, cls._flatten, cls._unflatten)
        cls._primitive = create_operator_primitive(cls)
        cls._sig = signature(cls)
        cls._name = cls.__name__

        param_names = cls._sig.parameters.keys()
        def_argnames = cls.static_argnames + cls.wire_argnames
        dyn_argnames = []

        for p in param_names:
            if p not in def_argnames:
                dyn_argnames.append(p)

        cls.dyn_argnames = tuple(dyn_argnames)

    @classproperty
    def resource_keys(cls):
        return set(cls._sig.parameters.keys())

    @property
    def resource_params(self):
        from pennylane.templates.core import AbstractArray

        resource_params = {s: self._bound_args.arguments[s] for s in self.static_argnames}
        if self.num_wires is not None and self.wire_argnames == ("wires",):
            resource_params["wires"] = AbstractArray((self.num_wires,), int)
        else:
            for w in self.wire_argnames:
                resource_params[w] = AbstractArray((len(self._bound_args.arguments[w]),), int)
        for d in self.dyn_argnames:
            d_arg = self._bound_args.arguments[d]
            resource_params[d] = AbstractArray(math.shape(d_arg), math.get_dtype_name(d_arg))

        return resource_params

    @classproperty
    def op_type(cls):
        return cls

    params = resource_params

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        """This class method should match the call signature of the class itself.

        When plxpr is enabled, this method is used to bind the arguments and keyword arguments
        to the primitive via ``cls._primitive.bind``.

        """
        if cls._primitive is None:
            # guard against this being called when primitive is not defined.
            return type.__call__(cls, *args, **kwargs)

        array_types = (jax.numpy.ndarray, np.ndarray)
        iterable_wires_types = (
            list,
            tuple,
            qml.wires.Wires,
            range,
            qml.capture.autograph.ag_primitives.PRange,
            set,
            *array_types,
        )

        bound_args = cls._sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        args_dict = bound_args.arguments

        dyn_args = []
        for d in cls.dyn_argnames:
            dyn_args.append(args_dict[d])

        all_wires = []
        wire_lengths = []
        for w in cls.wire_argnames:
            cur_wires = args_dict[w]
            if isinstance(cur_wires, array_types) and cur_wires.shape == ():
                all_wires.append(cur_wires)
                wire_lengths.append(1)
            elif isinstance(cur_wires, iterable_wires_types):
                all_wires += list(cur_wires)
                wire_lengths.append(len(cur_wires))
            else:
                all_wires.append(cur_wires)
                wire_lengths.append(1)

        static_args = {}
        for s in cls.static_argnames:
            static_args[s] = args_dict[s]

        prim_args = (*dyn_args, *all_wires)
        prim_kwargs = {
            **static_args,
            "wire_lengths": tuple(wire_lengths),
            "dyn_argnames": cls.dyn_argnames,
            "wire_argnames": cls.wire_argnames,
        }
        return cls._primitive.bind(*prim_args, **prim_kwargs)

    @property
    def hash(self) -> int:
        """int: Integer hash that uniquely represents the operator."""
        return hash((str(self.name), self._sig, tuple(self._bound_args.arguments.values())))

    def __eq__(self, other) -> bool:
        if type(other) != type(self):
            return False

        return all(
            other._bound_args.arguments[k] == v for k, v in self._bound_args.arguments.items()
        )

    def __hash__(self) -> int:
        return self.hash

    @property
    def name(self) -> str:
        """String for the name of the operator."""
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def pauli_rep(self) -> Optional["qml.pauli.PauliSentence"]:
        """A :class:`~.PauliSentence` representation of the Operator, or ``None`` if it doesn't have one."""
        return self._pauli_rep

    @property
    def is_verified_hermitian(self) -> bool:
        """This property determines if an operator is verified to be Hermitian."""
        return False

    @staticmethod
    def compute_matrix(*args, **kwargs) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        raise MatrixUndefinedError

    # pylint: disable=no-self-argument, comparison-with-callable
    @classproperty
    def has_matrix(cls) -> bool:
        r"""Bool: Whether or not the Operator returns a defined matrix."""
        return cls.compute_matrix != Operator2.compute_matrix or cls.matrix != Operator2.matrix

    def matrix(self, wire_order: WiresLike | None = None) -> TensorLike:
        r"""Representation of the operator as a matrix in the computational basis."""
        canonical_matrix = self.compute_matrix(**self._bound_args.arguments)

        if (
            wire_order is None
            or self.wires == Wires(wire_order)
            or (
                self.name in qml.ops.qubit.attributes.symmetric_over_all_wires
                and set(self.wires) == set(wire_order)
            )
        ):
            return canonical_matrix

        return math.expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    @staticmethod
    def compute_sparse_matrix(*args, format: str = "csr", **kwargs) -> spmatrix:
        r"""Representation of the operator as a sparse matrix in the computational basis (static method)."""
        raise SparseMatrixUndefinedError

    # pylint: disable=no-self-argument, comparison-with-callable
    @classproperty
    def has_sparse_matrix(cls) -> bool:
        r"""Bool: Whether the Operator returns a defined sparse matrix."""
        return (
            cls.compute_sparse_matrix != Operator2.compute_sparse_matrix
            or cls.sparse_matrix != Operator2.sparse_matrix
        )

    def sparse_matrix(self, wire_order: WiresLike | None = None, format="csr") -> spmatrix:
        r"""Representation of the operator as a sparse matrix in the computational basis."""
        canonical_sparse_matrix = self.compute_sparse_matrix(
            **self._bound_args.arguments, format="csr"
        )

        return math.expand_matrix(
            canonical_sparse_matrix, wires=self.wires, wire_order=wire_order
        ).asformat(format)

    @staticmethod
    def compute_decomposition(*args, **kwargs) -> list["Operator2"]:
        r"""Representation of the operator as a product of other operators (static method)."""
        raise DecompositionUndefinedError

    @classproperty
    def has_decomposition(cls) -> bool:
        r"""Bool: Whether or not the Operator returns a defined decomposition."""
        return (
            cls.compute_decomposition != Operator2.compute_decomposition
            or cls.decomposition != Operator2.decomposition
        )

    def decomposition(self) -> list["Operator2"]:
        r"""Representation of the operator as a product of other operators."""
        return self.compute_decomposition(**self._bound_args.arguments)

    @staticmethod
    def compute_diagonalizing_gates(*args, **kwargs) -> list["Operator2"]:
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method)."""
        raise DiagGatesUndefinedError

    # pylint: disable=no-self-argument, comparison-with-callable
    @classproperty
    def has_diagonalizing_gates(cls) -> bool:
        r"""Bool: Whether or not the Operator returns defined diagonalizing gates."""
        return (
            cls.compute_diagonalizing_gates != Operator2.compute_diagonalizing_gates
            or cls.diagonalizing_gates != Operator2.diagonalizing_gates
        )

    def diagonalizing_gates(self) -> list["Operator2"]:  # pylint:disable=no-self-use
        r"""Sequence of gates that diagonalize the operator in the computational basis."""
        return self.compute_diagonalizing_gates(**self._bound_args.arguments)

    # pylint: disable=no-self-argument
    @classproperty
    def has_generator(cls) -> bool:
        r"""Bool: Whether or not the Operator returns a defined generator."""
        return cls.generator != Operator2.generator

    def generator(self) -> "Operator2":  # pylint: disable=no-self-use
        r"""Generator of an operator that is in single-parameter-form."""
        raise GeneratorUndefinedError(f"Operation {self.name} does not have a generator")

    @staticmethod
    def compute_eigvals(*args, **kwargs) -> TensorLike:
        r"""Eigenvalues of the operator in the computational basis (static method)."""
        raise EigvalsUndefinedError

    def eigvals(self) -> TensorLike:
        r"""Eigenvalues of the operator in the computational basis."""
        try:
            return self.compute_eigvals(**self._bound_args.arguments)
        except EigvalsUndefinedError as e:
            # By default, compute the eigenvalues from the matrix representation if one is defined.
            if self.has_matrix:  # pylint: disable=using-constant-test
                return math.linalg.eigvals(self.matrix())
            raise EigvalsUndefinedError from e

    def terms(self) -> tuple[list[TensorLike], list["Operator2"]]:  # pylint: disable=no-self-use
        r"""Representation of the operator as a linear combination of other operators."""
        raise TermsUndefinedError

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        r"""A customizable string representation of the operator."""
        op_label = base_label or self.__class__.__name__

        # The only argument is `self`
        if len(self._bound_args.arguments) == 0:
            return op_label

        def _format(x):
            """Format a scalar parameter or retrieve/store a matrix-valued parameter
            from/to cache, formatting its position in the cache as parameter string."""
            if len(math.shape(x)) == 0:
                # Scalar case
                if decimals is None:
                    return ""
                try:
                    return format(math.toarray(x), f".{decimals}f")
                except ValueError:
                    # If the parameter can't be displayed as a float
                    return format(x)

            if cache is None or not isinstance(mat_cache := cache.get("matrices", None), list):
                # No caching; matrices are not printed out fully, so no printing of this parameter
                return ""

            # Retrieve matrix location in cache, or write the matrix to cache as new entry
            for i, mat in enumerate(mat_cache):
                if math.shape(x) == math.shape(mat) and math.allclose(x, mat):
                    return f"M{i}"
            mat_num = len(mat_cache)
            mat_cache.append(x)
            return f"M{mat_num}"

        # Format each parameter individually, excluding those that lead to empty strings
        param_strings = [
            out
            for n, p in self._bound_args.arguments.items()
            if n in self.dyn_argnames and (out := _format(p)) != ""
        ]
        inner_string = ",\n".join(param_strings)
        if inner_string == "":
            return f"{op_label}"
        return f"{op_label}\n({inner_string})"

    def __repr__(self) -> str:
        """Constructor-call-like representation."""
        if self._bound_args.arguments:
            params = ", ".join([repr(self._bound_args.arguments[d]) for d in self.dyn_argnames])
            return f"{self.name}({params}, wires={self.wires.tolist()})"
        return f"{self.name}(wires={self.wires.tolist()})"

    @property
    def wires(self) -> Wires:
        """Wires that the operator acts on."""
        return Wires.all_wires(self._bound_args.arguments[w] for w in self.wire_argnames)

    @property
    def parameters(self) -> list[TensorLike]:
        """Trainable parameters that the operator depends on."""
        if self._parameters is None:
            params = []
            for n in self.dyn_argnames:
                p = self._bound_args.arguments[n]
                if math.ndim(p) == 0 and "float" in math.get_dtype_name(p):
                    params.append(p)
            self._parameters = tuple(params)

        return self._parameters

    @property
    def data(self):
        return self.parameters

    def pow(self, z: float) -> list["Operator2"]:
        """A list of new operators equal to this one raised to the given power. This method is used to simplify
        :class:`~.Pow` instances created by :func:`~.pow` or ``op ** power``.
        """
        # Child methods may call super().pow(z%period) where op**period = I
        # For example, PauliX**2 = I, SX**4 = I, TShift**3 = I (for qutrit)
        # Hence we define the non-negative integer cases here as a repeated list
        if z == 0:
            return []
        if isinstance(z, int) and z > 0:
            if QueuingManager.recording():
                return [qml.apply(self) for _ in range(z)]
            return [copy.copy(self) for _ in range(z)]
        raise PowUndefinedError

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self  # so pre-constructed Observable instances can be queued and returned in a single statement

    @property
    def _queue_category(self) -> Literal["_ops", "_measurements", None]:
        """Used for sorting objects into their respective lists in `QuantumTape` objects."""
        return "_ops"

    # pylint: disable=no-self-argument
    @classproperty
    def has_adjoint(cls) -> bool:
        r"""Bool: Whether or not the Operator can compute its own adjoint."""
        return cls.adjoint != Operator2.adjoint

    def adjoint(self) -> "Operator2":  # pylint:disable=no-self-use
        """Create an operation that is the adjoint of this one. Used to simplify
        :class:`~.Adjoint` operators constructed by :func:`~.adjoint`.
        """
        raise AdjointUndefinedError

    @property
    def arithmetic_depth(self) -> int:
        """Arithmetic depth of the operator."""
        return 0

    def map_wires(self, wire_map: dict[Hashable, Hashable]) -> "Operator2":
        """Returns a copy of the current operator with its wires changed according to the given
        wire map.
        """
        new_op = copy.copy(self)
        for n in self.wire_argnames:
            new_op._bound_args.arguments[n] = Wires(
                [wire_map.get(w, w) for w in self._bound_args.arguments[n]]
            )
        if (p_rep := self.pauli_rep) is not None:
            new_op._pauli_rep = p_rep.map_wires(wire_map)
        return new_op

    def simplify(self) -> "Operator2":
        """Reduce the depth of nested operators to the minimum."""
        return self

    def __add__(self, other: Union["Operator2", TensorLike]) -> "Operator2":
        """The addition operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, Operator2):
            return qml.sum(self, other, lazy=False)
        if isinstance(other, TensorLike):
            if math.allequal(other, 0):
                return self
            return qml.sum(
                self,
                qml.s_prod(scalar=other, operator=qml.Identity(self.wires), lazy=False),
                lazy=False,
            )
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other: Callable | TensorLike) -> "Operator2":
        """The scalar multiplication between scalars and Operators."""
        if isinstance(other, TensorLike):
            return qml.s_prod(scalar=other, operator=self, lazy=False)
        return NotImplemented

    def __truediv__(self, other: TensorLike):
        """The division between an Operator and a number."""
        if isinstance(other, TensorLike):
            return self.__mul__(1 / other)
        return NotImplemented

    __rmul__ = __mul__

    def __matmul__(self, other: "Operator2") -> "Operator2":
        """The product operation between Operator objects."""
        return qml.prod(self, other, lazy=False) if isinstance(other, Operator2) else NotImplemented

    def __sub__(self, other: Union["Operator2", TensorLike]) -> "Operator2":
        """The subtraction operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, Operator2):
            return self + qml.s_prod(-1, other, lazy=False)
        if isinstance(other, TensorLike):
            return self + (math.multiply(-1, other))
        return NotImplemented

    def __rsub__(self, other: Union["Operator2", TensorLike]):
        """The reverse subtraction operation of Operator-Operator objects and Operator-scalar."""
        return -self + other

    def __neg__(self):
        """The negation operation of an Operator object."""
        return qml.s_prod(scalar=-1, operator=self, lazy=False)

    def __pow__(self, other: TensorLike) -> "Operator2":
        r"""The power operation of an Operator2 object."""
        if isinstance(other, TensorLike):
            return qml.pow(self, z=other)
        return NotImplemented

    def _flatten(self) -> FlatPytree:
        """Serialize the operation into trainable and non-trainable components."""
        dyn_data = []
        hashable_data = []
        for k, v in self._bound_args.arguments.items():
            if k in self.dyn_argnames:
                dyn_data.append(v)
                hashable_data.append((k, None))
            else:
                hashable_data.append(k, v)

        return tuple(dyn_data), tuple(hashable_data)

    @classmethod
    def _unflatten(cls, data: Iterable[Any], metadata: Hashable):
        """Recreate an operation from its serialized format."""
        args = {}
        dyn_idx = 0

        for k, v in metadata:
            if v is None:
                args[k] = data[dyn_idx]
                dyn_idx += 1
            else:
                args[k] = v

        return cls(**args)

    def __getattr__(self, name):
        if "_bound_args" in vars(self) and name in self._bound_args.arguments:
            return self._bound_args.arguments[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if "_bound_args" in vars(self) and name in self._bound_args.arguments:
            self._bound_args.arguments[name] = value
        else:
            object.__setattr__(self, name, value)


# =============================================================================
# Base Operation class
# =============================================================================


class Operation2(Operator2):
    r"""Base class representing quantum gates or channels applied to quantum states."""

    grad_recipe = None
    r"""tuple(Union(list[list[float]], None)) or None: Gradient recipe for the
        parameter-shift method."""

    @property
    def grad_method(self) -> Literal["A", "F", None]:
        """Gradient computation method."""
        if self.num_params == 0:
            return None
        if self.grad_recipe != [None] * self.num_params:
            return "A"
        try:
            self.parameter_frequencies  # pylint:disable=pointless-statement
            return "A"
        except ParameterFrequenciesUndefinedError:
            return "F"

    # Attributes for compilation transforms
    @property
    def basis(self) -> Literal["X", "Y", "Z", None]:
        """str or None: The basis of an operation, or for controlled gates, of the
        target operation. If not ``None``, should take a value of ``"X"``, ``"Y"``,
        or ``"Z"``.
        """
        return None

    @property
    def control_wires(self) -> Wires:  # pragma: no cover
        r"""Control wires of the operator."""
        return Wires([])

    def single_qubit_rot_angles(self) -> tuple[float, float, float]:
        r"""The parameters required to implement a single-qubit gate as an
        equivalent ``Rot`` gate, up to a global phase.
        """
        raise NotImplementedError

    @property
    def parameter_frequencies(self) -> list[tuple[float | int]]:
        r"""Returns the frequencies for each operator parameter with respect
        to an expectation value of the form
        :math:`\langle \psi | U(\mathbf{p})^\dagger \hat{O} U(\mathbf{p})|\psi\rangle`.
        """
        if self.num_params == 1:
            # if the operator has a single parameter, we can query the
            # generator, and if defined, use its eigenvalues.
            try:
                gen = qml.generator(self, format="observable")
            except GeneratorUndefinedError as e:
                raise ParameterFrequenciesUndefinedError(
                    f"Operation {self.name} does not have parameter frequencies defined."
                ) from e

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".+ eigenvalues will be computed numerically\."
                )
                eigvals = qml.eigvals(gen, k=2 ** len(self.wires))

            eigvals = tuple(np.round(eigvals, 8))
            return [qml.gradients.eigvals_to_frequencies(eigvals)]

        raise ParameterFrequenciesUndefinedError(
            f"Operation {self.name} does not have parameter frequencies defined, "
            "and parameter frequencies can not be computed as no generator is defined."
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check the grad_recipe validity
        if self.grad_recipe is None:
            # Make sure grad_recipe is an iterable of correct length instead of None
            self.grad_recipe = [None] * self.num_params


class StatePrepBase2(Operation2):
    """An interface for state-prep operations."""

    grad_method = None

    @abc.abstractmethod
    def state_vector(self, wire_order: WiresLike | None = None) -> TensorLike:
        """
        Returns the initial state vector for a circuit given a state preparation.

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels
                from the operator's wires

        Returns:
            array: A state vector for all wires in a circuit
        """

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return "|Ψ⟩"


def operation_derivative(operation: Operation2) -> TensorLike:
    r"""Calculate the derivative of an operation."""
    generator = qml.matrix(
        qml.generator(operation, format="observable"), wire_order=operation.wires
    )
    return 1j * generator @ operation.matrix()


@qml.BooleanFn
def is_trainable(obj):
    """Returns ``True`` if any of the parameters of an operator is trainable
    according to ``math.requires_grad``.
    """
    return any(math.requires_grad(p) for p in obj.parameters)
