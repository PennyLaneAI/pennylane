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

"""Defines the base class for the power of operators."""

from functools import reduce
from importlib import import_module
from typing import Union, override

from scipy.linalg import fractional_matrix_power

import pennylane as qp
from pennylane import capture, math
from pennylane.capture.op_eqns import pop_op_eqns
from pennylane.core import Operator
from pennylane.core.apply import apply
from pennylane.core.operator import abstractify
from pennylane.decomposition.resources import (
    AbstractOperatorLike,
    CompressedResourceOp,
    pow_resource_rep,
)
from pennylane.decomposition.utils import to_name
from pennylane.exceptions import (
    AdjointUndefinedError,
    DecompositionUndefinedError,
    PowUndefinedError,
    SparseMatrixUndefinedError,
)
from pennylane.ops.op_math import adjoint

from .symbolicop2 import SymbolicOp2

_superscript = str.maketrans("0123456789.+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⋅⁺⁻")


class Pow2(SymbolicOp2):
    """Symbolic operator denoting an operator raised to a power.

    Args:
        base (~.operation.Operator): the operator to be raised to a power
        z=1 (float): the exponent

    **Example**

    >>> sqrt_x = Pow2(qp.X(0), 0.5)
    >>> Pow2.compute_decomposition(qp.X(0), 0.5)
    [SX(0)]
    >>> qp.matrix(sqrt_x)
    array([[0.5+0.5j, 0.5-0.5j],
           [0.5-0.5j, 0.5+0.5j]])
    >>> qp.matrix(qp.SX(0))
    array([[0.5+0.5j, 0.5-0.5j],
           [0.5-0.5j, 0.5+0.5j]])
    >>> qp.matrix(Pow2(qp.T(0), 1.234))
    array([[1.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 0.56...+0.8244...j]])

    """

    z: int | float

    wire_argnames = ()
    hybrid_argnames = ("base",)
    static_argnames = ("z",)

    def __init__(self, base: Operator, z: float):
        super().__init__(base, z)

        if isinstance(z, int) and z > 0:
            if (base_pauli_rep := getattr(self.base, "pauli_rep", None)) and (
                self.batch_size is None
            ):
                pr = base_pauli_rep
                for _ in range(z - 1):
                    pr = pr @ base_pauli_rep
                self._pauli_rep = pr
            else:
                self._pauli_rep = None
        else:
            self._pauli_rep = None

    @override
    def _bind_primitive(self):
        pop_op_eqns((self.base,))
        super()._bind_primitive()

    def __repr__(self):
        return (
            f"({self.base})**{self.z}"
            if self.base.arithmetic_depth > 0
            else f"{self.base}**{self.z}"
        )

    @property
    @override
    def basis(self):  # pylint: disable=missing-function-docstring
        return self.base.basis

    @property
    @override
    def name(self) -> str:
        return f"Pow({self.base.name})"

    @property
    @override
    def ndim_params(self):
        return self.base.ndim_params

    @property
    @override
    def data(self):
        return self.base.data

    @property
    @override
    def num_wires(self):
        """The number of wires this power op acts on."""
        return self.base.num_wires

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        z_string = format(self.z).translate(_superscript)
        base_label = self.base.label(decimals, base_label, cache=cache)
        return (
            f"({base_label}){z_string}" if self.base.arithmetic_depth > 0 else base_label + z_string
        )

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    @override
    def has_sparse_matrix(self) -> bool:
        return self.base.has_sparse_matrix and isinstance(self.z, int)

    @staticmethod
    @override
    def compute_matrix(base, z):  # pylint: disable=arguments-differ
        mat = qp.matrix(base)
        if isinstance(z, int):
            return math.linalg.matrix_power(mat, z)
        return fractional_matrix_power(mat, z)

    # pylint: disable=arguments-differ
    @staticmethod
    @override
    def compute_sparse_matrix(base=None, z=0, format="csr"):
        if isinstance(z, int):
            base_matrix = base.compute_sparse_matrix(**base.arguments)
            return (base_matrix**z).asformat(format)
        raise SparseMatrixUndefinedError

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    @override
    def has_decomposition(self):
        if isinstance(self.z, int) and self.z > 0:
            return True
        try:
            self.base.pow(self.z)
        except PowUndefinedError:
            return False
        except Exception as e:
            # some pow methods cant handle a batched z
            if math.ndim(self.z) != 0:
                return False
            raise e
        return True

    @staticmethod
    @override
    def compute_decomposition(base, z):
        try:
            return base.pow(z)
        except PowUndefinedError as e:
            if isinstance(z, int) and z > 0:
                return [apply(base) for _ in range(z)]
            # TODO: consider: what if z is an int and less than 0?
            # do we want Pow(base, -1) to be a "more fundamental" op
            raise DecompositionUndefinedError from e
        except Exception as e:
            raise DecompositionUndefinedError from e

    @property
    @override
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    @staticmethod
    @override
    def compute_diagonalizing_gates(base, z):  # pylint: disable=unused-argument
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates of an operator to a power is the same as the diagonalizing
        gates as the original operator. As we can see,

        .. math::

            O^2 = U \Sigma U^{\dagger} U \Sigma U^{\dagger} = U \Sigma^2 U^{\dagger}

        This formula can be extended to inversion and any rational number.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_diagonalizing_gates`.

        Args:
            base (~.Operation): the operator to be raised to a power
            z (int): the exponent

        Returns:
            list[.Operator] or None: a list of operators
        """
        return base.diagonalizing_gates()

    @override
    def eigvals(self):
        base_eigvals = self.base.eigvals()
        is_single_precision = math.get_dtype_name(base_eigvals) in ("float32", "complex64")
        complex_dtype = "complex64" if is_single_precision else "complex128"
        return math.cast(base_eigvals, complex_dtype) ** self.z

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    @override
    def has_generator(self):
        return self.base.has_generator

    @override
    def generator(self):
        r"""Generator of an operator that is in single-parameter-form.

        The generator of a power operator is ``z`` times the generator of the
        base matrix.

        .. math::

            U(\phi)^z = e^{i\phi (z G)}

        See also :func:`~.generator`
        """
        return self.z * self.base.generator()

    @override
    def pow(self, z):
        return [Pow2(base=self.base, z=self.z * z)]

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    @override
    def has_adjoint(self):
        return isinstance(self.z, int)

    @override
    def adjoint(self):
        """Create an operation that is the adjoint of this one.

        Adjointed operations are the conjugated and transposed version of the
        original operation. Adjointed ops are equivalent to the inverted operation for unitary
        gates.

        .. warning::

            The adjoint of a fractional power of an operator is not well-defined due to branch cuts in the power function.
            Therefore, an ``AdjointUndefinedError`` is raised when the power ``z`` is not an integer.

            The integer power check is a type check, so that floats like ``2.0`` are not considered to be integers.

        Returns:
            The adjointed operation.

        Raises:
            AdjointUndefinedError: If the exponent ``z`` is not of type ``int``.

        """
        if isinstance(self.z, int):
            return Pow2(base=adjoint(self.base), z=self.z)
        raise AdjointUndefinedError(
            "The adjoint of Pow operators only is well-defined for integer powers."
        )

    @override
    def simplify(self) -> Union["Pow", "Identity"]:
        from pennylane.ops.identity import Identity  # pylint: disable=import-outside-toplevel

        # try using pauli_rep:
        if pr := self.pauli_rep:
            pr.prune()
            return pr.operation(wire_order=self.wires)

        base = self.base if capture.enabled() else self.base.simplify()
        try:
            ops = base.pow(z=self.z)
            if not ops:
                return Identity(self.wires)
            if not capture.enabled():
                ops = [op.simplify() for op in ops]
            return reduce(lambda nxt, acc: nxt @ acc, ops) if len(ops) > 1 else ops[0]
        except PowUndefinedError:
            return Pow2(base=base, z=self.z)


def _pow_abstract(op: AbstractOperatorLike | type[Operator], z: int | float = 1):
    op = abstractify(op)
    if isinstance(op, CompressedResourceOp):
        return pow_resource_rep(op.op_type, op.params, z)
    return qp.pow(op, z)


@to_name.register
def _pow2_to_name(op: Pow2):
    return f"Pow({to_name(op.base)})"


def __getattr__(name):
    """Resolve the existing public names of relocated decomposition rules."""
    if name in {
        "flip_pow_adjoint",
        "merge_powers",
        "pow_involutory",
        "repeat_pow_base",
        "make_pow_decomp_with_period",
        "pow_rotation",
    }:
        rules = import_module("pennylane.decomposition_rules.op_math.pow2")

        return getattr(rules, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
