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
from typing import Union, override

from scipy.linalg import fractional_matrix_power

import pennylane as qp
from pennylane import capture, math
from pennylane.core import Operator
from pennylane.core.operator import abstractify
from pennylane.core.queuing import apply
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    DecompositionRule,
    get_fixed_decomp,
    list_decomps,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import (
    AbstractOperatorLike,
    CompressedResourceOp,
    pow_resource_rep,
)
from pennylane.decomposition.symbolic_decomposition import is_integer
from pennylane.exceptions import (
    AdjointUndefinedError,
    DecompositionUndefinedError,
    PowUndefinedError,
    SparseMatrixUndefinedError,
)
from pennylane.ops.identity import Identity
from pennylane.ops.op_math import adjoint

from .adjoint import Adjoint
from .adjoint2 import Adjoint2
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

    def __repr__(self):
        return (
            f"({self.base})**{self.z}"
            if self.base.arithmetic_depth > 0
            else f"{self.base}**{self.z}"
        )

    @property
    def ndim_params(self):
        return self.base.ndim_params

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        z_string = format(self.z).translate(_superscript)
        base_label = self.base.label(decimals, base_label, cache=cache)
        return (
            f"({base_label}){z_string}" if self.base.arithmetic_depth > 0 else base_label + z_string
        )

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_sparse_matrix(self) -> bool:
        return self.base.has_sparse_matrix and isinstance(self.z, int)

    @staticmethod
    def compute_matrix(base, z):  # pylint: disable=arguments-differ
        mat = qp.matrix(base)
        if isinstance(z, int):
            return math.linalg.matrix_power(mat, z)
        return fractional_matrix_power(mat, z)

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_sparse_matrix(base=None, z=0, format="csr"):
        if isinstance(z, int):
            base_matrix = base.compute_sparse_matrix(**base.arguments)
            return (base_matrix**z).asformat(format)
        raise SparseMatrixUndefinedError

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
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
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    @staticmethod
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

    @staticmethod
    def compute_eigvals(base, z):
        base_eigvals = base.eigvals()
        return [value**z for value in base_eigvals]

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_generator(self):
        return self.base.has_generator

    def generator(self):
        r"""Generator of an operator that is in single-parameter-form.

        The generator of a power operator is ``z`` times the generator of the
        base matrix.

        .. math::

            U(\phi)^z = e^{i\phi (z G)}

        See also :func:`~.generator`
        """
        return self.z * self.base.generator()

    def pow(self, z):
        return [Pow2(base=self.base, z=self.z * z)]

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_adjoint(self):
        return isinstance(self.z, int)

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

    def simplify(self) -> Union["Pow", Identity]:
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


# pylint: disable=protected-access,unused-argument
@register_condition(lambda z, **__: is_integer(z) and z >= 0)
@register_resources(lambda base, z: {abstractify(base): z})
def repeat_pow_base(base, z):
    """Decompose the power of an operator by repeating the base operator. Assumes z
    is a non-negative integer."""

    @qp.for_loop(0, z)
    def _loop(i):
        qp.apply(base)

    _loop()  # pylint: disable=no-value-for-parameter


# pylint: disable=protected-access,unused-argument
@register_resources(lambda base, z: {abstractify(base.base): z * base.z})
def merge_powers(base, z):
    """Decompose nested powers by combining them."""
    qp.pow(base.base, z * base.z)


def _flip_pow_adjoint_resource(base, z):
    # base class is adjoint, and the base of the base is the target class
    return {qp.adjoint(Pow2(base.base, z=z)): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_flip_pow_adjoint_resource)
def flip_pow_adjoint(base, z, **__):
    """Decompose the power of an adjoint by power to the base of the adjoint and
    then taking the adjoint of the power."""
    adjoint(qp.pow(base.base, z))


def make_pow_decomp_with_period(period) -> DecompositionRule:
    """Make a decomposition rule for the power of an op that has a period."""

    def _condition_fn(base, z):  # pylint: disable=unused-argument
        return math.shape(z) == () and z % period != z

    def _resource_fn(base, z):
        z_mod_period = z % period
        if z_mod_period == 0:
            return {}
        if z_mod_period == 1:
            return {abstractify(base): 1}
        return {_pow_abstract(base, z_mod_period): 1}

    @register_condition(_condition_fn)
    @register_resources(_resource_fn)
    def _impl(base, z, **__):  # pylint: disable=unused-argument
        z_mod_period = z % period
        if z_mod_period == 1:
            apply(base)
        elif z_mod_period > 0 and z_mod_period != period:
            qp.pow(base, z_mod_period)

    return _impl


pow_involutory = make_pow_decomp_with_period(2)


@list_decomps.register
def _list_pow_decomps(op: Pow2) -> DecompCollection:

    abs_op = abstractify(op)

    # fixed_decomps would override everything.
    if fixed_rule := get_fixed_decomp(op):
        return DecompCollection([fixed_rule])

    # special case of merging nested powers.
    if isinstance(abs_op.base, Pow2):
        return DecompCollection([merge_powers])

    # special case that the base is an Adjoint.
    if isinstance(abs_op.base, (Adjoint, Adjoint2)):
        return DecompCollection([flip_pow_adjoint])

    # Custom decomposition rules registered specifically to the pow operator
    custom_rules = list_decomps.dispatch(object)(abs_op)

    return custom_rules + [repeat_pow_base] if is_integer(op.z) else custom_rules
