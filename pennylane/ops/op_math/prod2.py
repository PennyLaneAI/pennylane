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
"""
This file contains the implementation of the ``Prod2`` class, an :class:`~.Operator2`-based
symbolic operator representing the product of operators.
"""

import itertools
from collections import Counter
from copy import copy
from functools import reduce
from typing import Union, override

from scipy.sparse import kron as sparse_kron

import pennylane as qp
from pennylane import math
from pennylane.core.operator import Operator, Operator2, abstractify
from pennylane.core.queuing import apply
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.utils import to_name
from pennylane.exceptions import SparseMatrixUndefinedError
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ..qubit.non_parametric_ops import PauliX, PauliY, PauliZ
from .composite import handle_recursion_error
from .composite2 import CompositeOp2
from .pow import Pow
from .sprod import SProd
from .sum import Sum

MAX_NUM_WIRES_KRON_PRODUCT = 9
"""The maximum number of wires up to which using ``math.kron`` is faster than ``math.dot`` for
computing the sparse matrix representation."""


def _swappable_ops(op1, op2, wire_map: dict = None) -> bool:
    """Boolean expression that indicates if op1 and op2 don't have intersecting wires and if they
    should be swapped when sorting them by wire values.

    Args:
        op1 (.Operator): First operator.
        op2 (.Operator): Second operator.
        wire_map (dict): Dictionary containing the wire values as keys and its indexes as values.
            Defaults to None.

    Returns:
        bool: True if operators should be swapped, False otherwise.
    """
    # one is broadcasted onto all wires.
    if not op1.wires:
        return True
    if not op2.wires:
        return False
    wires1 = op1.wires
    wires2 = op2.wires
    if wire_map is not None:
        wires1 = wires1.map(wire_map)
        wires2 = wires2.map(wire_map)
    wires1 = set(wires1)
    wires2 = set(wires2)
    # compare strings of wire labels so that we can compare arbitrary wire labels like 0 and "a"
    return False if wires1 & wires2 else str(wires1.pop()) > str(wires2.pop())


class Prod2(CompositeOp2):
    r"""Symbolic operator representing the product of operators, built on :class:`~.CompositeOp2`.

    Args:
        operands (Sequence[~.operation.Operator]): the operators to be multiplied together.

    .. seealso:: :class:`~.ops.op_math.Prod`

    **Example**

    >>> prod_op = qp.ops.Prod2([qp.X(0), qp.Z(1)])
    >>> prod_op
    X(0) @ Z(1)
    >>> prod_op.decomposition()
    [Z(1), X(0)]

    .. note::
        When a ``Prod2`` operator is applied in a circuit, its factors are applied in the reverse
        order (i.e ``Prod2([op1, op2])`` corresponds to :math:`\hat{op}_{1}\cdot\hat{op}_{2}`, which
        indicates first applying :math:`\hat{op}_{2}` then :math:`\hat{op}_{1}` in the circuit).
    """

    hybrid_argnames = ("operands", "_init_pauli_rep")

    _op_symbol = "@"
    _math_op = staticmethod(math.prod)

    @classmethod
    def _sort(cls, op_list, wire_map: dict = None) -> list[Operator]:
        """Insertion sort of product factors by wire indices, respecting commutativity.

        Sorting relies on concrete wires; for abstract or compressed operands (which appear as
        resource representations in the decomposition graph) the construction order is preserved.
        """
        op_list = list(op_list)

        for i in range(1, len(op_list)):
            key_op = op_list[i]

            j = i - 1
            while j >= 0 and _swappable_ops(op1=op_list[j], op2=key_op, wire_map=wire_map):
                op_list[j + 1] = op_list[j]
                j -= 1
            op_list[j + 1] = key_op

        return op_list

    def _build_pauli_rep(self):
        """PauliSentence representation of the product of operators."""
        if all(operand_pauli_reps := [op.pauli_rep for op in self.operands]):
            return reduce(lambda a, b: a @ b, operand_pauli_reps) if operand_pauli_reps else None
        return None

    @property
    @handle_recursion_error
    def has_matrix(self) -> bool:
        return all(op.has_matrix for op in self) or self.pauli_rep is not None

    @handle_recursion_error
    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        if self.pauli_rep:
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires)

        mats: list[TensorLike] = []
        batched: list[bool] = []
        for ops in self.overlapping_ops:
            gen = ((op.matrix(), op.wires) for op in ops)
            reduced_mat, _ = math.reduce_matrices(gen, reduce_func=math.matmul)

            if self.batch_size is not None:
                batched.append(any(op.batch_size is not None for op in ops))
            else:
                batched.append(False)

            mats.append(reduced_mat)

        if self.batch_size is None:
            full_mat = reduce(math.kron, mats)
        else:
            full_mat = math.stack(
                [
                    reduce(
                        math.kron, [m[i] if b else m for m, b in zip(mats, batched, strict=True)]
                    )
                    for i in range(self.batch_size)
                ]
            )
        return math.expand_matrix(full_mat, self.wires, wire_order=wire_order)

    @property
    @handle_recursion_error
    def has_sparse_matrix(  # pylint: disable=arguments-differ,invalid-overridden-method
        self,
    ) -> bool:
        return (
            all(op.has_sparse_matrix for op in self)
            or self.pauli_rep is not None
            # Sparse matrices are 2-D. Thus, batch sizes are not supported
            and self.batch_size is None
        )

    @handle_recursion_error
    def sparse_matrix(self, wire_order=None, format="csr"):
        if self.pauli_rep:
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires, format=format)

        if self.batch_size is not None:
            raise SparseMatrixUndefinedError(
                "Sparse matrices cannot be defined for batched operators."
            )

        if self.has_overlapping_wires or self.num_wires > MAX_NUM_WIRES_KRON_PRODUCT:
            gen = ((op.sparse_matrix(), op.wires) for op in self)
            reduced_mat, prod_wires = math.reduce_matrices(gen, reduce_func=math.dot)
            wire_order = wire_order or self.wires
            return math.expand_matrix(reduced_mat, prod_wires, wire_order=wire_order).asformat(
                format
            )

        mats = (op.sparse_matrix() for op in self)
        full_mat = reduce(sparse_kron, mats)
        return math.expand_matrix(full_mat, self.wires, wire_order=wire_order).asformat(format)

    @property
    @override
    def is_verified_hermitian(self) -> bool:
        """Non-exhaustive check for whether the product is Hermitian. Since the check
        is non-exhaustive, it may be possible for Hermitian operators to return ``False``.
        """
        from itertools import combinations  # pylint: disable=import-outside-toplevel

        for o1, o2 in combinations(self.operands, r=2):
            if Wires.shared_wires([o1.wires, o2.wires]):
                return False
        return all(op.is_verified_hermitian for op in self)

    @override
    def adjoint(self) -> "Prod2":
        return Prod2([qp.adjoint(factor) for factor in self[::-1]])

    @property
    @override
    def name(self) -> str:
        """The legacy 'Prod' name, so name-keyed dispatch keeps recognizing products."""
        return "Prod"

    def _simplify_factors(self, factors: tuple[Operator2]) -> tuple[complex, Operator2]:
        """Reduces the depth of nested factors and groups identical factors.

        Returns:
            Tuple[complex, List[~.operation.Operator]: tuple containing the global phase and a list
            of the simplified factors
        """
        new_factors = _ProductFactorsGrouping()

        for factor in factors:
            simplified_factor = factor.simplify()
            new_factors.add(factor=simplified_factor)
        new_factors.remove_factors(wires=self.wires)
        return new_factors.global_phase, new_factors.factors

    @handle_recursion_error
    def terms(self):
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`

        **Example**

        >>> op = qp.X(0) @ (0.5 * qp.X(1) + qp.X(2))
        >>> op.terms()
        ([np.float64(0.5), 1.0], [X(0) @ X(1), X(0) @ X(2)])

        """
        # try using pauli_rep:
        if pr := self.pauli_rep:
            with qp.QueuingManager.stop_recording():
                ops = [pauli.operation() for pauli in pr.keys()]
            return list(pr.values()), ops

        with qp.QueuingManager.stop_recording():
            global_phase, factors = self._simplify_factors(factors=self.operands)
            factors = list(itertools.product(*factors))

            factors = [
                Prod2(factor).simplify() if len(factor) > 1 else factor[0] for factor in factors
            ]

        # harvest coeffs and ops
        coeffs = []
        ops = []
        for factor in factors:
            if isinstance(factor, SProd):
                coeffs.append(global_phase * factor.scalar)
                ops.append(factor.base)
            else:
                coeffs.append(global_phase)
                ops.append(factor)
        return coeffs, ops

    @handle_recursion_error
    def simplify(self) -> Union["Prod2", Sum]:
        r"""
        Transforms any nested Prod instance into the form :math:`\sum c_i O_i` where
        :math:`c_i` is a scalar coefficient and :math:`O_i` is a single PL operator
        or pure product of single PL operators.
        """
        # try using pauli_rep:
        if pr := self.pauli_rep:
            pr.prune()
            return pr.operation(wire_order=self.wires)

        global_phase, factors = self._simplify_factors(factors=self.operands)

        factors = list(itertools.product(*factors))
        if len(factors) == 1:
            factor = factors[0]
            if len(factor) == 0:
                op = qp.Identity(self.wires)
            else:
                op = factor[0] if len(factor) == 1 else Prod2(*factor)
            return op if global_phase == 1 else qp.s_prod(global_phase, op)

        factors = [Prod2(factor).simplify() if len(factor) > 1 else factor[0] for factor in factors]
        op = Sum(*factors).simplify()
        return op if global_phase == 1 else qp.s_prod(global_phase, op).simplify()


@abstractify.register(Prod2)
def _abstractify_prod2(val: Prod2):
    """Abstractify ``Prod2``."""
    abstract_operands = tuple(abstractify(op) for op in val.operands)
    return Prod2(abstract_operands, _init_pauli_rep=None)


def _prod2_resources(operands, _init_pauli_rep=None):  # pylint: disable=unused-argument
    return dict(Counter(abstractify(op) for op in operands))


@register_resources(_prod2_resources)
def _prod2_decomp(operands, _init_pauli_rep=None):  # pylint: disable=unused-argument
    for op in reversed(operands):
        apply(op)


add_decomps(Prod2, _prod2_decomp)


@to_name.register
def _prod2_to_name(op):
    """Prod2.name is Prod for device dispatch. Need to keep Prod2 decomp registry unique."""
    return "Prod2"


class _ProductFactorsGrouping:
    """Utils class used for grouping identical product factors."""

    _identity_map = {
        "Identity": (1.0, "Identity"),
        "PauliX": (1.0, "PauliX"),
        "PauliY": (1.0, "PauliY"),
        "PauliZ": (1.0, "PauliZ"),
    }
    _x_map = {
        "Identity": (1.0, "PauliX"),
        "PauliX": (1.0, "Identity"),
        "PauliY": (1.0j, "PauliZ"),
        "PauliZ": (-1.0j, "PauliY"),
    }
    _y_map = {
        "Identity": (1.0, "PauliY"),
        "PauliX": (-1.0j, "PauliZ"),
        "PauliY": (1.0, "Identity"),
        "PauliZ": (1.0j, "PauliX"),
    }
    _z_map = {
        "Identity": (1.0, "PauliZ"),
        "PauliX": (1.0j, "PauliY"),
        "PauliY": (-1.0j, "PauliX"),
        "PauliZ": (1.0, "Identity"),
    }
    _pauli_mult = {"Identity": _identity_map, "PauliX": _x_map, "PauliY": _y_map, "PauliZ": _z_map}
    _paulis = {"PauliX": PauliX, "PauliY": PauliY, "PauliZ": PauliZ}

    def __init__(self):
        self._pauli_factors = {}  #  {wire: (pauli_coeff, pauli_word)}
        self._non_pauli_factors = {}  # {wires: [hash, exponent, operator]}
        self._factors = []
        self.global_phase = 1

    def add(self, factor: Operator):
        """Add factor.

        Args:
            factor (Operator): Factor to add.
        """
        wires = factor.wires
        if isinstance(factor, Prod2):
            for prod_factor in factor:
                self.add(prod_factor)
        elif isinstance(factor, Sum):
            self._remove_pauli_factors(wires=wires)
            self._remove_non_pauli_factors(wires=wires)
            self._factors += (factor.operands,)
        elif not isinstance(factor, qp.Identity):
            if isinstance(factor, SProd):
                self.global_phase *= factor.scalar
                factor = factor.base
            if isinstance(factor, (qp.Identity, qp.X, qp.Y, qp.Z)):
                self._add_pauli_factor(factor=factor, wires=wires)
                self._remove_non_pauli_factors(wires=wires)
            else:
                self._add_non_pauli_factor(factor=factor, wires=wires)
                self._remove_pauli_factors(wires=wires)

    def _add_pauli_factor(self, factor: Operator, wires: list[int]):
        """Adds the given Pauli operator to the temporary ``self._pauli_factors`` dictionary. If
        there was another Pauli operator acting on the same wire, the two operators are grouped
        together using the ``self._pauli_mult`` dictionary.

        Args:
            factor (Operator): Factor to be added.
            wires (List[int]): Factor wires. This argument is added to avoid calling
                ``factor.wires`` several times.
        """
        wire = wires[0]
        op2_name = factor.name
        old_coeff, old_word = self._pauli_factors.get(wire, (1, "Identity"))
        coeff, new_word = self._pauli_mult[old_word][op2_name]
        self._pauli_factors[wire] = old_coeff * coeff, new_word

    def _add_non_pauli_factor(self, factor: Operator, wires: list[int]):
        """Adds the given non-Pauli factor to the temporary ``self._non_pauli_factors`` dictionary.
        If there alerady exists an identical operator in the dictionary, the two are grouped
        together.

        If there isn't an identical operator in the dictionary, all non Pauli factors that act on
        the same wires are removed and added to the ``self._factors`` tuple.

        Args:
            factor (Operator): Factor to be added.
            wires (List[int]): Factor wires. This argument is added to avoid calling
                ``factor.wires`` several times.
        """
        if isinstance(factor, Pow):
            exponent = factor.z
            factor = factor.base
        else:
            exponent = 1
        op_hash = hash(factor)
        old_hash, old_exponent, old_op = self._non_pauli_factors.get(wires, [None, None, None])
        if isinstance(old_op, (qp.RX, qp.RY, qp.RZ)) and factor.name == old_op.name:
            self._non_pauli_factors[wires] = [
                op_hash,
                old_exponent,
                factor.__class__(factor.data[0] + old_op.data[0], wires).simplify(),
            ]
        elif op_hash == old_hash:
            self._non_pauli_factors[wires][1] += exponent
        else:
            self._remove_non_pauli_factors(wires=wires)
            self._non_pauli_factors[wires] = [op_hash, copy(exponent), factor]

    def _remove_non_pauli_factors(self, wires: list[int]):
        """Remove all factors from the ``self._non_pauli_factors`` dictionary that act on the given
        wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
        if not self._non_pauli_factors:
            return
        for wire in wires:
            for key, (_, exponent, op) in list(self._non_pauli_factors.items()):
                if wire in key:
                    self._non_pauli_factors.pop(key)
                    if exponent == 0:
                        continue
                    if exponent != 1:
                        op = Pow(base=op, z=exponent).simplify()
                    if not isinstance(op, qp.Identity):
                        self._factors += ((op,),)

    def _remove_pauli_factors(self, wires: list[int]):
        """Remove all Pauli factors from the ``self._pauli_factors`` dictionary that act on the
        given wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
        if not self._pauli_factors:
            return
        for wire in wires:
            pauli_coeff, pauli_word = self._pauli_factors.pop(wire, (1, "Identity"))
            if pauli_word != "Identity":
                pauli_op = self._paulis[pauli_word](wire)
                self._factors += ((pauli_op,),)
            self.global_phase *= pauli_coeff

    def remove_factors(self, wires: list[int]):
        """Remove all factors from the ``self._pauli_factors`` and ``self._non_pauli_factors``
        dictionaries that act on the given wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
        self._remove_pauli_factors(wires=wires)
        self._remove_non_pauli_factors(wires=wires)

    @property
    def factors(self):
        """Grouped factors tuple.

        Returns:
            tuple: Tuple of grouped factors.
        """
        return tuple(self._factors)
