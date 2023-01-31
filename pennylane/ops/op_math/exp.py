# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This submodule defines the symbolic operation that stands for an exponential of an operator.
"""
from copy import copy
from warnings import warn

import numpy as np
from scipy.sparse.linalg import expm as sparse_expm

import pennylane as qml
from pennylane import math
from pennylane.operation import (
    DecompositionUndefinedError,
    GeneratorUndefinedError,
    Operation,
    OperatorPropertyUndefined,
    Tensor,
    expand_matrix,
)
from pennylane.ops.qubit import Hamiltonian
from pennylane.wires import Wires

from .sprod import SProd
from .sum import Sum
from .symbolicop import ScalarSymbolicOp


def exp(op, coeff=1, num_steps=1, id=None):
    """Take the exponential of an Operator times a coefficient.

    Args:
        base (~.operation.Operator): The Operator to be exponentiated
        coeff=1 (Number): A scalar coefficient of the operator.

    Returns:
       :class:`Exp`: A :class`~.operation.Operator` representing an operator exponential.

    .. note::

        This operator supports a batched base, a batched coefficient and a combination of both:

        >>> op = qml.exp(qml.RX([1, 2, 3], wires=0), coeff=4)
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.exp(qml.RX(1, wires=0), coeff=[1, 2, 3])
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.exp(qml.RX([1, 2, 3], wires=0), coeff=[4, 5, 6])
        >>> qml.matrix(op).shape
        (3, 2, 2)

        But it doesn't support batching of operators:

        >>> op = qml.exp([qml.RX(1, wires=0), qml.RX(2, wires=0)], coeff=4)
        AttributeError: 'list' object has no attribute 'batch_size'

    **Example**

    This symbolic operator can be used to make general rotation operators:

    >>> x = np.array(1.23)
    >>> op = qml.exp( qml.PauliX(0), -0.5j * x)
    >>> qml.math.allclose(op.matrix(), qml.RX(x, wires=0).matrix())
    True

    This can even be used for more complicated generators:

    >>> t = qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1)
    >>> isingxy = qml.exp(t, 0.25j * x)
    >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(x, wires=(0,1)).matrix())
    True

    If the coefficient is purely imaginary and the base operator is Hermitian, then
    the gate can be used in a circuit, though it may not be supported by the device and
    may not be differentiable.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit(x):
    ...     qml.exp(qml.PauliX(0), -0.5j * x)
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(qml.draw(circuit)(1.23))
    0: ──Exp─┤  <Z>

    If the base operator is Hermitian and the coefficient is real, then the ``Exp`` operator
    can be measured as an observable:

    >>> obs = qml.exp(qml.PauliZ(0), 3)
    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit():
    ...     return qml.expval(obs)
    >>> circuit()
    tensor(20.08553692, requires_grad=True)

    """
    return Exp(op, coeff, num_steps=num_steps, id=id)


class Exp(ScalarSymbolicOp, Operation):
    """A symbolic operator representating the exponential of a operator.

    Args:
        base (~.operation.Operator): The Operator to be exponentiated
        coeff=1 (Number): A scalar coefficient of the operator.

    **Example**

    This symbolic operator can be used to make general rotation operators:

    >>> x = np.array(1.23)
    >>> op = Exp( qml.PauliX(0), -0.5j * x)
    >>> qml.math.allclose(op.matrix(), qml.RX(x, wires=0).matrix())
    True

    This can even be used for more complicated generators:

    >>> t = qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1)
    >>> isingxy = Exp(t, 0.25j * x)
    >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(x, wires=(0,1)).matrix())
    True

    If the coefficient is purely imaginary and the base operator is Hermitian, then
    the gate can be used in a circuit, though it may not be supported by the device and
    may not be differentiable.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit(x):
    ...     Exp(qml.PauliX(0), -0.5j * x)
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(qml.draw(circuit)(1.23))
    0: ──Exp─┤  <Z>

    If the base operator is Hermitian and the coefficient is real, then the ``Exp`` operator
    can be measured as an observable:

    >>> obs = Exp(qml.PauliZ(0), 3)
    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit():
    ...     return qml.expval(obs)
    >>> circuit()
    tensor(20.08553692, requires_grad=True)

    """

    control_wires = Wires([])
    _name = "Exp"

    # pylint: disable=too-many-arguments
    def __init__(self, base, coeff=1, num_steps=1, do_queue=True, id=None):
        super().__init__(base, scalar=coeff, do_queue=do_queue, id=id)
        self._name = "Exp"
        self._data = [[coeff], self.base.data]
        self.grad_recipe = [None]
        self.num_steps = num_steps

    def __repr__(self):
        return (
            f"Exp({self.coeff} {self.base})"
            if self.base.arithmetic_depth > 0
            else f"Exp({self.coeff} {self.base.name})"
        )

    # pylint: disable=attribute-defined-outside-init
    def __copy__(self):
        # this method needs to be overwritten because the base must be copied too.
        copied_op = object.__new__(type(self))
        # copied_op must maintain inheritance structure of self
        # Relevant for symbolic ops that mix in operation-specific components.

        for attr, value in vars(self).items():
            if attr not in {"_hyperparameters"}:
                setattr(copied_op, attr, value)

        copied_op._hyperparameters = copy(self.hyperparameters)
        copied_op.hyperparameters["base"] = copy(self.base)
        copied_op._data = copy(self._data)

        return copied_op

    @property
    def hash(self):
        return hash((str(self.name), self.base.hash, str(self.coeff)))

    @property
    def data(self):
        return self._data

    @property
    def coeff(self):
        """The numerical coefficient of the operator in the exponent."""
        return self.scalar

    @property
    def num_params(self):
        return self.base.num_params + 1

    @property
    def is_hermitian(self):
        return self.base.is_hermitian and math.allequal(math.imag(self.coeff), 0)

    @property
    def _queue_category(self):
        if self.base.is_hermitian and math.allequal(math.real(self.coeff), 0):
            return "_ops"
        return None

    @property
    def inverse(self):
        """Setting inverse is not defined for Exp, so the inverse is always False"""
        return False

    # pylint: disable=invalid-overridden-method
    @property
    def has_decomposition(self):
        if isinstance(self.base, Tensor) and len(self.base.wires) != len(self.base.obs):
            raise DecompositionUndefinedError(
                "Unable to determine if the exponential has a decomposition "
                "when the base operator is a Tensor object with overlapping wires. "
                f"Received base {self.base}."
            )
        base = self.base
        coeff = self.coeff
        if isinstance(base, SProd):
            coeff *= base.scalar
            base = base.base
        is_pauli_rot = qml.pauli.is_pauli_word(self.base) and math.real(self.coeff) == 0
        is_sum = isinstance(base, (Hamiltonian, Sum))
        return is_pauli_rot or is_sum

    # pylint: disable=too-many-branches
    def decomposition(self):
        r"""Representation of the operator as a product of other operators. Decomposes into
        :class:`~.PauliRot` if the coefficient is imaginary and the base is a Pauli Word.

        .. math:: O = O_1 O_2 \dots O_n

        A ``DecompositionUndefinedError`` is raised if the coefficient is not imaginary or the base
        is not a Pauli Word.

        Returns:
            list[PauliRot]: decomposition of the operator
        """
        base = self.base
        coeff = self.coeff / self.num_steps  # divide by trotter number
        if isinstance(base, SProd):
            coeff *= base.scalar
            base = base.base

        if isinstance(base, Tensor) and len(base.wires) != len(base.obs):
            raise DecompositionUndefinedError(
                "Unable to determine if the exponential has a decomposition "
                "when the base operator is a Tensor object with overlapping wires. "
                f"Received base {self.base}."
            )

        if qml.pauli.is_pauli_word(base) and math.real(coeff) == 0:
            pauli_word = qml.pauli.pauli_word_to_string(base)
            coeff = 2j * coeff  # need to cancel the coefficients added by PauliRot
            if pauli_word == "I" * base.num_wires:
                return qml.Identity(wires=base.wires)
            if len(pauli_word) == 1:
                return [getattr(qml, f"R{pauli_word}")(phi=coeff, wires=base.wires)]
            return [qml.PauliRot(theta=coeff, pauli_word=pauli_word, wires=base.wires)]

        if isinstance(base, (Hamiltonian, Sum)):
            coeffs, ops = (
                base.terms() if isinstance(base, Hamiltonian) else ([1] * len(base), base.operands)
            )
            op_list = []

            for c, op in zip(coeffs, ops):
                c *= coeff
                if isinstance(op, SProd):
                    c *= op.scalar
                    op = op.base
                if qml.pauli.is_pauli_word(op) and math.real(c) == 0:
                    pauli_word = qml.pauli.pauli_word_to_string(op)
                    if pauli_word != "I" * op.num_wires:
                        c = 2j * c  # need to cancel the coefficients added by PauliRot
                        op_list.append(qml.PauliRot(theta=c, pauli_word=pauli_word, wires=op.wires))
                else:
                    op_list.append(qml.exp(op=op, coeff=c))
            return op_list * self.num_steps  # apply operators ``num_steps`` times

        raise DecompositionUndefinedError

    def matrix(self, wire_order=None):
        coeff_interface = math.get_interface(self.scalar)
        if coeff_interface == "autograd" and math.requires_grad(self.scalar):
            # math.expm is not differentiable with autograd
            # So we try to do a differentiable construction if possible
            #
            # This won't catch situations when the base matrix is autograd,
            # but at least this provides as much trainablility as possible
            try:
                eigvals = self.eigvals()
                eigvals_mat = (
                    math.stack(math.diag(e) for e in eigvals)
                    if qml.math.ndim(self.scalar) > 0
                    else math.diag(eigvals)
                )
                if len(self.diagonalizing_gates()) == 0:
                    return expand_matrix(eigvals_mat, wires=self.wires, wire_order=wire_order)
                diagonalizing_mat = qml.matrix(self.diagonalizing_gates, wire_order=self.wires)()
                mat = diagonalizing_mat.conj().T @ eigvals_mat @ diagonalizing_mat
                return expand_matrix(mat, wires=self.wires, wire_order=wire_order)
            except OperatorPropertyUndefined:
                warn(
                    f"The autograd matrix for {self} is not differentiable. "
                    "Use a different interface if you need backpropagation.",
                    UserWarning,
                )
        return super().matrix(wire_order=wire_order)

    @staticmethod
    def _matrix(scalar, mat):
        return math.expm(scalar * mat)

    # pylint: disable=arguments-differ
    def sparse_matrix(self, wire_order=None, format="csr"):
        if wire_order is not None:
            raise NotImplementedError("Wire order is not implemented for sparse_matrix")

        return sparse_expm(self.coeff * self.base.sparse_matrix().tocsc()).asformat(format)

    # pylint: disable=arguments-renamed,invalid-overridden-method
    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def eigvals(self):
        r"""Eigenvalues of the operator in the computational basis.

        .. math::

            c \mathbf{M} \mathbf{v} = c \lambda \mathbf{v}
            \quad \Longrightarrow \quad
            e^{c \mathbf{M}} \mathbf{v} = e^{c \lambda} \mathbf{v}

        >>> obs = Exp(qml.PauliX(0), 3)
        >>> qml.eigvals(obs)
        array([20.08553692,  0.04978707])
        >>> np.exp(3 * qml.eigvals(qml.PauliX(0)))
        tensor([20.08553692,  0.04978707], requires_grad=True)

        """
        base_eigvals = math.convert_like(self.base.eigvals(), self.coeff)
        base_eigvals = math.cast_like(base_eigvals, self.coeff)
        if qml.math.ndim(self.scalar) > 0:
            # exp coeff is broadcasted
            return qml.math.stack([qml.math.exp(c * base_eigvals) for c in self.coeff])
        return qml.math.exp(self.coeff * base_eigvals)

    def label(self, decimals=None, base_label=None, cache=None):
        coeff = (
            self.coeff if decimals is None else format(math.toarray(self.coeff), f".{decimals}f")
        )
        return base_label or f"Exp({coeff} {self.base.label(decimals=decimals, cache=cache)})"

    def pow(self, z):
        return Exp(self.base, self.coeff * z)

    def simplify(self):
        new_base = self.base.simplify()
        if isinstance(new_base, qml.ops.op_math.SProd):  # pylint: disable=no-member
            return Exp(new_base.base, self.coeff * new_base.scalar)
        return Exp(new_base, self.coeff)

    def generator(self):
        r"""Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U.generator()
          (0.5) [Y0]
        + (1.0) [Z0 X1]

        """
        if self.base.is_hermitian and not np.real(self.coeff):
            return self.base

        raise GeneratorUndefinedError(
            f"Exponential with coefficient {self.coeff} and base operator {self.base} does not appear to have a "
            f"generator. Consider using op.simplify() to simplify before finding the generator, or define the operator "
            f"in the form exp(ixG) through the Evolution class."
        )
