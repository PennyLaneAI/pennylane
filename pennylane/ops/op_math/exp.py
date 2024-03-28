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
from typing import List
from warnings import warn

import numpy as np
from scipy.sparse.linalg import expm as sparse_expm

import pennylane as qml
from pennylane import math
from pennylane.math import expand_matrix
from pennylane.operation import (
    AnyWires,
    DecompositionUndefinedError,
    GeneratorUndefinedError,
    Operation,
    Operator,
    OperatorPropertyUndefined,
    Tensor,
)
from pennylane.wires import Wires

from .sprod import SProd
from .sum import Sum
from .linear_combination import LinearCombination
from .symbolicop import ScalarSymbolicOp
from ..qubit.hamiltonian import Hamiltonian


def exp(op, coeff=1, num_steps=None, id=None):
    """Take the exponential of an Operator times a coefficient.

    Args:
        base (~.operation.Operator): The Operator to be exponentiated
        coeff (float): a scalar coefficient of the operator
        num_steps (int): The number of steps used in the decomposition of the exponential operator,
            also known as the Trotter number. If this value is `None` and the Suzuki-Trotter
            decomposition is needed, an error will be raised.
        id (str): id for the Exp operator. Default is None.

    Returns:
       :class:`Exp`: An :class:`~.operation.Operator` representing an operator exponential.

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
    >>> op = qml.exp(qml.X(0), -0.5j * x)
    >>> qml.math.allclose(op.matrix(), qml.RX(x, wires=0).matrix())
    True

    This can even be used for more complicated generators:

    >>> t = qml.X(0) @ qml.X(1) + qml.Y(0) @ qml.Y(1)
    >>> isingxy = qml.exp(t, 0.25j * x)
    >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(x, wires=(0,1)).matrix())
    True

    If the coefficient is purely imaginary and the base operator is Hermitian, then
    the gate can be used in a circuit, though it may not be supported by the device and
    may not be differentiable.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit(x):
    ...     qml.exp(qml.X(0), -0.5j * x)
    ...     return qml.expval(qml.Z(0))
    >>> print(qml.draw(circuit)(1.23))
    0: ──Exp─┤  <Z>

    If the base operator is Hermitian and the coefficient is real, then the ``Exp`` operator
    can be measured as an observable:

    >>> obs = qml.exp(qml.Z(0), 3)
    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit():
    ...     return qml.expval(obs)
    >>> circuit()
    tensor(20.08553692, requires_grad=True)

    """
    return Exp(op, coeff, num_steps=num_steps, id=id)


class Exp(ScalarSymbolicOp, Operation):
    """A symbolic operator representing the exponential of a operator.

    Args:
        base (~.operation.Operator): The Operator to be exponentiated
        coeff=1 (Number): A scalar coefficient of the operator.
        num_steps (int): The number of steps used in the decomposition of the exponential operator,
            also known as the Trotter number. If this value is `None` and the Suzuki-Trotter
            decomposition is needed, an error will be raised.
        id (str): id for the Exp operator. Default is None.

    **Example**

    This symbolic operator can be used to make general rotation operators:

    >>> x = np.array(1.23)
    >>> op = Exp( qml.X(0), -0.5j * x)
    >>> qml.math.allclose(op.matrix(), qml.RX(x, wires=0).matrix())
    True

    This can even be used for more complicated generators:

    >>> t = qml.X(0) @ qml.X(1) + qml.Y(0) @ qml.Y(1)
    >>> isingxy = Exp(t, 0.25j * x)
    >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(x, wires=(0,1)).matrix())
    True

    If the coefficient is purely imaginary and the base operator is Hermitian, then
    the gate can be used in a circuit, though it may not be supported by the device and
    may not be differentiable.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit(x):
    ...     Exp(qml.X(0), -0.5j * x)
    ...     return qml.expval(qml.Z(0))
    >>> print(qml.draw(circuit)(1.23))
    0: ──Exp─┤  <Z>

    If the base operator is Hermitian and the coefficient is real, then the ``Exp`` operator
    can be measured as an observable:

    >>> obs = Exp(qml.Z(0), 3)
    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit():
    ...     return qml.expval(obs)
    >>> circuit()
    tensor(20.08553692, requires_grad=True)

    """

    control_wires = Wires([])
    _name = "Exp"

    def _flatten(self):
        return (self.base, self.data[0]), (self.num_steps,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], data[1], num_steps=metadata[0])

    # pylint: disable=too-many-arguments
    def __init__(self, base, coeff=1, num_steps=None, id=None):
        if not isinstance(base, Operator):
            raise TypeError(f"base is expected to be of type Operator, but received {type(base)}")
        super().__init__(base, scalar=coeff, id=id)
        self.grad_recipe = [None]
        self.num_steps = num_steps

        self.hyperparameters["num_steps"] = num_steps

    def __repr__(self):
        return (
            f"Exp({self.coeff} {self.base})"
            if self.base.arithmetic_depth > 0
            else f"Exp({self.coeff} {self.base.name})"
        )

    @property
    def hash(self):
        return hash((str(self.name), self.base.hash, str(self.coeff)))

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
        return "_ops"

    # pylint: disable=invalid-overridden-method, arguments-renamed
    @property
    def has_decomposition(self):
        # TODO: Support nested sums in method
        if isinstance(self.base, Tensor) and len(self.base.wires) != len(self.base.obs):
            return False
        base = self.base
        coeff = self.coeff
        if isinstance(base, SProd):
            coeff *= base.scalar
            base = base.base
        is_pauli_rot = qml.pauli.is_pauli_word(self.base) and math.real(self.coeff) == 0
        is_hamiltonian = isinstance(base, (Hamiltonian, LinearCombination))
        is_sum_of_pauli_words = isinstance(base, Sum) and all(
            qml.pauli.is_pauli_word(o) for o in base
        )
        return is_pauli_rot or is_hamiltonian or is_sum_of_pauli_words

    def decomposition(self):
        r"""Representation of the operator as a product of other operators. Decomposes into
        :class:`~.PauliRot` if the coefficient is imaginary and the base is a Pauli Word.

        .. math:: O = O_1 O_2 \dots O_n

        A ``DecompositionUndefinedError`` is raised if the coefficient is not imaginary or the base
        is not a Pauli Word.

        Returns:
            list[PauliRot]: decomposition of the operator
        """
        with qml.QueuingManager.stop_recording():
            d = self._recursive_decomposition(self.base, self.coeff)

        if qml.QueuingManager.recording():
            for op in d:
                qml.apply(op)

        return d

    # pylint:disable=too-many-branches
    def _recursive_decomposition(self, base: Operator, coeff: complex):
        """Decompose the exponential of ``base`` multiplied by ``coeff``.

        Args:
            base (Operator): exponentiated operator
            coeff (complex): coefficient multiplying the exponentiated operator

        Returns:
            List[Operator]: decomposition
        """
        if isinstance(base, Tensor) and len(base.wires) != len(base.obs):
            raise DecompositionUndefinedError(
                "Unable to determine if the exponential has a decomposition "
                "when the base operator is a Tensor object with overlapping wires. "
                f"Received base {base}."
            )

        # Change base to `Sum`/`Prod`
        if isinstance(base, (Hamiltonian, LinearCombination)):
            base = qml.dot(base.coeffs, base.ops)
        elif isinstance(base, Tensor):
            base = qml.prod(*base.obs)

        if isinstance(base, SProd):
            return self._recursive_decomposition(base.base, base.scalar * coeff)

        if self.num_steps is not None and isinstance(base, Sum):
            # Apply trotter decomposition
            coeffs, ops = [1] * len(base), base.operands
            coeffs = [c * coeff for c in coeffs]
            return self._trotter_decomposition(ops, coeffs)

        # Store operator classes with generators
        has_generator_types = []
        has_generator_types_anywires = []
        for op_name in qml.ops.qubit.__all__:  # pylint:disable=no-member
            op_class = getattr(qml.ops.qubit, op_name)  # pylint:disable=no-member
            if op_class.has_generator:
                if op_class.num_wires == AnyWires:
                    has_generator_types_anywires.append(op_class)
                elif op_class.num_wires == len(base.wires):
                    has_generator_types.append(op_class)
        # Ensure op_class.num_wires == base.num_wires before op_class.num_wires == AnyWires
        has_generator_types.extend(has_generator_types_anywires)

        for op_class in has_generator_types:
            # PauliRot and PCPhase have different positional args
            if op_class not in {qml.PauliRot, qml.PCPhase}:
                g, c = qml.generator(op_class)(coeff, base.wires)
                # Some generators are not wire-ordered (e.g. OrbitalRotation)
                mapped_wires_g = qml.map_wires(g, dict(zip(g.wires, base.wires)))

                if qml.equal(mapped_wires_g, base) and math.real(coeff) == 0:
                    coeff = math.real(
                        -1j / c * coeff
                    )  # cancel the coefficients added by the generator
                    return [op_class(coeff, g.wires)]

                # could have absorbed the coefficient.
                simplified_g = qml.simplify(qml.s_prod(c, mapped_wires_g))

                if qml.equal(simplified_g, base) and math.real(coeff) == 0:
                    coeff = math.real(-1j * coeff)  # cancel the coefficients added by the generator
                    return [op_class(coeff, g.wires)]

        if qml.pauli.is_pauli_word(base) and math.real(coeff) == 0:
            # Check if the exponential can be decomposed into a PauliRot gate
            return self._pauli_rot_decomposition(base, coeff)

        error_msg = f"The decomposition of the {self} operator is not defined. "

        if not self.num_steps:  # if num_steps was not set
            error_msg += (
                "Please set a value to ``num_steps`` when instantiating the ``Exp`` operator "
                "if a Suzuki-Trotter decomposition is required. "
            )

        if math.real(self.coeff) != 0 and self.base.is_hermitian:
            error_msg += (
                "Decomposition is not defined for real coefficients of hermitian operators."
            )

        raise DecompositionUndefinedError(error_msg)

    @staticmethod
    def _pauli_rot_decomposition(base: Operator, coeff: complex):
        """Decomposes the exponential of a Pauli word into a PauliRot.

        Args:
            base (Operator): exponentiated operator
            coeff (complex): coefficient multiplying the exponentiated operator

        Returns:
            List[Operator]: list containing the PauliRot operator
        """
        coeff = math.real(
            2j * coeff
        )  # need to cancel the coefficients added by PauliRot and Ising gates
        pauli_word = qml.pauli.pauli_word_to_string(base)
        if pauli_word == "I" * base.num_wires:
            return []
        return [qml.PauliRot(theta=coeff, pauli_word=pauli_word, wires=base.wires)]

    def _trotter_decomposition(self, ops: List[Operator], coeffs: List[complex]):
        """Uses the Suzuki-Trotter approximation to decompose the exponential of the linear
        combination of ``coeffs`` and ``ops``.

        Args:
            ops (List[Operator]): list of operators of the linear combination
            coeffs (List[complex]): list of coefficients of the linear combination

        Raises:
            ValueError: if the Trotter number (``num_steps``) is not defined
            DecompositionUndefinedError: if the linear combination contains operators that are not
                Pauli words

        Returns:
            List[Operator]: a list of operators containing the decomposition
        """
        op_list = []
        for c, op in zip(coeffs, ops):
            c /= self.num_steps  # divide by trotter number
            if isinstance(op, SProd):
                c *= op.scalar
                op = op.base
            op_list.extend(self._recursive_decomposition(op, c))

        return op_list * self.num_steps  # apply operators ``num_steps`` times

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
                    math.stack([math.diag(e) for e in eigvals])
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

        >>> obs = Exp(qml.X(0), 3)
        >>> qml.eigvals(obs)
        array([20.08553692,  0.04978707])
        >>> np.exp(3 * qml.eigvals(qml.X(0)))
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

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_generator(self):
        return self.base.is_hermitian and not np.real(self.coeff)

    def generator(self):
        r"""Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U.generator()
          0.5 * Y(0) + Z(0) @ X(1)

        """
        if self.base.is_hermitian and not np.real(self.coeff):
            return self.base

        raise GeneratorUndefinedError(
            f"Exponential with coefficient {self.coeff} and base operator {self.base} does not appear to have a "
            f"generator. Consider using op.simplify() to simplify before finding the generator, or define the operator "
            f"in the form exp(-ixG) through the Evolution class."
        )
