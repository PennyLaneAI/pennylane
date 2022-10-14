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
This submodule defines the symbolic operation that stands for an exponential of an operator.
"""
from warnings import warn

from scipy.sparse.linalg import expm as sparse_expm

import pennylane as qml
from pennylane import math
from pennylane import numpy as np
from pennylane.operation import OperatorPropertyUndefined, expand_matrix
from pennylane.wires import Wires

from .symbolicop import SymbolicOp

COEFF_PRECISION = 10
"""Coeff precision used to compare Exp operators."""


def exp(op, coeff=1, id=None):
    """Take the exponential of an Operator times a coefficient.

    Args:
        base (~.operation.Operator): The Operator to be exponentiated
        coeff=1 (Number): A scalar coefficient of the operator.

    Returns:
       :class:`Exp`: A :class`~.operation.Operator` representing an operator exponential.

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
    return Exp(op, coeff, id=id)


class Exp(SymbolicOp):
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

    coeff = 1
    """The numerical coefficient of the operator in the exponent."""

    control_wires = Wires([])

    def __init__(self, base=None, coeff=1, do_queue=True, id=None):
        self.coeff = coeff
        super().__init__(base, do_queue=do_queue, id=id)
        self._name = "Exp"

    def __repr__(self):
        return (
            f"Exp({self.coeff} {self.base})"
            if self.base.arithmetic_depth > 0
            else f"Exp({self.coeff} {self.base.name})"
        )

    @property
    def data(self):
        return [[self.coeff], self.base.data]

    @data.setter
    def data(self, new_data):
        self.coeff = new_data[0][0]
        self.base.data = new_data[1]

    @property
    def num_params(self):
        return self.base.num_params + 1

    @property
    def is_hermitian(self):
        return self.base.is_hermitian and math.imag(self.coeff) == 0

    @property
    def _queue_category(self):
        if self.base.is_hermitian and math.real(self.coeff) == 0:
            return "_ops"
        return None

    def matrix(self, wire_order=None):

        coeff_interface = math.get_interface(self.coeff)
        if coeff_interface == "autograd" and math.requires_grad(self.coeff):
            # math.expm is not differentiable with autograd
            # So we try to do a differentiable construction if possible
            #
            # This won't catch situations when the base matrix is autograd,
            # but at least this provides as much trainablility as possible
            try:
                if len(self.diagonalizing_gates()) == 0:
                    eigvals_mat = math.diag(self.eigvals())
                    return expand_matrix(eigvals_mat, wires=self.wires, wire_order=wire_order)
                diagonalizing_mat = qml.matrix(self.diagonalizing_gates, wire_order=self.wires)()
                eigvals_mat = math.diag(self.eigvals())
                mat = diagonalizing_mat.conj().T @ eigvals_mat @ diagonalizing_mat
                return expand_matrix(mat, wires=self.wires, wire_order=wire_order)
            except OperatorPropertyUndefined:
                warn(
                    f"The autograd matrix for {self} is not differentiable. "
                    "Use a different interface if you need backpropagation.",
                    UserWarning,
                )
        base_mat = (
            qml.matrix(self.base) if isinstance(self.base, qml.Hamiltonian) else self.base.matrix()
        )
        if coeff_interface == "torch":
            # other wise get `RuntimeError: Can't call numpy() on Tensor that requires grad.`
            base_mat = math.convert_like(base_mat, self.coeff)
        mat = math.expm(self.coeff * base_mat)

        return expand_matrix(mat, wires=self.wires, wire_order=wire_order)

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
        return qml.math.exp(self.coeff * base_eigvals)

    def label(self, decimals=None, base_label=None, cache=None):
        coeff = (
            self.coeff if decimals is None else format(math.toarray(self.coeff), f".{decimals}f")
        )
        return base_label or f"Exp({coeff} {self.base.label(decimals=decimals, cache=cache)})"

    def pow(self, z):
        return Exp(self.base, self.coeff * z)

    def simplify(self):
        if isinstance(self.base, qml.ops.op_math.SProd):  # pylint: disable=no-member
            return Exp(self.base.base.simplify(), self.coeff * self.base.scalar)
        return Exp(self.base.simplify(), self.coeff)

    @property
    def hash(self):
        # We cast the self.coeff to numpy because the other interfaces might have
        # different string representations
        interface = math.get_interface(self.coeff)
        if interface == "torch":
            # Can't cast torch tensor to numpy when requires_grad = True --> Use .detach()
            coeff = math.convert_like(self.coeff.detach(), np.array(1.0))
        elif interface == "jax":
            # Can't cast jax Traced arrays to numpy --> use jax instead
            coeff = self.coeff
        else:
            coeff = math.convert_like(self.coeff, np.array(1.0))
        # We use the string of the euler representation to avoid having different hashes
        # for equal complex values: str(-3j) = '(-0-3j)' // str(0-3j) = '-3j'
        scalar_str = str(math.round(math.abs(coeff), COEFF_PRECISION))
        angle_str = str(math.round(math.angle(coeff), COEFF_PRECISION))
        euler_str = scalar_str + angle_str
        return hash((super().hash, euler_str))
