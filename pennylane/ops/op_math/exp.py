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
This submodule defines the symbolic operation that stands for an operator raised to a power.
"""
from scipy.sparse.linalg import expm as sparse_expm

import pennylane as qml
from pennylane import math
from pennylane.operation import expand_matrix, Tensor
from pennylane.wires import Wires

from .symbolicop import SymbolicOp


class Exp(SymbolicOp):
    """A symbolic operator representating the exponential of a operator.

    Args:
        base (~.operation.Operator)
        coeff=1

    **Example**

    This symbolic operator can be used to make general rotation operators:

    >>> x = np.array(1.23)
    >>> op = Exp( qml.PauliX(0), -0.5j * x)
    >>> qml.math.allclose(op.matrix(), qml.RX(x, wires=0).matrix())
    True

    This can even be used for more complicated generators:

    >>> t = qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1)
    >>> isingxy = Exp(t, 0.25j)
    >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(1, wires=(0,1)).matrix())
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
            if isinstance(self.base, Tensor)
            else f"Exp({self.coeff} {self.base.name})"
        )

    @property
    def data(self):
        return [self.coeff, self.base.data]

    @data.setter
    def data(self, new_data):
        self.coeff = new_data[0]
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
        mat = math.expm(self.coeff * qml.matrix(self.base))

        if wire_order is None or self.wires == Wires(wire_order):
            return mat

        return expand_matrix(mat, wires=self.wires, wire_order=wire_order)

    # pylint: disable=arguments-differ
    def sparse_matrix(self, wire_order=None, format="csr"):
        if wire_order is not None:
            raise NotImplementedError("Wire order is not implemented for sparse_matrix")

        return sparse_expm(self.coeff * self.base.sparse_matrix().tocsc()).asformat(format)

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def eigvals(self):
        r"""Eigenvalues of the operator in the computational basis.

        If:

        .. math:: c O \vec{v} = c \lambda \vec{v}

        then we can determine the eigenvalues of the exponential:

        .. math:: e^{c O} \vec{v} = e^{c \lambda} \vec{v}

        >>> obs = Exp(qml.PauliX(0), 3)
        >>> qml.eigvals(obs)
        array([20.08553692,  0.04978707])
        >>> np.exp(3 * qml.eigvals(qml.PauliX(0)))
        tensor([20.08553692,  0.04978707], requires_grad=True)

        """
        return qml.math.exp(self.coeff * self.base.eigvals())

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "Exp"

    def pow(self, z):
        return Exp(self.base, self.coeff * z)
