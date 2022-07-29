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

from scipy.linalg import expm
from scipy.sparse.linalg import expm as sparse_expm

import pennylane as qml
from pennylane.operation import expand_matrix
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


    """

    coeff = 1
    """The numerical coefficiant of the operator in the exponent."""

    def __init__(self, base=None, coeff=1, do_queue=True, id=None):
        self.coeff = coeff
        super().__init__(base, do_queue=do_queue, id=id)
        self._name = f"Exp({coeff} {base.name})"

    @property
    def data(self):
        return [[self.coeff], self.base.data]

    @data.setter
    def data(self, new_data):
        self.coeff = new_data[0][0]
        self.base.data = new_data[1:]

    @property
    def num_params(self):
        return self.base.num_params + 1

    def matrix(self, wire_order=None):
        mat = expm(self.coeff * qml.matrix(self.base))

        if wire_order is None or self.wires == Wires(wire_order):
            return mat

        return expand_matrix(mat, wires=self.wires, wire_order=wire_order)

    def sparse_matrix(self, wire_order=None):
        if wire_order is not None:
            raise NotImplementedError("Wire order is not implemented for sparse_matrix")

        base_smat = self.coeff * self.base.sparse_matrix()
        return sparse_expm(base_smat)

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def eigvals(self):
        return qml.math.exp(self.base.eigvals())

    def generator(self):
        if qml.math.iscomplex(self.coeff):
            return self.base
        return super().generator()
