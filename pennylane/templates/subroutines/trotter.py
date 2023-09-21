# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains templates for Suzuki-Trotter approximation based subroutines.
"""
import copy

import pennylane as qml

# from pennylane.math ...
from pennylane import numpy as np
from pennylane.operation import Operation
from pennylane.ops import Sum


def _scalar(order):
    """Assumes that order is an even integer > 2"""
    root = 1 / (order - 1)
    return (4 - 4**root) ** -1


def _recursive_op(x, order, ops):
    """Generate a list of operators."""
    if order == 1:
        return [qml.exp(op, x * 1j) for op in ops]

    if order == 2:
        return [qml.exp(op, x * 0.5j) for op in (ops + ops[::-1])]

    if order > 2 and order % 2 == 0:
        scalar_1 = _scalar(order)
        scalar_2 = 1 - 4 * scalar_1

        ops_lst_1 = _recursive_op(scalar_1 * x, order - 2, ops)
        ops_lst_2 = _recursive_op(scalar_2 * x, order - 2, ops)

        return (
            copy.deepcopy(ops_lst_1)
            + copy.deepcopy(ops_lst_1)
            + ops_lst_2
            + copy.deepcopy(ops_lst_1)
            + ops_lst_1
        )

    raise ValueError(f"The order of a Trotter Product must be 1 or an even integer, got {order}.")


class TrotterProduct(Operation):
    """Representing the Suzuki-Trotter product approximation"""

    def __init__(self, hamiltonian, time, n=1, order=1, check_hermitian=True, id=None):
        """Init method for the TrotterProduct class"""

        if isinstance(hamiltonian, qml.Hamiltonian):
            coeffs, ops = hamiltonian.terms()
            hamiltonian = qml.dot(coeffs, ops)

        if not isinstance(hamiltonian, Sum):
            raise ValueError(f"The given operator must be a PennyLane ~.Hamiltonian or ~.Sum got {hamiltonian}")

        if check_hermitian:
            pass

        self._hyperparameters = {"num_steps": n, "order": order, "base": hamiltonian}
        wires = hamiltonian.wires
        super().__init__(time, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(*args, **kwargs):
        time = args[0]
        n = kwargs["num_steps"]
        order = kwargs["order"]
        ops = kwargs["base"].operands

        with qml.QueuingManager.stop_recording():
            decomp = _recursive_op(time, order, ops)

        return decomp
