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
"""module docstring"""

import numpy as np

import pennylane as qml
from pennylane.tape import QuantumScript

from .apply_operation import apply_operation
from .python_execute import create_zeroes_state


def operation_derivative(operation: qml.operation.Operator) -> qml.operation.Operator:
    """This code is copied from operation.py.  It's moved here to reduce
    spagetti dependencies over an extremely simple function."""
    gen = qml.generator(operation, format="observable")
    return 1j * qml.prod(gen, operation)


def adjoint_diff_gradient(qscript: QuantumScript) -> tuple:
    """Calculate the gradient of a Quantum Script using adjoint differentiation with the provided
    simulator.

    Args:
        qscript (QuantumScript): the quantum script to take the gradient of

    """
    state = create_zeroes_state(qscript.num_wires)
    for op in qscript.operations:
        state = apply_operation(op, state)
    bra = apply_operation(qscript.measurements[0].obs, state)
    ket = state

    grads = []
    for op in reversed(qscript.operations):
        adj_op = qml.adjoint(op)
        ket = apply_operation(adj_op, ket)

        if op.num_params != 0:
            dU = operation_derivative(op)
            ket_temp = apply_operation(dU, ket)
            dM = 2 * np.real(np.vdot(bra, ket_temp))
            grads.append(dM)

        bra = apply_operation(adj_op, bra)

    grads = grads[::-1]
    return tuple(grads)
