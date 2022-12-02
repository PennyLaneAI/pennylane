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

from pennylane import adjoint, matrix, generator
from pennylane.tape import QuantumScript

from .numpy_simulator import PlainNumpySimulator


def operation_derivative(operation):
    """This code is copied from operation.py.  It's moved here to reduce
    spagetti dependencies over an extremely simple function."""
    gen = matrix(generator(operation, format="observable"), wire_order=operation.wires)
    return 1j * gen @ operation.matrix()


def adjoint_diff_gradient(qscript: QuantumScript, sim: PlainNumpySimulator) -> tuple:
    """Calculate the gradient of a Quantum Script using adjoint differentiation with the provided
    simulator.

    Args:
        qscript (QuantumScript): the quantum script to take the gradient of
        sim (PlainNumpySimulator): the simulator to perform the computation with

    """
    state = sim.create_zeroes_state(qscript.num_wires)
    for op in qscript.operations:
        state = sim.apply_operation(state, op)
    bra = sim.apply_operation(state, qscript.measurements[0].obs)
    ket = state

    grads = []
    for op in reversed(qscript.operations):
        adj_op = adjoint(op)
        ket = sim.apply_operation(ket, adj_op)

        if op.num_params != 0:
            dU = operation_derivative(op)
            ket_temp = sim.apply_matrix(ket, dU, op.wires)
            dM = 2 * np.real(np.vdot(bra, ket_temp))
            grads.append(dM)

        bra = sim.apply_operation(bra, adj_op)

    grads = grads[::-1]
    return tuple(grads)
