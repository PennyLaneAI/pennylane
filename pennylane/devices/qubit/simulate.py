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
"""Simulate a quantum script."""

# pylint: disable=protected-access
import pennylane as qml
from pennylane.wires import Wires
from pennylane.measurements import StateMeasurement

from .initialize_state import create_initial_state
from .apply_operation import apply_operation


def measure_state_diagonalizing_gates(state, measurementprocess: StateMeasurement):
    """Measure ``state`` using ``measurementprocess`` if measurement process is state based
    and has an observable that provides diagonalizing gates.."""
    total_indices = len(state.shape)
    wires = Wires(range(total_indices))

    for op in measurementprocess.obs.diagonalizing_gates():
        state = apply_operation(op, state)

    return measurementprocess.process_state(qml.math.flatten(state), wires)


def measure(mp: StateMeasurement, state):
    """Measure ``mp`` on the ``state``."""
    if isinstance(mp, StateMeasurement):

        if mp.obs is None:
            # no need to apply diagonalizing gates
            total_indices = len(state.shape)
            wires = Wires(range(total_indices))
            return mp.process_state(qml.math.flatten(state), wires)

        if mp.obs.has_diagonalizing_gates:
            return measure_state_diagonalizing_gates(state, mp)

    raise NotImplementedError


def simulate(circuit: qml.tape.QuantumScript):
    """Simulate a single quantum script.

    This is an internal function that will be called by the to-be-created ``PythonDevice``.

    Args:
        circuit: The single circuit to simulate

    Returns:
        tuple(ndarray): The results of the simulation

    This function assummes that wire labels denote indices and all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    state = create_initial_state(circuit.wires, circuit._prep[0] if circuit._prep else None)

    for op in circuit._ops:
        state = apply_operation(op, state)

    return tuple(measure(mp, state) for mp in circuit.measurements)
