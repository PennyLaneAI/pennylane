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
from pennylane.measurements import StateMeasurement, MeasurementProcess
from pennylane.typing import TensorLike

from .initialize_state import create_initial_state
from .apply_operation import apply_operation


def measure_state_diagonalizing_gates(
    measurementprocess: StateMeasurement, state: TensorLike
) -> TensorLike:
    """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state
        state (TensorLike): state to apply the measurement to

    Returns:
        TensorLike: the result of the measurement
    """
    total_indices = len(state.shape)
    wires = Wires(range(total_indices))

    for op in measurementprocess.obs.diagonalizing_gates():
        state = apply_operation(op, state)

    return measurementprocess.process_state(qml.math.flatten(state), wires)


def measure(measurementprocess: MeasurementProcess, state: TensorLike) -> TensorLike:
    """Apply a measurement process to a state.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (TensorLike): the state to measure

    Returns:
        Tensorlike: the result of the measurement
    """
    if isinstance(measurementprocess, StateMeasurement):
        if measurementprocess.obs is None:
            # no need to apply diagonalizing gates
            total_indices = len(state.shape)
            wires = Wires(range(total_indices))
            return measurementprocess.process_state(qml.math.flatten(state), wires)

        if measurementprocess.obs.has_diagonalizing_gates:
            return measure_state_diagonalizing_gates(measurementprocess, state)

    raise NotImplementedError


def simulate(circuit: qml.tape.QuantumScript) -> tuple:
    """Simulate a single quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    It does currently not support sampling or observables without diagonalizing gates.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    if set(circuit.wires) != set(range(circuit.num_wires)):
        wire_map = {w: i for i, w in enumerate(circuit.wires)}
        circuit = qml.map_wires(circuit, wire_map)

    state = create_initial_state(circuit.wires, circuit._prep[0] if circuit._prep else None)

    for op in circuit._ops:
        state = apply_operation(op, state)

    return tuple(measure(mp, state) for mp in circuit.measurements)
