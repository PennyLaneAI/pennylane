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
This module uses python functions to execute a quantum script via simulation.
"""
import numpy as np

from pennylane.measurements import MeasurementProcess, StateMeasurement
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

from .apply_operation import apply_operation


def create_zeroes_state(num_indices: int, dtype=np.complex128) -> np.ndarray:
    """Create a zeroes state with ``num_indices`` wires and of type ``dtype``."""
    state = np.zeros(2**num_indices, dtype=dtype)
    state[0] = 1
    state.shape = [2] * num_indices
    return state


def measure_state(state: np.ndarray, measurementprocess: MeasurementProcess):
    """Measure ``state`` using ``measurementprocess``."""
    if isinstance(measurementprocess, StateMeasurement):
        total_indices = len(state.shape)
        wires = Wires(range(total_indices))
        if (obs := measurementprocess.obs) is not None and obs.has_diagonalizing_gates:
            for op in obs.diagonalizing_gates():
                state = apply_operation(op, state)
        return measurementprocess.process_state(state.flatten(), wires)
    return state


def python_execute(qs: QuantumScript):
    """Execute a quantum script using python simulation methods.

    Args:
        qs (QuantumScript)

    Returns:
        tuple(ndarray)

    """
    state = create_zeroes_state(len(qs.wires))

    for op in qs.operations:
        state = apply_operation(op, state)

    measurements = tuple(measure_state(state, m) for m in qs.measurements)
    return measurements[0] if len(measurements) == 1 else measurements
