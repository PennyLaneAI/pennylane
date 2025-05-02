# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains the set_shots transform, which sets the number of shots for a given tape.
"""
from typing import Sequence, Union

from pennylane.measurements import Shots
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn


def null_postprocessing(results):
    """An empty post-processing function."""
    return results[0]


@transform
def set_shots(
    tape: QuantumScript, shots: Union[Shots, None, int, Sequence[Union[int, tuple[int, int]]]]
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Transform function to set or override the shots execution configuration
    for a quantum circuit.

    Args:
        tape (QuantumScript): The quantum circuit to be modified.
        shots (None or int or Sequence[int] or Sequence[tuple[int, int]] or pennylane.shots.Shots): The
            number of shots or shot execution configuration to apply to the circuit.
            This specification will override any shots value previously associated
            with the circuit or QNode during execution.
            Accepted values:
            * ``None``: Analytic execution (exact results).
            * ``int``: A single integer specifying the total number of shots.
            * ``Sequence[int]``: A sequence of integers defining a shot vector.
            * ``Sequence[tuple[int, int]]``: A sequence of tuples, each ``(shots, copies)``,
              defining a shot vector with shot batching.
            * ``pennylane.shots.Shots``: A pre-constructed ``Shots`` object.

    Returns:
        tuple[List[QuantumScript], function]: The transformed circuit as a batch of tapes and a
        post-processing function, as described in :func:`qml.transform <pennylane.transform>`. The output
        tape(s) will have their ``shots`` attribute set to the value provided in the ``shots`` argument.

    """
    if tape.shots != Shots(shots):
        tape = tape.copy(shots=shots)
    return (tape,), null_postprocessing
