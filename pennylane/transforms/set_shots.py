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


def null_postprocessing(results):  # pragma: no cover
    """An empty post-processing function."""
    return results[0]


@transform
def set_shots(
    tape: QuantumScript, shots: Union[Shots, None, int, Sequence[Union[int, tuple[int, int]]]]
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Sets the shot(s) of a given tape"""
    if tape.shots != shots:
        tape = tape.copy(shots=shots)
    return (tape,), null_postprocessing
