# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulate a quantum script for a qubit mixed state device."""
# pylint: skip-file
# black: skip-file
import pennylane as qml
from pennylane.typing import Result


def simulate(  # pylint: disable=too-many-arguments
    circuit: qml.tape.QuantumScript,
    rng=None,
    prng_key=None,
    debugger=None,
    interface=None,
    readout_errors=None,
) -> Result:
    raise NotImplementedError
