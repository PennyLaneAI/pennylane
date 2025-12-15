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
"""Test for gridsynth (not implemented in tape)."""


import pytest

import pennylane as qml
from pennylane.transforms.decompositions import gridsynth


def test_not_implemented():
    """Test that NotImplementedError is raised when trying to use gridsynth on tape."""
    with pytest.raises(
        NotImplementedError,
        match=r"This transform pass \(gridsynth\) has no tape based implementation. It can only be applied to QJIT-ed workflows after all purely tape transforms. For a tape transform, please use qml.transforms.clifford_t_decomposition.",
    ):
        with qml.tape.QuantumTape() as tape:
            qml.RZ(0.5, wires=0)
            qml.PhaseShift(0.2, wires=0)

        gridsynth(tape)
