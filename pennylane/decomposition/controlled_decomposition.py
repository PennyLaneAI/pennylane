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

"""This module contains special logic of decomposing controlled operations."""

import functools
import pennylane as qml


@functools.lru_cache()
def _special_ctrl_ops():
    """Gets a list of special operations with custom controlled versions."""

    ops_with_custom_ctrl_ops = {
        (qml.PauliZ, 1): qml.CZ,
        (qml.PauliZ, 2): qml.CCZ,
        (qml.PauliY, 1): qml.CY,
        (qml.CZ, 1): qml.CCZ,
        (qml.SWAP, 1): qml.CSWAP,
        (qml.Hadamard, 1): qml.CH,
        (qml.RX, 1): qml.CRX,
        (qml.RY, 1): qml.CRY,
        (qml.RZ, 1): qml.CRZ,
        (qml.Rot, 1): qml.CRot,
        (qml.PhaseShift, 1): qml.ControlledPhaseShift,
    }
    return ops_with_custom_ctrl_ops
