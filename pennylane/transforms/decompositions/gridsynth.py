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
"""Alias transform function for the Ross-Selinger decomposition (GridSynth) for qjit + capture."""

from pennylane.transforms.core import transform


@transform
def gridsynth(tape, *, epsilon, ppr_basis):
    r"""Decomposes RZ and PhaseShift gates into Clifford+T basis or PPR basis.

    This transform requires QJIT and capture to be enabled. This is a wrapper for the catalyst gridsynth transform pass.

    Args:
        tape (QNode): A quantum circuit.
        epsilon (float): The maximum error for the discretization.
        ppr_basis (bool): If True, decompose into the PPR basis. If False, decompose into the Clifford+T basis.

    """

    raise NotImplementedError(
        "This transform pass (gridsynth) is only implemented when using capture and QJIT. Otherwise, please use qml.transforms.clifford_t_decomposition."
    )
