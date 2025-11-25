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
    r"""A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""

    raise NotImplementedError(
        "This transform pass (gridsynth) is only implemented when using capture and QJIT. Otherwise, please use qml.transforms.clifford_t_decomposition."
    )
