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
r"""This module contains experimental resource estimation functionality."""

from .non_parametric_ops import (
    Hadamard,
    S,
    T,
    X,
    Y,
    Z,
    SWAP,
)
from .parametric_ops_single_qubit import (
    PhaseShift,
    RX,
    RY,
    RZ,
    Rot,
)

from .parametric_ops_multi_qubit import MultiRZ, PauliRot

from .qchem_ops import SingleExcitation

from .matrix_ops import QubitUnitary
