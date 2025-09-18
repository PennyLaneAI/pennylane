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
    ResourceHadamard,
    ResourceS,
    ResourceT,
    ResourceX,
    ResourceY,
    ResourceZ,
    ResourceSWAP,
)
from .parametric_ops_single_qubit import (
    ResourcePhaseShift,
    ResourceRot,
    ResourceRX,
    ResourceRY,
    ResourceRZ,
)
from .parametric_ops_multi_qubit import (
    ResourceMultiRZ,
    ResourcePauliRot,
    ResourceIsingXX,
    ResourceIsingYY,
    ResourceIsingXY,
    ResourceIsingZZ,
    ResourcePSWAP,
)
from .qchem_ops import ResourceSingleExcitation
