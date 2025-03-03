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
r"""This module contains resource operators for PennyLane Operators"""

from .identity import (
    ResourceGlobalPhase,
    ResourceIdentity,
)

from .qubit import (
    ResourceDoubleExcitation,
    ResourceDoubleExcitationMinus,
    ResourceDoubleExcitationPlus,
    ResourceFermionicSWAP,
    ResourceHadamard,
    ResourceIsingXX,
    ResourceIsingXY,
    ResourceIsingYY,
    ResourceIsingZZ,
    ResourceMultiRZ,
    ResourceOrbitalRotation,
    ResourcePauliRot,
    ResourcePhaseShift,
    ResourcePSWAP,
    ResourceRot,
    ResourceRX,
    ResourceRY,
    ResourceRZ,
    ResourceS,
    ResourceSingleExcitation,
    ResourceSingleExcitationMinus,
    ResourceSingleExcitationPlus,
    ResourceSWAP,
    ResourceT,
    ResourceX,
    ResourceY,
    ResourceZ,
)

from .op_math import (
    ResourceAdjoint,
    ResourceCY,
    ResourceCH,
    ResourceCZ,
    ResourceCSWAP,
    ResourceCCZ,
    ResourceCRot,
    ResourceCRX,
    ResourceCRY,
    ResourceCRZ,
    ResourceExp,
    ResourceToffoli,
    ResourceMultiControlledX,
    ResourceCNOT,
    ResourceControlled,
    ResourceControlledPhaseShift,
    ResourcePow,
    ResourceProd,
)
