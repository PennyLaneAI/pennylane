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

from .subroutines import (
    ResourceOutOfPlaceSquare,
    ResourcePhaseGradient,
    ResourceOutMultiplier,
    ResourceSemiAdder,
    ResourceQFT,
    ResourceAQFT,
    ResourceBasisRotation,
    ResourceSelect,
    ResourceQROM,
    ResourceSelectPauliRot,
    ResourceQubitUnitary,
    ResourceControlledSequence,
    ResourceQPE,
    ResourceIterativeQPE,
)
from .trotter import (
    ResourceTrotterProduct,
    ResourceTrotterCDF,
    ResourceTrotterTHC,
    ResourceTrotterVibrational,
    ResourceTrotterVibronic,
)

from .stateprep import (
    ResourceMPSPrep,
    ResourceQROMStatePreparation,
)

from .qubitize import ResourceQubitizeTHC
from .compact_hamiltonian import CompactHamiltonian
from .comparators import (
    ResourceIntegerComparator,
    ResourceSingleQubitComparator,
    ResourceTwoQubitComparator,
    ResourceRegisterComparator,
)
from .stateprep import ResourceUniformStatePrep, ResourceAliasSampling, ResourcePrepTHC
from .select import ResourceSelectTHC
