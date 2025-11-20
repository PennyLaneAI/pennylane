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
r"""This module contains resource templates."""

from .subroutines import (
    OutOfPlaceSquare,
    PhaseGradient,
    OutMultiplier,
    SemiAdder,
    QFT,
    AQFT,
    BasisRotation,
    Select,
    QROM,
    SelectPauliRot,
    ControlledSequence,
    QPE,
    IterativeQPE,
)

from .trotter import TrotterCDF, TrotterProduct, TrotterTHC, TrotterVibrational, TrotterVibronic

from .stateprep import MPSPrep, QROMStatePreparation, UniformStatePrep, AliasSampling, PrepTHC

from .comparators import (
    IntegerComparator,
    SingleQubitComparator,
    TwoQubitComparator,
    RegisterComparator,
)
from .qubitize import QubitizeTHC
from .select import SelectTHC
