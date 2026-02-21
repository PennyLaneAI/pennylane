# Copyright 2025-2026 Xanadu Quantum Technologies Inc.

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

from .arithmetic import (
    PhaseAdder,
    Adder,
    OutAdder,
    SemiAdder,
    Multiplier,
    OutMultiplier,
    ClassicalOutMultiplier,
    ModExp,
    OutOfPlaceSquare,
)

from .subroutines import (
    IQP,
    HybridQRAM,
    SelectOnlyQRAM,
    BBQRAM,
    PhaseGradient,
    QFT,
    AQFT,
    BasisRotation,
    Select,
    QROM,
    SelectPauliRot,
    ControlledSequence,
    QPE,
    IterativeQPE,
    UnaryIterationQPE,
    Reflection,
    Qubitization,
)

from .trotter import (
    TrotterCDF,
    TrotterProduct,
    TrotterTHC,
    TrotterVibrational,
    TrotterVibronic,
    TrotterPauli,
)

from .stateprep import (
    MPSPrep,
    QROMStatePreparation,
    UniformStatePrep,
    AliasSampling,
    PrepTHC,
    BasisState,
)

from .embeddings import BasisEmbedding

from .comparators import (
    IntegerComparator,
    SingleQubitComparator,
    TwoQubitComparator,
    RegisterComparator,
)
from .qubitize import QubitizeTHC
from .select import SelectTHC, SelectPauli
from .qsp import GQSP, GQSPTimeEvolution, QSP, QSVT
