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

r"""This module contains tools dedicated to logical resource estimation."""

from .wires_manager import Allocate, Deallocate, WireResourceManager

from .resources_base import Resources

from .resource_config import ResourceConfig

from .resource_operator import (
    ResourceOperator,
    CompressedResourceOp,
    GateCount,
    resource_rep,
)

from .estimate import estimate

from .ops.identity import Identity, GlobalPhase

from .ops.qubit import (
    X,
    Y,
    Z,
    SWAP,
    Hadamard,
    S,
    T,
    PhaseShift,
    RX,
    RY,
    RZ,
    Rot,
    MultiRZ,
    PauliRot,
    SingleExcitation,
    QubitUnitary,
)

from .ops.op_math import (
    CCZ,
    CH,
    CNOT,
    ControlledPhaseShift,
    CRot,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CY,
    CZ,
    MultiControlledX,
    TemporaryAND,
    Toffoli,
    Adjoint,
    Controlled,
    Pow,
    Prod,
    ChangeOpBasis,
)

from .templates import (
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
    PrepTHC,
    QubitizeTHC,
    SelectTHC,
    TrotterCDF,
    TrotterProduct,
    TrotterTHC,
    TrotterVibrational,
    TrotterVibronic,
    MPSPrep,
    QROMStatePreparation,
    UniformStatePrep,
    AliasSampling,
    IntegerComparator,
    SingleQubitComparator,
    TwoQubitComparator,
    RegisterComparator,
)

from .compact_hamiltonian import (
    CDFHamiltonian,
    THCHamiltonian,
    VibronicHamiltonian,
    VibrationalHamiltonian,
)
