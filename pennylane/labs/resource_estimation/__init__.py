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
r"""
This module contains experimental features for
resource estimation.

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

.. currentmodule:: pennylane.labs.resource_estimation


Resource Estimation Base Classes:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~Resources
    ~ResourceOperator
    ~CompressedResourceOp
    ~GateCount
    ~ResourceConfig

Resource Estimation Functions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~estimate
    ~resource_rep

Qubit Management Classes:
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~QubitManager
    ~AllocWires
    ~FreeWires

Arithmetic Operators:
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceAdjoint
    ~ResourceChangeBasisOp
    ~ResourceControlled
    ~ResourcePow
    ~ResourceProd

Operators:
~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceGlobalPhase
    ~ResourceHadamard
    ~ResourceIdentity
    ~ResourceS
    ~ResourceT
    ~ResourceX
    ~ResourceY
    ~ResourceZ
    ~ResourceRX
    ~ResourceRY
    ~ResourceRZ
    ~ResourceRot
    ~ResourcePhaseShift
    ~ResourceSWAP
    ~ResourceCH
    ~ResourceCY
    ~ResourceCZ
    ~ResourceCSWAP
    ~ResourceCCZ
    ~ResourceCNOT
    ~ResourceToffoli
    ~ResourceMultiControlledX
    ~ResourceCRX
    ~ResourceCRY
    ~ResourceCRZ
    ~ResourceCRot
    ~ResourceControlledPhaseShift
    ~ResourceTempAND
    ~ResourceMultiRZ
    ~ResourcePauliRot
    ~ResourceIsingXX
    ~ResourceIsingYY
    ~ResourceIsingXY
    ~ResourceIsingZZ
    ~ResourcePSWAP
    ~ResourceSingleExcitation

Templates:
~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceOutOfPlaceSquare
    ~ResourcePhaseGradient
    ~ResourceOutMultiplier
    ~ResourceSemiAdder
    ~ResourceQPE
    ~ResourceIterativeQPE
    ~ResourceControlledSequence
    ~ResourceQFT
    ~ResourceAQFT
    ~ResourceBasisRotation
    ~ResourceSelect
    ~ResourceQROM
    ~ResourceSingleQubitComparator
    ~ResourceTwoQubitComparator
    ~ResourceIntegerComparator
    ~ResourceRegisterComparator
    ~ResourceSelectPauliRot
    ~ResourceQubitUnitary
    ~ResourceTrotterProduct
    ~ResourceTrotterCDF
    ~ResourceTrotterTHC
    ~ResourceTrotterVibrational
    ~ResourceTrotterVibronic
    ~ResourceQubitizeTHC
    ~ResourceSelectTHC

State Preparation:
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceMPSPrep
    ~ResourceQROMStatePreparation
    ~ResourceUniformStatePrep
    ~ResourceAliasSampling
    ~ResourcePrepTHC

Compact Hamiltonian Class:
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~CompactHamiltonian

"""

from .qubit_manager import AllocWires, FreeWires, QubitManager
from .resources_base import Resources
from .resource_operator import (
    CompressedResourceOp,
    ResourceOperator,
    ResourcesNotDefined,
    resource_rep,
    GateCount,
)
from .resource_mapping import map_to_resource_op
from .resource_tracking import (
    StandardGateSet,
    DefaultGateSet,
    ResourceConfig,
    estimate,
)
from .ops import (
    ResourceHadamard,
    ResourceS,
    ResourceX,
    ResourceY,
    ResourceZ,
    ResourceRX,
    ResourceRY,
    ResourceRZ,
    ResourceT,
    ResourcePhaseShift,
    ResourceGlobalPhase,
    ResourceRot,
    ResourceIdentity,
    ResourceSWAP,
    ResourceCH,
    ResourceCY,
    ResourceCZ,
    ResourceCSWAP,
    ResourceCCZ,
    ResourceCNOT,
    ResourceToffoli,
    ResourceMultiControlledX,
    ResourceCRX,
    ResourceCRY,
    ResourceCRZ,
    ResourceCRot,
    ResourceControlledPhaseShift,
    ResourceMultiRZ,
    ResourcePauliRot,
    ResourceIsingXX,
    ResourceIsingYY,
    ResourceIsingXY,
    ResourceIsingZZ,
    ResourcePSWAP,
    ResourceTempAND,
    ResourceSingleExcitation,
    ResourceAdjoint,
    ResourceControlled,
    ResourceProd,
    ResourceChangeBasisOp,
    ResourcePow,
)
from .templates import (
    ResourceOutOfPlaceSquare,
    ResourcePhaseGradient,
    ResourceOutMultiplier,
    ResourceSemiAdder,
    ResourceQFT,
    ResourceAQFT,
    ResourceBasisRotation,
    ResourceSelect,
    ResourceQROM,
    ResourceTwoQubitComparator,
    ResourceIntegerComparator,
    ResourceSingleQubitComparator,
    ResourceRegisterComparator,
    ResourceQubitUnitary,
    ResourceSelectPauliRot,
    ResourceTrotterProduct,
    ResourceTrotterCDF,
    ResourceTrotterTHC,
    CompactHamiltonian,
    ResourceTrotterVibrational,
    ResourceTrotterVibronic,
    ResourceQubitizeTHC,
    ResourceMPSPrep,
    ResourceQROMStatePreparation,
    ResourceUniformStatePrep,
    ResourceAliasSampling,
    ResourceSelectTHC,
    ResourcePrepTHC,
    ResourceQPE,
    ResourceControlledSequence,
    ResourceIterativeQPE,
)
