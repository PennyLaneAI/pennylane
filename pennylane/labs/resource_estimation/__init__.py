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

Resource Estimation Functions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~estimate_resources
    ~resource_rep
    ~set_decomp
    ~set_adj_decomp
    ~set_ctrl_decomp
    ~set_pow_decomp

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

    ~ResourceTrotterCDF
    ~ResourceTrotterTHC
    ~ResourceTrotterVibrational
    ~ResourceTrotterVibronic
    ~ResourceQubitizeTHC
    ~ResourceOutOfPlaceSquare
    ~ResourcePhaseGradient
    ~ResourceOutMultiplier
    ~ResourceSemiAdder
    ~ResourceBasisRotation
    ~ResourceSelect
    ~ResourceQROM

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
    set_adj_decomp,
    set_ctrl_decomp,
    set_decomp,
    set_pow_decomp,
    GateCount,
)
from .resource_mapping import map_to_resource_op
from .resource_tracking import (
    StandardGateSet,
    DefaultGateSet,
    resource_config,
    estimate_resources,
)
from .templates import (
    CompactHamiltonian,
    ResourceBasisRotation,
    ResourceOutMultiplier,
    ResourceOutOfPlaceSquare,
    ResourcePhaseGradient,
    ResourceQROM,
    ResourceSelect,
    ResourceSemiAdder,
    ResourceTrotterCDF,
    ResourceTrotterTHC,
    ResourceTrotterVibrational,
    ResourceTrotterVibronic,
    ResourceQubitizeTHC,
)
from .ops import (
    ResourceAdjoint,
    ResourceCCZ,
    ResourceCH,
    ResourceChangeBasisOp,
    ResourceCNOT,
    ResourceControlled,
    ResourceControlledPhaseShift,
    ResourceCRot,
    ResourceCRX,
    ResourceCRY,
    ResourceCRZ,
    ResourceCSWAP,
    ResourceCY,
    ResourceCZ,
    ResourceGlobalPhase,
    ResourceHadamard,
    ResourceIdentity,
    ResourceIsingXX,
    ResourceIsingXY,
    ResourceIsingYY,
    ResourceIsingZZ,
    ResourceMultiControlledX,
    ResourceMultiRZ,
    ResourcePauliRot,
    ResourcePhaseShift,
    ResourcePow,
    ResourceProd,
    ResourcePSWAP,
    ResourceRot,
    ResourceRX,
    ResourceRY,
    ResourceRZ,
    ResourceS,
    ResourceSingleExcitation,
    ResourceSWAP,
    ResourceT,
    ResourceTempAND,
    ResourceToffoli,
    ResourceX,
    ResourceY,
    ResourceZ,
)
