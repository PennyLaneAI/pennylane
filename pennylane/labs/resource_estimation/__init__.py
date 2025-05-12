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

Operators
~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceCCZ
    ~ResourceCH
    ~ResourceCNOT
    ~ResourceControlledPhaseShift
    ~ResourceCRot
    ~ResourceCRX
    ~ResourceCRY
    ~ResourceCRZ
    ~ResourceCSWAP
    ~ResourceCY
    ~ResourceCZ
    ~ResourceDoubleExcitation
    ~ResourceDoubleExcitationMinus
    ~ResourceDoubleExcitationPlus
    ~ResourceFermionicSWAP
    ~ResourceGlobalPhase
    ~ResourceHadamard
    ~ResourceIdentity
    ~ResourceIsingXX
    ~ResourceIsingXY
    ~ResourceIsingYY
    ~ResourceIsingZZ
    ~ResourceMultiControlledX
    ~ResourceMultiRZ
    ~ResourceOrbitalRotation
    ~ResourcePauliRot
    ~ResourcePhaseShift
    ~ResourcePSWAP
    ~ResourceRot
    ~ResourceRX
    ~ResourceRY
    ~ResourceRZ
    ~ResourceS
    ~ResourceSingleExcitation
    ~ResourceSingleExcitationMinus
    ~ResourceSingleExcitationPlus
    ~ResourceSWAP
    ~ResourceT
    ~ResourceToffoli
    ~ResourceX
    ~ResourceY
    ~ResourceZ

Symbolic Operators
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceAdjoint
    ~ResourceControlled
    ~ResourceExp
    ~ResourcePow
    ~ResourceProd

Templates
~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceAmplitudeAmplification
    ~ResourceBasisRotation
    ~ResourcePrepSelPrep
    ~ResourceQFT
    ~ResourceQPE
    ~ResourceQuantumPhaseEstimation
    ~ResourceQubitization
    ~ResourceQROM
    ~ResourceReflection
    ~ResourceSelect
    ~ResourceTrotterProduct
    ~ResourceTrotterizedQfunc
    ~resource_trotterize
    ~ResourceControlledSequence
    ~ResourceModExp
    ~ResourceMultiplier
    ~ResourcePhaseAdder

State Preparation Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceBasisState
    ~ResourceStatePrep
    ~ResourceSuperposition
    ~ResourceMottonenStatePreparation

Tracking Resources
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~get_resources

Resource Object Functions:
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~add_in_series
    ~add_in_parallel
    ~mul_in_series
    ~mul_in_parallel
    ~substitute
"""

from .resource_operator import ResourceOperator, ResourcesNotDefined
from .resource_tracking import DefaultGateSet, get_resources, resource_config

from .resource_container import (
    CompressedResourceOp,
    Resources,
    add_in_series,
    add_in_parallel,
    mul_in_series,
    mul_in_parallel,
    substitute,
)

from .ops import (
    ResourceAdjoint,
    ResourceCCZ,
    ResourceCH,
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
    ResourceDoubleExcitation,
    ResourceDoubleExcitationMinus,
    ResourceDoubleExcitationPlus,
    ResourceExp,
    ResourceFermionicSWAP,
    ResourceGlobalPhase,
    ResourceHadamard,
    ResourceIdentity,
    ResourceIsingXX,
    ResourceIsingXY,
    ResourceIsingYY,
    ResourceIsingZZ,
    ResourceMultiControlledX,
    ResourceMultiRZ,
    ResourceOrbitalRotation,
    ResourcePauliRot,
    ResourcePow,
    ResourcePSWAP,
    ResourcePhaseShift,
    ResourceProd,
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
    ResourceToffoli,
    ResourceX,
    ResourceY,
    ResourceZ,
)

from .templates import (
    ResourceControlledSequence,
    ResourceModExp,
    ResourceMultiplier,
    ResourcePhaseAdder,
    ResourceBasisRotation,
    ResourcePrepSelPrep,
    ResourceQFT,
    ResourceQPE,
    ResourceQuantumPhaseEstimation,
    ResourceQubitization,
    ResourceQROM,
    ResourceReflection,
    ResourceSelect,
    ResourceStatePrep,
    ResourceTrotterProduct,
    ResourceTrotterizedQfunc,
    resource_trotterize,
    ResourceMottonenStatePreparation,
    ResourceSuperposition,
    ResourceAmplitudeAmplification,
    ResourceBasisState,
)
