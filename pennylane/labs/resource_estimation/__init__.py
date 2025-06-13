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

"""

from .ops import (
    ResourceGlobalPhase,
    ResourceHadamard,
    ResourceIdentity,
    ResourceS,
    ResourceT,
    ResourceX,
    ResourceY,
    ResourceZ,
)
from .qubit_manager import AllocWires, FreeWires, QubitManager
from .resource_mapping import map_to_resource_op
from .resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    ResourcesNotDefined,
    resource_rep,
    set_adj_decomp,
    set_ctrl_decomp,
    set_decomp,
    set_pow_decomp,
)
from .resource_tracking import DefaultGateSet, StandardGateSet, estimate_resources, resource_config
from .resources_base import Resources
