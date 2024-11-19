# Copyright 2024 Xanadu Quantum Technologies Inc.

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
    ~CompressedResourceOp
    ~ResourceOperator

Operators
~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceAdjoint
    ~ResourceCNOT
    ~ResourceControlled
    ~ResourceControlledPhaseShift
    ~ResourceHadamard
    ~ResourcePow
    ~ResourceRZ
    ~ResourceSWAP
    ~ResourceT

Templates
~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceQFT

Exceptions
~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourcesNotDefined
"""

from .resource_operator import ResourceOperator, ResourcesNotDefined
from .resource_container import CompressedResourceOp, Resources

from .ops import (
    ResourceAdjoint,
    ResourceCNOT,
    ResourceControlled,
    ResourceControlledPhaseShift,
    ResourceHadamard,
    ResourcePow,
    ResourceRZ,
    ResourceSWAP,
    ResourceT,
)

from .templates import (
    ResourceQFT,
)
