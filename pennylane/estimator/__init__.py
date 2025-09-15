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
This module contains tools for logical resource estimation.

.. currentmodule:: pennylane.estimator

Qubit Management Classes:
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~Allocate
    ~Deallocate
    ~WireResourceManager

Resource Estimation Base Classes:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~Resources
    ~ResourceOperator
    ~CompressedResourceOp
    ~GateCount


"""

from .wires_manager import Allocate, Deallocate, WireResourceManager

from .resources_base import Resources

from .resource_config import ResourceConfig

from .resource_operator import (
    ResourceOperator,
    CompressedResourceOp,
    GateCount,
    resource_rep,
)
