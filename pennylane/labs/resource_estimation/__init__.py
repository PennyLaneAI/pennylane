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

"""

from .qubit_manager import QubitManager, GrabWires, FreeWires
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
