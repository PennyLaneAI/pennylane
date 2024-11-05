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
r"""This module contains experimental resource estimation functionality. """

from .resource_constructor import ResourceConstructor, ResourcesNotDefined
from .resource_container import CompressedResourceOp, Resources
from .resource_tracking import get_resources, DefaultGateSet, _StandardGateSet, resource_config

from .ops import (
    ResourceCNOT,
    ResourceControlledPhaseShift,
    ResourceHadamard,
    ResourceRZ,
    ResourceSWAP,
    ResourceT,
)

from .templates import (
    ResourceQFT,
)