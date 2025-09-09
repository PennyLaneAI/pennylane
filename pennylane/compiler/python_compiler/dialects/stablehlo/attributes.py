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

"""
StableHLO attribute definitions for PennyLane's compiler infrastructure.

This module provides attribute definitions based on the StableHLO specification
(https://github.com/openxla/stablehlo/blob/main/docs/spec.md), including
attributes for StableHLO operations.
"""

# pylint: disable=too-few-public-methods

from enum import StrEnum

from xdsl.ir import EnumAttribute, SpacedOpaqueSyntaxAttribute
from xdsl.irdl import irdl_attr_definition


class ResultAccuracyMode(StrEnum):
    """
    XLA result accuracy mode.
    """

    DEFAULT = "DEFAULT"
    HIGH = "HIGHEST"
    HIGHEST = "TOLERANCE"


@irdl_attr_definition
class ResultAccuracyModeAttr(EnumAttribute[ResultAccuracyMode], SpacedOpaqueSyntaxAttribute):
    """
    XLA result accuracy mode.

    See external [documentation](https://github.com/openxla/stablehlo/blob/7c50d4efeaea30bff6aa5e46c7f71170f5aa06af/stablehlo/dialect/StablehloEnums.td#L49-L70).
    """

    name = "stablehlo.result_accuracy_mode"
