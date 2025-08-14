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
Extended StableHLO dialect that dynamically includes all upstream operations
plus custom operations for PennyLane's compiler infrastructure.

This module automatically imports all operations and attributes from the upstream
xdsl.dialects.stablehlo and adds custom ones without needing to hardcode
the upstream operation list.
"""

import xdsl.dialects.stablehlo as xstablehlo
from xdsl.ir import Dialect

from .elementwise.binary import (
    ComplexOp,
    DivideOp,
    MaximumOp,
    MinimumOp,
    PowerOp,
    RemainderOp,
)
from .elementwise.other import (
    ClampOp,
    CompareOp,
    MapOp,
    ReducePrecisionOp,
    SelectOp,
)

# Import all elementwise operations from organized files
from .elementwise.unary import (
    AbsOp,
    ConvertOp,
    CosineOp,
    ExponentialMinusOneOp,
    ExponentialOp,
    FloorOp,
    ImagOp,
    IsFiniteOp,
    LogisticOp,
    LogOp,
    LogPlusOneOp,
    NegateOp,
    RealOp,
    RoundNearestAfzOp,
    RoundNearestEvenOp,
    RsqrtOp,
    SignOp,
    SineOp,
    SqrtOp,
    TanhOp,
    TanOp,
)

# Operations to add to the dialect
OPERATIONS = [
    AbsOp,
    ClampOp,
    CompareOp,
    ComplexOp,
    ConvertOp,
    CosineOp,
    DivideOp,
    ExponentialMinusOneOp,
    ExponentialOp,
    FloorOp,
    ImagOp,
    IsFiniteOp,
    LogOp,
    LogPlusOneOp,
    LogisticOp,
    MapOp,
    MaximumOp,
    MinimumOp,
    NegateOp,
    PowerOp,
    RealOp,
    ReducePrecisionOp,
    RemainderOp,
    RoundNearestAfzOp,
    RoundNearestEvenOp,
    RsqrtOp,
    SelectOp,
    SignOp,
    SineOp,
    SqrtOp,
    TanOp,
    TanhOp,
]

# Create the extended StableHLO dialect by dynamically getting upstream components
StableHLO = Dialect(
    "stablehlo",
    [
        *xstablehlo.StableHLO.operations,
        *OPERATIONS,
    ],
    [
        *xstablehlo.StableHLO.attributes,
    ],
)
