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
StableHLO dialect package for PennyLane's compiler infrastructure.

This package contains organized elementwise operations and other StableHLO-related
functionality.
"""

# Import all elementwise operations explicitly
from .elementwise_unary import (
    ConvertOp,
    CosineOp,
    ExponentialMinusOneOp,
    ExponentialOp,
    FloorOp,
    ImagOp,
    IsFiniteOp,
    LogOp,
    LogPlusOneOp,
    LogisticOp,
    NegateOp,
    RealOp,
    RoundNearestAfzOp,
    RoundNearestEvenOp,
    RsqrtOp,
    SignOp,
    SineOp,
    SqrtOp,
    TanOp,
    TanhOp,
)

from .elementwise_binary import (
    ComplexOp,
    DivideOp,
    MaximumOp,
    MinimumOp,
    PowerOp,
    RemainderOp,
)

from .elementwise_other import (
    ClampOp,
    CompareOp,
    MapOp,
    ReducePrecisionOp,
    SelectOp,
)

# Import the main StableHLO dialect
from .dialect import StableHLO

# Export all operations and the dialect for external use
__all__ = [
    # Main dialect
    "StableHLO",
    # Unary operations
    "ConvertOp",
    "CosineOp",
    "ExponentialMinusOneOp",
    "ExponentialOp",
    "FloorOp",
    "ImagOp",
    "IsFiniteOp",
    "LogOp",
    "LogPlusOneOp",
    "LogisticOp",
    "NegateOp",
    "RealOp",
    "RoundNearestAfzOp",
    "RoundNearestEvenOp",
    "RsqrtOp",
    "SignOp",
    "SineOp",
    "SqrtOp",
    "TanOp",
    "TanhOp",
    # Binary operations
    "ComplexOp",
    "DivideOp",
    "MaximumOp",
    "MinimumOp",
    "PowerOp",
    "RemainderOp",
    # Other operations
    "ClampOp",
    "CompareOp",
    "MapOp",
    "ReducePrecisionOp",
    "SelectOp",
]
