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

# pylint: disable=too-few-public-methods

"""
Dynamism operations for the StableHLO dialect.
"""

from xdsl.dialects.builtin import DenseArrayBase, i64
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
)
from xdsl.traits import (
    ConditionallySpeculatable,
    NoMemoryEffect,
)

from ..xdsl_extras import TensorConstraint
from .types import HLO_AnyTensor, HLO_DimensionValue


@irdl_op_definition
class DynamicBroadcastInDimOp(IRDLOperation):
    """
    This operation is functionally identical to
    [broadcast_in_dim](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim)
    op, but the result shape is specified dynamically via `output_dimensions`.

    It also accepts optional attributes to express static knowledge about the
    expanding behavior of dimensions. If not specified, all dimensions are
    assumed to be possibly expanding. The sets of dimensions that are known to
    be expanding and the set of dimensions that are known to be non-expanding
    must be disjoint and they must be a subset of the operand's dimensions.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_broadcast_in_dim

    Example:
    ```mlir
    %operand = stablehlo.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
    %output_dimensions = stablehlo.constant dense<[2, 3, 2]> : tensor<3xi64>
    %result = "stablehlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
      broadcast_dimensions = array<i64: 2, 1>,
      known_expanding_dimensions = array<i64: 0>,
      known_nonexpanding_dimensions = array<i64: 1>
    } : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    ```
    """

    name = "stablehlo.dynamic_broadcast_in_dim"

    operand = operand_def(HLO_AnyTensor)
    output_dimensions = operand_def(TensorConstraint(element_type=HLO_DimensionValue, rank=1))
    broadcast_dimensions = attr_def(DenseArrayBase.constr(i64))
    known_expanding_dimensions = opt_attr_def(DenseArrayBase.constr(i64))
    known_nonexpanding_dimensions = opt_attr_def(DenseArrayBase.constr(i64))
    result = result_def(HLO_AnyTensor)

    assembly_format = (
        "$operand `,` $output_dimensions `,` `dims` `=` $broadcast_dimensions "
        "attr-dict `:` functional-type(operands, results)"
    )

    traits = traits_def(
        ConditionallySpeculatable(),
        NoMemoryEffect(),
        # TODO: InferShapedTypeOpInterface(),
    )

    # TODO: MLIR has a custom verifier for the dynamic_broadcast_in_dim operation.
