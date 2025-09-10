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

from xdsl.dialects import stablehlo as xstablehlo
from xdsl.dialects.builtin import DenseArrayBase, i64
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.irdl.operations import SameVariadicOperandSize
from xdsl.traits import (
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)

from .types import HLO_Tensor


@irdl_op_definition
class ReduceOp(IRDLOperation):
    """
    Applies a reduction function `body` to `inputs` and `init_values` along the
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce

    Example:
    ```mlir
    %result = "stablehlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    ```
    """

    name = "stablehlo.reduce"

    inputs = var_operand_def(HLO_Tensor)
    init_values = var_operand_def(HLO_Tensor)
    dimensions = attr_def(DenseArrayBase.constr(i64))
    result = var_result_def(HLO_Tensor)
    body = region_def("single_block")

    irdl_options = [SameVariadicOperandSize()]

    traits = traits_def(
        RecursiveMemoryEffect(),
        # TODO: InferShapedTypeOpInterface(),
        # TODO: HLO_RecursivelySpeculatableIfAllInputsStatic,
        # TODO: InferTensorTypeWithReify(),
        SingleBlockImplicitTerminator(xstablehlo.ReturnOp),
    )

    # TODO: MLIR has a custom verifier for the reduce operation.
    # TODO: MLIR has a custom assembly format implementation for the reduce operation.
