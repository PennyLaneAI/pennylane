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
Control flow operations for the StableHLO dialect.
"""

from typing import TypeVar

from xdsl.dialects.builtin import AnyTensorType
from xdsl.dialects.stablehlo import ReturnOp
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import (
    Pure,
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)

# Import our custom StableHLO types
from .types import HLO_PredTensor, HLO_TensorOrPerAxisQuantizedTensorOrToken, HLO_TensorOrToken

# Generic type variables for templating
T_IN = TypeVar("T_IN", bound=AnyTensorType)
T_OUT = TypeVar("T_OUT", bound=AnyTensorType)


@irdl_op_definition
class IfOp(IRDLOperation):
    """
    Produces the output from executing exactly one branch from `true_branch` or
    `false_branch` depending on the value of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#if

    Example:
    %result = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%result_true_branch) : (tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%result_false_branch) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
    """

    name = "stablehlo.if"

    pred = operand_def(HLO_PredTensor)

    res = var_result_def(HLO_TensorOrPerAxisQuantizedTensorOrToken)

    true_branch = region_def("single_block")

    false_branch = region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        RecursivelySpeculatable(),
        SingleBlockImplicitTerminator(ReturnOp),
        # TODO: InferTypeOpInterface
        # TODO: OpAsmOpInterface
    )

    # TODO: Add custom assembly format


@irdl_op_definition
class WhileOp(IRDLOperation):
    """
    Produces the output from executing `body` function 0 or more times while the
    `cond` function outputs `true`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while

    Example:
    ```mlir
    %results0, %results1 = stablehlo.while(%arg0 = %init_i, %arg1 = %init_sum) : tensor<i64>, tensor<i64>
    cond {
      %cond = stablehlo.compare LT, %arg0, %ten : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %new_sum = stablehlo.add %arg1, %one : tensor<i64>
      %new_i = stablehlo.add %arg0, %one : tensor<i64>
      stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
    }
    """

    name = "stablehlo.while"

    operand = var_operand_def(HLO_TensorOrPerAxisQuantizedTensorOrToken)

    res = var_result_def(HLO_TensorOrPerAxisQuantizedTensorOrToken)

    cond = region_def("single_block")

    body = region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        RecursivelySpeculatable(),
        SingleBlockImplicitTerminator(ReturnOp),
        # TODO: InferTypeOpInterface
        # TODO: OpAsmOpInterface
    )


@irdl_op_definition
class OptimizationBarrierOp(IRDLOperation):
    """
    Ensures that the operations that produce the `operand` are executed before any
    operations that depend on the `result` and prevents compiler transformations
    from moving operations across the barrier. Other than that, the operation is
    an identity, i.e. `result` = `operand`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#optimization_barrier

    Example:
    ```mlir
    %result0, %result1 = stablehlo.optimization_barrier %operand0, %operand1 : tensor<f32>, tensor<f32>
    ```
    """

    name = "stablehlo.optimization_barrier"

    operand = var_operand_def(HLO_TensorOrToken)

    res = var_result_def(HLO_TensorOrToken)

    traits = traits_def(
        Pure(),
        # TODO: HLO_PairwiseSameOperandAndResultType
        # TODO: InferTypeOpInterface
    )

    # TODO: Add custom assembly format
    # assembly_format = """
    #   attr-dict ($operand^ `:` custom<PairwiseOpType>(type($operand), type($result))):(`(` `)`)?
    # """
