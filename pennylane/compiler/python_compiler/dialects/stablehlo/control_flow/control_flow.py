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

from xdsl.dialects.builtin import (
    I1,
    AnyFloat,
    AnyTensorType,
    ComplexType,
    TensorType,
)
from xdsl.dialects.stablehlo import ReturnOp
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    region_def,
    traits_def,
    var_result_def,
)
from xdsl.traits import (
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)

# Type aliases
I1TensorType = TensorType[I1]
FloatTensorType = TensorType[AnyFloat]
FloatOrComplexType = AnyFloat | ComplexType
FloatOrComplexTensorType = TensorType[FloatOrComplexType]
ComplexTensorType = TensorType[ComplexType]

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

    pred = operand_def(I1TensorType)

    results = var_result_def(AnyTensorType)

    true_branch = region_def("single_block")

    false_branch = region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        RecursivelySpeculatable(),
        SingleBlockImplicitTerminator(ReturnOp),
        # TODO: InferTypeOpInterface
    )
