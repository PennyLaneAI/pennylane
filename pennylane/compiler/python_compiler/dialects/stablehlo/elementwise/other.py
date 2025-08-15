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
Other elementwise operations for the StableHLO dialect.
"""

from typing import ClassVar

import xdsl.dialects.stablehlo as xstablehlo
from xdsl.dialects.builtin import (
    I1,
    I32,
    AnyFloat,
    AnyTensorType,
    DenseArrayBase,
    IntegerAttr,
    TensorType,
    i64,
)
from xdsl.irdl import (
    IRDLOperation,
    VarConstraint,
    attr_def,
    base,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
)
from xdsl.traits import NoMemoryEffect, RecursiveMemoryEffect, SingleBlockImplicitTerminator

from pennylane.compiler.python_compiler.xdsl_extras import Elementwise, SameOperandsAndResultShape

# Type aliases
I1TensorType = TensorType[I1]
FloatTensorType = TensorType[AnyFloat]


@irdl_op_definition
class ClampOp(IRDLOperation):
    """Element-wise clamp with min and max bounds.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#clamp
    """

    name = "stablehlo.clamp"

    min = operand_def(AnyTensorType)
    operand = operand_def(AnyTensorType)
    max = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)

    # TODO: Implement CustomDirective
    # assembly_format = """
    # $min `,` $operand `,` $max attr-dict
    #   `:` custom<SameOperandsAndResultType>(type($min), type($operand), type($max), type($result))
    # """

    traits = traits_def(
        NoMemoryEffect(),
        # TODO: HLO_SpeculatableIfAllInputsStatic(),
        # TODO: HLO_CompatibleOperandsAndResultElementType(),
        # TODO: HLO_BroadcastingElementwise(),
        # TODO: InferTensorType(),
        # TODO: InferShapedTypeOpInterface(),
    )


@irdl_op_definition
class CompareOp(IRDLOperation):
    """Element-wise compare with direction and type attributes."""

    name = "stablehlo.compare"

    assembly_format = """
    $comparison_direction `,` $lhs `,` $rhs (`,` $comparison_type^)? attr-dict `:` functional-type(operands, results)
    """

    lhs = operand_def(AnyTensorType)
    rhs = operand_def(AnyTensorType)
    result = result_def(I1TensorType)
    comparison_direction = attr_def(xstablehlo.ComparisonDirectionAttr)
    comparison_type = opt_attr_def(xstablehlo.ComparisonTypeAttr)

    traits = traits_def(
        NoMemoryEffect(),
        Elementwise(),
        SameOperandsAndResultShape(),
        # TODO: HLO_SpeculatableIfAllInputsStatic(),
        # TODO: HLO_CompatibleOperandsElementType(),
        # TODO: InferTensorTypeWithReify(),
    )


@irdl_op_definition
class MapOp(IRDLOperation):
    """
    Applies a map function `computation` to `inputs` along the `dimensions` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#map

    Example:
    ```mlir
    %result = "stablehlo.map"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.multiply %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      dimensions = array<i64: 0, 1>
    } : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
    ```
    """

    name = "stablehlo.map"

    inputs = var_operand_def(AnyTensorType)
    result = result_def(AnyTensorType)
    dimensions = attr_def(DenseArrayBase.constr(i64))
    computation = var_region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        SameOperandsAndResultShape(),
        SingleBlockImplicitTerminator(xstablehlo.ReturnOp),
        # TODO: HLO_RecursivelySpeculatableIfAllInputsStatic(),
        # TODO: InferTypeOpInterface
        # TODO: InferShapedTypeOpInterface(),
    )


@irdl_op_definition
class ReducePrecisionOp(IRDLOperation):
    """
    Performs element-wise conversion of `operand` to another floating-point type
    that uses `exponent_bits` and `mantissa_bits` and back to the original
    floating-point type and produces an `output` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_precision

    Example:
    ```mlir
    %output = stablehlo.reduce_precision %operand, format = e5m10 : tensor<6xf64>
    ```
    """

    name = "stablehlo.reduce_precision"

    # TODO: Implement CustomDirective
    # assembly_format = """
    # $operand `,` `format` `=` custom<ExponentMantissa>($exponent_bits, $mantissa_bits)
    #   attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($output))
    # """

    operand = operand_def(FloatTensorType)
    result = result_def(FloatTensorType)

    exponent_bits = attr_def(IntegerAttr[I32])
    mantissa_bits = attr_def(IntegerAttr[I32])

    traits = traits_def(
        NoMemoryEffect(),
        Elementwise(),
        # TODO: HLO_CompatibleOperandsAndResultType(),
        # TODO: HLO_SpeculatableIfStaticDimInOutputIsStaticInInput(),
    )


@irdl_op_definition
class SelectOp(IRDLOperation):
    """
    Produces a `result` tensor where each element is selected from `on_true` or
    `on_false` tensor based on the value of the corresponding element of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select

    Example:
    ```mlir
    %result = stablehlo.select %pred, %on_true, %on_false : tensor<2x2xi1>, tensor<2x2xi32>
    ```
    """

    name = "stablehlo.select"

    # assembly_format = """
    # operands attr-dict `:`
    #   custom<SelectOpType>(type($pred), type($on_true), type($on_false), type($result))
    # """

    PRED: ClassVar = VarConstraint("PRED", base(I1TensorType))
    T: ClassVar = VarConstraint("T", base(AnyTensorType))

    pred = operand_def(PRED)
    on_true = operand_def(T)
    on_false = operand_def(T)
    result = result_def(T)

    traits = traits_def(
        NoMemoryEffect(),
    )
