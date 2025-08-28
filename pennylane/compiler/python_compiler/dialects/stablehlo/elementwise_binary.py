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
Binary elementwise operations for the StableHLO dialect.
"""

# pylint: disable=too-few-public-methods

import abc
from typing import Generic, TypeVar

from xdsl.dialects.builtin import AnyTensorType, ComplexType, Float32Type, Float64Type, TensorType
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import NoMemoryEffect

from pennylane.compiler.python_compiler.xdsl_extras import (
    Elementwise,
    SameOperandsAndResultShape,
    SameOperandsElementType,
)

from .types import (
    HLO_ComplexTensor,
    HLO_Fp32Or64Tensor,
    HLO_IntFpOrComplexOrQuantizedIntTensor,
    HLO_Tensor,
)

# Type aliases
F32Or64Type = Float32Type | Float64Type
F32Or64TensorType = TensorType[F32Or64Type]
ComplexTensorType = TensorType[ComplexType]

# Generic type variables for templating
T_LHS = TypeVar("T_LHS", bound=AnyTensorType)
T_RHS = TypeVar("T_RHS", bound=AnyTensorType)
T_OUT = TypeVar("T_OUT", bound=AnyTensorType)


class ElementwiseBinaryOperation(IRDLOperation, abc.ABC, Generic[T_LHS, T_RHS, T_OUT]):
    """
    Templated base class for elementwise binary operations.

    This class provides a flexible template for binary operations that can work
    with different tensor types.

    For more information about the semantics, see:
    https://openxla.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
    """

    lhs = operand_def(T_LHS)
    rhs = operand_def(T_RHS)
    result = result_def(T_OUT)

    traits = traits_def(
        NoMemoryEffect(),
        SameOperandsAndResultShape(),
        Elementwise(),
        # TODO: HLO_SpeculatableIfAllInputsStatic(),
    )

    # TODO: Implement CustomDirective
    # assembly_format = """
    # $lhs `,` $rhs attr-dict
    #   `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
    # """

    def __init__(self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = lhs.type
        super().__init__(operands=(lhs, rhs), result_types=(result_type,))


@irdl_op_definition
class ComplexOp(
    ElementwiseBinaryOperation[HLO_Fp32Or64Tensor, HLO_Fp32Or64Tensor, HLO_ComplexTensor]
):
    """
    Performs element-wise conversion to a complex value from a pair of real and
    imaginary values, `lhs` and `rhs`, and produces a `result` tensor.
    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#complex
    Example:
    ```mlir
    %result = stablehlo.complex %lhs, %rhs : tensor<2xcomplex<f64>>
    ```
    """

    name = "stablehlo.complex"

    # assembly_format = """
    # operands attr-dict
    #   `:` custom<ComplexOpType>(type($lhs), type($rhs), type($result))
    # """

    traits = traits_def(
        NoMemoryEffect(),
        SameOperandsElementType(),
        SameOperandsAndResultShape(),
        # TODO: HLO_SpeculatableIfAllInputsStatic(),
    )


@irdl_op_definition
class DivideOp(
    ElementwiseBinaryOperation[
        HLO_IntFpOrComplexOrQuantizedIntTensor,
        HLO_IntFpOrComplexOrQuantizedIntTensor,
        HLO_IntFpOrComplexOrQuantizedIntTensor,
    ]
):
    """
    Performs element-wise division of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide

    Example:
    ```mlir
    %result = stablehlo.divide %lhs, %rhs : tensor<4xf32>
    ```
    """

    name = "stablehlo.divide"


@irdl_op_definition
class MaximumOp(ElementwiseBinaryOperation[HLO_Tensor, HLO_Tensor, HLO_Tensor]):
    """
    Performs element-wise max operation on tensors `lhs` and `rhs` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#maximum

    Example:
    ```mlir
    %result = stablehlo.maximum %lhs, %rhs : tensor<4xf32>
    ```
    """

    name = "stablehlo.maximum"


@irdl_op_definition
class MinimumOp(ElementwiseBinaryOperation[HLO_Tensor, HLO_Tensor, HLO_Tensor]):
    """
    Performs element-wise min operation on tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#minimum

    Example:
    ```mlir
    %result = stablehlo.minimum %lhs, %rhs : tensor<4xf32>
    ```
    """

    name = "stablehlo.minimum"


@irdl_op_definition
class PowerOp(ElementwiseBinaryOperation[HLO_Tensor, HLO_Tensor, HLO_Tensor]):
    """
    Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power

    Example:
    ```mlir
    %result = stablehlo.power %lhs, %rhs : tensor<6xf64>
    ```
    """

    name = "stablehlo.power"


@irdl_op_definition
class RemainderOp(ElementwiseBinaryOperation[HLO_Tensor, HLO_Tensor, HLO_Tensor]):
    """
    Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#remainder

    Example:
    ```mlir
    %result = stablehlo.remainder %lhs, %rhs : tensor<4xi64>
    ```
    """

    name = "stablehlo.remainder"
