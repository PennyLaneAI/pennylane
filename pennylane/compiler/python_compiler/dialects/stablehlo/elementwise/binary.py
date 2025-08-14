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

import abc
from typing import Generic, TypeVar

from xdsl.dialects.builtin import AnyTensorType
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import NoMemoryEffect

from pennylane.compiler.python_compiler.xdsl_extras import Elementwise, SameOperandsAndResultShape

# Generic type variables for templating
T_IN = TypeVar("T_IN", bound=AnyTensorType)
T_OUT = TypeVar("T_OUT", bound=AnyTensorType)


class ElementwiseBinaryOperation(IRDLOperation, abc.ABC, Generic[T_IN, T_OUT]):
    """
    Templated base class for elementwise binary operations.

    This class provides a flexible template for binary operations that can work
    with different tensor types.

    For more informtation about the semantics, see:
    https://openxla.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
    """

    lhs = operand_def(T_IN)
    rhs = operand_def(T_IN)
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
class DivideOp(ElementwiseBinaryOperation[AnyTensorType, AnyTensorType]):
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
class MaximumOp(ElementwiseBinaryOperation[AnyTensorType, AnyTensorType]):
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
class MinimumOp(ElementwiseBinaryOperation[AnyTensorType, AnyTensorType]):
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
class PowerOp(ElementwiseBinaryOperation[AnyTensorType, AnyTensorType]):
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
class RemainderOp(ElementwiseBinaryOperation[AnyTensorType, AnyTensorType]):
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
