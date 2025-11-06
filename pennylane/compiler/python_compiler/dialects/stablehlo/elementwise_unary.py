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
Unary elementwise operations for the StableHLO dialect.
"""

import abc
from typing import Generic, TypeVar

from xdsl.dialects.builtin import (
    I1,
    AnyFloat,
    AnyTensorType,
    ComplexType,
    TensorType,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
)
from xdsl.traits import NoMemoryEffect

from ...xdsl_extras import Elementwise, SameOperandsAndResultShape
from .attributes import ResultAccuracyMode, ResultAccuracyModeAttr
from .types import (
    HLO_FloatTensor,
    HLO_FpComplexOrQuantizedIntTensor,
    HLO_FpOrComplexTensor,
    HLO_FpOrQuantizedIntTensor,
    HLO_IntFpOrComplexOrQuantizedIntTensor,
    HLO_NonQuantizedTensor,
    HLO_PredTensor,
    HLO_SIntFpComplexOrQuantizedIntTensor,
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


class ElementwiseUnaryOperation(IRDLOperation, abc.ABC, Generic[T_IN, T_OUT]):
    """
    Templated base class for elementwise unary operations.

    This class provides a flexible template for unary operations that can work
    with different tensor types.

    For more informtation about the semantics, see:
    https://openxla.org/xla/operation_semantics#element-wise_unary_functions
    """

    operand = operand_def(T_IN)
    result = result_def(T_OUT)

    # TODO: Implement CustomDirective
    # assembly_format = """
    # $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
    # """

    traits = traits_def(
        NoMemoryEffect(),
        SameOperandsAndResultShape(),
        Elementwise(),
        # TODO: InferShapedTypeOpInterface(),
        # TODO: HLO_SpeculatableIfStaticDimInOutputIsStaticInInput(),
    )

    def __init__(self, operand: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = operand.type
        super().__init__(operands=(operand,), result_types=(result_type,))


@irdl_op_definition
class ConvertOp(ElementwiseUnaryOperation[HLO_NonQuantizedTensor, HLO_NonQuantizedTensor]):
    """
    Performs an element-wise conversion from one element type to another on
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert

    Example:
    ```mlir
    %result = stablehlo.convert %operand : (tensor<3xi64>) -> tensor<3xcomplex<f64>>
    ```
    """

    name = "stablehlo.convert"

    traits = traits_def(SameOperandsAndResultShape())


@irdl_op_definition
class CosineOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise cosine operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cosine

    Example:
    ```mlir
    %result = stablehlo.cosine %operand : tensor<2xf32>
    ```
    """

    name = "stablehlo.cosine"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )
    # TODO: implement HLO_CompatibleOperandsAndResultType()
    # traits = traits_def(
    #     HLO_CompatibleOperandsAndResultType()
    # )


@irdl_op_definition
class ExponentialMinusOneOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise exponential minus one operation on `operand` tensor
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential_minus_one

    Example:
    ```mlir
    %result = stablehlo.exponential_minus_one %operand : tensor<2xf64>
    ```
    """

    name = "stablehlo.exponential_minus_one"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    # TODO: implement HLO_CompatibleOperandsAndResultType()
    # traits = traits_def(
    #     HLO_CompatibleOperandsAndResultType()
    # )


@irdl_op_definition
class ExponentialOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise exponential operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential

    Example:
    ```mlir
    %result = stablehlo.exponential %operand : tensor<2x2xf64>
    ```
    """

    name = "stablehlo.exponential"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    # TODO: implement HLO_CompatibleOperandsAndResultType()
    # traits = traits_def(
    #     HLO_CompatibleOperandsAndResultType()
    # )


@irdl_op_definition
class FloorOp(ElementwiseUnaryOperation[HLO_FpOrQuantizedIntTensor, HLO_FpOrQuantizedIntTensor]):
    """
    Performs element-wise floor of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#floor

    Example:
    ```mlir
    %result = stablehlo.floor %operand : tensor<2xf32>
    ```
    """

    name = "stablehlo.floor"


@irdl_op_definition
class ImagOp(ElementwiseUnaryOperation[HLO_FpOrComplexTensor, HLO_FloatTensor]):
    """
    Extracts the imaginary part, element-wise, from the `operand` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#imag

    Example:
    ```mlir
    %result = stablehlo.imag %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    ```
    """

    name = "stablehlo.imag"


@irdl_op_definition
class IsFiniteOp(ElementwiseUnaryOperation[HLO_FpOrQuantizedIntTensor, HLO_PredTensor]):
    """
    Performs element-wise check whether the value in `x` is finite (i.e. is
    neither +Inf, -Inf, nor NaN) and produces a `y` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#is_finite

    Example:
    ```mlir
    %y = stablehlo.is_finite %x : (tensor<7xf64>) -> tensor<7xi1>
    ```
    """

    name = "stablehlo.is_finite"


@irdl_op_definition
class LogOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise logarithm operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log

    Example:
    ```mlir
    %result = stablehlo.log %operand : tensor<2x2xf64>
    ```
    """

    name = "stablehlo.log"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )


@irdl_op_definition
class LogPlusOneOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise logarithm plus one operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log_plus_one

    Example:
    ```mlir
    %result = stablehlo.log_plus_one %operand : tensor<5xf64>
    ```
    """

    name = "stablehlo.log_plus_one"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )


@irdl_op_definition
class LogisticOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise logistic operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#logistic

    Example:
    ```mlir
    %result = stablehlo.logistic %operand : tensor<2x2xf64>
    ```
    """

    name = "stablehlo.logistic"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )


@irdl_op_definition
class NegateOp(
    ElementwiseUnaryOperation[
        HLO_IntFpOrComplexOrQuantizedIntTensor, HLO_IntFpOrComplexOrQuantizedIntTensor
    ]
):
    """
    Performs element-wise negation of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate

    Example:
    ```mlir
    %result = stablehlo.negate %operand : tensor<2x3xi32>
    ```
    """

    name = "stablehlo.negate"


@irdl_op_definition
class RealOp(ElementwiseUnaryOperation[HLO_FpOrComplexTensor, HLO_FloatTensor]):
    """
    Extracts the real part, element-wise, from the `operand` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#real

    Example:
    ```mlir
    %result = stablehlo.real %operand : tensor<2xcomplex<f32>> : tensor<2xf32>
    ```
    """

    name = "stablehlo.real"


@irdl_op_definition
class RoundNearestAfzOp(
    ElementwiseUnaryOperation[HLO_FpOrQuantizedIntTensor, HLO_FpOrQuantizedIntTensor]
):
    """
    Performs element-wise rounding towards the nearest integer, breaking ties
    away from zero, on the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_afz

    Example:
    ```mlir
    %result = stablehlo.round_nearest_afz %operand : tensor<5xf64>
    ```
    """

    name = "stablehlo.round_nearest_afz"


@irdl_op_definition
class RoundNearestEvenOp(
    ElementwiseUnaryOperation[HLO_FpOrQuantizedIntTensor, HLO_FpOrQuantizedIntTensor]
):
    """
    Performs element-wise rounding towards the nearest integer, breaking ties
    towards the even integer, on the `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_even

    Example:
    ```mlir
    %result = stablehlo.round_nearest_even %operand : tensor<5xf64>
    ```
    """

    name = "stablehlo.round_nearest_even"


@irdl_op_definition
class RsqrtOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise reciprocal square root operation on `operand` tensor
    and produces a `result` tensor, implementing the `rSqrt` operation from the
    IEEE-754 specification.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rsqrt

    Example:
    ```mlir
    %result = stablehlo.rsqrt %operand : tensor<2x2xf32>
    ```
    """

    name = "stablehlo.rsqrt"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )


@irdl_op_definition
class SignOp(
    ElementwiseUnaryOperation[
        HLO_SIntFpComplexOrQuantizedIntTensor, HLO_SIntFpComplexOrQuantizedIntTensor
    ]
):
    """
    Returns the sign of the `operand` element-wise and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sign

    Example:
    ```mlir
    %result = stablehlo.sign %operand : tensor<5xf64>
    ```
    """

    name = "stablehlo.sign"


@irdl_op_definition
class SineOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise sine operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sine

    Example:
    ```mlir
    %result = stablehlo.sine %operand : tensor<2xf32>
    ```
    """

    name = "stablehlo.sine"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )


@irdl_op_definition
class SqrtOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise square root operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sqrt

    Example:
    ```mlir
    %result = stablehlo.sqrt %operand : tensor<2x2xf32>
    ```
    """

    name = "stablehlo.sqrt"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )


@irdl_op_definition
class TanOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise tangent operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tan

    Example:
    ```mlir
    %result = stablehlo.tan %operand : tensor<2x2xf64>
    ```
    """

    name = "stablehlo.tan"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )


@irdl_op_definition
class TanhOp(
    ElementwiseUnaryOperation[HLO_FpComplexOrQuantizedIntTensor, HLO_FpComplexOrQuantizedIntTensor]
):
    """
    Performs element-wise hyperbolic tangent operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tanh

    Example:
    ```mlir
    %result = stablehlo.tanh %operand : tensor<2xf32>
    ```
    """

    name = "stablehlo.tanh"

    result_accuracy = opt_attr_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )
