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
StableHLO type definitions for PennyLane's compiler infrastructure.

This module provides type definitions based on the StableHLO specification
(https://github.com/openxla/stablehlo/blob/main/docs/spec.md), including
token types and other necessary type definitions for StableHLO operations.
"""

# pylint: disable=too-few-public-methods

from typing import TypeAlias

from xdsl.dialects.builtin import (
    AnyFloatConstr,
    ComplexType,
    Float32Type,
    Float64Type,
    IndexType,
    IntAttr,
    IntAttrConstraint,
    IntegerType,
    ParametrizedAttribute,
    Signedness,
    SignednessAttr,
    TensorType,
    i1,
)
from xdsl.dialects.stablehlo import TokenType
from xdsl.irdl import eq, irdl_attr_definition
from xdsl.irdl.attributes import EqAttrConstraint, ParamAttrConstraint
from xdsl.irdl.constraints import IntSetConstraint

from pennylane.compiler.python_compiler.xdsl_extras.constraints import (
    NestedTupleOfConstraint,
)


def _create_param_constrained_type(
    base_attr: type, widths: list[int], signedness: Signedness | None = None
):
    """Create an integer type constrained using ParamAttrConstraint with IntSetConstraint."""
    width_constraint = IntAttrConstraint(IntSetConstraint(frozenset(widths)))

    if signedness is None:
        signedness_constraint = None
    else:
        signedness_constraint = EqAttrConstraint(SignednessAttr(signedness))

    return ParamAttrConstraint(base_attr, [width_constraint, signedness_constraint])


# =============================================================================
# Core StableHLO types constraints
# =============================================================================

HLO_Pred = eq(i1)
HLO_PredTensor: TypeAlias = TensorType[HLO_Pred]

# NOTE: IntegerType is defined in the StableHLO spec as:
# IntegerType ::= SignedIntegerType | UnsignedIntegerType,
# but the MLIR implementation is using signless integers instead of signed,
# and there is a TODO to fix it.

_HLO_INT_WIDTHS = [2, 4, 8, 16, 32, 64]
HLO_SignedInt = _create_param_constrained_type(IntegerType, _HLO_INT_WIDTHS, Signedness.SIGNED)
HLO_UnsignedInt = _create_param_constrained_type(IntegerType, _HLO_INT_WIDTHS, Signedness.UNSIGNED)
HLO_SignlessInt = _create_param_constrained_type(IntegerType, _HLO_INT_WIDTHS, None)

HLO_Int: TypeAlias = HLO_UnsignedInt | HLO_SignlessInt
HLO_IntTensor: TypeAlias = TensorType[HLO_Int]

_HLO_INT_OR_PRED_WIDTHS = [1, 2, 4, 8, 16, 32, 64]
HLO_IntOrPred = _create_param_constrained_type(IntegerType, _HLO_INT_OR_PRED_WIDTHS, None)


HLO_AnyIntegerOrIndex: TypeAlias = IntegerType | IndexType
HLO_AnyIntegerOrIndexTensor: TypeAlias = TensorType.constr(HLO_AnyIntegerOrIndex)

HLO_DimensionValue: TypeAlias = HLO_Int | IndexType

# Constraint variants for use in unions with ParamAttrConstraint
HLO_Float: TypeAlias = AnyFloatConstr
HLO_Float32Or64: TypeAlias = Float32Type | Float64Type
HLO_FloatTensor: TypeAlias = TensorType.constr(HLO_Float)
HLO_Fp32Or64Tensor: TypeAlias = TensorType.constr(HLO_Float32Or64)

# Complex as a constraint over element types {f32,f64}
HLO_Complex: TypeAlias = ComplexType[HLO_Float32Or64]
HLO_ComplexTensor: TypeAlias = TensorType.constr(HLO_Complex)

# =============================================================================
# Quantized element type definitions
# =============================================================================


@irdl_attr_definition
class UniformQuantizedType(ParametrizedAttribute):
    """
    Placeholder for StableHLO per-tensor uniform quantized types.

    Parameterized by width to support different quantized integer widths
    (e.g., 8-bit, 16-bit quantization).
    """

    name = "stablehlo.uniform_quantized"
    width: IntAttr
    signedness: SignednessAttr


@irdl_attr_definition
class UniformQuantizedPerAxisType(ParametrizedAttribute):
    """
    Placeholder for StableHLO per-axis uniform quantized types.

    Parameterized by width to support different quantized integer widths
    (e.g., 8-bit, 16-bit quantization).
    """

    name = "stablehlo.uniform_quantized_per_axis"
    width: IntAttr
    signedness: SignednessAttr


# =============================================================================
# StableHLO quantized type aliases
# =============================================================================

_HLO_QUANTIZED_WIDTHS = [2, 4, 8, 16, 32]

# Constraint-based types for operation definitions
HLO_QuantizedSignedInt = _create_param_constrained_type(
    UniformQuantizedType, _HLO_QUANTIZED_WIDTHS, Signedness.SIGNED
)
HLO_QuantizedUnsignedInt = _create_param_constrained_type(
    UniformQuantizedType, _HLO_QUANTIZED_WIDTHS, Signedness.UNSIGNED
)
HLO_QuantizedAnySignednessInt = _create_param_constrained_type(
    UniformQuantizedType, _HLO_QUANTIZED_WIDTHS, None
)
HLO_QuantizedInt: TypeAlias = HLO_QuantizedSignedInt | HLO_QuantizedUnsignedInt

HLO_PerAxisQuantizedSignedInt = _create_param_constrained_type(
    UniformQuantizedPerAxisType, _HLO_QUANTIZED_WIDTHS, Signedness.SIGNED
)
HLO_PerAxisQuantizedUnsignedInt = _create_param_constrained_type(
    UniformQuantizedPerAxisType, _HLO_QUANTIZED_WIDTHS, Signedness.UNSIGNED
)
HLO_PerAxisQuantizedAnySignednessInt = _create_param_constrained_type(
    UniformQuantizedPerAxisType, _HLO_QUANTIZED_WIDTHS, None
)
HLO_PerAxisQuantizedInt: TypeAlias = HLO_PerAxisQuantizedSignedInt | HLO_PerAxisQuantizedUnsignedInt

# =============================================================================
# Main tensor type definitions
# =============================================================================

HLO_Tensor: TypeAlias = TensorType[HLO_Float | HLO_Complex | HLO_IntOrPred | HLO_QuantizedInt]
HLO_NonQuantizedTensor: TypeAlias = TensorType[HLO_Float | HLO_Complex | HLO_IntOrPred]

# Note: There is a discrepancy between the StableHLO spec and the MLIR implementation.
# The spec does not allow unranked tensors, but the MLIR implementation
# defines it as a tensor of any type and rank. There is a TODO to fix this in MLIR.
# Therefore, we use the correct ranked tensor type.
HLO_AnyTensor: TypeAlias = TensorType[
    HLO_Float | HLO_Complex | HLO_IntOrPred | HLO_QuantizedInt | HLO_PerAxisQuantizedInt
]
HLO_TensorOrToken: TypeAlias = HLO_Tensor | TokenType
HLO_TensorOrPerAxisQuantizedTensorOrToken: TypeAlias = HLO_AnyTensor | TokenType

# HLO_AnyTuple : NestedTupleOf<[HLO_AnyTensor, HLO_Token]>
HLO_AnyTuple = NestedTupleOfConstraint([HLO_AnyTensor, TokenType])

HLO_CustomCallValue: TypeAlias = HLO_Tensor | TokenType | HLO_AnyTuple

# =============================================================================
# HLO combined type definitions
# =============================================================================

HLO_PredOrIntTensor: TypeAlias = TensorType.constr(HLO_IntOrPred)

HLO_FpOrComplexTensor: TypeAlias = TensorType.constr(HLO_Float | HLO_Complex)
HLO_FpOrQuantizedIntTensor: TypeAlias = TensorType.constr(HLO_Float | HLO_QuantizedInt)
HLO_FpComplexOrQuantizedIntTensor: TypeAlias = TensorType.constr(
    HLO_Float | HLO_Complex | HLO_QuantizedInt
)
HLO_IntFpOrComplexOrQuantizedIntTensor: TypeAlias = TensorType.constr(
    HLO_Int | HLO_Float | HLO_Complex | HLO_QuantizedInt
)
HLO_SIntFpComplexOrQuantizedIntTensor: TypeAlias = TensorType.constr(
    HLO_SignedInt | HLO_Float | HLO_Complex | HLO_QuantizedInt
)


__all__ = [
    # Core types
    "HLO_Pred",
    "HLO_PredTensor",
    "HLO_Int",
    "HLO_IntTensor",
    "HLO_AnyIntegerOrIndex",
    "HLO_AnyIntegerOrIndexTensor",
    "HLO_DimensionValue",
    "HLO_Float",
    "HLO_Float32Or64",
    "HLO_FloatTensor",
    "HLO_Fp32Or64Tensor",
    "HLO_ComplexTensor",
    "HLO_SignedInt",
    "HLO_UnsignedInt",
    "HLO_SignlessInt",
    "HLO_QuantizedSignedInt",
    "HLO_QuantizedUnsignedInt",
    "HLO_QuantizedAnySignednessInt",
    "HLO_QuantizedInt",
    "HLO_PerAxisQuantizedSignedInt",
    "HLO_PerAxisQuantizedUnsignedInt",
    "HLO_PerAxisQuantizedAnySignednessInt",
    "HLO_PerAxisQuantizedInt",
    # Quantized types
    "UniformQuantizedType",
    "UniformQuantizedPerAxisType",
    "HLO_Tensor",
    "HLO_NonQuantizedTensor",
    "HLO_AnyTensor",
    "HLO_TensorOrToken",
    "HLO_TensorOrPerAxisQuantizedTensorOrToken",
    "HLO_CustomCallValue",
    # Combined types
    "HLO_PredOrIntTensor",
    "HLO_FpOrComplexTensor",
    "HLO_FpOrQuantizedIntTensor",
    "HLO_FpComplexOrQuantizedIntTensor",
    "HLO_IntFpOrComplexOrQuantizedIntTensor",
    "HLO_SIntFpComplexOrQuantizedIntTensor",
]
