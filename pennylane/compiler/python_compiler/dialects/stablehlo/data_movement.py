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
Data movement operations for the StableHLO dialect.
"""

from xdsl.dialects.builtin import BoolAttr, DenseArrayBase, IntegerAttr, TensorType, i64
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.irdl.attributes import eq
from xdsl.irdl.constraints import AtLeast
from xdsl.irdl.operations import SameVariadicOperandSize
from xdsl.traits import (
    ConditionallySpeculatable,
    NoMemoryEffect,
    Pure,
    RecursiveMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.type import get_element_type_or_self

from ...xdsl_extras import (
    AllMatchSameOperatorTrait,
    SameOperandsAndResultElementType,
    TensorConstraint,
)
from .attributes import GatherDimensionNumbers, ScatterDimensionNumbers
from .types import HLO_AnyIntegerOrIndexTensor, HLO_AnyTensor, HLO_Int, HLO_IntTensor, HLO_Tensor


@irdl_op_definition
class BroadcastInDimOp(IRDLOperation):
    """
    Expands the dimensions and/or rank of an input tensor by duplicating the
    data in the ``operand`` tensor and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

    Example:
    ```mlir
    %result = stablehlo.broadcast_in_dim %operand, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    ```
    """

    name = "stablehlo.broadcast_in_dim"
    operand = operand_def(HLO_AnyTensor)
    broadcast_dimensions = prop_def(DenseArrayBase.constr(i64))
    result = result_def(HLO_AnyTensor)

    assembly_format = """
        $operand `,` `dims` `=` $broadcast_dimensions
          attr-dict `:` functional-type(operands, results)
    """

    traits = traits_def(
        NoMemoryEffect(),
        # TODO: HLO_SpeculatableIfAllInputsStatic,
        # TODO: HLO_CompatibleOperandsAndResultElementType,
    )

    def verify_(self) -> None:
        """Verify non-quantized broadcast_in_dim constraints."""
        o_type = self.operand_types[0]
        r_type = self.result_types[0]

        # These are constrained to tensors by the op definition
        assert isinstance(o_type, TensorType) and isinstance(r_type, TensorType)

        # broadcast_in_dim_c2: broadcast_dimensions size == operand rank
        dims = tuple(self.broadcast_dimensions.get_values())
        operand_rank = o_type.get_num_dims()
        if len(dims) != operand_rank:
            raise VerifyException(
                "broadcast_dimensions size ("
                f"{len(dims)}"
                ") does not match operand rank ("
                f"{operand_rank}"
                ")"
            )

        # broadcast_in_dim_c4: broadcast_dimensions should not have duplicates
        if len(set(dims)) != len(dims):
            raise VerifyException("broadcast_dimensions should not have duplicates")

        # Result rank and per-dimension checks
        result_rank = r_type.get_num_dims()
        o_shape = o_type.get_shape()
        r_shape = r_type.get_shape()

        for i, dim_index in enumerate(dims):
            # broadcast_in_dim_c3: each dim index in bounds of result rank
            if dim_index < 0 or dim_index >= result_rank:
                raise VerifyException(
                    "broadcast_dimensions contains invalid value "
                    f"{dim_index} for result with rank {result_rank}"
                )

            # If operand dim is static, enforce broadcast_in_dim_c5
            if o_shape[i] != -1:
                dim_size = o_shape[i]
                result_dim_size = r_shape[dim_index]
                if dim_size not in (1, result_dim_size):
                    raise VerifyException(
                        "size of operand dimension "
                        f"{i} ({dim_size}) is not equal to 1 or size of result dimension "
                        f"{dim_index} ({result_dim_size})"
                    )


@irdl_op_definition
class ConcatenateOp(IRDLOperation):
    """
    Concatenates a variadic number of tensors in ``inputs`` along ``dimension``
    dimension in the same order as the given arguments and produces a ``result``
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate

    Example:
    ```mlir
    %result = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    ```
    """

    name = "stablehlo.concatenate"

    inputs = var_operand_def(HLO_Tensor)
    result = result_def(HLO_Tensor)
    dimension = prop_def(IntegerAttr.constr(type=eq(i64), value=AtLeast(0)))

    traits = traits_def(
        NoMemoryEffect(),
        ConditionallySpeculatable(),
        SameOperandsAndResultElementType(),
        # InferTypeOpInterface(),
    )

    # TODO: Implement CustomDirective
    # assembly_format = """
    #   custom<VariadicOperandWithAttribute>($inputs) `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
    # """


@irdl_op_definition
class DynamicSliceOp(IRDLOperation):
    """
    Extracts a slice from the ``operand`` using dynamically-computed starting
    indices and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_slice

    Example:
    ```mlir
    %result = stablehlo.dynamic_slice %operand, %start_indices0, %start_indices1, sizes = [2, 2]
      : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
    ```
    """

    name = "stablehlo.dynamic_slice"
    operand = operand_def(HLO_Tensor)
    start_indices = var_operand_def(TensorConstraint(element_type=HLO_Int, rank=0))
    slice_sizes = prop_def(DenseArrayBase.constr(i64))
    result = result_def(HLO_Tensor)

    # TODO: Implement CustomDirective
    # assembly_format = """
    #     $operand `,` custom<VariadicOperandWithAttribute>($start_indices)
    #     `sizes` `=` $slice_sizes attr-dict `:` functional-type(operands, results)
    # """

    traits = traits_def(
        Pure(),
        AllMatchSameOperatorTrait(
            ("operand", "result"), lambda x: get_element_type_or_self(x.type), "element type"
        ),
        # TODO: InferTensorType(),
    )


@irdl_op_definition
class GatherOp(IRDLOperation):
    """
    Gathers slices from ``operand`` tensor from offsets specified in
    ``start_indices`` and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather

    Example:
    ```mlir
    %result = "stablehlo.gather"(%operand, %start_indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [3, 4],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [1],
        start_index_map = [2, 1],
        index_vector_dim = 3>,
      slice_sizes = array<i64: 1, 1, 2, 2>,
      indices_are_sorted = false
    } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi64>
    ```
    """

    name = "stablehlo.gather"
    operand = operand_def(HLO_Tensor)
    start_indices = operand_def(HLO_IntTensor)
    dimension_numbers = prop_def(GatherDimensionNumbers)
    slice_sizes = prop_def(DenseArrayBase.constr(i64))
    indices_are_sorted = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    result = result_def(HLO_Tensor)

    traits = traits_def(
        NoMemoryEffect(),
        ConditionallySpeculatable(),
        AllMatchSameOperatorTrait(
            ("operand", "result"), lambda x: get_element_type_or_self(x.type), "element type"
        ),
        # TODO: InferTensorTypeWithReify(),
    )

    # TODO: Implement CustomDirective
    # assembly_format = """
    #   custom<VariadicOperandWithAttribute>($inputs) `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
    # """


@irdl_op_definition
class ReshapeOp(IRDLOperation):
    """
    Performs reshape of ``operand`` tensor to a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

    Example:
    ```mlir
    %result = stablehlo.reshape %operand : (tensor<2xf32>) -> tensor<1x2xf32>
    ```
    """

    name = "stablehlo.reshape"
    operand = operand_def(HLO_AnyTensor)
    result = result_def(HLO_AnyTensor)

    assembly_format = """
    operands attr-dict `:` functional-type(operands, results)
    """

    traits = traits_def(
        NoMemoryEffect(),
        ConditionallySpeculatable(),
        # TODO: HLO_CompatibleOperandsAndResultElementType,
    )

    def verify_(self) -> None:
        """Verify that the operation has the same shape for all operands and results."""
        o_type = self.operand_types[0]
        r_type = self.result_types[0]

        # These are constrained to tensors by the op definition
        assert isinstance(o_type, TensorType) and isinstance(r_type, TensorType)

        # If o_type or r_type is dynamically shaped there is nothing to verify.
        if not o_type.has_static_shape() or not r_type.has_static_shape():
            return

        # If the operand type is statically shaped (not required) the number of
        # elements must match that of the result type.
        num_operand_elements = 1
        for dim in o_type.get_shape():
            num_operand_elements *= dim

        num_result_elements = 1
        for dim in r_type.get_shape():
            num_result_elements *= dim

        if num_result_elements != num_operand_elements:
            raise VerifyException(
                "number of output elements ("
                f"{num_result_elements}"
                ") doesn't match expected number of elements ("
                f"{num_operand_elements}"
                ")"
            )


@irdl_op_definition
class ScatterOp(IRDLOperation):
    """
     Produces ``results`` tensors which are equal to ``inputs`` tensors except that
     several slices specified by ``scatter_indices`` are updated with the values
     ``updates`` using ``update_computation``.

     See:
     https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter

    Example:
    ```mlir
    %result = "stablehlo.scatter"(%input, %scatter_indices, %update) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [3, 4],
        inserted_window_dims = [1],
        input_batching_dims = [0],
        scatter_indices_batching_dims = [1],
        scatter_dims_to_operand_dims = [2, 1],
        index_vector_dim = 3>,
      indices_are_sorted = false,
      unique_indices = false
    } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
    ```
    """

    name = "stablehlo.scatter"
    inputs = var_operand_def(HLO_Tensor)
    scatter_indices = operand_def(HLO_AnyIntegerOrIndexTensor)
    updates = var_operand_def(HLO_Tensor)
    scatter_dimension_numbers = prop_def(ScatterDimensionNumbers)
    indices_are_sorted = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    unique_indices = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    result = var_result_def(HLO_Tensor)
    update_computation = region_def("single_block")
    # TODO: The MLIR implementation doesn't have the SingleBlockImplicitTerminator trait,
    # However, it is checked to have a terminator in the verifier,
    # which does not specifically check the terminator to be stablehlo.return.

    traits = traits_def(
        RecursiveMemoryEffect(),
        ConditionallySpeculatable(),
        # TODO: InferTypeOpInterface(),
    )

    irdl_options = [SameVariadicOperandSize()]

    # TODO: MLIR has a custom verifier for the scatter operation.


@irdl_op_definition
class SliceOp(IRDLOperation):
    """
    Extracts a slice from the ``operand`` using statically-computed starting
    indices and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = stablehlo.slice %operand [1:3, 4:8:2]
       : (tensor<3x8xi64>) -> tensor<2x2xi64>

    // Same in generic form: the `1:3` above is mapped to the first entry in
    // `start_indices` and `limit_indices`, while `strides` is implicitly 1.
    // The `4:8:2` above is parsed into the second entry of `start_indices`,
    // `limit_indices` and `strides` respectively.
    %result = "stablehlo.slice" (%operand) {
      start_indices = array<i64: 1, 4>,
      limit_indices = array<i64: 3, 8>,
      strides = array<i64: 1, 2>
    } : (tensor<3x8xi64>) -> tensor<2x2xi64>
    ```
    """

    name = "stablehlo.slice"

    operand = operand_def(HLO_Tensor)
    start_indices = prop_def(DenseArrayBase.constr(i64))
    limit_indices = prop_def(DenseArrayBase.constr(i64))
    strides = prop_def(DenseArrayBase.constr(i64))
    result = result_def(HLO_Tensor)

    # TODO: Implement CustomDirective
    # assembly_format = """
    #   $operand custom<SliceRanges>($start_indices, $limit_indices, $strides)
    #   attr-dict `:` functional-type(operands, results)
    # """

    traits = traits_def(
        NoMemoryEffect(),
        ConditionallySpeculatable(),
        AllMatchSameOperatorTrait(("start_indices", "limit_indices", "strides"), len, "size"),
        SameOperandsAndResultElementType(),
    )
