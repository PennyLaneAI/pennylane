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

from dataclasses import dataclass
from typing import ClassVar, TypeVar

from xdsl.dialects import stablehlo as xstablehlo
from xdsl.dialects.builtin import BoolAttr, DenseArrayBase, IntegerAttr, i64
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
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
    OpTrait,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException

from .attributes import GatherDimensionNumbers, ScatterDimensionNumbers
from .types import HLO_AnyIntegerOrIndexTensor, HLO_AnyTensor, HLO_IntTensor, HLO_Tensor


@dataclass(frozen=True)
class SliceArraysSameSizeTrait(OpTrait):
    """
    Trait that ensures start_indices, limit_indices, and strides arrays
    all have the same size (equivalent to MLIR's AllMatchSameOperatorTrait).
    """

    def verify(self, op) -> None:
        """Verify that all three slice arrays have the same size."""
        # Access the attributes from the operation
        start_indices = getattr(op, "start_indices", None)
        limit_indices = getattr(op, "limit_indices", None)
        strides = getattr(op, "strides", None)

        if start_indices is None or limit_indices is None or strides is None:
            return

        if not (
            isinstance(start_indices, DenseArrayBase)
            and isinstance(limit_indices, DenseArrayBase)
            and isinstance(strides, DenseArrayBase)
        ):
            return

        # Use the built-in __len__ method of DenseArrayBase
        start_size = len(start_indices)
        limit_size = len(limit_indices)
        strides_size = len(strides)

        if not start_size == limit_size == strides_size:
            raise VerifyException(
                f"all of {{start_indices, limit_indices, strides}} have same size: "
                f"got sizes {start_size}, {limit_size}, {strides_size}"
            )


@irdl_op_definition
class BroadcastInDimOp(IRDLOperation):
    """
    Expands the dimensions and/or rank of an input tensor by duplicating the
    data in the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

    Example:
    ```mlir
    %result = stablehlo.broadcast_in_dim %operand, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    ```
    """

    name = "stablehlo.broadcast_in_dim"
    operand = operand_def(HLO_AnyTensor)
    broadcast_dimensions = attr_def(DenseArrayBase.constr(i64))
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

    # TODO: MLIR has a custom verifier for the broadcast_in_dim operation.


@irdl_op_definition
class ConcatenateOp(IRDLOperation):
    """
    Concatenates a variadic number of tensors in `inputs` along `dimension`
    dimension in the same order as the given arguments and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate

    Example:
    ```mlir
    %result = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    ```
    """

    name = "stablehlo.concatenate"

    T: ClassVar = TypeVar("T", bound=HLO_Tensor)
    inputs = var_operand_def(T)
    result = result_def(T)
    dimension = attr_def(IntegerAttr.constr(type=eq(i64), value=AtLeast(0)))

    traits = traits_def(
        NoMemoryEffect(),
        ConditionallySpeculatable(),
        SingleBlockImplicitTerminator(xstablehlo.ReturnOp),
        # InferTypeOpInterface(),
    )

    # TODO: Implement CustomDirective
    # assembly_format = """
    #   custom<VariadicOperandWithAttribute>($inputs) `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
    # """


@irdl_op_definition
class GatherOp(IRDLOperation):
    """
    Gathers slices from `operand` tensor from offsets specified in
    `start_indices` and produces a `result` tensor.

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
    T: ClassVar = TypeVar("T", bound=HLO_Tensor)
    operand = operand_def(T)
    start_indices = operand_def(HLO_IntTensor)
    dimension_numbers = attr_def(GatherDimensionNumbers)
    slice_sizes = attr_def(DenseArrayBase.constr(i64))
    indices_are_sorted = opt_attr_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    result = result_def(T)

    traits = traits_def(
        NoMemoryEffect(),
        ConditionallySpeculatable(),
        # TODO: InferTensorTypeWithReify(),
    )

    # TODO: Implement CustomDirective
    # assembly_format = """
    #   custom<VariadicOperandWithAttribute>($inputs) `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
    # """


@irdl_op_definition
class ReshapeOp(IRDLOperation):
    """
    Performs reshape of `operand` tensor to a `result` tensor.

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

    # TODO: MLIR has a custom verifier for the reshape operation.


@irdl_op_definition
class ScatterOp(IRDLOperation):
    """
     Produces `results` tensors which are equal to `inputs` tensors except that
     several slices specified by `scatter_indices` are updated with the values
     `updates` using `update_computation`.

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
    scatter_dimension_numbers = attr_def(ScatterDimensionNumbers)
    indices_are_sorted = opt_attr_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    unique_indices = opt_attr_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    result = var_result_def(HLO_Tensor)
    update_computation = region_def("single_block")

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
    Extracts a slice from the `operand` using statically-computed starting
    indices and produces a `result` tensor.

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

    T: ClassVar = TypeVar("T", bound=HLO_Tensor)
    operand = operand_def(T)
    start_indices = attr_def(DenseArrayBase.constr(i64))
    limit_indices = attr_def(DenseArrayBase.constr(i64))
    strides = attr_def(DenseArrayBase.constr(i64))
    result = result_def(T)

    # TODO: Implement CustomDirective
    # assembly_format = """
    #   $operand custom<SliceRanges>($start_indices, $limit_indices, $strides)
    #   attr-dict `:` functional-type(operands, results)
    # """

    traits = traits_def(
        NoMemoryEffect(),
        ConditionallySpeculatable(),
        SliceArraysSameSizeTrait(),
        # TODO: HLO_SpeculatableIfStaticDimInOutputIsStaticInInput,
        # TODO: InferTypeOpInterface(),
    )
