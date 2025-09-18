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

from xdsl.dialects.builtin import DenseArrayBase, TensorType, i64
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import (
    ConditionallySpeculatable,
    NoMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException

from ...xdsl_extras import TensorConstraint
from .types import HLO_AnyTensor, HLO_DimensionValue


@irdl_op_definition
class DynamicBroadcastInDimOp(IRDLOperation):
    """
    This operation is functionally identical to
    [broadcast_in_dim](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim)
    op, but the result shape is specified dynamically via ``output_dimensions``.

    It also accepts optional attributes to express static knowledge about the
    expanding behavior of dimensions. If not specified, all dimensions are
    assumed to be possibly expanding. The sets of dimensions that are known to
    be expanding and the set of dimensions that are known to be non-expanding
    must be disjoint and they must be a subset of the operand's dimensions.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_broadcast_in_dim

    Example:
    ```mlir
    %operand = stablehlo.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
    %output_dimensions = stablehlo.constant dense<[2, 3, 2]> : tensor<3xi64>
    %result = "stablehlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
      broadcast_dimensions = array<i64: 2, 1>,
      known_expanding_dimensions = array<i64: 0>,
      known_nonexpanding_dimensions = array<i64: 1>
    } : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    ```
    """

    name = "stablehlo.dynamic_broadcast_in_dim"

    operand = operand_def(HLO_AnyTensor)
    output_dimensions = operand_def(TensorConstraint(element_type=HLO_DimensionValue, rank=1))
    broadcast_dimensions = prop_def(DenseArrayBase.constr(i64))
    known_expanding_dimensions = opt_prop_def(DenseArrayBase.constr(i64))
    known_nonexpanding_dimensions = opt_prop_def(DenseArrayBase.constr(i64))
    result = result_def(HLO_AnyTensor)

    assembly_format = (
        "$operand `,` $output_dimensions `,` `dims` `=` $broadcast_dimensions "
        "attr-dict `:` functional-type(operands, results)"
    )

    traits = traits_def(
        ConditionallySpeculatable(),
        NoMemoryEffect(),
        # TODO: InferShapedTypeOpInterface(),
    )

    irdl_options = [ParsePropInAttrDict()]

    # pylint: disable=too-many-branches
    def verify_(self):
        """Verify the operation."""
        # Operand and result must be tensors
        operand_ty = self.operand_types[0]
        result_ty = self.result_types[0]
        assert isinstance(operand_ty, TensorType) and isinstance(result_ty, TensorType)

        # dynamic_broadcast_in_dim_c2: broadcast_dimensions size == operand rank
        bcast_dims = tuple(self.broadcast_dimensions.get_values())  # pylint: disable=no-member
        operand_rank = operand_ty.get_num_dims()
        if len(bcast_dims) != operand_rank:
            raise VerifyException(
                "broadcast_dimensions size ("
                f"{len(bcast_dims)}"
                ") does not match operand rank ("
                f"{operand_rank}"
                ")"
            )

        # dynamic_broadcast_in_dim_c3: result rank >= operand rank
        result_rank = result_ty.get_num_dims()
        if result_rank < operand_rank:
            raise VerifyException(
                "result rank ("
                f"{result_rank}"
                ") is less than operand rank ("
                f"{operand_rank}"
                ")"
            )

        # dynamic_broadcast_in_dim_c4: broadcast_dimensions should not have duplicates
        if len(set(bcast_dims)) != len(bcast_dims):
            raise VerifyException("broadcast_dimensions should not have duplicates")

        # dynamic_broadcast_in_dim_c5: bounds and per-dimension compatibility
        operand_shape = operand_ty.get_shape()
        result_shape = result_ty.get_shape()
        for i, dim_index in enumerate(bcast_dims):
            if dim_index < 0 or dim_index >= result_rank:
                raise VerifyException(
                    "broadcast_dimensions contains invalid value "
                    f"{dim_index} for result with rank {result_rank}"
                )
            op_dim = operand_shape[i]
            res_dim = result_shape[dim_index]
            # If operand dim is static and not size-1, require compatibility with result dim
            if op_dim not in (-1, 1):
                if res_dim not in (-1, op_dim):
                    raise VerifyException(
                        "size of operand dimension "
                        f"{i} ({op_dim}) is not compatible with size of result dimension "
                        f"{dim_index} ({res_dim})"
                    )

        # dynamic_broadcast_in_dim_c7: output_dimensions shape compatible with result rank
        out_dims_ty = self.output_dimensions.type  # pylint: disable=no-member
        assert isinstance(out_dims_ty, TensorType)
        # Must be rank-1 tensor (enforced by type constraint), and length must match result rank when statically known
        out_shape = out_dims_ty.get_shape()
        if len(out_shape) != 1:
            raise VerifyException("output_dimensions must be a 1D tensor")
        if out_shape[0] != -1 and out_shape[0] != result_rank:
            raise VerifyException(
                "length of output_dimensions ("
                f"{out_shape[0]}"
                ") is not compatible with result rank ("
                f"{result_rank}"
                ")"
            )

        # dynamic_broadcast_in_dim_c8: no duplicate expansion hints across both lists
        hints = []
        if self.known_expanding_dimensions is not None:
            hints.extend(self.known_expanding_dimensions.get_values())  # pylint: disable=no-member
        if self.known_nonexpanding_dimensions is not None:
            hints.extend(
                self.known_nonexpanding_dimensions.get_values()  # pylint: disable=no-member
            )
        if len(set(hints)) != len(hints):
            raise VerifyException("duplicate expansion hint for at least one operand dimension")

        # dynamic_broadcast_in_dim_c9/c10: each hint must reference a valid operand dimension
        for h in set(hints):
            if h < 0 or h >= operand_rank:
                raise VerifyException(
                    "hint for expanding dimension "
                    f"{h} does not refer to a valid operand dimension"
                )
