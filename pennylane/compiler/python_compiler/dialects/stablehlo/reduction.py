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

from xdsl.dialects import stablehlo as xstablehlo
from xdsl.dialects.builtin import DenseArrayBase, i64
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.irdl.operations import SameVariadicOperandSize
from xdsl.traits import (
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException

from .types import HLO_Tensor


@irdl_op_definition
class ReduceOp(IRDLOperation):
    """
    Applies a reduction function ``body`` to ``inputs`` and ``init_values`` along the
    ``dimensions`` and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce

    Example:
    ```mlir
    %result = "stablehlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    ```
    """

    name = "stablehlo.reduce"

    inputs = var_operand_def(HLO_Tensor)
    init_values = var_operand_def(HLO_Tensor)
    dimensions = prop_def(DenseArrayBase.constr(i64))
    result = var_result_def(HLO_Tensor)
    body = region_def("single_block")

    irdl_options = [SameVariadicOperandSize()]

    traits = traits_def(
        RecursiveMemoryEffect(),
        # TODO: InferShapedTypeOpInterface(),
        # TODO: HLO_RecursivelySpeculatableIfAllInputsStatic,
        # TODO: InferTensorTypeWithReify(),
        SingleBlockImplicitTerminator(xstablehlo.ReturnOp),
    )

    # pylint: disable=no-member
    # pylint: disable=too-many-branches
    def verify_(self):
        """Verify the ReduceOp."""
        # Gather shaped operand/result types
        input_types = [op.type for op in self.inputs]
        init_types = [op.type for op in self.init_values]

        # reduce_c1/c4/c5/i3: verify inputs and infer shape compatibility
        dims_attr = self.dimensions
        dims = tuple(dims_attr.get_values()) if dims_attr is not None else tuple()

        # Basic structural checks mirroring verifyReduceOpInputsAndInferShape
        if len(input_types) == 0:
            raise VerifyException("expected at least 1 input for reduce")
        if len(input_types) != len(init_types):
            raise VerifyException("number of inputs must match number of init_values")

        # All inputs must have equal rank; dimensions must be within rank and unique
        # and not empty.
        ranks = []
        for t in input_types:
            # Tensors by op definition
            assert hasattr(t, "get_num_dims")
            ranks.append(t.get_num_dims())
        rank0 = ranks[0]
        if any(r != rank0 for r in ranks):
            raise VerifyException("all inputs must have the same rank")

        if len(dims) == 0:
            raise VerifyException("dimensions cannot be empty for reduce")
        if len(set(dims)) != len(dims):
            raise VerifyException("dimensions should not have duplicates")
        if any(d < 0 or d >= rank0 for d in dims):
            raise VerifyException("dimensions contains an invalid value")

        # Element type compatibility between each input and its init value
        for it, iv in zip(input_types, init_types):
            it_elem = it.get_element_type()
            iv_elem = iv.get_element_type()
            if it_elem != iv_elem:
                raise VerifyException("input and init_value must have the same element type")

        # reduce_c2/c6: verify reducer region shape
        # Expect block with arity 2 * number of inputs, with matching tensor element types and 0D tensors
        if len(self.body.blocks) != 1:
            raise VerifyException("reducer must have a single block")
        block = self.body.blocks[0]

        expected_args = 2 * len(input_types)
        if len(block.args) != expected_args:
            raise VerifyException(
                f"reducer must take {expected_args} arguments, got {len(block.args)}"
            )

        # Each pair (arg_i, arg_{i+N}) must be 0D tensors of the input element type
        for i, it in enumerate(input_types):
            it_elem = it.get_element_type()
            acc = block.args[i]
            val = block.args[i + len(input_types)]
            for a in (acc, val):
                a_ty = a.type
                if not hasattr(a_ty, "get_num_dims") or a_ty.get_num_dims() != 0:
                    raise VerifyException("reducer arguments must be rank-0 tensors")
                if a_ty.get_element_type() != it_elem:
                    raise VerifyException(
                        "reducer argument element types must match input element type"
                    )

        # Region must terminate with exactly len(inputs) results
        ret = block.ops.last
        if len(ret.operands) != len(input_types):
            raise VerifyException("reducer must return exactly one value per input")
        for i, it in enumerate(input_types):
            it_elem = it.get_element_type()
            rty = ret.operands[i].type
            if not hasattr(rty, "get_num_dims") or rty.get_num_dims() != 0:
                raise VerifyException("reducer return values must be rank-0 tensors")
            if rty.get_element_type() != it_elem:
                raise VerifyException("reducer return element types must match input element type")
