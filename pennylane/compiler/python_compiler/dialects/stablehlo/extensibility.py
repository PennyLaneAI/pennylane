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


from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    DenseIntElementsAttr,
    DictionaryAttr,
    FlatSymbolRefAttr,
    StringAttr,
    TensorType,
    TupleType,
)
from xdsl.ir import Attribute
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import (
    MemoryEffect,
)
from xdsl.utils.exceptions import VerifyException

from .attributes import CustomCallApiVersion, CustomCallApiVersionAttr, OutputOperandAlias


@irdl_op_definition
class CustomCallOp(IRDLOperation):
    """
    Encapsulates an implementation-defined operation ``call_target_name`` that
    takes ``inputs`` and ``called_computations`` and produces ``results``.

    Depending on the API version there are two ways to pass extra bits of static
    information to the external function:
    1. Use ``API_VERSION_TYPED_FFI`` which allows passing a dictionary attribute.
    2. Use a previous API version with a ``StringAttr`` to encode backend config.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#custom_call

    Example:
    ```mlir
    %results = stablehlo.custom_call @foo(%input0) {
      backend_config = {bar = 42 : i32},
      api_version = 4 : i32,
      called_computations = [@foo]
    } : (tensor<f64>) -> tensor<f64>
    ```
    """

    name = "stablehlo.custom_call"

    inputs = var_operand_def(AnyAttr())
    call_target_name = prop_def(StringAttr)
    has_side_effect = prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    backend_config = opt_prop_def(DictionaryAttr | StringAttr)
    api_version = prop_def(
        CustomCallApiVersionAttr,
        default_value=CustomCallApiVersionAttr(CustomCallApiVersion.API_VERSION_ORIGINAL),
    )
    called_computations = opt_prop_def(ArrayAttr[FlatSymbolRefAttr], default_value=ArrayAttr([]))
    operand_layouts = opt_prop_def(ArrayAttr[DenseIntElementsAttr])
    result_layouts = opt_prop_def(ArrayAttr[DenseIntElementsAttr])
    output_operand_aliases = prop_def(ArrayAttr[OutputOperandAlias])

    result = var_result_def(AnyAttr())

    traits = traits_def(
        MemoryEffect(),
    )

    # TODO: Implement CustomDirective
    # assembly_format = """
    #     custom<CustomCallTarget>($call_target_name) `(` $inputs `)`
    #     attr-dict `:` functional-type(operands, results)
    # """

    def verify_(self) -> None:
        """Verify the CustomCallOp."""
        # If both operand and result layout attributes are not specified then nothing to verify.
        if self.operand_layouts is None and self.result_layouts is None:
            return

        # Layout constraints for either both operands & results or none should be specified.
        if (self.operand_layouts is None) != (self.result_layouts is None):
            raise VerifyException(
                "Layout attributes should be specified for either both operands and results or none."
            )

        assert self.operand_layouts is not None and self.result_layouts is not None

        def verify_types_and_layouts(
            types: tuple[Attribute, ...], layouts: ArrayAttr, value_name: str
        ):
            if len(types) != len(layouts.data):
                raise VerifyException(
                    "Number of "
                    f"{value_name}s must match the number of {value_name} layouts, "
                    f"{len(types)} != {len(layouts.data)}"
                )

            for index, (ty, layout_attr) in enumerate(zip(types, layouts.data)):
                # Tuple types are not fully supported with layout constraints yet
                if isinstance(ty, TupleType):
                    raise VerifyException(
                        "Tuple types are not fully supported with layout constraints yet"
                    )

                try:
                    dims = list(layout_attr.get_values())
                except Exception as exc:
                    raise VerifyException("invalid layout attribute") from exc

                # For non-tensor types, layout must be empty
                if not isinstance(ty, TensorType):
                    if len(dims) == 0:
                        continue
                    raise VerifyException(
                        "Only tensor types can have non-empty layout: "
                        f"{value_name} #{index} of type {ty} has layout {dims}"
                    )

                # For ranked tensors, require permutation of [0, rank)
                rank = ty.get_num_dims()
                if rank != len(dims) or sorted(dims) != list(range(rank)):
                    raise VerifyException(
                        f"incorrect layout {dims} for type {ty}, layout must be a permutation of [0, {rank})"
                    )

        # Operand types
        operand_types: tuple[Attribute, ...] = tuple(op.type for op in self.operands)

        # Result types: if single tuple result, use its element types
        if len(self.result_types) == 1 and isinstance(self.result_types[0], TupleType):
            tuple_ty: TupleType = self.result_types[0]
            result_types = tuple(tuple_ty.types.data)
        else:
            result_types = tuple(self.result_types)

        # Verify that operands and operand layouts match.
        verify_types_and_layouts(operand_types, self.operand_layouts, "operand")
        # Verify that results and result layouts match.
        verify_types_and_layouts(result_types, self.result_layouts, "result")
