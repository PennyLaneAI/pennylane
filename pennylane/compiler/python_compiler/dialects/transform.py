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
This file contains an updated version of the transform dialect.
As of the time of writing, xDSL uses the MLIR released with LLVM's
version 20.1.7. However, https://github.com/PennyLaneAI/catalyst/pull/1916
will be updating MLIR where the transform dialect has the
`apply_registered_pass` operation re-defined.

See the following changelog on the above PR

    Things related to transform.apply_registered_pass op:

    It now takes in a dynamic_options

    [MLIR][Transform] Allow ApplyRegisteredPassOp to take options as
    a param llvm/llvm-project#142683. We don't need to use this as all our pass options are static.
    https://github.com/llvm/llvm-project/pull/142683

    The options it takes in are now dictionaries instead of strings
    [MLIR][Transform] apply_registered_pass op's options as a dict llvm/llvm-project#143159
    https://github.com/llvm/llvm-project/pull/143159

This file will re-define the apply_registered_pass operation in xDSL
and the transform dialect.

Once xDSL moves to a newer version of MLIR, these changes should
be contributed upstream.
"""

from xdsl.dialects.builtin import Dialect

# pylint: disable=too-few-public-methods
from xdsl.dialects.transform import ApplyRegisteredPassOp as xApplyRegisteredPassOp
from xdsl.dialects.transform import (
    DictionaryAttr,
    StringAttr,
)
from xdsl.dialects.transform import Transform as xTransform
from xdsl.dialects.transform import (
    TransformHandleType,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import IRDLOperation, ParsePropInAttrDict


@irdl_op_definition
class ApplyRegisteredPassOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformapply_registered_pass-transformapplyregisteredpassop).
    """

    name = "transform.apply_registered_pass"

    options = prop_def(DictionaryAttr, default_value=DictionaryAttr({}))
    pass_name = prop_def(StringAttr)
    target = operand_def(TransformHandleType)
    result = result_def(TransformHandleType)
    # While this assembly format doesn't match
    # the one in upstream MLIR,
    # this is because xDSL currently lacks CustomDirectives
    # https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-directives
    # https://github.com/xdslproject/xdsl/pull/4829
    # However, storing the property in the attribute should still work
    # specially when parsing and printing in generic format.
    # Which is how Catalyst and XDSL currently communicate at the moment.
    # TODO: Add test.
    assembly_format = "$pass_name `to` $target attr-dict `:` functional-type(operands, results)"
    irdl_options = [ParsePropInAttrDict()]

    def __init__(
        self,
        pass_name: str | StringAttr,
        target: SSAValue,
        options: dict[str | StringAttr, Attribute | str | bool | int] | None = None,
    ):
        if isinstance(pass_name, str):
            pass_name = StringAttr(pass_name)

        if isinstance(options, dict):
            options = DictionaryAttr(options)

        super().__init__(
            properties={
                "pass_name": pass_name,
                "options": options,
            },
            operands=[target],
            result_types=[target.type],
        )


# Copied over from xDSL's sources
# the main difference will be the use
# of a different ApplyRegisteredPassOp
operations = list(xTransform.operations)
del operations[operations.index(xApplyRegisteredPassOp)]
operations.append(ApplyRegisteredPassOp)

Transform = Dialect(
    "transform",
    [
        *operations,
    ],
    [
        *xTransform.attributes,
    ],
)
