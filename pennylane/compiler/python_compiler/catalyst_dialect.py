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
This file contains the Catalyst dialect for the Python compiler.

This file is was originally ported automatically by xDSL and modified manually
to support the Python compiler.

The catalyst dialect serves as a standard library for the Catalyst compiler.
It contains data structures that support core compiler functionality.
"""

# ruff: noqa: F403, F405

from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *


@irdl_attr_definition
class ArrayListType(ParametrizedAttribute, TypeAttribute):
    """a dynamically resizable array"""

    name = "catalyst.arraylist"


@irdl_op_definition
class AssertionOp(IRDLOperation):
    """Asserts condition at runtime."""

    name = "catalyst.assert"

    assertion = operand_def(EqAttrConstraint(IntegerType(1)))

    error = prop_def(BaseAttr(StringAttr))


@irdl_op_definition
class CallbackCallOp(IRDLOperation):
    name = "catalyst.callback_call"

    assembly_format = """
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
      """

    callee = prop_def(AnyAttr())

    inputs = var_operand_def(AnyAttr())

    arg_attrs = opt_prop_def(AnyAttr())

    res_attrs = opt_prop_def(AnyAttr())

    v1 = var_result_def(AnyAttr())

    irdl_options = [ParsePropInAttrDict()]


@irdl_op_definition
class CallbackOp(IRDLOperation):
    """Operation denoting a symbol to refer to user callbacks."""

    name = "catalyst.callback"

    sym_name = prop_def(BaseAttr(StringAttr))

    function_type = prop_def(AnyAttr())

    id = prop_def(IntegerAttr.constr(type=EqAttrConstraint(IntegerType(64))))

    argc = prop_def(IntegerAttr.constr(type=EqAttrConstraint(IntegerType(64))))

    resc = prop_def(IntegerAttr.constr(type=EqAttrConstraint(IntegerType(64))))

    arg_attrs = opt_prop_def(AnyAttr())

    res_attrs = opt_prop_def(AnyAttr())

    body = region_def()


@irdl_op_definition
class CustomCallOp(IRDLOperation):
    """CustomCall operation"""

    name = "catalyst.custom_call"

    assembly_format = """
        `fn` `(`$call_target_name`)` `(` $inputs `)`
          attr-dict `:` functional-type(operands, results)
      """

    inputs = var_operand_def(AnyAttr())

    call_target_name = prop_def(BaseAttr(StringAttr))

    number_original_arg = opt_prop_def(AnyAttr())

    v2 = var_result_def(AnyAttr())

    irdl_options = [ParsePropInAttrDict()]


@irdl_op_definition
class LaunchKernelOp(IRDLOperation):
    name = "catalyst.launch_kernel"

    assembly_format = """
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
      """

    callee = prop_def(AnyAttr())

    inputs = var_operand_def(AnyAttr())

    arg_attrs = opt_prop_def(AnyAttr())

    res_attrs = opt_prop_def(AnyAttr())

    v3 = var_result_def(AnyAttr())

    irdl_options = [ParsePropInAttrDict()]


@irdl_op_definition
class ListDeallocOp(IRDLOperation):
    """Deallocate the underlying memory of an arraylist."""

    name = "catalyst.list_dealloc"

    assembly_format = """ $list attr-dict `:` type($list) """

    list = operand_def(BaseAttr(ArrayListType))


@irdl_op_definition
class ListInitOp(IRDLOperation):
    """Initialize a dynamically resizable arraylist."""

    name = "catalyst.list_init"

    assembly_format = """ attr-dict `:` type($list) """

    list = result_def(BaseAttr(ArrayListType))


@irdl_op_definition
class ListLoadDataOp(IRDLOperation):
    """Get the underlying memref storing the data of an array list."""

    name = "catalyst.list_load_data"

    assembly_format = """ $list attr-dict `:` type($list) `->` type($data) """

    list = operand_def(BaseAttr(ArrayListType))

    data = result_def(AnyAttr())

# TODO: The following class has been modified from the original xDSL generated code.
# The modification was unsuccessfully modified to capture the functionality of TypesMatchWith in
# MLIR to match the type of result and the type of the list elements.
#
# @irdl_op_definition
# class ListPopOp(IRDLOperation):
#     """Remove an element from the end of an array list and return it."""

#     name = "catalyst.list_pop"

#     assembly_format = """ $list attr-dict `:` type($list) """

#     T = VarConstraint("T", AnyAttr())

#     list = operand_def(BaseAttr(ArrayListType(T)))

#     result = result_def(T)


# @irdl_op_definition
# class ListPushOp(IRDLOperation):
#     """Append an element to the end of an array list."""

#     name = "catalyst.list_push"

#     assembly_format = """ $value `,` $list attr-dict `:` type($list) """

#     value = operand_def(AnyAttr())

#     list = operand_def(BaseAttr(ArrayListType))


@irdl_op_definition
class PrintOp(IRDLOperation):
    """Prints numeric values or constant strings at runtime."""

    name = "catalyst.print"

    val = opt_operand_def(AnyAttr())

    const_val = opt_prop_def(BaseAttr(StringAttr))

    print_descriptor = prop_def(EqAttrConstraint(UnitAttr()))


Catalyst_Dialect = Dialect(
    "catalyst",
    [
        AssertionOp,
        CallbackCallOp,
        CallbackOp,
        CustomCallOp,
        LaunchKernelOp,
        ListDeallocOp,
        ListInitOp,
        ListLoadDataOp,
        # ListPopOp,
        # ListPushOp,
        PrintOp,
    ],
    [
        ArrayListType,
    ],
)
