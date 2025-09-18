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

This file was originally ported automatically by xDSL (using the ``xdsl-tblgen`` tool) and modified manually
to support the Python compiler.

The catalyst dialect serves as a standard library for the Catalyst compiler.
It contains data structures that support core compiler functionality.
"""

# pylint: disable=too-few-public-methods

from typing import ClassVar

from xdsl.dialects.builtin import (
    I64,
    ArrayAttr,
    DenseArrayBase,
    DictionaryAttr,
    FlatSymbolRefAttrConstr,
    FunctionType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    i32,
)
from xdsl.ir import AttributeCovT, Dialect, Generic, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    ParsePropInAttrDict,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)


@irdl_attr_definition
class ArrayListType(Generic[AttributeCovT], ParametrizedAttribute, TypeAttribute):
    """A dynamically resizable array"""

    name = "catalyst.arraylist"

    element_type: AttributeCovT


@irdl_op_definition
class AssertionOp(IRDLOperation):
    """Asserts condition at runtime."""

    name = "catalyst.assert"

    assertion = operand_def(IntegerType(1))

    error = prop_def(StringAttr)


@irdl_op_definition
class CallbackCallOp(IRDLOperation):
    """CallbackCallOp operation."""

    name = "catalyst.callback_call"

    assembly_format = """
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
      """

    callee = prop_def(FlatSymbolRefAttrConstr)

    inputs = var_operand_def()

    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    callback_results = var_result_def()

    irdl_options = [ParsePropInAttrDict()]


@irdl_op_definition
class CallbackOp(IRDLOperation):
    """Operation denoting a symbol to refer to user callbacks."""

    name = "catalyst.callback"

    sym_name = prop_def(StringAttr)

    function_type = prop_def(FunctionType)

    id = prop_def(IntegerAttr[I64])

    argc = prop_def(IntegerAttr[I64])

    resc = prop_def(IntegerAttr[I64])

    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    body = region_def()


@irdl_op_definition
class CustomCallOp(IRDLOperation):
    """CustomCall operation"""

    name = "catalyst.custom_call"

    assembly_format = """
        `fn` `(`$call_target_name`)` `(` $inputs `)`
          attr-dict `:` functional-type(operands, results)
      """

    inputs = var_operand_def()

    call_target_name = prop_def(StringAttr)

    number_original_arg = opt_prop_def(DenseArrayBase.constr(i32))

    custom_results = var_result_def()

    irdl_options = [ParsePropInAttrDict()]


@irdl_op_definition
class LaunchKernelOp(IRDLOperation):
    """LaunchKernelOp operation."""

    name = "catalyst.launch_kernel"

    assembly_format = """
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
      """

    callee = prop_def(SymbolRefAttr)

    inputs = var_operand_def()

    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    kernel_results = var_result_def()

    irdl_options = [ParsePropInAttrDict()]


@irdl_op_definition
class ListDeallocOp(IRDLOperation):
    """Deallocate the underlying memory of an arraylist."""

    name = "catalyst.list_dealloc"

    assembly_format = """ $list attr-dict `:` type($list) """

    list = operand_def(ArrayListType)


@irdl_op_definition
class ListInitOp(IRDLOperation):
    """Initialize a dynamically resizable arraylist."""

    name = "catalyst.list_init"

    assembly_format = """ attr-dict `:` type($list) """

    list = result_def(ArrayListType)


@irdl_op_definition
class ListLoadDataOp(IRDLOperation):
    """Get the underlying memref storing the data of an array list."""

    name = "catalyst.list_load_data"

    assembly_format = """ $list attr-dict `:` type($list) `->` type($data) """

    list = operand_def(ArrayListType)

    data = result_def(MemRefType)


@irdl_op_definition
class ListPopOp(IRDLOperation):
    """Remove an element from the end of an array list and return it."""

    name = "catalyst.list_pop"

    assembly_format = """ $list attr-dict `:` type($list) """

    T: ClassVar = VarConstraint("T", AnyAttr())

    list = operand_def(base(ArrayListType[T]))

    result = result_def(T)


@irdl_op_definition
class ListPushOp(IRDLOperation):
    """Append an element to the end of an array list."""

    name = "catalyst.list_push"

    assembly_format = """ $value `,` $list attr-dict `:` type($list) """

    T: ClassVar = VarConstraint("T", AnyAttr())

    value = operand_def(T)

    list = operand_def(base(ArrayListType[T]))


@irdl_op_definition
class PrintOp(IRDLOperation):
    """Prints numeric values or constant strings at runtime."""

    name = "catalyst.print"

    val = opt_operand_def()

    const_val = opt_prop_def(StringAttr)

    print_descriptor = prop_def(UnitAttr)


Catalyst = Dialect(
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
        ListPopOp,
        ListPushOp,
        PrintOp,
    ],
    [
        ArrayListType,
    ],
)
