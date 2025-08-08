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

"""Unit test module for pennylane/compiler/python_compiler/dialects/catalyst.py."""

import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

pytestmark = pytest.mark.external

from pennylane.compiler.python_compiler.dialects import Catalyst

all_ops = list(Catalyst.operations)
all_attrs = list(Catalyst.attributes)

expected_ops_names = {
    "AssertionOp": "catalyst.assert",
    "CallbackCallOp": "catalyst.callback_call",
    "CallbackOp": "catalyst.callback",
    "CustomCallOp": "catalyst.custom_call",
    "LaunchKernelOp": "catalyst.launch_kernel",
    "ListDeallocOp": "catalyst.list_dealloc",
    "ListInitOp": "catalyst.list_init",
    "ListLoadDataOp": "catalyst.list_load_data",
    "ListPopOp": "catalyst.list_pop",
    "ListPushOp": "catalyst.list_push",
    "PrintOp": "catalyst.print",
}

expected_attrs_names = {
    "ArrayListType": "catalyst.arraylist",
}


def test_catalyst_dialect_name():
    """Test that the Catalyst dialect name is correct."""
    assert Catalyst.name == "catalyst"


@pytest.mark.parametrize("op", all_ops)
def test_all_operations_names(op):
    """Test that all operations have the expected name."""
    op_class_name = op.__name__
    expected_name = expected_ops_names.get(op_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected operation {op_class_name} found in Catalyst dialect"
    assert op.name == expected_name


@pytest.mark.parametrize("attr", all_attrs)
def test_all_attributes_names(attr):
    """Test that all attributes have the expected name."""
    attr_class_name = attr.__name__
    expected_name = expected_attrs_names.get(attr_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected attribute {attr_class_name} found in Catalyst dialect"
    assert attr.name == expected_name


def test_assembly_format(run_filecheck):
    """Test the assembly format of the catalyst ops."""
    program = """
    // CHECK: [[LIST:%.+]] = catalyst.list_init : !catalyst.arraylist<f64>
    %list = catalyst.list_init : !catalyst.arraylist<f64>

    // CHECK: [[DATA:%.+]] = catalyst.list_load_data [[LIST]] : !catalyst.arraylist<f64> -> memref<?xf64>
    %data = catalyst.list_load_data %list : !catalyst.arraylist<f64> -> memref<?xf64>

    // CHECK: [[VAL:%.+]] = "test.op"() : () -> f64
    %val = "test.op"() : () -> f64

    // CHECK: [[POP_RESULT:%.+]] = catalyst.list_pop [[LIST]] : !catalyst.arraylist<f64>
    %pop_result = catalyst.list_pop %list : !catalyst.arraylist<f64>

    // CHECK: catalyst.list_push [[VAL]], [[LIST]] : !catalyst.arraylist<f64>
    catalyst.list_push %val, %list : !catalyst.arraylist<f64>

    // CHECK: catalyst.list_dealloc [[LIST]] : !catalyst.arraylist<f64>
    catalyst.list_dealloc %list : !catalyst.arraylist<f64>

    // CHECK: [[CUSTOM_RESULT:%.+]] = catalyst.custom_call fn("custom_function") ([[VAL]]) : (f64) -> f64
    %custom_result = catalyst.custom_call fn("custom_function")(%val) : (f64) -> f64

    // CHECK: [[KERNEL_RESULT:%.+]] = catalyst.launch_kernel @kernel_name([[VAL]]) : (f64) -> f64
    %kernel_result = catalyst.launch_kernel @kernel_name(%val) : (f64) -> f64

    // CHECK: [[CALLBACK_RESULT:%.+]] = catalyst.callback_call @callback_func([[VAL]]) : (f64) -> f64
    %callback_result = catalyst.callback_call @callback_func(%val) : (f64) -> f64
    """

    run_filecheck(program, roundtrip=True)
