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

"""Unit test module for pennylane/compiler/python_compiler/dialects/qec.py."""

import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

pytestmark = pytest.mark.external

from pennylane.compiler.python_compiler.dialects import QEC

all_ops = list(QEC.operations)
all_attrs = list(QEC.attributes)

expected_ops_names = {
    "FabricateOp": "qec.fabricate",
    "LayerOp": "qec.layer",
    "PPMeasurementOp": "qec.ppm",
    "PPRotationOp": "qec.ppr",
    "PrepareStateOp": "qec.prepare",
    "SelectPPMeasurementOp": "qec.select.ppm",
    "YieldOp": "qec.yield",
}

expected_attrs_names = {
    "LogicalInit": "qec.enum",
}


def test_qec_dialect_name():
    """Test that the QEC dialect name is correct."""
    assert QEC.name == "qec"


@pytest.mark.parametrize("op", all_ops)
def test_all_operations_names(op):
    """Test that all operations have the expected name."""
    op_class_name = op.__name__
    expected_name = expected_ops_names.get(op_class_name)
    assert expected_name is not None, f"Unexpected operation {op_class_name} found in QEC dialect"
    assert op.name == expected_name


@pytest.mark.parametrize("attr", all_attrs)
def test_all_attributes_names(attr):
    """Test that all attributes have the expected name."""
    attr_class_name = attr.__name__
    expected_name = expected_attrs_names.get(attr_class_name)
    assert expected_name is not None, f"Unexpected attribute {attr_class_name} found in QEC dialect"
    assert attr.name == expected_name


def test_assembly_format(run_filecheck):
    """Test the assembly format of the qec ops."""
    program = """
    // CHECK: [[QUBIT:%.+]] = "test.op"() : () -> !quantum.bit
    %qubit = "test.op"() : () -> !quantum.bit

    // CHECK: [[COND:%.+]] = "test.op"() : () -> i1
    %cond = "test.op"() : () -> i1

    // CHECK: [[FABRICATED:%.+]] = qec.fabricate magic : !quantum.bit
    %fabricated = qec.fabricate magic : !quantum.bit

    // CHECK: [[PREPARED:%.+]] = qec.prepare zero [[QUBIT]] : !quantum.bit
    %prepared = qec.prepare zero %qubit : !quantum.bit

    // CHECK: [[ROTATED:%.+]] = qec.ppr ["X", "I", "Z"](4) [[QUBIT]] : !quantum.bit
    %rotated = qec.ppr ["X", "I", "Z"](4) %qubit : !quantum.bit

    // CHECK: [[MEASURED:%.+]], [[OUT_QUBITS:%.+]] = qec.ppm ["X", "I", "Z"] [[QUBIT]] : i1, !quantum.bit
    %measured, %out_qubits = qec.ppm ["X", "I", "Z"] %qubit : i1, !quantum.bit

    // CHECK: [[MEASURED_COND:%.+]], [[OUT_QUBITS_COND:%.+]] = qec.ppm ["X", "I", "Z"] [[QUBIT]] cond([[COND]]) : i1, !quantum.bit
    %measured_cond, %out_qubits_cond = qec.ppm ["X", "I", "Z"] %qubit cond(%cond) : i1, !quantum.bit

    // CHECK: [[SELECT_MEASURED:%.+]], [[SELECT_OUT:%.+]] = qec.select.ppm([[COND]], ["X"], ["Z"]) [[QUBIT]] : i1, !quantum.bit
    %select_measured, %select_out = qec.select.ppm (%cond, ["X"], ["Z"]) %qubit : i1, !quantum.bit

    """

    run_filecheck(program)
