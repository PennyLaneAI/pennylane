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

"""Unit test module for pennylane/compiler/python_compiler/mbqc_dialect.py."""

import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

pytestmark = pytest.mark.external

from xdsl.dialects import arith, builtin, test

from pennylane.compiler.python_compiler.mbqc_dialect import MBQCDialect
from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect

all_ops = list(MBQCDialect.operations)
all_attrs = list(MBQCDialect.attributes)

expected_ops_names = {
    "MeasureInBasisOp": "mbqc.measure_in_basis",
}

expected_attrs_names = {
    "MeasurementPlaneAttr": "mbqc.measurement_plane",
}


def test_mbqc_dialect_name():
    """Test that the MBQCDialect name is correct."""
    assert MBQCDialect.name == "mbqc"


@pytest.mark.parametrize("op", all_ops)
def test_all_operations_names(op):
    """Test that all operations have the expected name."""
    op_class_name = op.__name__
    expected_name = expected_ops_names.get(op_class_name)
    assert expected_name is not None, f"Unexpected operation {op_class_name} found in MBQCDialect"
    assert op.name == expected_name


@pytest.mark.parametrize("attr", all_attrs)
def test_all_attributes_names(attr):
    """Test that all attributes have the expected name."""
    attr_class_name = attr.__name__
    expected_name = expected_attrs_names.get(attr_class_name)
    assert expected_name is not None, f"Unexpected attribute {attr_class_name} found in MBQCDialect"
    assert attr.name == expected_name


def test_assembly_format(run_filecheck):
    """Test the assembly format of the mbqc ops

    ..
        FIXME: There's a bug with the postselect parameter; currently debugging...
    """
    program = r"""
    // CHECK: [[angle:%.+]] = arith.constant {{.+}} : f64
    %angle = arith.constant 3.141592653589793 : f64

    // CHECK: [[qubit:%.+]] = "test.op"() : () -> !quantum.bit
    %qubit = "test.op"() : () -> !quantum.bit

    // CHECK: [[res0:%.+]], [[new_q0:%.+]] = mbqc.measure_in_basis{{\s*}}[XY, [[angle]]] [[qubit]] : i1, !quantum.bit
    %res0, %new_q0 = mbqc.measure_in_basis [XY, %angle] %qubit : i1, !quantum.bit

    // CHECK: [[res1:%.+]], [[new_q1:%.+]] = mbqc.measure_in_basis{{\s*}}[YZ, [[angle]]] [[qubit]] : i1, !quantum.bit
    %res1, %new_q1 = mbqc.measure_in_basis [YZ, %angle] %qubit : i1, !quantum.bit

    // CHECK: [[res2:%.+]], [[new_q2:%.+]] = mbqc.measure_in_basis{{\s*}}[ZX, [[angle]]] [[qubit]] : i1, !quantum.bit
    %res2, %new_q2 = mbqc.measure_in_basis [ZX, %angle] %qubit : i1, !quantum.bit

    // CHECK: [[res3:%.+]], [[new_q3:%.+]] = mbqc.measure_in_basis{{\s*}}[XY, [[angle]]] [[qubit]] postselect 0 : i1, !quantum.bit
    %res3, %new_q3 = mbqc.measure_in_basis [XY, %angle] %qubit postselect 0 : i1, !quantum.bit

    // CHECK: [[res4:%.+]], [[new_q4:%.+]] = mbqc.measure_in_basis{{\s*}}[XY, [[angle]]] [[qubit]] postselect 1 : i1, !quantum.bit
    %res4, %new_q4 = mbqc.measure_in_basis [XY, %angle] %qubit postselect 1 : i1, !quantum.bit
    """

    ctx = xdsl.context.Context()

    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(test.Test)
    ctx.load_dialect(QuantumDialect)
    ctx.load_dialect(MBQCDialect)

    module = xdsl.parser.Parser(ctx, program).parse_module()

    run_filecheck(program, module)
