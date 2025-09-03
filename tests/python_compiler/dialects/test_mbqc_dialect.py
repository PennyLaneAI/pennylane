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

"""Unit test module for pennylane/compiler/python_compiler/dialects/mbqc.py."""

import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

pytestmark = pytest.mark.external

from xdsl.dialects import arith, builtin, test
from xdsl.utils.exceptions import VerifyException

from pennylane.compiler.python_compiler.dialects import Quantum, mbqc

all_ops = list(mbqc.MBQC.operations)
all_attrs = list(mbqc.MBQC.attributes)

expected_ops_names = {
    "MeasureInBasisOp": "mbqc.measure_in_basis",
    "GraphStatePrepOp": "mbqc.graph_state_prep",
}

expected_attrs_names = {
    "MeasurementPlaneAttr": "mbqc.measurement_plane",
}


def test_mbqc_dialect_name():
    """Test that the MBQCDialect name is correct."""
    assert mbqc.MBQC.name == "mbqc"


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
    """Test the assembly format of the mbqc ops."""
    program = r"""
    // CHECK: [[angle:%.+]] = "test.op"() : () -> f64
    %angle = "test.op"() : () -> f64

    // CHECK: [[qubit:%.+]] = "test.op"() : () -> !quantum.bit
    %qubit = "test.op"() : () -> !quantum.bit

    // CHECK: [[mres0:%.+]], [[out_qubit0:%.+]] = mbqc.measure_in_basis{{\s*}}[XY, [[angle]]] [[qubit]] : i1, !quantum.bit
    %mres0, %out_qubit0 = mbqc.measure_in_basis [XY, %angle] %qubit : i1, !quantum.bit

    // CHECK: [[mres1:%.+]], [[out_qubit1:%.+]] = mbqc.measure_in_basis{{\s*}}[YZ, [[angle]]] [[qubit]] : i1, !quantum.bit
    %mres1, %out_qubit1 = mbqc.measure_in_basis [YZ, %angle] %qubit : i1, !quantum.bit

    // CHECK: [[mres2:%.+]], [[out_qubit2:%.+]] = mbqc.measure_in_basis{{\s*}}[ZX, [[angle]]] [[qubit]] : i1, !quantum.bit
    %mres2, %out_qubit2 = mbqc.measure_in_basis [ZX, %angle] %qubit : i1, !quantum.bit

    // CHECK: [[mres3:%.+]], [[out_qubit3:%.+]] = mbqc.measure_in_basis{{\s*}}[XY, [[angle]]] [[qubit]] postselect 0 : i1, !quantum.bit
    %mres3, %out_qubit3 = mbqc.measure_in_basis [XY, %angle] %qubit postselect 0 : i1, !quantum.bit

    // CHECK: [[mres4:%.+]], [[out_qubit4:%.+]] = mbqc.measure_in_basis{{\s*}}[XY, [[angle]]] [[qubit]] postselect 1 : i1, !quantum.bit
    %mres4, %out_qubit4 = mbqc.measure_in_basis [XY, %angle] %qubit postselect 1 : i1, !quantum.bit

    // COM: Check generic format
    // CHECK: {{%.+}}, {{%.+}} = mbqc.measure_in_basis[XY, [[angle]]] [[qubit]] postselect 0 : i1, !quantum.bit
    %res:2 = "mbqc.measure_in_basis"(%qubit, %angle) <{plane = #mbqc<measurement_plane XY>, postselect = 0 : i32}> : (!quantum.bit, f64) -> (i1, !quantum.bit)

    // CHECK: [[adj_matrix:%.+]] = arith.constant {{.*}} : tensor<6xi1>
    // CHECK: [[graph_reg:%.+]] = mbqc.graph_state_prep{{\s*}}([[adj_matrix]] : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
    %adj_matrix = arith.constant dense<[1, 0, 1, 0, 0, 1]> : tensor<6xi1>
    %graph_reg = mbqc.graph_state_prep (%adj_matrix : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
    """

    run_filecheck(program, roundtrip=True)


class TestMeasureInBasisOp:
    """Unit tests for the mbqc.measure_in_basis op."""

    @pytest.mark.parametrize("plane,", ["XY", "YZ", "ZX"])
    @pytest.mark.parametrize("postselect", ["", "postselect 0", "postselect 1"])
    def test_measure_in_basis_properties(self, plane, postselect):
        """Test the parsing of the mbqc.measure_in_basis op's properties."""
        program = rf"""
        %angle = "test.op"() : () -> f64
        %qubit = "test.op"() : () -> !quantum.bit

        %mres, %out_qubit = mbqc.measure_in_basis [{plane}, %angle] %qubit {postselect} : i1, !quantum.bit
        """

        ctx = xdsl.context.Context()

        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)
        ctx.load_dialect(mbqc.MBQC)

        module = xdsl.parser.Parser(ctx, program).parse_module()

        measure_in_basis_op: mbqc.MeasureInBasisOp = module.ops.last
        assert isinstance(measure_in_basis_op, mbqc.MeasureInBasisOp)

        assert measure_in_basis_op.properties["plane"].data == plane

        if postselect:
            assert measure_in_basis_op.properties["postselect"].value.data == int(postselect[-1])
        else:
            assert measure_in_basis_op.properties.get("postselect") is None

    @pytest.mark.parametrize("postselect", [-1, 2])
    def test_invalid_postselect_raises_on_verify(self, postselect):
        """Test that using an invalid postselect value (a value other than 0 or 1) raises a
        VerifyException during verification."""

        program = rf"""
        %angle = "test.op"() : () -> f64
        %qubit = "test.op"() : () -> !quantum.bit

        %mres, %out_qubit = mbqc.measure_in_basis [XY, %angle] %qubit postselect {postselect} : i1, !quantum.bit
        """

        ctx = xdsl.context.Context()

        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)
        ctx.load_dialect(mbqc.MBQC)

        module = xdsl.parser.Parser(ctx, program).parse_module()

        measure_in_basis_op: mbqc.MeasureInBasisOp = module.ops.last
        assert isinstance(measure_in_basis_op, mbqc.MeasureInBasisOp)

        with pytest.raises(VerifyException, match="'postselect' must be 0 or 1"):
            measure_in_basis_op.verify_()

    @pytest.mark.parametrize("init_op", ["Hadamard", builtin.StringAttr(data="Hadamard")])
    @pytest.mark.parametrize("entangle_op", ["CZ", builtin.StringAttr(data="CZ")])
    def test_graph_state_prep_instantiation(self, init_op, entangle_op):
        """Test the instantiation of a mbqc.graph_state_prep op."""
        adj_matrix = [1, 0, 1, 0, 0, 1]
        adj_matrix_op = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(builtin.IntegerType(1), shape=(6,)), data=adj_matrix
            )
        )
        graph_state_prep_op = mbqc.GraphStatePrepOp(adj_matrix_op.result, init_op, entangle_op)

        assert graph_state_prep_op.adj_matrix == adj_matrix_op.result
        assert graph_state_prep_op.init_op == builtin.StringAttr(data="Hadamard")
        assert graph_state_prep_op.entangle_op == builtin.StringAttr(data="CZ")
