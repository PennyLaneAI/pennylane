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

"""Test module for the xDSL transform helpers module."""

from typing import Optional

import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
from xdsl.dialects import builtin, func, tensor, test
from xdsl.ir import Operation

from pennylane.compiler.python_compiler import quantum_dialect as quantum
from pennylane.compiler.python_compiler.transforms import transform_helpers


@pytest.fixture(name="context", scope="function")
def fixture_context():
    """A fixture that prepares the context for unit tests of the xDSL transform helpers."""
    ctx = xdsl.context.Context(allow_unregistered=True)
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(tensor.Tensor)
    ctx.load_dialect(test.Test)
    ctx.load_dialect(quantum.QuantumDialect)

    yield ctx


class TestGetGateWires:
    """TODO"""

    def test_1_wire(self, context):
        """TODO"""
        program = """
        func.func @test_func() {
            %0 = "quantum.alloc"() <{nqubits_attr = 1 : i64}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
            %2 = quantum.custom "U0a"() %1 : !quantum.bit
            %3 = quantum.custom "U0b"() %2 : !quantum.bit
            return
        }
        """
        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U0a = ops[2]
        _assert_gate_is_expected(U0a, gate_name="U0a")
        assert transform_helpers.get_gate_wires(U0a) == [0]

        U0b = ops[3]
        _assert_gate_is_expected(U0b, gate_name="U0b")
        assert transform_helpers.get_gate_wires(U0b) == [0]

    @pytest.mark.parametrize(
        "inner_ops",
        [
            pytest.param(
                """%2 = "test.op"() : () -> tensor<2xi1>
                %3 = quantum.set_basis_state (%2) %1 : (tensor<2xi1>, !quantum.bit) -> !quantum.bit""",
                id="quantum.set_basis_state",
            ),
            pytest.param(
                """%2 = "test.op"() : () -> tensor<2xcomplex<f64>>
                %3 = quantum.set_state (%2) %1 : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit""",
                id="quantum.set_state",
            ),
            pytest.param(
                """%2 = "test.op"() : () -> tensor<2x2xcomplex<f64>>
                %3 = quantum.unitary (%2 : tensor<2x2xcomplex<f64>>) %1 : !quantum.bit""",
                id="quantum.unitary",
            ),
        ],
    )
    def test_1_wire_with_non_custom_op(self, context, inner_ops):
        """TODO"""
        program = rf"""
        func.func @test_func() {{
            %0 = "quantum.alloc"() <{{nqubits_attr = 1 : i64}}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{{idx_attr = 0 : i64}}> : (!quantum.reg) -> !quantum.bit
            {inner_ops}
            %4 = quantum.custom "U0a"() %3 : !quantum.bit
            %5 = quantum.custom "U0b"() %4 : !quantum.bit
            return
        }}
        """
        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U0a = ops[4]
        _assert_gate_is_expected(U0a, gate_name="U0a")
        assert transform_helpers.get_gate_wires(U0a) == [0]

        U0b = ops[5]
        _assert_gate_is_expected(U0b, gate_name="U0b")
        assert transform_helpers.get_gate_wires(U0b) == [0]

    def test_1_wire_with_mcm(self, context):
        """TODO"""
        program = """
        func.func @test_func() {
            %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
            %2 = quantum.custom "U0a"() %1 : !quantum.bit
            %mres, %3 = quantum.measure %2 : i1, !quantum.bit
            %4 = quantum.custom "U0b"() %3 : !quantum.bit
            return
        }
        """
        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U0a = ops[2]
        _assert_gate_is_expected(U0a, gate_name="U0a")
        assert transform_helpers.get_gate_wires(U0a) == [0]

        U0b = ops[4]
        _assert_gate_is_expected(U0b, gate_name="U0b")
        assert transform_helpers.get_gate_wires(U0b) == [0]

    def test_2_wire_with_1_qubit_ops(self, context):
        """TODO"""
        program = """
        func.func @test_func() {
            %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
            %2 = "quantum.extract"(%0) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
            %3 = quantum.custom "U0a"() %1 : !quantum.bit
            %4 = quantum.custom "U1a"() %2 : !quantum.bit
            %5 = quantum.custom "U1b"() %4 : !quantum.bit
            return
        }
        """
        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U0a = ops[3]
        _assert_gate_is_expected(U0a, gate_name="U0a")
        assert transform_helpers.get_gate_wires(U0a) == [0]

        U1a = ops[4]
        _assert_gate_is_expected(U1a, gate_name="U1a")
        assert transform_helpers.get_gate_wires(U1a) == [1]

        U1b = ops[5]
        _assert_gate_is_expected(U1b, gate_name="U1b")
        assert transform_helpers.get_gate_wires(U1b) == [1]

    def test_2_wire_with_2_qubit_ops(self, context):
        """TODO"""
        program = """
        func.func @test_func() {
            %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
            %2 = "quantum.extract"(%0) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
            %3 = quantum.custom "U01"() %1, %2 : !quantum.bit
            %4 = quantum.custom "U10"() %2, %1 : !quantum.bit
            return
        }
        """
        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U01 = ops[3]
        _assert_gate_is_expected(U01, gate_name="U01")
        assert transform_helpers.get_gate_wires(U01) == [0, 1]

        U10 = ops[4]
        _assert_gate_is_expected(U10, gate_name="U10")
        assert transform_helpers.get_gate_wires(U10) == [1, 0]

    @pytest.mark.parametrize(
        "inner_op",
        [
            pytest.param(
                """%3 = "test.op"() : () -> tensor<2xi1>
                %4:2 = quantum.set_basis_state(%3) %1, %2 : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)""",
                id="quantum.set_basis_state",
            ),
            pytest.param(
                """%3 = "test.op"() : () -> tensor<4xcomplex<f64>>
                %4:2 = quantum.set_state(%3) %1, %2 : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)""",
                id="quantum.set_state",
            ),
            pytest.param(
                """%3 = "test.op"() : () -> tensor<4x4xcomplex<f64>>
                %4:2 = quantum.unitary(%3 : tensor<4x4xcomplex<f64>>) %1, %2 : !quantum.bit, !quantum.bit""",
                id="quantum.unitary",
            ),
        ],
    )
    def test_2_wire_with_non_custom_op(self, context, inner_op):
        """TODO"""
        program = rf"""
        func.func @test_func() {{
            %0 = "quantum.alloc"() <{{nqubits_attr = 2 : i64}}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{{idx_attr = 0 : i64}}> : (!quantum.reg) -> !quantum.bit
            %2 = "quantum.extract"(%0) <{{idx_attr = 1 : i64}}> : (!quantum.reg) -> !quantum.bit
            {inner_op}
            %5 = quantum.custom "U0"() %4#0 : !quantum.bit
            %6 = quantum.custom "U1"() %4#1 : !quantum.bit
            %7 = quantum.custom "U01"() %6, %6 : !quantum.bit
            return
        }}
        """
        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U0 = ops[5]
        _assert_gate_is_expected(U0, gate_name="U0")
        assert transform_helpers.get_gate_wires(U0) == [0]

        U1 = ops[6]
        _assert_gate_is_expected(U1, gate_name="U1")
        assert transform_helpers.get_gate_wires(U1) == [0]

        U01 = ops[7]
        _assert_gate_is_expected(U01, gate_name="U01")
        assert transform_helpers.get_gate_wires(U01) == [0, 1]

    def test_2_wire_with_1_qubit_op_1_ctrl(self, context):
        """TODO"""
        program = """
        func.func @test_func() {
            %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
            %2 = "quantum.extract"(%0) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
            %3 = "test.op"() : () -> i1
            %40, %41 = quantum.custom "U0ctrl1"() %1 ctrls(%2) ctrlvals(%3) : !quantum.bit ctrls !quantum.bit
            %5 = quantum.custom "U0"() %40 : !quantum.bit
            %6 = quantum.custom "U1"() %41 : !quantum.bit
            return
        }
        """

        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U0ctrl1 = ops[4]
        _assert_gate_is_expected(U0ctrl1, gate_name="U0ctrl1")
        assert transform_helpers.get_gate_wires(U0ctrl1) == [0, 1]
        assert transform_helpers.get_gate_wires(U0ctrl1, separate_control=True) == ([0], [1])

        U0 = ops[5]
        _assert_gate_is_expected(U0, gate_name="U0")
        assert transform_helpers.get_gate_wires(U0) == [0]

        U1 = ops[6]
        _assert_gate_is_expected(U1, gate_name="U1")
        assert transform_helpers.get_gate_wires(U1) == [1]

    def test_3_wire_with_1_qubit_op_2_ctrl(self, context):
        """TODO"""
        program = """
        func.func @test_func() {
            %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
            %2 = "quantum.extract"(%0) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
            %3 = "quantum.extract"(%0) <{idx_attr = 2 : i64}> : (!quantum.reg) -> !quantum.bit
            %4 = "test.op"() : () -> i1
            %50, %51:2 = quantum.custom "U0ctrl12"() %1 ctrls(%2, %3) ctrlvals(%4, %4) : !quantum.bit ctrls !quantum.bit, !quantum.bit
            %6 = quantum.custom "U0"() %50 : !quantum.bit
            %7 = quantum.custom "U1"() %51#0 : !quantum.bit
            %8 = quantum.custom "U2"() %51#1 : !quantum.bit
            return
        }
        """

        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U0ctrl12 = ops[5]
        _assert_gate_is_expected(U0ctrl12, gate_name="U0ctrl12")
        assert transform_helpers.get_gate_wires(U0ctrl12) == [0, 1]
        assert transform_helpers.get_gate_wires(U0ctrl12, separate_control=True) == ([0], [1, 2])

        U0 = ops[6]
        _assert_gate_is_expected(U0, gate_name="U0")
        assert transform_helpers.get_gate_wires(U0) == [0]

        U1 = ops[7]
        _assert_gate_is_expected(U1, gate_name="U1")
        assert transform_helpers.get_gate_wires(U1) == [1]

        U1 = ops[8]
        _assert_gate_is_expected(U1, gate_name="U1")
        assert transform_helpers.get_gate_wires(U1) == [2]

    def test_3_wire_with_2_qubit_op_1_ctrl(self, context):
        """TODO"""
        program = """
        func.func @test_func() {
            %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
            %2 = "quantum.extract"(%0) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
            %3 = "quantum.extract"(%0) <{idx_attr = 2 : i64}> : (!quantum.reg) -> !quantum.bit
            %4 = "test.op"() : () -> i1
            %50:2, %51 = quantum.custom "U01ctrl2"() %1, %2 ctrls(%3) ctrlvals(%4) : !quantum.bit, !quantum.bit ctrls !quantum.bit
            %6 = quantum.custom "U0"() %50#0 : !quantum.bit
            %7 = quantum.custom "U12"() %50#1, %51 : !quantum.bit
            return
        }
        """

        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U01ctrl2 = ops[5]
        _assert_gate_is_expected(U01ctrl2, gate_name="U01ctrl2")
        assert transform_helpers.get_gate_wires(U01ctrl2) == [0, 1, 2]
        assert transform_helpers.get_gate_wires(U01ctrl2, separate_control=True) == ([0, 1], [2])

        U0 = ops[6]
        _assert_gate_is_expected(U0, gate_name="U0")
        assert transform_helpers.get_gate_wires(U0) == [0]

        U12 = ops[7]
        _assert_gate_is_expected(U12, gate_name="U12")
        assert transform_helpers.get_gate_wires(U12) == [1, 2]

    def test_4_wire_with_2_qubit_op_2_ctrl(self, context):
        """TODO"""
        program = """
        func.func @test_func() {
            %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
            %1 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
            %2 = "quantum.extract"(%0) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
            %3 = "quantum.extract"(%0) <{idx_attr = 2 : i64}> : (!quantum.reg) -> !quantum.bit
            %4 = "quantum.extract"(%0) <{idx_attr = 3 : i64}> : (!quantum.reg) -> !quantum.bit
            %5 = "test.op"() : () -> i1
            %60:2, %61:2 = quantum.custom "U01ctrl23"() %1, %2 ctrls(%3, %4) ctrlvals(%5, %5) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit
            %7 = quantum.custom "U02"() %60#0, %61#0 : !quantum.bit
            %8 = quantum.custom "U1"() %60#1 : !quantum.bit
            %9 = quantum.custom "U3"() %61#1 : !quantum.bit
            return
        }
        """

        module = xdsl.parser.Parser(context, program).parse_module()

        func_op = module.ops.first
        assert isinstance(func_op, func.FuncOp), "Expected func.FuncOp"

        ops = list(func_op.body.ops)

        U01ctrl23 = ops[6]
        _assert_gate_is_expected(U01ctrl23, gate_name="U01ctrl23")
        assert transform_helpers.get_gate_wires(U01ctrl23) == [0, 1, 2, 3]
        assert transform_helpers.get_gate_wires(U01ctrl23, separate_control=True) == (
            [0, 1],
            [2, 3],
        )

        U02 = ops[7]
        _assert_gate_is_expected(U02, gate_name="U02")
        assert transform_helpers.get_gate_wires(U02) == [0, 2]

        U1 = ops[8]
        _assert_gate_is_expected(U1, gate_name="U1")
        assert transform_helpers.get_gate_wires(U1) == [1]

        U3 = ops[9]
        _assert_gate_is_expected(U3, gate_name="U3")
        assert transform_helpers.get_gate_wires(U3) == [3]


def _assert_gate_is_expected(
    op: Operation,
    expected_op_type: type[Operation] = quantum.CustomOp,
    gate_name: Optional[str] = None,
):
    """Helper function that asserts that `op` (generally representing a quantum gate) is of type
    `expected_op_type`.

    If `gate_name` is also provided, it also asserts that the ``gate_name`` string attribute of `op`
    is equal to `gate_name`. This option only works if `expected_op_type` is a quantum.custom op.

    This function is generally for sanity checks only; if the tests have been written correctly,
    none of these asserts should be triggered, even if there are issues with the production code
    being tested.

    Args:
        op (Operation): The operation whose type is to be checked. This is generally a quantum op
            representing a quantum gate, but any op can be used.
        expected_op_type (type[Operation], optional): The expected operation type. This is generally
            ``quantum.CustomOp`` (the default), representing a quantum gate, but any op type can be
            used.
        gate_name (str, optional): The expected gate name. Defaults to None, in which case the
            ``gate_name`` attribute of `op` is not checked.

    Raises:
        TypeError: If `gate_name` is provided but `expected_op_type` is not a quantum.custom op.
    """
    if gate_name is None:
        assert isinstance(
            op, expected_op_type
        ), f"Expected `{expected_op_type.name}`, but got `{op.name}`"

    else:
        if expected_op_type is not quantum.CustomOp:
            raise TypeError(
                f"incorrect usage: if `gate_name` is supplied, `expected_op_type` must be a "
                f"quantum.custom, but got {expected_op_type.name}"
            )

        if hasattr(op, gate_name):
            assert (
                isinstance(op, expected_op_type) and op.gate_name.data == gate_name
            ), f'Expected `{expected_op_type.name} "{gate_name}"`, but got `{op.name} "{op.gate_name.data}"`'
        else:
            assert (
                isinstance(op, expected_op_type) and op.gate_name.data == gate_name
            ), f'Expected `{expected_op_type.name} "{gate_name}"`, but got `{op.name}`'
