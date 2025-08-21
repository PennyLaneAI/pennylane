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

"""Test suite for the xdsl_extras module."""

import pytest

pytestmark = pytest.mark.external

pytest.importorskip("xdsl")

# pylint: disable=wrong-import-position
from xdsl.context import Context
from xdsl.dialects import builtin, test
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.utils.exceptions import VerifyException

from pennylane.compiler.unified_compiler.jax_utils import QuantumParser
from pennylane.compiler.unified_compiler.xdsl_extras import (
    MemRefRankConstraint,
    TensorRankConstraint,
)


@pytest.fixture(scope="module", name="my_dialect")
def my_dialect_fixture():
    """Returns a test dialect, called 'my_dialect', with simple ops that operate on memref and
    tensor types.
    """

    @irdl_op_definition
    class MyMemrefOp(IRDLOperation):
        """A test op with rank-1 memref types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.memref_op"
        in_value = operand_def(MemRefRankConstraint(1))
        out_value = result_def(MemRefRankConstraint(1))

    @irdl_op_definition
    class MyTensorOp(IRDLOperation):
        """A test op with rank-1 tensor types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.tensor_op"
        in_value = operand_def(TensorRankConstraint(1))
        out_value = result_def(TensorRankConstraint(1))

    MyDialect = Dialect(
        "my_dialect",
        [MyMemrefOp, MyTensorOp],
    )

    return MyDialect


class TestMemRefRankConstraint:
    """Tests for the MemRefRankConstraint class."""

    def test_memref_rank_constraint(self):
        """Test that the properties of MemRefRankConstraint object are correct."""
        rank = 1
        constraint = MemRefRankConstraint(rank)

        assert constraint.expected_rank == rank
        assert constraint.expected_type == builtin.MemRefType
        assert constraint.type_name == "memref"
        assert constraint.mapping_type_vars({}) is constraint

    @pytest.mark.parametrize("rank", [0, 1, 2])
    def test_memref_rank_constraint_verify_valid(self, rank):
        """Test that verifying a MemRefType attribute with the same rank as the MemRefRankConstraint
        does not raise an exception."""
        constraint = MemRefRankConstraint(rank)
        attr = builtin.MemRefType(builtin.i32, [1] * rank)

        constraint.verify(attr, None)

    @pytest.mark.parametrize("rank", [0, 1, 2])
    def test_memref_rank_constraint_verify_invalid_rank(self, rank):
        """Test that verifying a MemRefType attribute with a different rank as the
        MemRefRankConstraint raises a VerifyException."""
        constraint = MemRefRankConstraint(rank)
        attr = builtin.MemRefType(builtin.i32, [1] * (rank + 1))

        with pytest.raises(
            VerifyException,
            match=f"Expected memref rank to be {constraint.expected_rank}, got {attr.get_num_dims()}",
        ):
            constraint.verify(attr, None)

    def test_memref_rank_constraint_verify_invalid_type(self):
        """Test that verifying an attribute with a type other than MemRefType raises a
        VerifyException."""
        constraint = MemRefRankConstraint(1)
        attr = builtin.TensorType(builtin.i32, [1])

        with pytest.raises(VerifyException, match=f"{attr} should be of type MemRefType"):
            constraint.verify(attr, None)

    def test_memref_rank_constraint_integration(self, my_dialect):
        """Test that verification of a legal my_dialect.memref_op does not raise an exception."""
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> memref<2xi64>
            %1 = "my_dialect.memref_op"(%0) : (memref<2xi64>) -> memref<2xi64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()
        xdsl_module.verify()

    def test_memref_rank_constraint_integration_invalid_rank(self, my_dialect):
        """Test that verification of an illegal my_dialect.memref_op raises a VerifyException.

        Here, the input operand to my_dialect.memref_op is a rank-2 memref, while it is constrained
        to be rank 1.
        """
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> memref<2x2xi64>
            %1 = "my_dialect.memref_op"(%0) : (memref<2x2xi64>) -> memref<2x2xi64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()

        with pytest.raises(VerifyException, match="Expected memref rank to be 1, got 2"):
            xdsl_module.verify()

    def test_memref_rank_constraint_integration_invalid_type(self, my_dialect):
        """Test that verification of an illegal my_dialect.memref_op raises a VerifyException.

        Here, the input operand to my_dialect.memref_op is a rank-1 tensor, while it is constrained
        to be of type memref.
        """
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> tensor<2xi64>
            %1 = "my_dialect.memref_op"(%0) : (tensor<2xi64>) -> memref<2xi64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()

        with pytest.raises(VerifyException, match="tensor<2xi64> should be of type MemRefType"):
            xdsl_module.verify()


class TestTensorRankConstraint:
    """Tests for the TensorRankConstraint class."""

    def test_tensor_rank_constraint(self):
        """Test that the properties of TensorRankConstraint object are correct."""
        rank = 1
        constraint = TensorRankConstraint(rank)

        assert constraint.expected_rank == rank
        assert constraint.expected_type == builtin.TensorType
        assert constraint.type_name == "tensor"
        assert constraint.mapping_type_vars({}) is constraint

    @pytest.mark.parametrize("rank", [0, 1, 2])
    def test_tensor_rank_constraint_verify_valid(self, rank):
        """Test that verifying a MemRefType attribute with the same rank as the MemRefRankConstraint
        does not raise an exception."""
        constraint = TensorRankConstraint(rank)
        attr = builtin.TensorType(builtin.i32, [1] * rank)

        constraint.verify(attr, None)

    @pytest.mark.parametrize("rank", [0, 1, 2])
    def test_tensor_rank_constraint_verify_invalid_rank(self, rank):
        """Test that verifying a TensorType attribute with a different rank as the
        TensorRankConstraint raises a VerifyException."""
        constraint = TensorRankConstraint(rank)
        attr = builtin.TensorType(builtin.i32, [1] * (rank + 1))

        with pytest.raises(
            VerifyException,
            match=f"Expected tensor rank to be {constraint.expected_rank}, got {attr.get_num_dims()}",
        ):
            constraint.verify(attr, None)

    def test_tensor_rank_constraint_verify_invalid_type(self):
        """Test that verifying an attribute with a type other than TensorType raises a
        VerifyException."""
        constraint = TensorRankConstraint(1)
        attr = builtin.MemRefType(builtin.i32, [1])

        with pytest.raises(VerifyException, match=f"{attr} should be of type TensorType"):
            constraint.verify(attr, None)

    def test_tensor_rank_constraint_integration(self, my_dialect):
        """Test that verification of a legal my_dialect.tensor_op does not raise an exception."""
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> tensor<2xi64>
            %1 = "my_dialect.tensor_op"(%0) : (tensor<2xi64>) -> tensor<2xi64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()
        xdsl_module.verify()

    def test_tensor_rank_constraint_integration_invalid_rank(self, my_dialect):
        """Test that verification of an illegal my_dialect.tensor_op raises a VerifyException.

        Here, the input operand to my_dialect.tensor_op is a rank-2 tensor, while it is constrained
        to be rank 1.
        """
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> tensor<2x2xi64>
            %1 = "my_dialect.tensor_op"(%0) : (tensor<2x2xi64>) -> tensor<2x2xi64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()

        with pytest.raises(VerifyException, match="Expected tensor rank to be 1, got 2"):
            xdsl_module.verify()

    def test_tensor_rank_constraint_integration_invalid_type(self, my_dialect):
        """Test that verification of an illegal my_dialect.tensor_op raises a VerifyException.

        Here, the input operand to my_dialect.tensor_op is a rank-1 memref, while it is constrained
        to be of type tensor.
        """
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> memref<2xi64>
            %1 = "my_dialect.tensor_op"(%0) : (memref<2xi64>) -> tensor<2xi64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()

        with pytest.raises(VerifyException, match="memref<2xi64> should be of type TensorType"):
            xdsl_module.verify()
