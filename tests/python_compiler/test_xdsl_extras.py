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

from pennylane.compiler.python_compiler.jax_utils import QuantumParser
from pennylane.compiler.python_compiler.xdsl_extras import MemRefConstraint, TensorConstraint


@pytest.fixture(scope="module", name="my_dialect")
def my_dialect_fixture():
    """Returns a test dialect, called 'my_dialect', with simple ops that operate on memref and
    tensor types.
    """

    @irdl_op_definition
    class Float64MemrefOp(IRDLOperation):
        """A test op with float memref types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.memref_float64"
        in_value = operand_def(MemRefConstraint(element_type=builtin.Float64Type()))
        out_value = result_def(MemRefConstraint(element_type=builtin.Float64Type()))

    @irdl_op_definition
    class Float64TensorOp(IRDLOperation):
        """A test op with float tensor types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.tensor_float64"
        in_value = operand_def(TensorConstraint(element_type=builtin.Float64Type()))
        out_value = result_def(TensorConstraint(element_type=builtin.Float64Type()))

    @irdl_op_definition
    class Rank1MemrefOp(IRDLOperation):
        """A test op with rank-1 memref types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.memref_rank1"
        in_value = operand_def(MemRefConstraint(rank=1))
        out_value = result_def(MemRefConstraint(rank=1))

    @irdl_op_definition
    class Rank1TensorOp(IRDLOperation):
        """A test op with rank-1 tensor types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.tensor_rank1"
        in_value = operand_def(TensorConstraint(rank=1))
        out_value = result_def(TensorConstraint(rank=1))

    @irdl_op_definition
    class Rank2Or4MemrefOp(IRDLOperation):
        """A test op with rank-2 or -4 memref types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.memref_rank24"
        in_value = operand_def(MemRefConstraint(rank=(2, 4)))
        out_value = result_def(MemRefConstraint(rank=(2, 4)))

    @irdl_op_definition
    class Rank2Or4TensorOp(IRDLOperation):
        """A test op with rank-2 or -4 tensor types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.tensor_rank24"
        in_value = operand_def(TensorConstraint(rank=(2, 4)))
        out_value = result_def(TensorConstraint(rank=(2, 4)))

    @irdl_op_definition
    class Shape123MemrefOp(IRDLOperation):
        """A test op with shape-(1, 2, 3) memref types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.memref_shape123"
        in_value = operand_def(MemRefConstraint(shape=(1, 2, 3)))
        out_value = result_def(MemRefConstraint(shape=(1, 2, 3)))

    @irdl_op_definition
    class Shape123TensorOp(IRDLOperation):
        """A test op with shape-(1, 2, 3) tensor types"""

        # pylint: disable=too-few-public-methods

        name = "my_dialect.tensor_shape123"
        in_value = operand_def(TensorConstraint(shape=(1, 2, 3)))
        out_value = result_def(TensorConstraint(shape=(1, 2, 3)))

    MyDialect = Dialect(
        "my_dialect",
        [
            Float64MemrefOp,
            Float64TensorOp,
            Rank1MemrefOp,
            Rank1TensorOp,
            Rank2Or4MemrefOp,
            Rank2Or4TensorOp,
            Shape123MemrefOp,
            Shape123TensorOp,
        ],
    )

    return MyDialect


class TestMemRefConstraint:
    """Tests for the MemRefConstraint class."""

    def test_memref_constraint_init_invalid_element_type(self):
        """Test that an error is raised if the provided element_type is invalid"""

        element_type = int
        with pytest.raises(
            TypeError, match="is not a valid constraint for the 'element_type' argument"
        ):
            MemRefConstraint(element_type=element_type)

    def test_memref_constraint_init_invalid_shape(self):
        """Test that an error is raised if the provided shape is invalid"""

        shape = {1, 2}
        with pytest.raises(TypeError, match="is not a valid constraint for the 'shape' argument"):
            MemRefConstraint(shape=shape)

    def test_memref_constraint_init_invalid_rank(self):
        """Test that an error is raised if the provided rank is invalid"""

        rank = 1.5
        with pytest.raises(TypeError, match="is not a valid constraint for the 'rank' argument"):
            MemRefConstraint(rank=rank)

    def test_memref_constraint_properties(self):
        """Test that the properties of MemRefConstraint object are correct."""
        rank = 1
        constraint = MemRefConstraint(rank=rank)

        assert constraint.expected_type == builtin.MemRefType
        assert constraint.type_name == "memref"
        assert constraint.mapping_type_vars({}) is constraint

    @pytest.mark.parametrize("rank", [0, 1, 2])
    def test_memref_single_rank_constraint_verify_valid(self, rank):
        """Test that verifying a MemRefType attribute with the same rank as the MemRefConstraint
        does not raise an exception."""
        constraint = MemRefConstraint(rank=rank)
        attr = builtin.MemRefType(builtin.i32, [1] * rank)

        constraint.verify(attr, None)

    def test_memref_multi_rank_constraint_verify_valid(self):
        """Test that verifying a MemRefType attribute with any of the ranks as the MemRefConstraint
        does not raise an exception."""
        rank = (3, 4, 5)
        constraint = MemRefConstraint(rank=rank)

        for r in rank:
            attr = builtin.MemRefType(builtin.i32, [1] * r)
            constraint.verify(attr, None)

    def test_memref_shape_constraint_verify_valid(self):
        """Test that verifying an attribute with a valid shape does not raise an exception."""
        constraint = MemRefConstraint(shape=(1, 2, 3))
        attr = builtin.MemRefType(builtin.i32, (1, 2, 3))

        constraint.verify(attr, None)

    def test_memref_element_type_constraint_verify_valid(self):
        """Test that verifying an attribute with a valid element type does not raise an exception."""
        constraint = MemRefConstraint(element_type=builtin.i32)
        attr = builtin.MemRefType(builtin.i32, [1])

        constraint.verify(attr, None)

    @pytest.mark.parametrize("rank", [0, 1, 2])
    def test_memref_single_rank_constraint_verify_invalid(self, rank):
        """Test that verifying a MemRefType attribute with a different rank as the
        MemRefConstraint raises a VerifyException."""
        constraint = MemRefConstraint(rank=rank)
        attr = builtin.MemRefType(builtin.i32, [1] * (rank + 1))

        with pytest.raises(VerifyException, match=f"Invalid value {rank + 1}, expected {rank}"):
            constraint.verify(attr, None)

    def test_memref_multi_rank_constraint_verify_invalid(self):
        """Test that verifying a MemRefType attribute with a different rank as the
        MemRefConstraint raises a VerifyException."""

        rank = {3, 4, 5}
        invalid_rank = 2
        constraint = MemRefConstraint(rank=rank)
        attr = builtin.MemRefType(builtin.i32, [1] * invalid_rank)

        with pytest.raises(
            VerifyException, match=f"Invalid value {invalid_rank}, expected one of {rank}"
        ):
            constraint.verify(attr, None)

    def test_memref_constraint_verify_invalid_type(self):
        """Test that verifying an attribute with a type other than MemRefType raises a
        VerifyException."""
        constraint = MemRefConstraint(rank=1)
        attr = builtin.TensorType(builtin.i32, [1])

        with pytest.raises(VerifyException, match=f"{attr} should be of type MemRefType"):
            constraint.verify(attr, None)

    def test_memref_shape_constraint_verify_invalid(self):
        """Test that verifying an attribute with an invalid shape raises an exception."""
        constraint = MemRefConstraint(shape=(1, 2, 3))
        attr = builtin.MemRefType(builtin.i32, (2, 2, 3))

        with pytest.raises(VerifyException, match=r"Expected attribute \[.*\] but got \[.*\]"):
            constraint.verify(attr, None)

    def test_memref_element_type_constraint_verify_invalid(self):
        """Test that verifying an attribute with an invalid element type raises an exception."""
        constraint = MemRefConstraint(element_type=builtin.i32)
        attr = builtin.MemRefType(builtin.i64, [1])

        with pytest.raises(
            VerifyException, match=f"Expected attribute {builtin.i32} but got {builtin.i64}"
        ):
            constraint.verify(attr, None)

    def test_memref_constraint_integration(self, my_dialect):
        """Test that verification of legal operations with memref operand/result constraints does
        not raise an exception."""
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> memref<2xf64>
            %1 = "test.op"() : () -> memref<2x4xf64>
            %2 = "test.op"() : () -> memref<2x4x5x3xf64>
            %3 = "test.op"() : () -> memref<1x2x3xf64>
            %4 = "my_dialect.memref_float64"(%0) : (memref<2xf64>) -> memref<2xf64>
            %5 = "my_dialect.memref_rank1"(%0) : (memref<2xf64>) -> memref<2xf64>
            %6 = "my_dialect.memref_rank24"(%1) : (memref<2x4xf64>) -> memref<2x4xf64>
            %7 = "my_dialect.memref_rank24"(%2) : (memref<2x4x5x3xf64>) -> memref<2x4x5x3xf64>
            %8 = "my_dialect.memref_shape123"(%3) : (memref<1x2x3xf64>) -> memref<1x2x3xf64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()
        xdsl_module.verify()

    def test_memref_constraint_integration_invalid(self, my_dialect):
        """Test that verification of illegal operations with memref operands/result constraints
        raises a VerifyException."""
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> memref<2x2xi64>
            %1 = "my_dialect.memref_float64"(%0) : (memref<2x2xi64>) -> memref<2x2xi64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()

        with pytest.raises(
            VerifyException,
            match=f"Expected attribute {builtin.Float64Type()} but got {builtin.i64}",
        ):
            xdsl_module.verify()


class TestTensorConstraint:
    """Tests for the TensorConstraint class."""

    def test_tensor_constraint_init_invalid_element_type(self):
        """Test that an error is raised if the provided element_type is invalid"""

        element_type = int
        with pytest.raises(
            TypeError, match="is not a valid constraint for the 'element_type' argument"
        ):
            TensorConstraint(element_type=element_type)

    def test_tensor_constraint_init_invalid_shape(self):
        """Test that an error is raised if the provided shape is invalid"""

        shape = {1, 2}
        with pytest.raises(TypeError, match="is not a valid constraint for the 'shape' argument"):
            TensorConstraint(shape=shape)

    def test_tensor_constraint_init_invalid_rank(self):
        """Test that an error is raised if the provided rank is invalid"""

        rank = 1.5
        with pytest.raises(TypeError, match="is not a valid constraint for the 'rank' argument"):
            TensorConstraint(rank=rank)

    def test_tensor_constraint_properties(self):
        """Test that the properties of TensorConstraint object are correct."""
        rank = 1
        constraint = TensorConstraint(rank=rank)

        assert constraint.expected_type == builtin.TensorType
        assert constraint.type_name == "tensor"
        assert constraint.mapping_type_vars({}) is constraint

    @pytest.mark.parametrize("rank", [0, 1, 2])
    def test_tensor_single_rank_constraint_verify_valid(self, rank):
        """Test that verifying a TensorType attribute with the same rank as the TensorConstraint
        does not raise an exception."""
        constraint = TensorConstraint(rank=rank)
        attr = builtin.TensorType(builtin.i32, [1] * rank)

        constraint.verify(attr, None)

    def test_tensor_multi_rank_constraint_verify_valid(self):
        """Test that verifying a TensorType attribute with any of the ranks as the TensorConstraint
        does not raise an exception."""
        rank = (3, 4, 5)
        constraint = TensorConstraint(rank=rank)

        for r in rank:
            attr = builtin.TensorType(builtin.i32, [1] * r)
            constraint.verify(attr, None)

    def test_tensor_shape_constraint_verify_valid(self):
        """Test that verifying an attribute with a valid shape does not raise an exception."""
        constraint = TensorConstraint(shape=(1, 2, 3))
        attr = builtin.TensorType(builtin.i32, (1, 2, 3))

        constraint.verify(attr, None)

    def test_tensor_element_type_constraint_verify_valid(self):
        """Test that verifying an attribute with a valid element type does not raise an exception."""
        constraint = TensorConstraint(element_type=builtin.i32)
        attr = builtin.TensorType(builtin.i32, [1])

        constraint.verify(attr, None)

    @pytest.mark.parametrize("rank", [0, 1, 2])
    def test_tensor_single_rank_constraint_verify_invalid(self, rank):
        """Test that verifying a TensorType attribute with a different rank as the
        TensorConstraint raises a VerifyException."""
        constraint = TensorConstraint(rank=rank)
        attr = builtin.TensorType(builtin.i32, [1] * (rank + 1))

        with pytest.raises(VerifyException, match=f"Invalid value {rank + 1}, expected {rank}"):
            constraint.verify(attr, None)

    def test_tensor_multi_rank_constraint_verify_invalid(self):
        """Test that verifying a TensorType attribute with a different rank as the
        TensorConstraint raises a VerifyException."""

        rank = {3, 4, 5}
        invalid_rank = 2
        constraint = TensorConstraint(rank=rank)
        attr = builtin.TensorType(builtin.i32, [1] * invalid_rank)

        with pytest.raises(
            VerifyException, match=f"Invalid value {invalid_rank}, expected one of {rank}"
        ):
            constraint.verify(attr, None)

    def test_tensor_constraint_verify_invalid_type(self):
        """Test that verifying an attribute with a type other than TensorType raises a
        VerifyException."""
        constraint = TensorConstraint(rank=1)
        attr = builtin.MemRefType(builtin.i32, [1])

        with pytest.raises(VerifyException, match=f"{attr} should be of type TensorType"):
            constraint.verify(attr, None)

    def test_tensor_shape_constraint_verify_invalid(self):
        """Test that verifying an attribute with an invalid shape raises an exception."""
        constraint = TensorConstraint(shape=(1, 2, 3))
        attr = builtin.TensorType(builtin.i32, (2, 2, 3))

        with pytest.raises(VerifyException, match=r"Expected attribute \[.*\] but got \[.*\]"):
            constraint.verify(attr, None)

    def test_tensor_element_type_constraint_verify_invalid(self):
        """Test that verifying an attribute with an invalid element type raises an exception."""
        constraint = TensorConstraint(element_type=builtin.i32)
        attr = builtin.TensorType(builtin.i64, [1])

        with pytest.raises(
            VerifyException, match=f"Expected attribute {builtin.i32} but got {builtin.i64}"
        ):
            constraint.verify(attr, None)

    def test_tensor_constraint_integration(self, my_dialect):
        """Test that verification of legal operations with tensor operand/result constraints does
        not raise an exception."""
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> tensor<2xf64>
            %1 = "test.op"() : () -> tensor<2x4xf64>
            %2 = "test.op"() : () -> tensor<2x4x5x3xf64>
            %3 = "test.op"() : () -> tensor<1x2x3xf64>
            %4 = "my_dialect.tensor_float64"(%0) : (tensor<2xf64>) -> tensor<2xf64>
            %5 = "my_dialect.tensor_rank1"(%0) : (tensor<2xf64>) -> tensor<2xf64>
            %6 = "my_dialect.tensor_rank24"(%1) : (tensor<2x4xf64>) -> tensor<2x4xf64>
            %7 = "my_dialect.tensor_rank24"(%2) : (tensor<2x4x5x3xf64>) -> tensor<2x4x5x3xf64>
            %8 = "my_dialect.tensor_shape123"(%3) : (tensor<1x2x3xf64>) -> tensor<1x2x3xf64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()
        xdsl_module.verify()

    def test_tensor_constraint_integration_invalid(self, my_dialect):
        """Test that verification of illegal operations with tensor operands/result constraints
        raises a VerifyException."""
        program = """
        func.func public @test_workload() -> () {
            %0 = "test.op"() : () -> tensor<2x2xi64>
            %1 = "my_dialect.tensor_float64"(%0) : (tensor<2x2xi64>) -> tensor<2x2xi64>
            func.return
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module: builtin.ModuleOp = QuantumParser(
            ctx, program, extra_dialects=(test.Test, my_dialect)
        ).parse_module()

        with pytest.raises(
            VerifyException,
            match=f"Expected attribute {builtin.Float64Type()} but got {builtin.i64}",
        ):
            xdsl_module.verify()
