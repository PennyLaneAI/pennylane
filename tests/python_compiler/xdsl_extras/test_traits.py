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

"""Test the traits defined within the xdsl_extras module."""

from typing import TypeAlias

import pytest

# pylint: disable=wrong-import-position
# pylint: disable=too-few-public-methods

xdsl = pytest.importorskip("xdsl")

from xdsl.dialects.builtin import AnyAttr, TensorType, f32, i32
from xdsl.ir import Attribute

# xdsl imports
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

# Import the custom traits we want to test
from pennylane.compiler.python_compiler.xdsl_extras.traits import (
    Elementwise,
    SameOperandsAndResultShape,
    SameOperandsElementType,
)

pytestmark = pytest.mark.external

AnyTensorType: TypeAlias = TensorType[Attribute]


# Test the SameOperandsAndResultShape trait
def test_same_operands_and_result_shape_trait():
    """Test the SameOperandsAndResultShape trait."""

    @irdl_op_definition
    class ShapeTestOp(IRDLOperation):
        """Test operation for the SameOperandsAndResultShape trait."""

        name = "test.shape_test"
        traits = traits_def(SameOperandsAndResultShape())
        operand = operand_def(AnyTensorType)
        result = result_def(AnyTensorType)

    assert ShapeTestOp.has_trait(SameOperandsAndResultShape)

    operand = create_ssa_value(TensorType(i32, [2, 3]))
    op = ShapeTestOp.create(operands=[operand], result_types=[TensorType(i32, [2, 3])])
    op.verify()

    op = ShapeTestOp.create(operands=[operand], result_types=[TensorType(i32, [3, 2])])
    with pytest.raises(VerifyException, match="requires the same shape"):
        op.verify()


# Test the SameOperandsElementType trait
def test_same_operands_element_type_trait():
    """Test the SameOperandsElementType trait."""

    @irdl_op_definition
    class ShapeTestOp(IRDLOperation):
        """Test operation for the SameOperandsElementType trait."""

        name = "test.element_type_test"
        traits = traits_def(SameOperandsElementType())

        operand1 = operand_def(AnyTensorType)
        operand2 = operand_def(AnyTensorType)
        result = result_def(AnyTensorType)

    assert ShapeTestOp.has_trait(SameOperandsElementType)

    operand1 = create_ssa_value(TensorType(i32, [2, 3]))
    operand2 = create_ssa_value(TensorType(i32, [3, 2]))
    op = ShapeTestOp.create(operands=[operand1, operand2], result_types=[TensorType(i32, [2, 2])])
    op.verify()

    operand3 = create_ssa_value(TensorType(f32, [2, 3]))
    op = ShapeTestOp.create(operands=[operand1, operand3], result_types=[TensorType(f32, [2, 3])])

    with pytest.raises(VerifyException, match="requires the same element type for all operands"):
        op.verify()


# Test the Elementwise trait
@irdl_op_definition
class ElementwiseTestOp(IRDLOperation):
    """Test operation for the Elementwise trait."""

    name = "test.elementwise_test"
    traits = traits_def(Elementwise())

    operand = var_operand_def(AnyAttr())
    result = result_def(AnyAttr())


def test_elementwise_trait():
    """Test the Elementwise trait."""
    assert ElementwiseTestOp.has_trait(Elementwise)

    operand = create_ssa_value(TensorType(i32, [2, 3]))
    op = ElementwiseTestOp.create(operands=[operand], result_types=[TensorType(i32, [2, 3])])
    op.verify()

    operand = create_ssa_value(i32)
    op = ElementwiseTestOp.create(operands=[operand], result_types=[i32])
    op.verify()

    scalar_operand = create_ssa_value(i32)
    tensor_operand = create_ssa_value(TensorType(i32, [2, 3]))
    op = ElementwiseTestOp.create(
        operands=[scalar_operand, tensor_operand], result_types=[TensorType(i32, [2, 3])]
    )
    op.verify()


def test_elementwise_trait_failure_no_tensor_result():
    """Test that Elementwise trait fails when operand is tensor but result is scalar."""
    operand = create_ssa_value(TensorType(i32, [2, 3]))
    op = ElementwiseTestOp.create(operands=[operand], result_types=[i32])

    with pytest.raises(
        VerifyException,
        match="if an operand is non-scalar, then there must be at least one non-scalar result",
    ):
        op.verify()


def test_elementwise_trait_failure_no_tensor_operand():
    """Test that Elementwise trait fails when result is tensor but operand is scalar."""
    operand = create_ssa_value(i32)
    op = ElementwiseTestOp.create(operands=[operand], result_types=[TensorType(i32, [2, 3])])

    with pytest.raises(
        VerifyException,
        match="if a result is non-scalar, then at least one operand must be non-scalar",
    ):
        op.verify()


def test_elementwise_trait_failure_shape_mismatch():
    """Test that Elementwise trait fails for shape mismatches."""
    operand1 = create_ssa_value(TensorType(i32, [2, 3]))
    operand2 = create_ssa_value(TensorType(i32, [3, 2]))
    op = ElementwiseTestOp.create(
        operands=[operand1, operand2], result_types=[TensorType(i32, [2, 3])]
    )

    with pytest.raises(
        VerifyException,
        match="all non-scalar operands/results must have the same shape and base type",
    ):
        op.verify()
