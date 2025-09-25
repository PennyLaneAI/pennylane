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
Traits for xDSL operations.

This module provides operation traits that can be used to define operation invariants,
additional semantic information, or to group operations that have similar properties.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import TensorType, VectorType
from xdsl.ir import Attribute, Operation
from xdsl.traits import OpTrait
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.type import get_element_type_or_self, have_compatible_shape


@dataclass(frozen=True)
class SameOperandsAndResultShape(OpTrait):
    """Constrain the operation to have the same shape for all operands and results."""

    # TODO: This trait should be added to ElementwiseBinaryOperation and
    # ElementwiseUnaryOperation operations when upstreaming to xdsl.

    def verify(self, op: Operation) -> None:
        """Verify that the operation has the same shape for all operands and results."""

        if len(op.results) < 1 or len(op.operands) < 1:
            raise VerifyException(f"'{op.name}' requires at least one result or operand")

        # Get all types (operands and results) to check for compatible shapes
        all_types = list(op.operand_types) + list(op.result_types)

        # Check that all types have compatible shapes
        for type_to_check in all_types[1:]:
            if not have_compatible_shape(all_types[0], type_to_check):

                raise VerifyException(
                    f"'{op.name}' requires the same shape for all operands and results"
                )


@dataclass(frozen=True)
class SameOperandsElementType(OpTrait):
    """Constrain the operation to have the same element type for all operands."""

    # TODO: This trait should be added to ElementwiseBinaryOperation and
    # ElementwiseUnaryOperation operations when upstreaming to xdsl.

    def verify(self, op: Operation) -> None:
        """Verify that the operation has the same element type for all operands."""

        if len(op.operands) <= 1:
            return

        # Get the element type of the first operand
        first_elem_type = get_element_type_or_self(op.operand_types[0])

        # Check that all other operands have the same element type
        for operand_type in op.operand_types[1:]:
            elem_type = get_element_type_or_self(operand_type)
            if elem_type != first_elem_type:
                raise VerifyException(
                    f"'{op.name}' requires the same element type for all operands"
                )


@dataclass(frozen=True)
class SameOperandsAndResultElementType(OpTrait):
    """Constrain the operation to have the same element type for all operands and results."""

    def verify(self, op: Operation) -> None:
        """Verify that the operation has the same element type for all operands and results."""

        if len(op.results) < 1 or len(op.operands) < 1:
            raise VerifyException(f"'{op.name}' requires at least one result or operand")

        # Get the element type of the first operand
        first_elem_type = get_element_type_or_self(op.operand_types[0])

        all_types = list(op.operand_types) + list(op.result_types)

        # Check that all other operands have the same element type
        for type_to_check in all_types[1:]:
            elem_type = get_element_type_or_self(type_to_check)
            if elem_type != first_elem_type:
                raise VerifyException(
                    f"'{op.name}' requires the same element type for all operands and results"
                )


@dataclass(frozen=True)
class Elementwise(OpTrait):
    """
    The following is the definition of the `Elementwise` trait from MLIR:

    https://github.com/llvm/llvm-project/blob/f8cb7987c64dcffb72414a40560055cb717dbf74/mlir/include/mlir/IR/OpDefinition.h#L1378-L1409

    TODO: Add this trait to all the elementwise operations in xdsl when upstreaming.

    Tags elementwise operations on vectors or tensors.

    NOTE: Not all ops that are "elementwise" in some abstract sense satisfy this trait.
    In particular, broadcasting behavior is not allowed.

    An `Elementwise` op must satisfy the following properties:

    1. If any result is a vector/tensor then at least one operand must also be a
       vector/tensor.
    2. If any operand is a vector/tensor then there must be at least one result
       and all results must be vectors/tensors.
    3. All operand and result vector/tensor types must be of the same shape. The
       shape may be dynamic in which case the op's behaviour is undefined for
       non-matching shapes.
    4. The operation must be elementwise on its vector/tensor operands and
       results. When applied to single-element vectors/tensors, the result must
       be the same per element.

    Rationale:
    - 1. and 2. guarantee a well-defined iteration space and exclude the cases
      of 0 non-scalar operands or 0 non-scalar results, which complicate a
      generic definition of the iteration space.
    - 3. guarantees that folding can be done across scalars/vectors/tensors with
      the same pattern, as otherwise lots of special handling for type
      mismatches would be needed.
    - 4. guarantees that no error handling is needed. Higher-level dialects
      should reify any needed guards or error handling code before lowering to
      an Elementwise op.
    """

    def verify(self, op: Operation) -> None:
        """Verify that the operation is elementwise."""

        # Filter mappable types from results and operands (vectors/tensors only)
        result_mappable_types = [t for t in op.result_types if Elementwise.is_mappable_type(t)]
        operand_mappable_types = [t for t in op.operand_types if Elementwise.is_mappable_type(t)]

        # If the op only has scalar operand/result types, then we have nothing to check
        if not result_mappable_types and not operand_mappable_types:
            return

        # If a result is non-scalar, then at least one operand must be non-scalar
        if result_mappable_types and not operand_mappable_types:
            raise VerifyException(
                f"'{op.name}': if a result is non-scalar, then at least one "
                "operand must be non-scalar"
            )

        # At this point, operand_mappable_types should not be empty
        assert operand_mappable_types, "At least one operand must be a vector or tensor"

        # If an operand is non-scalar, then there must be at least one non-scalar result
        if not result_mappable_types:
            raise VerifyException(
                f"'{op.name}': if an operand is non-scalar, then there must be at "
                "least one non-scalar result"
            )

        # If an operand is non-scalar, then all results must be non-scalar
        if len(result_mappable_types) != len(op.results):
            raise VerifyException(
                f"'{op.name}': if an operand is non-scalar, then all results must be non-scalar"
            )

        # All non-scalar operands/results must have the same shape and base type
        all_types = operand_mappable_types + result_mappable_types

        # Check that all types have compatible shapes
        for type_to_check in all_types[1:]:
            if not have_compatible_shape(all_types[0], type_to_check):
                raise VerifyException(
                    f"'{op.name}': all non-scalar operands/results must have the "
                    "same shape and base type"
                )

    @staticmethod
    def is_mappable_type(attr_type: Attribute) -> bool:
        """Return True if the type is elementwise-mappable (vector or tensor).

        There is a TODO in MLIR to generalize this trait to avoid hardcoding vector/tensor.
        We should update this when the TODO is resolved.
        """
        return isinstance(attr_type, (VectorType, TensorType))


@dataclass(frozen=True)
class AllMatchSameOperatorTrait(OpTrait):
    """
    Verify that a list of operation attributes all match under the same operator
    (e.g., size, rank, type, shape, element type).

    Parameters:
    - attr_names: attribute names on the op to compare
    - operator: callable taking the attribute value and returning a comparable value
    - summary: human-readable name of the property used in error messages
    """

    attr_names: tuple[str, ...]
    operator: Callable[[Any], Any]
    summary: str

    def verify(self, op: Operation) -> None:
        """Verify that the operation attributes all match under the same operator."""
        attributes = []
        for name in self.attr_names:
            value = getattr(op, name, None)
            if value is None:
                return
            attributes.append(value)

        if len(attributes) <= 1:
            return

        names_str = ", ".join(self.attr_names)
        try:
            results = [self.operator(attr) for attr in attributes]
        except (TypeError, ValueError, AttributeError) as e:
            raise VerifyException(f"cannot compute {self.summary} for {{{names_str}}}: {e}") from e

        first = results[0]
        if any(res != first for res in results[1:]):
            results_str = ", ".join(str(r) for r in results)
            raise VerifyException(
                f"all of {{{names_str}}} must have the same {self.summary}: got {self.summary}s {results_str}"
            )
