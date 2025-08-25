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

"""This module contains additional type and attribute constraints that are currently not available
upstream in xDSL."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from typing_extensions import TypeVar
from xdsl.dialects import builtin
from xdsl.dialects.builtin import TupleType
from xdsl.ir import Attribute
from xdsl.irdl import (
    AttrConstraint,
    ConstraintContext,
    IntConstraint,
    irdl_to_attr_constraint,
)
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True)
class _RankConstraint(AttrConstraint, ABC):
    """Internal base class to constrain an attribute to be of a given rank.

    Subclasses must provide 'expected_type' and 'type_name'.
    """

    expected_rank: int
    """The expected rank."""

    @property
    @abstractmethod
    def expected_type(self) -> type:
        """The expected IR type class (e.g., builtin.TensorType)."""

    @property
    @abstractmethod
    def type_name(self) -> str:
        """The name of the type for use in error messages (e.g., 'tensor')."""

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        # pylint: disable=unused-argument,missing-function-docstring
        if not isinstance(attr, self.expected_type):
            raise VerifyException(f"{attr} should be of type {self.expected_type.__name__}.")
        if attr.get_num_dims() != self.expected_rank:
            raise VerifyException(
                f"Expected {self.type_name} rank to be {self.expected_rank}, got {attr.get_num_dims()}."
            )

    def mapping_type_vars(
        self, type_var_mapping: dict[TypeVar, AttrConstraint]
    ) -> "_RankConstraint":
        # pylint: disable=unused-argument,missing-function-docstring
        return self


@dataclass(frozen=True)
class MemRefRankConstraint(_RankConstraint):
    """
    Constrain a memref to be of a given rank.
    """

    @property
    def expected_type(self) -> type:
        return builtin.MemRefType

    @property
    def type_name(self) -> str:
        return "memref"


@dataclass(frozen=True)
class TensorRankConstraint(_RankConstraint):
    """
    Constrain a tensor to be of a given rank.
    """

    @property
    def expected_type(self) -> type:
        return builtin.TensorType

    @property
    def type_name(self) -> str:
        return "tensor"


@dataclass(frozen=True)
class NestedTupleOfConstraint(AttrConstraint[TupleType]):
    """Constrain a nested tuple whose flattened leaves all match any allowed constraints."""

    elem_constraints: tuple[AttrConstraint, ...]

    def __init__(self, elem_constraints: Sequence[object]):
        object.__setattr__(
            self,
            "elem_constraints",
            tuple(irdl_to_attr_constraint(c) for c in elem_constraints),
        )

    def get_flattened(self, a: Attribute):
        """Get the flattened leaves of a tuple."""
        if isinstance(a, TupleType):
            for t in a.types.data:
                yield from self.get_flattened(t)
        else:
            yield a

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        """Verify that the attribute is a tuple of allowed types."""
        if not isinstance(attr, TupleType):
            raise VerifyException(f"expected TupleType, got {type(attr)}")

        leaves = list(self.get_flattened(attr))

        for i, leaf in enumerate(leaves):
            matched = False
            for constr in self.elem_constraints:
                try:
                    constr.verify(leaf, constraint_context)
                    matched = True
                    break
                except VerifyException:
                    # Try next allowed constraint
                    pass
            if not matched:
                raise VerifyException(f"tuple leaf {i} failed all allowed constraints: {leaf}")

    def mapping_type_vars(
        self,
        type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint],
    ) -> AttrConstraint:
        """Map type variables to constraints."""
        # pylint: disable=unused-argument
        return self
