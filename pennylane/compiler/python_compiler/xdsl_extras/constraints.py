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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar, get_args, get_origin

from xdsl.dialects.builtin import ArrayAttr, IntAttr, IntAttrConstraint, MemRefType, TensorType
from xdsl.ir import Attribute, AttributeInvT
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AttrConstraint,
    ConstraintContext,
    EqIntConstraint,
    IntSetConstraint,
    IRDLAttrConstraint,
    RangeLengthConstraint,
    RangeOf,
)
from xdsl.utils.exceptions import VerifyException

# Get all bases of type alias
_constr_bases = get_args(IRDLAttrConstraint)
# If base is generic, get the origin
_valid_constraint_bases = tuple(
    c if (origin := get_origin(c)) is None else origin for c in _constr_bases
)


@dataclass(frozen=True)
class ContainerConstraint(AttrConstraint, ABC):
    r"""Internal base class for constraining the element type and shape of container types.

    The shape of the container can be constrained by providing a constraint either directly
    for the shape, or by providing a constraint for the rank. There are two ways to provide
    an explicit shape constraint:

        * By providing an ``IRDLAttrConstraint`` for the ``shape`` argument.
        * By providing a sequence of ``int``\ s specifying the concrete expected shape for
          the ``shape`` argument.

    There are three ways to provide the rank constraint:

        * By providing an ``IRDLAttrConstraint`` for the ``rank`` argument.
        * By providing an ``int`` representing the concrete expected rank for the ``rank``
          argument.
        * By providing a sequence of ``int``\ s specifying the various allowed ranks for the
          ``rank`` argument.

    .. note::

        Either the ``shape`` or ``rank`` constraint must be provided, not both.

    Args:
        element_type (IRDLAttrConstraint | None): The constraint for the element type.
            Default is ``None``, which indicates that any element type is allowed.
        shape (IRDLAttrConstraint | Sequence[int] | None): The constraint for the shape.
            Default is ``None``, which indicates that any shape is allowed.
        shape (IRDLAttrConstraint | Sequence[int] | int | None): The constraint for the
            rank. Default is ``None``, which indicates that any rank is allowed.
    """

    element_type: IRDLAttrConstraint[AttributeInvT]

    shape: IRDLAttrConstraint[AttributeInvT]

    def __init__(
        self,
        element_type: IRDLAttrConstraint[AttributeInvT] | None = None,
        shape: IRDLAttrConstraint[AttributeInvT] | Sequence[int] | None = None,
        rank: IRDLAttrConstraint[AttributeInvT] | Sequence[int] | int | None = None,
    ):
        self.element_type = element_type or AnyAttr()
        if shape is None and rank is None:
            self.shape = AnyAttr()
            return

        if shape is not None and rank is not None:
            raise ValueError("Only one of 'shape' or 'rank' must be provided.")

        if shape is not None:
            if isinstance(shape, Sequence):
                shape = ArrayAttr([IntAttr(s) for s in shape])
            assert isinstance(shape, _valid_constraint_bases)
            self.shape = shape
            return

        # rank is not None
        if isinstance(rank, (int, Sequence)):
            # Constrain shape to have length `rank` if `rank` is an int
            if isinstance(rank, int):
                length_constr = EqIntConstraint(rank)
            else:
                length_constr = IntSetConstraint(frozenset(rank))
            rank = ArrayAttr.constr(
                RangeLengthConstraint(
                    constraint=RangeOf(IntAttrConstraint(AnyInt())), length=length_constr
                )
            )
        assert isinstance(rank, _valid_constraint_bases)
        self.shape = rank

    @property
    @abstractmethod
    def expected_type(self) -> type[Attribute]:
        """The expected IR type class (e.g., builtin.TensorType)."""

    @property
    def type_name(self) -> str:
        """The name of the type for use in error messages (e.g., 'tensor')."""
        return self.expected_type.name

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        # pylint: disable=missing-function-docstring
        if not isinstance(attr, self.expected_type):
            raise VerifyException(f"{attr} should be of type {self.expected_type.__name__}.")
        self.element_type.verify(attr.element_type, constraint_context=constraint_context)
        self.shape.verify(attr.shape, constraint_context=constraint_context)

    def mapping_type_vars(
        self, type_var_mapping: dict[TypeVar, AttrConstraint]
    ) -> "ContainerConstraint":
        # pylint: disable=unused-argument,missing-function-docstring
        return self


@dataclass(frozen=True)
class TensorConstraint(ContainerConstraint):
    """TensorType constraint for element type and shape."""

    @property
    def expected_type(self):
        return TensorType


@dataclass(frozen=True)
class MemRefConstraint(ContainerConstraint):
    """MemRefType constraint for element type and shape."""

    @property
    def expected_type(self):
        return MemRefType
