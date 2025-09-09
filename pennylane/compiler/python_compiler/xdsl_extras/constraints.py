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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    IntAttrConstraint,
    MemRefType,
    TensorType,
    TupleType,
)
from xdsl.ir import Attribute, AttributeInvT
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AttrConstraint,
    ConstraintContext,
    EqIntConstraint,
    IntConstraint,
    IntSetConstraint,
    IRDLAttrConstraint,
    RangeLengthConstraint,
    RangeOf,
    irdl_to_attr_constraint,
)
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True, init=False)
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
        * By providing a collection of ``int``\ s specifying the various allowed ranks for the
          ``rank`` argument.

    .. note::

        Only one of the ``shape`` or ``rank`` constraint may be provided, not both.

    Args:
        element_type (IRDLAttrConstraint | None): The constraint for the element type.
            Default is ``None``, which indicates that any element type is allowed.
        shape (IRDLAttrConstraint | Sequence[int] | None): The constraint for the shape.
            Default is ``None``, which indicates that any shape is allowed.
        rank (IRDLAttrConstraint | Collection[int] | int | None): The constraint for the
            rank. Default is ``None``, which indicates that any rank is allowed.
    """

    element_type: IRDLAttrConstraint[AttributeInvT]

    shape: IRDLAttrConstraint[AttributeInvT]

    def __init__(
        self,
        *,
        element_type: IRDLAttrConstraint[AttributeInvT] | None = None,
        shape: IRDLAttrConstraint[AttributeInvT] | Sequence[int] | None = None,
        rank: IRDLAttrConstraint[AttributeInvT] | Collection[int] | int | None = None,
    ):
        element_type = element_type or AnyAttr()
        element_type_constr = element_type
        shape_constr = None

        if shape is not None and rank is not None:
            raise ValueError("Only one of 'shape' or 'rank' may be provided.")

        if shape is None and rank is None:
            shape_constr = AnyAttr()

        elif shape is not None:
            shape_constr = shape
            if isinstance(shape_constr, Sequence):
                shape_constr = ArrayAttr([IntAttr(s) for s in shape])

        # rank is not None
        else:
            shape_constr = rank
            if isinstance(shape_constr, (int, Collection)):
                # Constrain shape to have length `rank` if `rank` is an int
                if isinstance(shape_constr, int):
                    length_constr = EqIntConstraint(shape_constr)
                else:
                    length_constr = IntSetConstraint(frozenset(shape_constr))
                shape_constr = ArrayAttr.constr(
                    RangeLengthConstraint(
                        constraint=RangeOf(IntAttrConstraint(AnyInt())), length=length_constr
                    )
                )

        if not isinstance(element_type_constr, (Attribute, AttrConstraint)):
            raise TypeError(
                f"{element_type} is not a valid constraint for the 'element_type' argument. "
                "'element_type' must be an AttrConstraint or Attribute."
            )
        if not isinstance(shape_constr, (Attribute, AttrConstraint)):
            if shape is not None:
                raise TypeError(
                    f"{shape} is not a valid constraint for the 'shape' argument. 'shape' "
                    "must be an AttrConstraint, Attribute, or sequence of integers."
                )
            raise TypeError(
                f"{rank} is not a valid constraint for the 'rank' argument. 'rank' must be "
                "an AttrConstraint, Attribute, integer, or collection of integers."
            )

        object.__setattr__(self, "element_type", element_type_constr)
        object.__setattr__(self, "shape", shape_constr)

    @property
    @abstractmethod
    def expected_type(self) -> type[Attribute]:
        """The expected IR type class (e.g., builtin.TensorType)."""

    @property
    def type_name(self) -> str:
        """The name of the type for use in error messages (e.g., 'tensor')."""
        return self.expected_type.name

    def get_bases(self) -> set[type[Attribute]]:
        """Get a set of base types that can satisfy this constraint (e.g., {builtin.TensorType})."""
        return {self.expected_type}

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        # pylint: disable=missing-function-docstring
        if not isinstance(attr, self.expected_type):
            raise VerifyException(f"{attr} should be of type {self.expected_type.__name__}.")
        constr = self.expected_type.constr(element_type=self.element_type, shape=self.shape)
        constr.verify(attr, constraint_context)

    def mapping_type_vars(
        self, type_var_mapping: dict[TypeVar, AttrConstraint]
    ) -> "ContainerConstraint":
        # pylint: disable=unused-argument,missing-function-docstring
        return self


@dataclass(frozen=True, init=False)
class TensorConstraint(ContainerConstraint):
    """TensorType constraint for element type and shape."""

    @property
    def expected_type(self):
        return TensorType


@dataclass(frozen=True, init=False)
class MemRefConstraint(ContainerConstraint):
    """MemRefType constraint for element type and shape."""

    @property
    def expected_type(self):
        return MemRefType


@dataclass(frozen=True, init=False)
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
