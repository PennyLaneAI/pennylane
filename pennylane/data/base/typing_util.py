# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a sentinel object, common type objects utilities for parsing types
and converting them to strings."""

from collections.abc import MutableMapping
from enum import Enum
from types import GenericAlias
from typing import Any, List, Literal, Tuple, TypeVar, Union, get_args, get_origin

from numpy.typing import ArrayLike

# Type aliases for Zarr objects.
ZarrArray = ArrayLike
ZarrGroup = MutableMapping
ZarrAny = Union[ZarrArray, ZarrGroup]
Zarr = TypeVar("Zarr", ZarrArray, ZarrGroup, ZarrAny)

# Generic type variable
T = TypeVar("T")


class UnsetType(Enum):
    """Sentintel object - used for defaults where None
    may be a valid (non-default) value.

    This class is an enum so it may used as a Literal
    type annotation, e.g Literal[UNSET]."""

    UNSET = "UNSET"

    def __bool__(self) -> Literal[False]:
        return False


UNSET = UnsetType.UNSET


def get_type_str(cls_or_obj: Union[object, type]) -> str:
    """Return a string representing the type of `cls_or_obj`.

    If cls_or_obj is a built-in type, such as 'str', returns the unqualified
        name.

    If cls_or_obj is a parametrized generic of a built-in type, such as list[str],
        returns the string representation of that generic (e.g 'list[str]').

    Otherwise, returns the fully-qualified class name, including the module.

    """
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
    if isinstance(cls_or_obj, GenericAlias):
        return str(cls_or_obj)

    if cls.__module__ == "builtins":
        return cls.__name__

    return f"{cls.__module__}.{cls.__qualname__}"


def resolve_special_type(type_: Any) -> Tuple[type, List[type]]:
    """Converts special typing forms (Union[...], Optional[...]), and parametrized
    generics (list[...], dict[...]) into a 2-tuple of its base type and arguments.
    If ``type_`` is a regular type, or an object, this function will return
    ``None``.

    For example:
        resolve_special_type(Union[str, int]) == (Union, [str, int])
        resolve_special_type(list[str]) == (list, [str])
        resolve_special_type(list) == (list, [])
    """

    orig_type = get_origin(type_)
    if orig_type is None:
        return (type_, [])

    args = list(get_args(type_))
    type_ = orig_type

    for i, arg in enumerate(args):
        orig_type = get_origin(arg)
        if orig_type:
            args[i] = orig_type

    return (type_, args)
