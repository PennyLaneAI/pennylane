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
from functools import lru_cache
from typing import (
    Any,
    ForwardRef,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from numpy.typing import ArrayLike

# Type aliases for Zarr objects.
ZarrAttrs = MutableMapping
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


@lru_cache
def get_type_str(cls: Union[type, str, None]) -> str:
    """Return a string representing the type ``cls``.

    If cls is a built-in type, such as 'str', returns the unqualified
        name.

    If cls is a parametrized generic such as List[str], or a special typing
        form such as Optional[int], returns the string representation of cls.

    Otherwise, returns the fully-qualified class name, including the module.
    """
    if cls is None:
        return "None"

    if isinstance(cls, str):
        return cls

    if isinstance(cls, ForwardRef):
        return cls.__forward_arg__

    orig_type = get_origin(cls)
    if orig_type is not None:
        orig_args = get_args(cls)
        if orig_args:
            return f"{get_type_str(orig_type)}[{','.join(get_type_str(arg) for arg in orig_args)}]"
        return get_type_str(orig_type)

    if getattr(cls, "__module__", None) in ("builtins", None):
        return cls.__name__

    return f"{cls.__module__}.{cls.__qualname__}"


def resolve_special_type(type_: Any) -> Optional[Tuple[type, List[type]]]:
    """Converts special typing forms (Union[...], Optional[...]), and parametrized
    generics (List[...], Dict[...]) into a 2-tuple of its base type and arguments.
    If ``type_`` is a regular type, or an object, this function will return
    ``None``.

    For example:
        resolve_special_type(Union[str, int]) == (Union, [str, int])
        resolve_special_type(List[str]) == (list, [str])
        resolve_special_type(list) == (list, [])
    """

    orig_type = get_origin(type_)
    if orig_type is None:
        return None

    args = list(get_args(type_))
    type_ = orig_type

    for i, arg in enumerate(args):
        orig_type = get_origin(arg)
        if orig_type:
            args[i] = orig_type

    return (type_, args)
