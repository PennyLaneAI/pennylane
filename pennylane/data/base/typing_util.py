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

from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    Dict,
    ForwardRef,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    _SpecialForm,
    get_args,
    get_origin,
)

JSON = Union[str, int, bool, float, None, Dict[str, Any], List[Any]]

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


def get_type(type_or_obj: Union[object, Type]) -> Type:
    """Given an object or an object type, returns the underlying class.

    Examples:
        >>> _get_type(list)
        <class 'list'>
        >>> _get_type(List[int])
        <class 'list'>
        >>> _get_type([])
        <class 'list'>
    """

    # First, check if this is a special type, e.g a parametrized
    # generic like List[int]
    special_args = resolve_special_type(type_or_obj)
    if special_args is not None:
        type_, _ = special_args
    elif isinstance(type_or_obj, type):
        type_ = type_or_obj
    else:
        type_ = type(type_or_obj)

    return type_


@lru_cache
def get_type_str(cls: Union[type, str, None]) -> str:  # pylint: disable=too-many-return-statements
    """Return a string representing the type ``cls``.

    If cls is a built-in type, such as 'str', returns the unqualified
        name.

    If cls is a parametrized generic such as List[str], or a special typing
        form such as Optional[int], returns the string representation of cls.

    Otherwise, returns the fully-qualified class name, including the module.
    """
    if cls is None or cls is type(None):
        return "None"

    if isinstance(cls, str):
        return cls

    if isinstance(cls, ForwardRef):
        # String annotations, as in List['MyClass']
        return cls.__forward_arg__

    if isinstance(cls, _SpecialForm):
        # These are typing constructs like Union, Literal etc that
        # are not parametrized
        return cls._name  # pylint: disable=protected-access

    orig_type = get_origin(cls)
    if orig_type is not None:
        # This is either a parametrized generic or parametrized special form
        orig_args = get_args(cls)
        if orig_args:
            return f"{get_type_str(orig_type)}[{', '.join(get_type_str(arg) for arg in orig_args)}]"

        return get_type_str(orig_type)

    if getattr(cls, "__module__", None) in ("builtins", None):
        # This is a built-in type
        return cls.__name__

    # Regular class
    return f"{cls.__module__}.{cls.__qualname__}"


def resolve_special_type(type_: Any) -> Optional[Tuple[type, List[type]]]:
    """Converts special typing forms (Union[...], Optional[...]), and parametrized
    generics (List[...], Dict[...]) into a 2-tuple of its base type and arguments.
    If ``type_`` is a regular type, or an object, this function will return
    ``None``.

    Note that this function will only perform one level of recursion - the
    arguments of nested types will not be resolved:

        >>> resolve_special_type(List[List[int]])
        (<class 'list'>, [<class 'list'>])

    Further examples:
        >>> resolve_special_type(Union[str, int])
        (typing.Union, [<class 'str'>, <class 'int'>])
        >>> resolve_special_type(List[int])
        (<class 'list'>, [<class 'int'>])
        >>> resolve_special_type(List)
        (<class 'list'>, [])
        >>> resolve_special_type(list)
        None
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
