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
"""
Contains types representing dataset parameters and the dataset
folder map.
"""

import enum
import typing
from collections.abc import Mapping
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any, Dict, FrozenSet, List, Literal, Union, cast


class ParamArg(enum.Enum):
    """Enum representing special args to ``load()``.

    FULL: used to request all attributes
    DEFAULT: used to request the default attribute
    """

    FULL = "full"
    DEFAULT = "default"

    @classmethod
    @lru_cache(maxsize=1)
    def values(cls) -> FrozenSet[str]:
        """Returns all values."""
        return frozenset(arg.value for arg in cls)

    @classmethod
    def is_arg(cls, val: Union["ParamArg", str]) -> bool:
        """Returns true if ``val`` is a ``ParamArg``, or one
        of its values."""
        return isinstance(val, ParamArg) or val in cls.values()

    def __str__(self) -> str:
        return cast(str, self.value)


DEFAULT = ParamArg.DEFAULT
FULL = ParamArg.FULL

ParamName = str
ParamVal = str
ParamArgs = Dict[ParamName, Union[ParamArg, List[ParamVal]]]
Description = Dict[ParamName, ParamVal]
DataPath = PurePosixPath


class FolderMapView(Mapping):
    """Provides a read-only view of the ``foldermap.json`` file in
    the datasets bucket. The folder map is a nested mapping of
    dataset parameters to their path, relative to ``foldermap.json``.

    A dictionary in the folder map can optionally specify a default
    paramater using the '__default' key. This view hides that
    key, and allows the default parameter to be accessed.
    """

    __PRIVATE_KEYS = {"__default"}

    def __init__(self, __foldermap: typing.Mapping[Any, Any]) -> None:
        """Initialize the mapping.

        Args:
            __foldermap: The raw foldermap
        """
        self.__foldermap = __foldermap

    def get_default_key(self) -> str:
        """Get the default key for this level of the foldermap.
        Raises a ValueError if it does not have a default.
        """
        try:
            return self.__foldermap["__default"]
        except KeyError as exc:
            raise ValueError("No default available") from exc

    def __getitem__(self, __key: Union[str, Literal[ParamArg.DEFAULT]]) -> Any:
        """Gets the item with key. If key is ``ParamArg.DEFAULT``, return the
        item under the default parameter, or raise a ``ValueError`` if no
        default exists."""
        if __key in self.__PRIVATE_KEYS:
            raise KeyError(__key)

        if __key == ParamArg.DEFAULT:
            return self[self.get_default_key()]

        elem = self.__foldermap[__key]
        if isinstance(elem, Mapping):
            return FolderMapView(elem)

        return elem

    def __iter__(self) -> typing.Iterator[str]:
        return (key for key in self.__foldermap if key not in self.__PRIVATE_KEYS)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def __repr__(self) -> str:
        items_repr = ", ".join((f"{repr(k)}: {repr(v)}") for k, v in self.items())

        return f"{{{items_repr}}}"

    def __str__(self) -> str:
        items_str = ", ".join((f"{str(k)}: {str(v)}") for k, v in self.items())

        return f"{{{items_str}}}"
