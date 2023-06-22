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
Contains ``FolderMapView`` for reading the ``foldermap.json`` file in the
datasets bucket.
"""


import typing
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, List, Literal, Optional, Tuple, Union

from .params import Description, ParamArg, ParamName, ParamVal

# Type for a dataset path, relative to the foldermap.json file
DataPath = PurePosixPath


class FolderMapView(typing.Mapping[str, Union["FolderMapView", DataPath]]):
    """Provides a read-only view of the ``foldermap.json`` file in
    the datasets bucket. The folder map is a nested mapping of
    dataset parameters to their path, relative to the ``foldermap.json``
    file.

    A dictionary in the folder map can optionally specify a default
    paramater using the '__default' key. This view hides that
    key, and allows the default parameter to be accessed.

    For example:

        {
            "__name": "data_name",
            "qchem": {
                "__name": "molname",
                "O2": {
                    "__name": "basis",
                    "__default": "STO-3G",
                    "STO-3G": {
                        "__name": "bondlength",
                        "__default": "0.5",
                        "0.5": "qchem/O2/STO-3G/0.5.h5",
                        "0.6": "qchem/O2/STO-3G/0.6.h5"
                    }
                },
                "H2": {
                    "__name": "molname",
                    "__default": "STO-3G",
                    "STO-3G": {
                        "__name": "bondlength",
                        "__default": "0.7",
                        "0.7": "qchem/H2/STO-3G/0.7.h5"
                    }
                }
            },
        }
    """

    __PRIVATE_KEYS = {"__default", "__name"}

    def __init__(self, __curr_level: typing.Mapping[str, Any]) -> None:
        """Initialize the mapping.

        Args:
            __data_struct: The top level foldermap
        """
        self.__curr_level = __curr_level

    @property
    def name(self) -> str:
        """Returns the the name of the current parameter, e.g 'data_name', 'molname'."""
        return self.__curr_level["__name"]

    def get_default_key(self) -> str:
        """Get the default key for this level of the foldermap.
        Raises a ValueError if it does not have a default.
        """
        try:
            return self.__curr_level["__default"]
        except KeyError as exc:
            raise ValueError(f"No default available for parameter '{self.name}'") from exc

    def find(
        self,
        data_name: str,
        missing_default: Optional[ParamArg] = ParamArg.DEFAULT,
        **params: Union[typing.Iterable[ParamVal], ParamArg],
    ) -> List[Tuple[Description, DataPath]]:
        """Returns a 2-tuple of dataset description and paths, for each dataset that
        matches ``params``."""
        if self.name != "data_name":
            raise RuntimeError("Can only call 'find()' from top level.")

        try:
            data_name_map = self[data_name]
        except KeyError as exc:
            raise ValueError(f"No such data_name: '{data_name}'") from exc

        todo: List[
            Tuple[Union[FolderMapView, DataPath], Tuple[Tuple[ParamName, ParamVal], ...]]
        ] = []
        done: List[Tuple[Description, DataPath]] = []

        todo.append((data_name_map, tuple()))

        while todo:
            curr_level, params_acc = todo.pop()
            if isinstance(curr_level, DataPath):
                done.append((Description(params_acc), curr_level))
                continue

            param_arg = params.get(curr_level.name, missing_default)
            if param_arg is None:
                raise ValueError(f"Missing argument for parameter '{curr_level.name}'")

            if param_arg == ParamArg.FULL:
                next_params = curr_level
            elif param_arg == ParamArg.DEFAULT:
                next_params = (curr_level.get_default_key(),)
            elif isinstance(param_arg, str):
                next_params = (param_arg,)
            else:
                next_params = param_arg

            todo.extend(
                (curr_level[next_param], (*params_acc, (curr_level.name, next_param)))
                for next_param in next_params
            )

        return done

    def __getitem__(
        self, __key: Union[str, Literal[ParamArg.DEFAULT]]
    ) -> Union["FolderMapView", DataPath]:
        """Gets the item with key. If key is ``ParamArg.DEFAULT``, return the
        item under the default parameter, or raise a ``ValueError`` if no
        default exists."""
        if __key in self.__PRIVATE_KEYS:
            raise KeyError(__key)

        if __key == ParamArg.DEFAULT:
            return self[self.get_default_key()]

        elem = self.__curr_level[__key]
        if isinstance(elem, Mapping):
            return FolderMapView(elem)

        return DataPath(elem)

    def __iter__(self) -> typing.Iterator[str]:
        return (key for key in self.__curr_level if key not in self.__PRIVATE_KEYS)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def __repr__(self) -> str:
        items_repr = ", ".join((f"{repr(k)}: {repr(v)}") for k, v in self.items())

        return f"{{{items_repr}}}"

    def __str__(self) -> str:
        items_str = ", ".join((f"{str(k)}: {str(v)}") for k, v in self.items())

        return f"{{{items_str}}}"
