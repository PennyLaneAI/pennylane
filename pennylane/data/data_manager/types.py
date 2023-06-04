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
from typing import Any, Dict, FrozenSet, List, Literal, Tuple, TypedDict, Union


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
        return self.value


DEFAULT = ParamArg.DEFAULT
FULL = ParamArg.FULL

# Type for the name of a parameter, e.g 'molname', 'bondlength'
ParamName = str
# Type for a concrete paramter value, e.g 'H2', '0.5'
ParamVal = str
# TODO: needed?
ParamArgs = Dict[ParamName, Union[ParamArg, typing.Iterable[ParamVal]]]
# Description is a dictionary that contains all the parameter values
# for a dataset
Description = Dict[ParamName, ParamVal]
# Type for a dataset path, relative to the foldermap.json file
DataPath = PurePosixPath


class DataStuct(TypedDict):
    params: List[str]
    attributes: List[str]


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
            "qchem": {
                "__data_struct": {
                    "params": ["molname", "basis", "bondlength"],
                    "attributes": [
                        "attributes":
                        "molecule",
                        "hamiltonian",
                        "sparse_hamiltonian",
                        "hf_state",
                        "meas_groupings",
                        "fci_energy",
                        "fci_spectrum",
                        "dipole_op",
                        ...
                        "vqe_params",
                        "vqe_energy"
                    ]
                }
                "O2": {
                    "__default": "STO-3G",
                    "STO-3G": {
                        "__default": "0.5",
                        "0.5": "qchem/O2/STO-3G/0.5.h5",
                        "0.6": "qchem/O2/STO-3G/0.6.h5"
                    }
                },
                "H2": {
                    "__default": "STO-3G",
                    "STO-3G": {
                        "__default": "0.7",
                        "0.7": "qchem/H2/STO-3G/0.7.h5"
                    }
                }
            },
        }
    """

    __PRIVATE_KEYS = {"__default", "__data_struct"}

    def __init__(self, __foldermap: typing.Mapping[str, Any]) -> None:
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

    def find(
        self, category: str, **params: Union[typing.Iterable[ParamVal], ParamArg]
    ) -> List[Tuple[Description, DataPath]]:
        """Returns a 2-tuple of dataset description and paths, for each dataset that
        matches ``params``."""

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
