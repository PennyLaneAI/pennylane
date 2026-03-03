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
Contains types and functions for dataset parameters.
"""

import enum
from collections.abc import Iterable, Iterator, Mapping
from functools import lru_cache
from typing import Any, Union


class ParamArg(enum.Enum):
    """Enum representing special args to ``load()``.

    FULL: used to request all attributes
    DEFAULT: used to request the default attribute
    """

    FULL = "full"
    DEFAULT = "default"

    @classmethod
    @lru_cache(maxsize=1)
    def values(cls) -> frozenset[str]:
        """Returns all values."""
        return frozenset(arg.value for arg in cls)

    @classmethod
    def is_arg(cls, val: Union["ParamArg", str]) -> bool:
        """Returns true if ``val`` is a ``ParamArg``, or one
        of its values."""
        return isinstance(val, ParamArg) or (isinstance(val, str) and val in cls.values())

    def __str__(self) -> str:
        return self.value


DEFAULT = ParamArg.DEFAULT
FULL = ParamArg.FULL

# Type for the name of a parameter, e.g 'molname', 'bondlength'
ParamName = str
# Type for a concrete paramter value, e.g 'H2', '0.5'
ParamVal = str


class Description(Mapping[ParamName, ParamVal]):
    """An immutable and hashable dictionary that contains all the parameter
    values for a dataset."""

    def __init__(self, params: Iterable[tuple[ParamName, ParamVal]]):
        self.__data = dict(params)
        self.__hash = None

    def __getitem__(self, __key: ParamName) -> ParamVal:
        return self.__data[__key]

    def __iter__(self) -> Iterator[ParamName]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __hash__(self) -> int:
        if not self.__hash:
            self.__hash = hash(tuple(self.__data))

        return self.__hash

    def __str__(self) -> str:
        return str(self.__data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.__data)})"


# pylint:disable=too-many-branches
def format_param_args(param: ParamName, details: Any) -> ParamArg | list[ParamVal]:
    """Ensures each user-inputted parameter is a properly typed list.
    Also provides custom support for certain parameters."""
    if not isinstance(details, list):
        details = [details]

    for detail in details:
        if ParamArg.is_arg(detail):
            return ParamArg(detail)

    if param == "layout":
        # if a user inputs layout=[1,2], they wanted "1x2"
        # note that the above conversion to a list of details wouldn't work as expected here
        if all(isinstance(dim, int) for dim in details):
            return ["x".join(map(str, details))]
        # will turn [(1,2), [3,4], "5x6"] into ["1x2", "3x4", "5x6"]
        for i, detail in enumerate(details):
            if isinstance(detail, Iterable) and all(isinstance(dim, int) for dim in detail):
                details[i] = "x".join(map(str, detail))
            elif not isinstance(detail, str):
                raise TypeError(
                    f"Invalid layout value of '{detail}'. Must be a string or a tuple of ints."
                )
    elif param == "bondlength":
        for i, detail in enumerate(details):
            if isinstance(detail, float):
                details[i] = str(detail)
            elif isinstance(detail, int):
                details[i] = f"{detail:.1f}"
            elif not isinstance(detail, str):
                raise TypeError(f"Invalid bondlength '{detail}'. Must be a string, int or float.")

    for detail in details:
        if not isinstance(detail, str):
            raise TypeError(f"Invalid type '{type(detail).__name__}' for parameter '{param}'")

    return details


def format_params(**params: Any) -> list[dict[str:ParamName, str : ParamArg | ParamVal]]:
    """Converts params to a list of dictionaries whose values are parameter names and
    single ``ParamaterArg`` objects or lists of parameter values."""

    input_params = {
        param_name: format_param_args(param_name, param) for param_name, param in params.items()
    }
    return [{"name": k, "values": v} for k, v in input_params.items()]


def provide_defaults(
    data_name: str, params: list[dict[str:ParamName, str : ParamArg | ParamVal]]
) -> list[dict[str:ParamName, str : ParamArg | ParamVal]]:
    """
    Provides default parameters to the qchem and qspin query parameters if the parameter
    names are missing from the provided ``params``.
    """
    param_names = [param["name"] for param in params]
    if data_name == "qchem":
        if "basis" not in param_names:
            params.append({"default": True, "name": "basis"})
        if "bondlength" not in param_names:
            params.append({"default": True, "name": "bondlength"})

    if data_name == "qspin":
        if "periodicity" not in param_names:
            params.append({"default": True, "name": "periodicity"})
        if "lattice" not in param_names:
            params.append({"default": True, "name": "lattice"})
        if "layout" not in param_names:
            params.append({"default": True, "name": "layout"})

    return params
