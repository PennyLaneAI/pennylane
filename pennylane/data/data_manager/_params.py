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
Contains utility functions for resolving dataset paths from parameter
arguments.
"""


from collections.abc import Iterable
from typing import Any, Dict, List, Tuple, Union

from .types import (
    DEFAULT,
    FULL,
    DataPath,
    Description,
    FolderMapView,
    ParamArg,
    ParamArgs,
    ParamName,
    ParamVal,
)


# pylint:disable=too-many-branches
def format_param_args(param: ParamName, details: Any) -> Union[ParamArg, List[ParamVal]]:
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


def format_params(**params: Any) -> ParamArgs:
    """Converts params is a dictionary whose keys are parameter names and
    whose values are single ``ParamaterArg`` objects or lists of parameter values."""
    return {
        param_name: format_param_args(param_name, param) for param_name, param in params.items()
    }


def resolve_params(
    data_struct: Dict, foldermap: FolderMapView, category: str, **params: Any
) -> List[Tuple[Description, DataPath]]:
    """
    Returns a 2-tuple of dataset parameters and paths, for each dataset that
    matches ``params``.
    """
    curr: List[Tuple[Tuple[ParamVal, ...], FolderMapView]] = [(tuple(), foldermap[category])]
    next_: List[Tuple[Tuple[ParamVal, ...], FolderMapView]] = []

    category_params = data_struct[category]["params"]
    param_args = format_params(**params)

    invalid_names = ",".join(
        f"'{param_name}'" for param_name in param_args if param_name not in category_params
    )
    if invalid_names:
        valid_names = ",".join(f"'{param_name}'" for param_name in category_params)
        raise ValueError(
            f"Invalid parameter(s) for '{category}': {invalid_names}. Valid parameters are: {valid_names}"
        )

    for param_name in data_struct[category]["params"]:
        while curr:
            resolved, dir_map = curr.pop()
            params = param_args.get(param_name, DEFAULT)
            if params == FULL:
                next_params = (param for param in dir_map)
            elif params == DEFAULT:
                try:
                    next_params = (dir_map.get_default_key(),)
                except ValueError as exc:
                    raise ValueError(
                        f"No default available for parameter '{param_name}' of '{category}'."
                    ) from exc
            else:
                next_params = (param for param in params)

            try:
                next_.extend(((*resolved, param), dir_map[param]) for param in next_params)
            except KeyError as exc:
                raise ValueError(
                    f"No '{category}' dataset exists with {param_name}={exc.args[0]}"
                ) from exc

        curr, next_ = next_, curr

    return [(dict(zip(category_params, resolved)), data_path) for resolved, data_path in curr]
