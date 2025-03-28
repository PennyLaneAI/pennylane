# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Contains the Enum classes and helper functions for mid-circuit measurement
configuration options.
"""

from enum import Enum
from typing import Literal, Union


class MCM_METHOD(Enum):
    """Canonical set of mid-circuit measurement methods supported."""

    DEFERRED = "deferred"
    ONE_SHOT = "one-shot"
    TREE_TRAVERSAL = "tree-traversal"
    SINGLE_BRANCH_STATISTICS = "single-branch-statistics"

    def __eq__(self, mcm_method):
        if not isinstance(mcm_method, MCM_METHOD):
            return super().__eq__(get_canonical_mcm_method(mcm_method))
        return super().__eq__(mcm_method)


MCM_METHOD_MAP = {
    None: None,
    "deferred": MCM_METHOD.DEFERRED,
    "one-shot": MCM_METHOD.ONE_SHOT,
    "tree-traversal": MCM_METHOD.TREE_TRAVERSAL,
    "single-branch-statistics": MCM_METHOD.SINGLE_BRANCH_STATISTICS,
}
SupportedMCMMethodUserInput = Literal[tuple(MCM_METHOD_MAP.keys())]

SUPPORTED_MCM_METHODS = list(MCM_METHOD)


def get_canonical_mcm_method(user_input: Union[str, MCM_METHOD, None]) -> MCM_METHOD:
    """Helper function to convert user input to a canonical MCM_METHOD.

    Args:
        user_input (str, None): The user input to convert.
    Raises:
        ValueError: key does not exist in MCM_METHOD_MAP
    Returns:
        MCM_METHOD: The canonical MCM_METHOD.

    """
    if not user_input:
        return None

    if isinstance(user_input, MCM_METHOD):
        return user_input

    try:
        return MCM_METHOD_MAP[user_input]
    except KeyError as exc:
        raise ValueError(
            f"Invalid mcm method {user_input}, must be one of {SUPPORTED_MCM_METHODS}."
        ) from exc


class POSTSELECT_MODE(Enum):
    """Canonical set of postselection modes supported."""

    HW_LIKE = "hw-like"
    FILL_SHOTS = "fill-shots"
    PAD_INVALID_SAMPLES = "pad-invalid-samples"

    def __eq__(self, postselect_mode):
        if not isinstance(postselect_mode, MCM_METHOD):
            return super().__eq__(get_canonical_postselect_mode(postselect_mode))
        return super().__eq__(postselect_mode)


POSTSELECT_MODE_MAP = {
    "hw-like": POSTSELECT_MODE.HW_LIKE,
    "fill-shots": POSTSELECT_MODE.FILL_SHOTS,
    "pad-invalid-samples": POSTSELECT_MODE.PAD_INVALID_SAMPLES,
}
SupportedPostSelectModeUserInput = Literal[tuple(POSTSELECT_MODE_MAP.keys())]

SUPPORTED_POSTSELECT_MODES = list(POSTSELECT_MODE)


def get_canonical_postselect_mode(user_input: Union[str, POSTSELECT_MODE, None]) -> POSTSELECT_MODE:
    """Helper function to convert user input to a canonical POSTSELECT_MODE.

    Args:
        user_input (str, None): The user input to convert.
    Raises:
        ValueError: key does not exist in POSTSELECT_MODE_MAP
    Returns:
        POSTSELECT_MODE: The canonical POSTSELECT_MODE.
    """
    if not user_input:
        return None

    if isinstance(user_input, POSTSELECT_MODE):
        return user_input

    try:
        return POSTSELECT_MODE_MAP[user_input]
    except KeyError as exc:
        raise ValueError(
            f"Invalid postselection mode {user_input}, must be one of {SUPPORTED_POSTSELECT_MODES}."
        ) from exc
