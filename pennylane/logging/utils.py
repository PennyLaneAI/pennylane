# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file provides developer-facing functionality for the PennyLane logging module."""
import inspect
import sys

from .decorators import debug_logger


def _is_local_fn(f, mod_name):
    """
    Predicate that validates if argument ``f`` is a local function belonging to module ``mod_name``.
    """
    is_func = inspect.isfunction(f)
    is_local_to_mod = inspect.getmodule(f).__name__ == mod_name
    return is_func and is_local_to_mod


def _add_logging_all(mod_name):
    """
    Modifies the module ``mod_name`` to add logging implicitly to all free-functions.
    """
    l_func = inspect.getmembers(
        sys.modules[mod_name], predicate=lambda x: _is_local_fn(x, mod_name)
    )
    for f_name, f in l_func:
        globals()[f_name] = debug_logger(f)
