# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Public/internal API for the AutoGraph module.
"""

import functools

from pennylane.exceptions import AutoGraphWarning
from .transformer import autograph_source, run_autograph, disable_autograph

AUTOGRAPH_WRAPPER_ASSIGNMENTS = tuple(
    attr for attr in functools.WRAPPER_ASSIGNMENTS if attr != "__module__"
)


def wraps(target):
    """Wrap another function using functools.wraps. For use with AutoGraph, the __module__ attribute
    should be preserved in order for the AutoGraph conversion allow/block listing to work properly.
    """
    return functools.wraps(target, assigned=AUTOGRAPH_WRAPPER_ASSIGNMENTS)


__all__ = (
    "autograph_source",
    "run_autograph",
    "disable_autograph",
    "wraps",
)
