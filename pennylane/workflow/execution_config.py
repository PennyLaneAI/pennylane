# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains that dataclass that specifies the hyperparameters of
an execution.

"""

from typing import Union, Callable
from dataclasses import dataclass


# pylint: disable=too-many-instance-attributes
@dataclass
class ExecutionConfig:
    """Configuration dataclass to support runtime execution of given workloads."""

    # important ones for now
    shots: int = 0
    interface: Union[None, str] = None
    diff_method: Union[None, str, Callable] = None
    order: int = 1

    cache_size: int = 10000  # Set to 0 to disable cache#
    max_expansion: int = 10
    max_diff: int = 1
    grad_args: Union[None, dict] = None
