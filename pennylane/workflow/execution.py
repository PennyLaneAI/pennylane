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
Contains the :class:`ExecutionConfig` data class.

shots
gradient method/ function
gradient hyperparameters
device options
ml framework
"""
from dataclasses import dataclass, field
from typing import Callable, Union

from pennylane.interfaces import SUPPORTED_INTERFACES


@dataclass
class ExecutionConfig:
    """
    A configuration class to describe an execution of a quantum circuit on a device.
    """

    shots: int = 1000
    """The number of shots for an execution"""

    gradient_method: Union[None, Callable] = None
    """The method used to compute the gradient of the quantum circuit being executed"""

    gradient_hyperparameters: dict = field(default_factory=dict)
    """The non-trainable parameters that the execution depends on"""

    device_options: dict = field(default_factory=dict)
    """Various options for the device executing a quantum circuit"""

    framework: str = "jax"
    """The machine learning framework to use"""

    def __post_init__(self):
        if self.framework not in SUPPORTED_INTERFACES:
            raise ValueError(
                f"framework must be in {SUPPORTED_INTERFACES}, got {self.framework} instead."
            )
