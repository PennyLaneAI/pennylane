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
"""
from dataclasses import dataclass  # , field
from typing import Optional, Tuple, Union

from pennylane.interfaces import SUPPORTED_INTERFACES
from pennylane.gradients import SUPPORTED_GRADIENT_KWARGS

SUPPORTED_GRADIENT_METHODS = [
    "best",
    "parameter-shift",
    "backprop",
    "finite-diff",
    "device",
    "adjoint",
]


@dataclass
class ExecutionConfig:
    """
    A configuration class to describe an execution of a quantum circuit on a device.

    Args:
        shots (Optional[Union[int, Tuple[int]]]): The number of shots for an execution
        gradient_method (Optional[str]): The method used to compute the gradient of the quantum circuit being executed
        gradient_keyword_arguments (dict): Arguments used to control a gradient transform
        device_options (dict): Various options for the device executing a quantum circuit
        framework (str): The machine learning framework to use
        derivative_order (int): The derivative order to compute while evaluating a gradient
    """

    shots: Optional[Union[int, Tuple[int]]] = None
    gradient_method: Optional[str] = None
    gradient_keyword_arguments: dict = None  # field(default_factory=dict)
    device_options: dict = None  # field(default_factory=dict)
    framework: str = "jax"
    derivative_order: int = 1

    def __post_init__(self):
        """
        Validate the configured execution options.

        Note that this hook is automatically called after init via the dataclass integration.
        """
        if self.framework not in SUPPORTED_INTERFACES:
            raise ValueError(
                f"framework must be in {SUPPORTED_INTERFACES}, got {self.framework} instead."
            )

        if (
            self.gradient_method is not None
            and self.gradient_method not in SUPPORTED_GRADIENT_METHODS
        ):
            raise ValueError(
                f"gradient_method must be in {SUPPORTED_GRADIENT_METHODS}, got {self.gradient_method} instead."
            )

        if any(arg not in SUPPORTED_GRADIENT_KWARGS for arg in self.gradient_keyword_arguments):
            raise ValueError(
                f"All gradient_keyword_arguments keys must be in {SUPPORTED_GRADIENT_KWARGS}, got unexpected values: {set(self.gradient_keyword_arguments) - set(SUPPORTED_GRADIENT_KWARGS)}"
            )
