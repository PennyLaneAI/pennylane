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
from dataclasses import dataclass, field
from typing import Optional, Union

from pennylane.transforms.core import TransformDispatcher


# pylint: disable=too-many-instance-attributes
@dataclass
class ExecutionConfig:
    """
    A class to configure the execution of a quantum circuit on a device.

    See the Attributes section to learn more about the various configurable options.
    """

    grad_on_execution: Optional[bool] = None
    """Whether or not to compute the gradient at the same time as the execution.

    If ``None``, then the device or execution pipeline can decide which one is most efficient for the situation.
    """

    use_device_gradient: Optional[bool] = None
    """Whether or not to compute the gradient on the device.

    ``None`` indicates to use the device if possible, but to fall back to pennylane behaviour if it isn't.

    True indicates a request to either use the device gradient or fail.
    """

    use_device_jacobian_product: Optional[bool] = None
    """Whether or not to use the device provided vjp or jvp to compute gradients.

    ``None`` indicates to use the device if possible, but to fall back to the device Jacobian
    or PennyLane behaviour if it isn't.

    ``True`` indicates to either use the device Jacobian products or fail.
    """

    gradient_method: Optional[Union[str, TransformDispatcher]] = "best"
    """The method used to compute the gradient of the quantum circuit being executed"""

    gradient_keyword_arguments: dict = field(default_factory=dict)
    """Arguments used to control a gradient transform"""

    device_options: dict = field(default_factory=dict)
    """Various options for the device executing a quantum circuit"""

    interface: Optional[str] = "auto"
    """The machine learning framework to use"""

    derivative_order: int = 1
    """The derivative order to compute while evaluating a gradient"""


DefaultExecutionConfig = ExecutionConfig()
