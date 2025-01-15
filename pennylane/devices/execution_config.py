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

from pennylane.math import Interface, get_canonical_interface_name
from pennylane.transforms.core import TransformDispatcher


@dataclass
class MCMConfig:
    """A class to store mid-circuit measurement configurations."""

    mcm_method: Optional[str] = None
    """The mid-circuit measurement strategy to use. Use ``"deferred"`` for the deferred
    measurements principle and ``"one-shot"`` if using finite shots to execute the circuit for
    each shot separately. Any other value will be passed to the device, and the device is
    expected to handle mid-circuit measurements using the requested method. If not specified,
    the device will decide which method to use."""

    postselect_mode: Optional[str] = None
    """How postselection is handled with finite-shots. If ``"hw-like"``, invalid shots will be
    discarded and only results for valid shots will be returned. In this case, fewer samples
    may be returned than the original number of shots. If ``"fill-shots"``, the returned samples
    will be of the same size as the original number of shots. If not specified, the device will
    decide which mode to use. Note that internally ``"pad-invalid-samples"`` is used internally
    instead of ``"hw-like"`` when using jax/catalyst"""

    def __post_init__(self):
        """Validate the configured mid-circuit measurement options."""
        if self.postselect_mode not in ("hw-like", "fill-shots", "pad-invalid-samples", None):
            raise ValueError(f"Invalid postselection mode '{self.postselect_mode}'.")


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

    gradient_method: Optional[Union[str, TransformDispatcher]] = None
    """The method used to compute the gradient of the quantum circuit being executed"""

    gradient_keyword_arguments: Optional[dict] = None
    """Arguments used to control a gradient transform"""

    device_options: Optional[dict] = None
    """Various options for the device executing a quantum circuit"""

    interface: Interface = Interface.NUMPY
    """The machine learning framework to use"""

    derivative_order: int = 1
    """The derivative order to compute while evaluating a gradient"""

    mcm_config: MCMConfig = field(default_factory=MCMConfig)
    """Configuration options for handling mid-circuit measurements"""

    convert_to_numpy: bool = True
    """Whether or not to convert parameters to numpy before execution.

    If ``False`` and using the jax-jit, no pure callback will occur and the device
    execution itself will be jitted.
    """

    def __post_init__(self):
        """
        Validate the configured execution options.

        Note that this hook is automatically called after init via the dataclass integration.
        """
        self.interface = get_canonical_interface_name(self.interface)

        if self.grad_on_execution not in {True, False, None}:
            raise ValueError(
                f"grad_on_execution must be True, False, or None. Got {self.grad_on_execution} instead."
            )

        if self.device_options is None:
            self.device_options = {}

        if self.gradient_keyword_arguments is None:
            self.gradient_keyword_arguments = {}

        if not (
            isinstance(self.gradient_method, (str, TransformDispatcher))
            or self.gradient_method is None
        ):
            raise ValueError(
                f"Differentiation method {self.gradient_method} must be a str, TransformDispatcher, or None. Got {type(self.gradient_method)} instead."
            )

        if isinstance(self.mcm_config, dict):
            self.mcm_config = MCMConfig(**self.mcm_config)  # pylint: disable=not-a-mapping

        elif not isinstance(self.mcm_config, MCMConfig):
            raise ValueError(f"Got invalid type {type(self.mcm_config)} for 'mcm_config'")


DefaultExecutionConfig = ExecutionConfig()
