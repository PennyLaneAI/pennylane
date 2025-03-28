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
from enum import Enum
from typing import Literal, Optional, Union

from pennylane.math import Interface, get_canonical_interface_name
from pennylane.transforms.core import TransformDispatcher


class MCM_METHOD(Enum):
    """Canonical set of mid-circuit measurement methods supported."""

    AUTO = None
    DEFERRED = "deferred"
    ONE_SHOT = "one-shot"
    TREE_TRAVERSAL = "tree-traversal"
    SINGLE_BRANCH_STATISTICS = "single-branch-statistics"

    def __eq__(self, mcm_method):
        if isinstance(mcm_method, str):
            raise TypeError("Cannot compare MCMMethod Enum with str")
        return super().__eq__(mcm_method)


MCM_METHOD_MAP = {
    None: MCM_METHOD.AUTO,
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
    if isinstance(user_input, MCM_METHOD) and user_input in SUPPORTED_MCM_METHODS:
        return user_input

    try:
        return MCM_METHOD_MAP[user_input]
    except KeyError as exc:
        raise ValueError(
            f"Unknown mcm method {user_input}, must be one of {SUPPORTED_MCM_METHODS}."
        ) from exc


class POSTSELECT_MODE(Enum):
    """Canonical set of postselection modes supported."""

    AUTO = None
    HW_LIKE = "hw-like"
    FILL_SHOTS = "fill-shots"

    def __eq__(self, mcm_method):
        if isinstance(mcm_method, str):
            raise TypeError("Cannot compare POSTSELECT_MODE Enum with str")
        return super().__eq__(mcm_method)


POSTSELECT_MODE_MAP = {
    None: POSTSELECT_MODE.AUTO,
    "hw-like": POSTSELECT_MODE.HW_LIKE,
    "fill-shots": POSTSELECT_MODE.FILL_SHOTS,
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
    if isinstance(user_input, POSTSELECT_MODE) and user_input in SUPPORTED_POSTSELECT_MODES:
        return user_input

    try:
        return POSTSELECT_MODE_MAP[user_input]
    except KeyError as exc:
        raise ValueError(
            f"Unknown post select mode {user_input}, must be one of {SUPPORTED_POSTSELECT_MODES}."
        ) from exc


@dataclass
class MCMConfig:
    """A class to store mid-circuit measurement configurations."""

    mcm_method: MCM_METHOD = MCM_METHOD.AUTO
    """The mid-circuit measurement strategy to use. Use ``"deferred"`` for the deferred
    measurements principle and ``"one-shot"`` if using finite shots to execute the circuit for
    each shot separately. Any other value will be passed to the device, and the device is
    expected to handle mid-circuit measurements using the requested method. If not specified,
    the device will decide which method to use."""

    postselect_mode: POSTSELECT_MODE = POSTSELECT_MODE.AUTO
    """How postselection is handled with finite-shots. If ``"hw-like"``, invalid shots will be
    discarded and only results for valid shots will be returned. In this case, fewer samples
    may be returned than the original number of shots. If ``"fill-shots"``, the returned samples
    will be of the same size as the original number of shots. If not specified, the device will
    decide which mode to use. Note that internally ``"pad-invalid-samples"`` is used internally
    instead of ``"hw-like"`` when using jax/catalyst"""

    def __post_init__(self):
        """Validate the configured mid-circuit measurement options."""
        self.mcm_method = get_canonical_mcm_method(self.mcm_method)
        self.postselect_mode = get_canonical_postselect_mode(self.postselect_mode)


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
