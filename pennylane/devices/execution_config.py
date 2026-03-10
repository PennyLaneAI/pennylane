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
Contains the :class:`ExecutionConfig` and :class:`MCMConfig` data classes.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from pennylane.concurrency.executors.backends import ExecBackends, get_executor
from pennylane.math.interface_utils import Interface
from pennylane.transforms.core import Transform

if TYPE_CHECKING:
    from pennylane.concurrency.executors.base import RemoteExec


class FrozenMapping(MutableMapping):
    """
    Custom immutable mapping.
    Inherit from MutableMapping to ensure all mutable methods are implemented.
    """

    def __init__(self, *args, **kwargs):
        self._data = dict(*args, **kwargs)
        self._hash = None  # Cache the hash value

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __setitem__(self, key, value):
        raise TypeError(
            "FrozenMapping is immutable. To update this field please use `dataclasses.replace`. "
        )

    def __delitem__(self, key):
        raise TypeError(
            "FrozenMapping is immutable. To update this field please use `dataclasses.replace`. "
        )

    def __repr__(self):
        return f"{self._data}"

    def __hash__(self):
        """Makes the object hashable, allowing it to be used in sets and as a dict key."""
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def copy(self):
        """Returns a standard, mutable shallow copy of the data."""
        return self._data.copy()

    def __copy__(self):
        """Supports copy.copy() by returning a mutable dict."""
        return self.copy()

    def __deepcopy__(self, memo=None):
        """Supports copy.deepcopy() by returning a mutable dict with deep-copied contents."""
        return deepcopy(self._data, memo)


class MCM_METHOD(StrEnum):
    """Canonical set up supported mid-circuit measurement methods."""

    DEFERRED = "deferred"
    ONE_SHOT = "one-shot"
    TREE_TRAVERSAL = "tree-traversal"
    SINGLE_BRANCH_STATISTICS = "single-branch-statistics"
    DEVICE = "device"


class POSTSELECT_MODE(StrEnum):
    """Canonical set up supported postselect modes."""

    HW_LIKE = "hw-like"
    FILL_SHOTS = "fill-shots"
    PAD_INVALID_SAMPLES = "pad-invalid-samples"
    DEVICE = "device"


@dataclass(frozen=True)
class MCMConfig:
    """A class to store mid-circuit measurement configurations."""

    mcm_method: MCM_METHOD | str | None = None
    """The mid-circuit measurement strategy to use. Use ``"deferred"`` for the deferred
    measurements principle and ``"one-shot"`` if using finite shots to execute the circuit for
    each shot separately. Any other value will be passed to the device, and the device is
    expected to handle mid-circuit measurements using the requested method. If not specified,
    the device will decide which method to use."""

    postselect_mode: POSTSELECT_MODE | str | None = None
    """How postselection is handled with finite-shots. If ``"hw-like"``, invalid shots will be
    discarded and only results for valid shots will be returned. In this case, fewer samples
    may be returned than the original number of shots. If ``"fill-shots"``, the returned samples
    will be of the same size as the original number of shots. If not specified, the device will
    decide which mode to use. Note that internally ``"pad-invalid-samples"`` is used internally
    instead of ``"hw-like"`` when using jax/catalyst."""

    def _validate_inputs(self, mcm_method, postselect_mode) -> None:
        """Validate inputs to MCMConfig.

        Args:
            mcm_method (MCM_METHOD | str | None): Mid-circuit measurement method.
            postselect_mode (POSTSELECT_MODE | str | None): Postselection mode.

        """
        _valid_mcm_methods: list[str | None] = [None] + [item.value for item in MCM_METHOD]
        _valid_postselection_modes: list[str | None] = [None] + [
            item.value for item in POSTSELECT_MODE
        ]

        if mcm_method not in _valid_mcm_methods:
            raise ValueError(
                f"'{mcm_method}' is not a valid mcm_method. Please use one of the supported mid-circuit measurement methods available: {_valid_mcm_methods}"
            )

        if postselect_mode not in _valid_postselection_modes:
            raise ValueError(
                f"'{postselect_mode}' is not a valid postselect_mode. Please use one of the supported postselection modes available: {_valid_postselection_modes}"
            )

    def __post_init__(self):
        """Validate the configured mid-circuit measurement options."""
        self._validate_inputs(self.mcm_method, self.postselect_mode)

        object.__setattr__(
            self,
            "mcm_method",
            self.mcm_method if self.mcm_method is None else MCM_METHOD(self.mcm_method),
        )
        object.__setattr__(
            self,
            "postselect_mode",
            (
                self.postselect_mode
                if self.postselect_mode is None
                else POSTSELECT_MODE(self.postselect_mode)
            ),
        )

    def __repr__(self):
        """Custom __repr__ for displaying the MCMConfig."""
        mcm_method: str | None = (
            self.mcm_method
            if self.mcm_method is None or isinstance(self.mcm_method, str)
            else self.mcm_method.value
        )
        postselect_mode: str | None = (
            self.postselect_mode
            if self.postselect_mode is None or isinstance(self.postselect_mode, str)
            else self.postselect_mode.value
        )
        if mcm_method:
            mcm_method = "'" + mcm_method + "'"
        if postselect_mode:
            postselect_mode = "'" + postselect_mode + "'"
        return f"MCMConfig(mcm_method={mcm_method}, postselect_mode={postselect_mode})"


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class ExecutionConfig:
    """
    A class to configure the execution of a quantum circuit on a device.

    See the Attributes section to learn more about the various configurable options.
    """

    grad_on_execution: bool | None = None
    """Whether or not to compute the gradient at the same time as the execution.

    If ``None``, then the device or execution pipeline can decide which one is most efficient for the situation.
    """

    use_device_gradient: bool | None = None
    """Whether or not to compute the gradient on the device.

    ``None`` indicates to use the device if possible, but to fall back to pennylane behaviour if it isn't.

    True indicates a request to either use the device gradient or fail.
    """

    use_device_jacobian_product: bool | None = None
    """Whether or not to use the device provided vjp or jvp to compute gradients.

    ``None`` indicates to use the device if possible, but to fall back to the device Jacobian
    or PennyLane behaviour if it isn't.

    ``True`` indicates to either use the device Jacobian products or fail.
    """

    gradient_method: str | Transform | None = None
    """The method used to compute the gradient of the quantum circuit being executed"""

    gradient_keyword_arguments: dict = field(default_factory=FrozenMapping)
    """Arguments used to control a gradient transform"""

    device_options: dict = field(default_factory=FrozenMapping)
    """Various options for the device executing a quantum circuit"""

    interface: str | Interface | None = Interface.NUMPY
    """The machine learning framework to use"""

    derivative_order: int = 1
    """The derivative order to compute while evaluating a gradient"""

    mcm_config: MCMConfig | dict = field(default_factory=MCMConfig)
    """Configuration options for handling mid-circuit measurements"""

    convert_to_numpy: bool = True
    """Whether or not to convert parameters to numpy before execution.

    If ``False`` and using the jax-jit, no pure callback will occur and the device
    execution itself will be jitted.
    """

    executor_backend: RemoteExec | None = None
    """
    Defines the class for the executor backend.
    """

    def __post_init__(self):
        """
        Validate the configured execution options.

        Note that this hook is automatically called after init via the dataclass integration.
        """
        object.__setattr__(self, "interface", Interface(self.interface))

        if self.grad_on_execution not in {True, False, None}:
            raise ValueError(
                f"grad_on_execution must be True, False, or None. Got {self.grad_on_execution} instead."
            )

        def _validate_and_freeze_dict(field_name: str):
            value = getattr(self, field_name)
            if not isinstance(value, (dict, FrozenMapping)):
                raise TypeError(f"Got invalid type {type(value)} for '{field_name}'")
            # This handles the case when `dataclasses.replace` is used and
            # the field is not being modified.
            if isinstance(value, dict):
                object.__setattr__(self, field_name, FrozenMapping(value))

        _validate_and_freeze_dict("device_options")
        _validate_and_freeze_dict("gradient_keyword_arguments")

        if not (isinstance(self.gradient_method, (str, Transform)) or self.gradient_method is None):
            raise ValueError(
                f"Differentiation method {self.gradient_method} must be a str, Transform, or None. Got {type(self.gradient_method)} instead."
            )

        if isinstance(self.mcm_config, dict):
            object.__setattr__(self, "mcm_config", MCMConfig(**self.mcm_config))
        elif not isinstance(self.mcm_config, MCMConfig):
            raise ValueError(f"Got invalid type {type(self.mcm_config)} for 'mcm_config'")

        if self.executor_backend is None:
            object.__setattr__(self, "executor_backend", get_executor(backend=ExecBackends.MP_Pool))
