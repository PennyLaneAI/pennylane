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
from typing import Optional, Tuple, Union, Sequence

from pennylane.interfaces import SUPPORTED_INTERFACES
from pennylane.gradients import SUPPORTED_GRADIENT_KWARGS
from pennylane._device import _process_shot_sequence, DeviceError, ShotTuple

SUPPORTED_GRADIENT_METHODS = [
    "best",
    "parameter-shift",
    "backprop",
    "finite-diff",
    "device",
    "adjoint",
]


@dataclass(frozen=True)
class ExecutionConfig:
    """
    A class to configure the execution of a quantum circuit on a device.

    See the Attributes section to learn more about the various configurable options.
    """

    # pylint: disable=too-many-instance-attributes
    
    shots: int = None
    """The number of shots for an execution."""

    _shots: int = field(init=False, repr=False)
    _shot_vector: list[ShotTuple] = field(init=False, repr=False)
    _raw_shot_sequence: list[int] = field(init=False, repr=False)
    """Private attributes for storing shot information"""

    gradient_method: Optional[str] = None
    """The method used to compute the gradient of the quantum circuit being executed"""

    gradient_keyword_arguments: dict = None
    """Arguments used to control a gradient transform"""

    device_options: dict = None
    """Various options for the device executing a quantum circuit"""

    interface: str = "autograd"
    """The machine learning framework to use"""

    derivative_order: int = 1
    """The derivative order to compute while evaluating a gradient"""

    def __post_init__(self):
        """
        Validate the configured execution options.

        Note that this hook is automatically called after init via the dataclass integration.
        """
        if self.interface not in SUPPORTED_INTERFACES:
            raise ValueError(
                f"interface must be in {SUPPORTED_INTERFACES}, got {self.interface} instead."
            )

        if (
            self.gradient_method is not None
            and self.gradient_method not in SUPPORTED_GRADIENT_METHODS
        ):
            raise ValueError(
                f"gradient_method must be in {SUPPORTED_GRADIENT_METHODS}, got {self.gradient_method} instead."
            )

        if self.device_options is None:
            self.device_options = {}

        if self.gradient_keyword_arguments is None:
            self.gradient_keyword_arguments = {}

        if any(arg not in SUPPORTED_GRADIENT_KWARGS for arg in self.gradient_keyword_arguments):
            raise ValueError(
                f"All gradient_keyword_arguments keys must be in {SUPPORTED_GRADIENT_KWARGS}, got unexpected values: {set(self.gradient_keyword_arguments) - set(SUPPORTED_GRADIENT_KWARGS)}"
            )

    @property
    def shots(self):
        """The number of shots for an execution"""
        return self._shots

    @shots.setter
    def shots(self, shots: Union[int, Tuple[int]]):
        """Changes the number of shots.

        Args:
            shots (Union[int, Tuple[int]]): number of circuit evaluations/random samples used to estimate
                expectation values of observables

        Raises:
            DeviceError: if number of shots is less than 1
        """
        if shots is None:
            # analytic mode
            self._shots = shots
            self._shot_vector = None
            self._raw_shot_sequence = None

        elif isinstance(shots, int):
            # sampling mode (unbatched)
            if shots < 1:
                raise DeviceError(
                    f"The specified number of shots needs to be at least 1. Got {shots}."
                )

            self._shots = shots
            self._shot_vector = None

        elif isinstance(shots, Sequence) and not isinstance(shots, str):
            # batched sampling mode
            self._shots, self._shot_vector = _process_shot_sequence(shots)
            self._raw_shot_sequence = shots

        else:
            raise DeviceError(
                "Shots must be a single non-negative integer or a sequence of non-negative integers."
            )

    @property
    def shot_vector(self):
        """list[.ShotTuple[int, int]]: Returns the shot vector, a sparse
        representation of the shot sequence"""
        return self._shot_vector
