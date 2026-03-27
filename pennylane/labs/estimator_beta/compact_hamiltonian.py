# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains classes used to compactly store the metadata of various Hamiltonians which are relevant for resource estimation.
"""
import copy
from dataclasses import dataclass
from typing import Iterable


def _validate_positive_int(name, value):
    """Helper to validate positive integers."""
    if not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer, got {value}")


@dataclass(frozen=True)
class FirstQuantizedHamiltonian:
    """For a first quantized Hamiltonian, stores the minimum necessary information pertaining to resource estimation.

    The form of this Hamiltonian is described in ``_.

    Args:
        num_plane_waves (int): number of plane waves
        num_electrons (int): number of electrons
        omega (float): unit-cell volume
        charge (int): total charge of the system

    Raises:
        TypeError: if ``num_plane_waves``, ``num_electrons`` is not a positive integer
        TypeError: if ``omega`` is not a positive float
        TypeError: if ``charge`` is not an integer

    .. seealso::
        :class:`~.estimator.templates.prep_op.PrepFirstQuantized`
        :class:`~.estimator.templates.select_op.SelectFirstQuantized`
    """

    num_plane_waves: int
    num_electrons: int
    omega: float
    charge: int

    def __post_init__(self):
        """Checks the types of the inputs."""

        _validate_positive_int("num_plane_waves", self.num_plane_waves)
        _validate_positive_int("num_electrons", self.num_electrons)
        if not isinstance(self.omega, (float, int)) or self.omega <= 0:
            raise TypeError(f"omega must be a positive float, got {self.omega}")
        if not isinstance(self.charge, int):
            raise TypeError(f"charge must be an integer, got {self.charge}")

