# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains operations for the default phase space observables"""

from .expectation import Expectation


class Fock(Expectation):
    r"""Returns the photon-number expectation value in the phase space.

    Args:
        wires (int): the subsystem the operation acts on.
    """
    def __init__(self, wires):
        super().__init__('Fock', [], wires)


class X(Expectation):
    r"""Returns the position expectation value in the phase space.

    Args:
        wires (int): the subsystem the operation acts on.
    """
    def __init__(self, wires):
        super().__init__('X', [], wires)


class P(Expectation):
    r"""Returns the momentum expectation value in the phase space.

    Args:
        wires (int): the subsystem the operation acts on.
    """
    def __init__(self, wires):
        super().__init__('P', [], wires)


class Homodyne(Expectation):
    r"""Returns the homodyne expectation value in the phase space.

    Args:
        phi (float): axis in the phase space at which to calculate
            the homodyne measurement.
        wires (int): the subsystem the operation acts on.
    """
    def __init__(self, phi, wires):
        super().__init__('Homodyne', [phi], wires)


class Heterodyne(Expectation):
    r"""Returns the displacement expectation value in the phase space.

    Args:
        wires (int): the subsystem the operation acts on.
    """
    def __init__(self, phi, wires):
        super().__init__('Heterodyne', [], wires)
