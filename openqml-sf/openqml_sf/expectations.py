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
"""This module contains the expectation operations"""
from collections import Sequence


def PNR(state, wires, params):
    """Function that returns the qm.Fock observable in Strawberry Fields.

    Args:
        state (strawberryfields.backends.states.BaseState): the quantum state
        wires (int): the measured mode
        params (Sequence): sequence of parameters (not used).

    Returns:
        tuple(float, float): Mean photon number and variance.
    """
    # pylint: disable=unused-argument
    if isinstance(wires, Sequence):
        return state.mean_photon(wires[0])

    return state.mean_photon(wires)


def Homodyne(phi=None):
    """Function factory that returns the qm.Homodyne observable function for Strawberry Fields.

    ``Homodyne(phi)`` returns a function

    .. code-block:: python

        homodyne_expectation(state, wires, phi)

    that is used to determine the Homodyne expectation value of a wire within a SF state,
    measured along a particular phase space angle ``phi``.

    Note that:

    * If ``phi`` is not None, the returned function will be hardcoded to return the
      homodyne expectation value at angle ``phi`` in the phase space.
    * If ``phi`` the value of ``phi`` must be set when calling the returned function.

    Args:
        phi (float): the default phase space axis to perform the Homodyne measurement.

    Returns:
        function: A function that accepts a SF state, the wire to measure,
            and phase space angle phi, and returns the quadrature expectation
            value and variance.
    """
    if phi is not None:
        return lambda state, wires, params: state.quad_expectation(wires, phi)

    return lambda state, wires, params: state.quad_expectation(wires, *params)
