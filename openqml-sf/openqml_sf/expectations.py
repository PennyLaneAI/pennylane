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
"""This module contains the expectation value operations."""
from collections import Sequence

import numpy as np

import openqml.expectation


def PNR(state, wires, params):
    """Computes the EV of the qm.Fock observable in Strawberry Fields.

    Args:
        state (strawberryfields.backends.states.BaseState): the quantum state
        wires (Sequence[int]): the measured mode
        params (Sequence): sequence of parameters (not used).

    Returns:
        float, float: mean photon number and its variance.
    """
    # pylint: disable=unused-argument
    return state.mean_photon(wires[0])


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


def Order2Poly(state, wires, params):
    r"""Computes the EV of an observable that is a second-order polynomial in :math:\{x_i, p_i\}_i`.

    Args:
        state (strawberryfields.backends.states.BaseState): the quantum state
        wires (Sequence[int]): measured modes
        params (Sequence[array]): Q is a matrix or vector of coefficients
            using the (I, x1,p1, x2,p2, ...) ordering

    Returns:
        float, float: expectation value, variance
    """
    Q = params[0]

    # HACK, we need access to the Poly instance in order to expand the matrix!
    op = openqml.expectation.PolyXP(Q, wires=wires, do_queue=False)
    Q = op.heisenberg_obs(state.num_modes)

    if Q.ndim == 1:
        d = np.r_[Q[1::2], Q[2::2]]
        return state.poly_quad_expectation(None, d, Q[0])

    # convert to the (I, x1,x2,..., p1,p2...) ordering
    M = np.vstack((Q[0:1,:], Q[1::2,:], Q[2::2,:]))
    M = np.hstack((M[:,0:1], M[:,1::2], M[:,2::2]))
    d1 = M[1:, 0]
    d2 = M[0, 1:]
    return state.poly_quad_expectation(M[1:,1:], d1+d2, M[0,0])
