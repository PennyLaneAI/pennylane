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

import numpy as np

from openqml.operation import CVExpectation



class PhotonNumber(CVExpectation):
    r"""Returns the photon number expectation value.

    The photon number operator is :math:`n = a^\dagger a = \frac{1}{2\hbar}(x^2 +p^2) -\I/2`.
    """
    ev_order = 2
    @staticmethod
    def _heisenberg_rep(p):
        hbar = 2
        return np.diag([-0.5, 0.5/hbar, 0.5/hbar])


class X(CVExpectation):
    r"""Returns the position expectation value in the phase space.
    """
    ev_order = 1
    @staticmethod
    def _heisenberg_rep(p):
        return np.array([0, 1, 0])


class P(CVExpectation):
    r"""Returns the momentum expectation value in the phase space.

    """
    ev_order = 1
    @staticmethod
    def _heisenberg_rep(p):
        return np.array([0, 0, 1])


class Homodyne(CVExpectation):
    r"""Returns the homodyne expectation value in the phase space.

    Args:
        phi (float): axis in the phase space at which to calculate
            the homodyne measurement.
    """
    n_params = 1
    ev_order = 1
    @staticmethod
    def _heisenberg_rep(p):
        phi = p[0]
        return np.array([0, np.cos(phi), np.sin(phi)])  # TODO check


class Heterodyne(CVExpectation):
    r"""Returns the displacement expectation value in the phase space.

    """


class PolyXP(CVExpectation):
    r"""Second order polynomial observable.

    Represents an arbitrary observable Q that is a second order polynomial in the basis
    :math:`\vec{E} = (\I, x_0, p_0, x_1, p_1, \ldots)`.

    For first-order observables the representation is a real vector q such that :math:`Q = \sum_i q_i E_i`.
    For second-order observables the representation is a real symmetric matrix q such that :math:`Q = \sum_{ij} q_{ij} E_i E_j`.

    Used by :meth:`QNode._pd_analytic` for evaluating arbitrary order-2 CV observables.

    Args:
      q (array[float]): expansion coefficients
    """
    n_wires  = 0
    n_params = 1
    par_domain = 'A'
    ev_order = 2
    @staticmethod
    def _heisenberg_rep(p):
        return p[0]


all_ops = [Heterodyne, Homodyne, PhotonNumber, P, X, PolyXP]

__all__ = [cls.__name__ for cls in all_ops]
