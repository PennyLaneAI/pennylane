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
"""
CV quantum expectations
=======================

.. currentmodule:: openqml.expval.cv

This section contains the available built-in continuous-variable
quantum operations supported by OpenQML, as well as their conventions.

.. note:: Currently, all expectation commands act on only one wire.

Summary
-------

.. autosummary::
    PhotonNumber
    X
    P
    Homodyne
    PolyXP

Code details
~~~~~~~~~~~~
"""

import numpy as np

from openqml.operation import CVExpectation


class PhotonNumber(CVExpectation):
    r"""openqml.expval.PhotonNumber(wires)
    Returns the photon number expectation value.

    This expectation command returns the value
    :math:`\braket{\hat{n}}` where the number operator is
    :math:`\hat{n} = \a^\dagger \a = \frac{1}{2\hbar}(\x^2 +\p^2) -\I/2`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Expectation order: 2nd order in the quadrature operators.
    * Heisenberg representation:

      .. math:: M = \frac{1}{2\hbar}\begin{bmatrix}
            -\hbar & 0 & 0\\
            0 & 1 & 0\\
            0 & 0 & 1
        \end{bmatrix}

    Args:
        wires (Sequence[int] or int): the wire the operation acts on.
    """
    num_wires = 1
    num_params = 0
    par_domain = None

    ev_order = 2

    @staticmethod
    def _heisenberg_rep(p):
        hbar = 2
        return np.diag([-0.5, 0.5/hbar, 0.5/hbar])


class X(CVExpectation):
    r"""openqml.expval.X(wires)
    Returns the position expectation value in phase space.

    This expectation command returns the value :math:`\braket{\x}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Expectation order: 1st order in the quadrature operators.
    * Heisenberg representation:

      .. math:: d = [0, 1, 0]

    Args:
        wires (Sequence[int] or int): the wire the operation acts on.
    """
    num_wires = 1
    num_params = 0
    par_domain = None

    ev_order = 1

    @staticmethod
    def _heisenberg_rep(p):
        return np.array([0, 1, 0])


class P(CVExpectation):
    r"""openqml.expval.P(wires)
    Returns the momentum expectation value in phase space.

    This expectation command returns the value :math:`\braket{\p}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Expectation order: 1st order in the quadrature operators.
    * Heisenberg representation:

      .. math:: d = [0, 0, 1]

    Args:
        wires (Sequence[int] or int): the wire the operation acts on.
    """
    num_wires = 1
    num_params = 0
    par_domain = None

    ev_order = 1

    @staticmethod
    def _heisenberg_rep(p):
        return np.array([0, 0, 1])


class Homodyne(CVExpectation):
    r"""openqml.expval.Homodyne(wires)
    Returns the homodyne expectation value in phase space.

    This expectation command returns the value :math:`\braket{\x_\phi}`,
    where :math:`\x_\phi = \x cos\phi+\p\sin\phi` is the generalised
    quadrature operator.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Expectation order: 1st order in the quadrature operators.
    * Heisenberg representation:

      .. math:: d = [0, \cos\phi, \sin\phi]

    Args:
        phi (float): axis in the phase space at which to calculate
            the homodyne measurement.
        wires (Sequence[int] or int): the wire the operation acts on.
    """
    num_wires = 1
    num_params = 1
    par_domain = 'R'

    grad_method = 'A'
    ev_order = 1

    @staticmethod
    def _heisenberg_rep(p):
        phi = p[0]
        return np.array([0, np.cos(phi), np.sin(phi)])  # TODO check


class PolyXP(CVExpectation):
    r"""openqml.expval.PolyXP(wires)
    Second order polynomial observable.

    Represents an arbitrary observable :math:`P(\x,\p)` that is a second order
    polynomial in the basis :math:`\mathbf{r} = (\I, x_0, p_0, x_1, p_1, \ldots)`.

    For first-order observables the representation is a real vector
    :math:`\mathbf{d}` such that :math:`P(\x,\p) = \mathbf{d}^T \mathbf{r}`.

    For second-order observables the representation is a real symmetric
    matrix :math:`A` such that :math:`P(\x,\p) = \mathbf{r}^T A \mathbf{r}`.

    Used by :meth:`QNode._pd_analytic` for evaluating arbitrary order-2 CV observables.

    **Details:**

    * Number of wires: None (applied to the entire system).
    * Number of parameters: 1
    * Expectation order: 2nd order in the quadrature operators.
    * Heisenberg representation: :math:`A`

    Args:
        q (array[float]): expansion coefficients
    """
    num_wires = 0
    num_params = 1
    par_domain = 'A'

    grad_method = 'F'
    ev_order = 2

    @staticmethod
    def _heisenberg_rep(p):
        return p[0]


all_ops = [Homodyne, PhotonNumber, P, X, PolyXP]

__all__ = [cls.__name__ for cls in all_ops]
