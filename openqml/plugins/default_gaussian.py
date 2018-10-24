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
# pylint: disable=inconsistent-return-statements
"""
Default Gaussian plugin
=======================

**Module name:** :mod:`openqml.plugins.default_gaussian`

**Short name:** ``"default.gaussian"``

.. currentmodule:: openqml.plugins.default_gaussian

The default plugin is meant to be used as a template for writing CV OpenQML
device plugins for new backends.

It implements all the :class:`~openqml._device.Device` methods as well as all built-in
continuous-variable Gaussian operations and expectations, and provides
a very simple simulation of a Gaussian-based quantum circuit architecture.

Auxillary functions
-------------------

.. autosummary::
    bloch_messiah
    density_matrix


Gates and operations
--------------------

.. autosummary::
    rotation
    displacement
    squeezing
    quadratic_phase
    beamsplitter
    two_mode_squeezing
    controlled_addition
    controlled_phase

State preparation
-----------------

.. autosummary::
    squeezed_cov
    vacuum_state
    coherent_state
    squeezed_state
    displaced_squeezed_state
    thermal_state
    gaussian_state
    set_state


Expectations
------------

.. autosummary::
    photon_number
    homodyne
    poly_quad_expectations
    fock_expectation


Classes
-------

.. autosummary::
   DefaultGaussian

Code details
~~~~~~~~~~~~
"""
# pylint: disable=attribute-defined-outside-init
import logging as log

import numpy as np
from numpy.polynomial.hermite import hermval

from scipy.special import binom, factorial as fac

import openqml as qm
from openqml import Device

log.getLogger()

# tolerance for numerical errors
tolerance = 1e-10


#========================================================
#  auxillary functions
#========================================================

def bloch_messiah(cov):
    r"""Performs the Bloch-Messiah decomposition of single mode
    Gaussian state.

    Args:
        cov (array): :math:`2\times 2` covariance matrix.

    Returns:
        tuple: mean photon number, rotation angle, and
        squeezing magnitude of the Gaussian state.
    """
    det = np.linalg.det(cov)
    nbar = (np.sqrt(det)-1)/2

    mm = cov/np.sqrt(det)
    a = mm[0, 0]
    b = mm[0, 1]

    r = -0.5*np.arccosh((1+a*a+b*b)/(2*a))
    phi = 0.5*np.arctan2((2*a*b), (-1+a*a-b*b))
    return nbar, phi, r


def density_matrix(mu, cov, cutoff, hbar=2.):
    r"""Returns the density matrix for a single mode Gaussian state.

    Args:
        mu (array): length-:math:`2N` means vector
        cov (array): :math:`2\times 2` covariance matrix
        m (int): :math:`m`th row of the density matrix
        n (int): :math:`n`th column of the density matrix

    Returns:
        complex: density matrix element :math:`\rho_{mn}`
    """
    # calculate mean photon number, rotation, and squeezing
    nbar, phi, r = bloch_messiah(cov)
    # calculate the displacement
    beta = (mu[0] + mu[1]*1j)/np.sqrt(2*hbar)

    # change signs to account for convention
    beta = -beta
    r = -r
    phi = -2*phi

    m = np.arange(cutoff+1).reshape(-1, 1, 1)
    n = np.arange(cutoff+1).reshape(1, -1, 1)
    i = np.arange(cutoff+1).reshape(1, 1, -1)

    # we only perform the sum when 0 <= i <= min(m, n)
    mask = i <= np.minimum(m, n)
    m_i = mask*(m-i)
    n_i = mask*(n-i)

    if np.abs(r) <= tolerance:
        # squeezing is trivial
        terms = ((-1)**m_i) * (beta**n_i) * (beta.conj()**m_i) * binom(m, i)/fac(n_i)
        coeff = np.sqrt(fac(n)/(fac(m))) * np.exp(-np.abs(beta)**2/2)
    else:
        # squeezing is non-trivial
        v = np.exp(-1j*phi)*np.sinh(r)
        u = np.cosh(r)

        alpha = beta*u-np.conjugate(beta)*v

        H_m = hermval(-alpha.conj()/np.sqrt(-2*u*v.conj()), np.identity(cutoff+1))
        H_n = hermval(beta/np.sqrt(2*u*v), np.identity(cutoff+1))

        terms = (binom(m, i)/fac(n_i)) * ((2/(u*v))**(i/2)) * ((-v.conj()/(2*u))**((m_i)/2)) * H_n[n_i] * H_m[m_i]
        coeff = np.sqrt(fac(n)/(fac(m)*u)) * (v/(2*u))**(n/2) * np.exp(-(np.abs(beta)**2-v.conj()*beta**2/u)/2)

    f = np.sum(coeff*terms*mask, axis=2).conj()

    if abs(nbar) < tolerance:
        # thermal population is trivial
        psi = f[:, 0]
        return np.outer(psi, psi.conj())

    # thermal population is non-trivial
    ratio = nbar/(1+nbar)

    n = np.arange(cutoff+1).reshape(1, -1)
    psi = np.sqrt(ratio**n) * f
    rho = np.einsum('in,jn->ij', psi, psi.conj())/(1+nbar)

    return rho


#========================================================
#  parametrized gates
#========================================================

def rotation(phi):
    """Rotation in the phase space.

    Args:
        phi (float): rotation parameter.

    Returns:
        array: symplectic transformation matrix.
    """
    return np.array([[np.cos(phi), -np.sin(phi)],
                     [np.sin(phi), np.cos(phi)]])


def displacement(state, wire, alpha, hbar=2):
    """Displacement in the phase space

    Args:
        state (tuple): contains means vector and covariance matrix.
        wire (int): wire that the displacement acts on.
        alpha (float): complex displacement.

    Returns:
        tuple: contains the vector of means and covariance matrix.
    """
    mu = state[0]
    mu[wire] += alpha.real*np.sqrt(2*hbar)
    mu[wire+len(mu)//2] += alpha.imag*np.sqrt(2*hbar)
    return mu, state[1]


def squeezing(r, phi):
    """Squeezing in the phase space

    Args:
        r (float): squeezing magnitude.
        phi (float): rotation parameter.

    Returns:
        array: symplectic transformation matrix.
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ch = np.cosh(r)
    sh = np.sinh(r)
    return np.array([[ch-cp*sh, -sp*sh],
                     [-sp*sh, ch+cp*sh]])


def quadratic_phase(s):
    """Quadratic phase shift.

    Args:
        s (float): parameter.

    Returns:
        array: symplectic transformation matrix.
    """
    return np.array([[1, 0],
                     [s, 1]])


def beamsplitter(theta, phi):
    r"""Beamsplitter.

    Args:
        theta (float): transmittivity angle (:math:`t=\cos\theta`).
        phi (float): phase angle (:math:`r=e^{i\phi}\sin\theta`).

    Returns:
        array: symplectic transformation matrix.
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)

    S = np.array([[ct, -cp*st, 0, -st*sp],
                  [cp*st, ct, -st*sp, 0],
                  [0, st*sp, ct, -cp*st],
                  [st*sp, 0, cp*st, ct]])

    return S


def two_mode_squeezing(r, phi):
    """two-mode squeezing

    Args:
        r (float): squeezing magnitude.
        phi (float): rotation parameter.

    Returns:
        array: symplectic transformation matrix.
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ch = np.cosh(r)
    sh = np.sinh(r)

    S = np.array([[ch, cp*sh, 0, sp*sh],
                  [cp*sh, ch, sp*sh, 0],
                  [0, sp*sh, ch, -cp*sh],
                  [sp*sh, 0, -cp*sh, ch]])

    return S


def controlled_addition(s):
    """The CX gate.

    Args:
        s (float): parameter.

    Returns:
        array: symplectic transformation matrix.
    """
    S = np.array([[1, 0, 0, 0],
                  [s, 1, 0, 0],
                  [0, 0, 1, -s],
                  [0, 0, 0, 1]])

    return S


def controlled_phase(s):
    """The CZ gate.

    Args:
        s (float): parameter.

    Returns:
        array: symplectic transformation matrix.
    """
    S = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, s, 1, 0],
                  [s, 0, 0, 1]])

    return S


#========================================================
#  Arbitrary states and operators
#========================================================

def squeezed_cov(r, phi, hbar=2):
    r"""Returns the squeezed covariance matrix of a squeezed state

    Args:
        r (float): the squeezing magnitude
        p (float): the squeezing phase :math:`\phi`
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the squeezed state
    """
    cov = np.array([[np.exp(-2*r), 0],
                    [0, np.exp(2*r)]]) * hbar/2

    R = rotation(phi/2)

    return R @ cov @ R.T


def vacuum_state(wires, hbar=2.):
    r""" Returns the vacuum state.

    Args:
        basis (str): Returns the vector of means and the covariance matrix.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the vacuum state
    """
    means = np.zeros((2*wires))
    cov = np.identity(2*wires) * hbar/2
    state = [means, cov]
    return state


def coherent_state(a, phi=0, hbar=2.):
    r""" Returns the coherent state.

    Args:
        a (complex) : the displacement
        phi (float): the phase
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the coherent state
    """
    alpha = a*np.exp(1j*phi)
    means = np.array([alpha.real, alpha.imag]) * np.sqrt(2*hbar)
    cov = np.identity(2) * hbar/2
    state = [means, cov]
    return state


def squeezed_state(r, phi, hbar=2.):
    r""" Returns the squeezed state

    Args:
        r (float): the squeezing magnitude
        phi (float): the squeezing phase :math:`\phi`
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the squeezed state
    """
    means = np.zeros((2))
    state = [means, squeezed_cov(r, phi, hbar)]
    return state


def displaced_squeezed_state(a, r, phi, hbar=2.):
    r""" Returns the squeezed coherent state

    Args:
        a (complex): the displacement.
        r (float): the squeezing magnitude
        phi (float): the squeezing phase :math:`\phi`
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the squeezed coherent state
    """
    means = np.array([a.real, a.imag]) * np.sqrt(2*hbar)
    state = [means, squeezed_cov(r, phi, hbar)]
    return state


def thermal_state(nbar, hbar=2.):
    r""" Returns the thermal state.

    Args:
        nbar (float): the mean photon number.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the thermal state
    """
    means = np.zeros([2])
    state = [means, (2*nbar+1)*np.identity(2)*hbar/2]
    return state


def gaussian_state(mu, cov, hbar=2.):
    r""" Returns the Gaussian state.

    This is simply a bare wrapper function,
    since the means vector and covariance matrix
    can be passed via the parameters unchanged.

    Note that both the means vector and covariance
    matrix should be in :math:`(\x_1,\dots, \x_N, \p_1, \dots, \p_N)`
    ordering.

    Args:
        mu (array): vector means. Must be length-:math:`2N`,
            where N is the number of modes.
        cov (array): covariance matrix. Must be dimension :math:`2N\times 2N`,
            where N is the number of modes.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the thermal state
    """
    # pylint: disable=unused-argument
    return mu, cov


def set_state(state, wire, mu, cov):
    r"""Inserts a single mode Gaussian into the
    state representation of the complete system.

    Args:
        state (tuple): contains means vector
            and covariance matrix of existing state.
        wire (int): wire corresponding to the new Gaussian state.
        mu (array): vector of means to insert.
        cov (array): covariance matrix to insert.

    Returns:
        tuple: contains the vector of means and covariance matrix.
    """
    mu0 = state[0]
    cov0 = state[1]
    N = len(mu0)//2

    # insert the new state into the means vector
    mu0[[wire, wire+N]] = mu

    # insert the new state into the covariance matrix
    ind = np.concatenate([np.array([wire]), np.array([wire])+N])
    rows = ind.reshape(-1, 1)
    cols = ind.reshape(1, -1)
    cov0[rows, cols] = cov

    return mu0, cov0


#========================================================
#  expectations
#========================================================


def photon_number(mu, cov, wires, params, hbar=2.):
    r"""Calculates the mean photon number for a given one-mode state.

    Args:
        mu (array): length-2 vector of means.
        cov (array): :math:`2\times 2` covariance matrix.
        wires (Sequence[int]): wires to calculate the expectation for.
        params (None): no parameters are used for this expectation value.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        tuple: contains the photon number expectation and variance.
    """
    # pylint: disable=unused-argument
    ex = (np.trace(cov) + mu.T @ mu)/(2*hbar) - 1/2
    var = (np.trace(cov @ cov) + 2*mu.T @ cov @ mu)/(2*hbar**2) - 1/4
    return ex, var


def homodyne(phi=None):
    """Function factory that returns the Homodyne expectation of a one mode state.

    Args:
        phi (float): the default phase space axis to perform the Homodyne measurement.

    Returns:
        function: A function that accepts a single mode means vector, covariance matrix,
        and phase space angle phi, and returns the quadrature expectation
        value and variance.
    """
    if phi is not None:
        def _homodyne(mu, cov, wires, params, hbar=2.):
            """Arbitrary angle homodyne expectation."""
            # pylint: disable=unused-argument
            rot = rotation(phi)
            muphi = rot.T @ mu
            covphi = rot.T @ cov @ rot
            return muphi[0], covphi[0, 0]
        return _homodyne

    def _homodyne(mu, cov, wires, params, hbar=2.):
        """Arbitrary angle homodyne expectation."""
        # pylint: disable=unused-argument
        rot = rotation(params[0])
        muphi = rot.T @ mu
        covphi = rot.T @ cov @ rot
        return muphi[0], covphi[0, 0]
    return _homodyne


def poly_quad_expectations(mu, cov, wires, params, hbar=2.):
    r"""Calculates the expectation and variance for an arbitrary
    polynomial of quadrature operators.

    Args:
        mu (array): length-2 vector of means.
        cov (array): :math:`2\times 2` covariance matrix.
        wires (Sequence[int]): wires to calculate the expectation for.
        params (array): a :math:`(2N+1)\times (2N+1)` array containing the linear
            and quadratic coefficients of the quadrature operators
            :math:`(I, \x_0, \p_0, \x_1, \p_1,\dots)`.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        tuple: contains the quadrature expectation and variance.
    """
    Q = params[0]
    N = len(mu)//2

    # HACK, we need access to the Poly instance in order to expand the matrix!
    op = qm.expval.PolyXP(Q, wires=wires, do_queue=False)
    Q = op.heisenberg_obs(N)

    if Q.ndim == 1:
        d = np.r_[Q[1::2], Q[2::2]]
        return d.T @ mu + Q[0], d.T @ cov @ d

    # convert to the (I, x1,x2,..., p1,p2...) ordering
    M = np.vstack((Q[0:1, :], Q[1::2, :], Q[2::2, :]))
    M = np.hstack((M[:, 0:1], M[:, 1::2], M[:, 2::2]))
    d1 = M[1:, 0]
    d2 = M[0, 1:]

    A = M[1:, 1:]
    d = d1 + d2
    k = M[0, 0]

    d2 = 2*A @ mu + d
    k2 = mu.T @ A @ mu + mu.T @ d + k

    ex = np.trace(A @ cov) + k2
    var = 2*np.trace(A @ cov @ A @ cov) + d2.T @ cov @ d2

    modes = np.arange(2*N).reshape(2, -1).T
    groenewald_correction = np.sum([np.linalg.det(hbar*A[:, m][n]) for m in modes for n in modes])
    var -= groenewald_correction

    return ex, var


def fock_expectation(mu, cov, wires, params, hbar=2.):
    r"""Calculates the expectation and variance for a single mode
    Fock state probability.

    Args:
        mu (array): length-2 vector of means.
        cov (array): :math:`2\times 2` covariance matrix.
        wires (Sequence[int]): wires to calculate the expectation for.
        params (int): the Fock state to return the expectation value for.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        tuple: contains the Fock state expectation and variance.
    """
    n = params[0]
    cutoff = n + 5 # is there a better heuristic for this?
    dm = density_matrix(mu, cov, cutoff, hbar=hbar)

    # note: currently not sure how to return the variance for this
    return np.abs(dm[n, n]), 0


#========================================================
#  device
#========================================================


class DefaultGaussian(Device):
    r"""Default Gaussian device for OpenQML.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): How many times should the circuit be evaluated (or sampled) to estimate
            the expectation values. 0 yields the exact result.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    """
    name = 'Default Gaussian OpenQML plugin'
    short_name = 'default.gaussian'
    api_version = '0.1.0'
    version = '0.1.0'
    author = 'Xanadu Inc.'

    _operation_map = {
        'Beamsplitter': beamsplitter,
        'ControlledAddition': controlled_addition,
        'ControlledPhase': controlled_phase,
        'Displacement': displacement,
        'QuadraticPhase': quadratic_phase,
        'Rotation': rotation,
        'Squeezing': squeezing,
        'TwoModeSqueezing': two_mode_squeezing,
        'CoherentState': coherent_state,
        'DisplacedSqueezedState': displaced_squeezed_state,
        'SqueezedState': squeezed_state,
        'ThermalState': thermal_state,
        'GaussianState': gaussian_state
    }

    _expectation_map = {
        'PhotonNumber': photon_number,
        'X': homodyne(0),
        'P': homodyne(np.pi/2),
        'Homodyne': homodyne(None),
        'PolyXP': poly_quad_expectations,
        'NumberState': fock_expectation
    }

    _circuits = {}

    def __init__(self, wires, *, shots=0, hbar=2):
        super().__init__(self.short_name, wires, shots)
        self.eng = None
        self.hbar = hbar
        self.reset()

    def pre_apply(self):
        self.reset()

    def apply(self, gate_name, wires, par):
        if gate_name == 'Displacement':
            self._state = displacement(self._state, wires[0], par[0]*np.exp(1j*par[1]))
            return # we are done here

        if gate_name == 'GaussianState':
            if wires != list(range(self.num_wires)):
                raise ValueError("GaussianState means vector or covariance matrix is "
                                 "the incorrect size for the number of subsystems.")
            self._state = self._operation_map[gate_name](*par, hbar=self.hbar)
            return # we are done here

        if 'State' in gate_name:
            # set the new device state
            mu, cov = self._operation_map[gate_name](*par, hbar=self.hbar)
            # state preparations only act on at most 1 subsystem
            self._state = set_state(self._state, wires[0], mu, cov)
            return # we are done here

        # get the symplectic matrix
        S = self._operation_map[gate_name](*par)

        # expand the symplectic to act on the proper subsystem
        if len(wires) == 1:
            S = self.expand_one(S, wires[0])
        elif len(wires) == 2:
            S = self.expand_two(S, wires)

        # apply symplectic matrix to the means vector
        means = S @ self._state[0]
        # apply symplectic matrix to the covariance matrix
        cov = S @ self._state[1] @ S.T

        self._state = [means, cov]

    def expand_one(self, S, wire):
        r"""Expands a one-mode Symplectic matrix S to act on the entire subsystem.

        Args:
            S (array): :math:`2\times 2` Symplectic matrix.
            wire (int): the wire S acts on.

        Returns:
            array: the resulting :math:`2N\times 2N` Symplectic matrix.
        """
        S2 = np.identity(2*self.num_wires)

        ind = np.concatenate([np.array([wire]), np.array([wire])+self.num_wires])
        rows = ind.reshape(-1, 1)
        cols = ind.reshape(1, -1)
        S2[rows, cols] = S.copy()

        return S2

    def expand_two(self, S, wires):
        r"""Expands a two-mode Symplectic matrix S to act on the entire subsystem.

        Args:
            S (array): :math:`4\times 4` Symplectic matrix.
            wires (Sequence[int]): the list of two wires S acts on.

        Returns:
            array: the resulting :math:`2N\times 2N` Symplectic matrix.
        """
        S2 = np.identity(2*self.num_wires)
        w = np.array(wires)

        S2[w.reshape(-1, 1), w.reshape(1, -1)] = S[:2, :2].copy() #X
        S2[(w+self.num_wires).reshape(-1, 1), (w+self.num_wires).reshape(1, -1)] = S[2:, 2:].copy() #P
        S2[w.reshape(-1, 1), (w+self.num_wires).reshape(1, -1)] = S[:2, 2:].copy() #XP
        S2[(w+self.num_wires).reshape(-1, 1), w.reshape(1, -1)] = S[2:, :2].copy() #PX

        return S2

    def expval(self, expectation, wires, par):
        mu, cov = self.reduced_state(wires)

        if expectation == 'PolyXP':
            mu, cov = self._state

        ev, var = self._expectation_map[expectation](mu, cov, wires, par, hbar=self.hbar)

        if self.shots != 0:
            # estimate the ev
            # use central limit theorem, sample normal distribution once, only ok if n_eval is large
            # (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ev = np.random.normal(ev, np.sqrt(var / self.shots))

        return ev

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        self._state = vacuum_state(self.num_wires, self.hbar)

    def reduced_state(self, wires):
        r""" Returns the vector of means and the covariance matrix of the specified wires.

        Args:
            wires (int of Sequence[int]): indices of the requested wires

        Returns:
            tuple (means, cov): where means is an array containing the vector of means,
            and cov is a square array containing the covariance matrix.
        """
        if wires == list(range(self.num_wires)):
            # reduced state is full state
            return self._state

        # reduce rho down to specified subsystems
        if isinstance(wires, int):
            wires = [wires]

        if np.any(np.array(wires) > self.num_wires):
            raise ValueError("The specified wires cannot "
                             "be larger than the number of subsystems.")

        ind = np.concatenate([np.array(wires), np.array(wires)+self.num_wires])
        rows = ind.reshape(-1, 1)
        cols = ind.reshape(1, -1)

        return self._state[0][ind], self._state[1][rows, cols]
