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

It implements all the :class:`~openqml.device.Device` methods as well as all built-in
continuous-variable Gaussian gates and observables, and provides
a very simple simulation of a Gaussian-based quantum circuit architecture.

Gates and operations
--------------------

.. autosummary::
    rotation
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

Classes
-------

.. autosummary::
   DefaultGaussian

----
"""
# pylint: disable=attribute-defined-outside-init
import logging as log

import numpy as np

import openqml as qm
from openqml.device import Device

log.getLogger()

# tolerance for numerical errors
tolerance = 1e-10


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


def squeezing(r, phi):
    """squeezing in the phase space

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

    R = np.array([[np.cos(phi/2), -np.sin(phi/2)],
                  [np.sin(phi/2), np.cos(phi/2)]])

    return np.dot(np.dot(R, cov), R.T)


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


#========================================================
#  device
#========================================================


class DefaultGaussian(Device):
    """Default qubit device for OpenQML.

    Args:
      wires (int): the number of modes to initialize the device in
      shots (int): How many times should the circuit be evaluated (or sampled) to estimate
        the expectation values? 0 yields the exact result.
    """
    name = 'Default Gaussian OpenQML plugin'
    short_name = 'default.gaussian'
    api_version = '0.1.0'
    version = '0.1.0'
    author = 'Xanadu Inc.'

    _operator_map = {
        'Beamsplitter': beamsplitter,
        'ControlledAddition': controlled_addition,
        'ControlledPhase': controlled_phase,
        'Displacement': None,
        'QuadraticPhase': quadratic_phase,
        'Rotation': rotation,
        'Squeezing': squeezing,
        'TwoModeSqueezing': two_mode_squeezing,
        'CoherentState': coherent_state,
        'DisplacedSqueezedState': displaced_squeezed_state,
        'SqueezedState': squeezed_state,
        'ThermalState': thermal_state,
        'GaussianState': lambda *p, hbar=2: p
    }

    _observable_map = {
        'PhotonNumber': None,
        'X': None,
        'P': None,
        'Homodyne': None,
        'PolyXP': None
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
            # modify the means
            alpha = par[0]*np.exp(1j*par[1])
            self._state[0][wires[0]] += alpha.real*np.sqrt(2*self.hbar)
            self._state[0][wires[0]+self.wires] += alpha.imag*np.sqrt(2*self.hbar)
            return

        if gate_name == 'GaussianState':
            if wires != list(range(self.wires)):
                raise ValueError("GaussianState means vector or covariance matrix is "
                                 "the incorrect size for the number of subsystems.")
            self._state = self._operator_map[gate_name](*par, hbar=self.hbar)
            return

        if 'State' in gate_name:
            # set the new device state
            w = wires[0]
            mu, cov = self._operator_map[gate_name](*par, hbar=self.hbar)

            # insert the new state into the means vector
            self._state[0][[w, w+self.wires]] = mu

            # insert the new state into the covariance matrix
            ind = np.concatenate([np.array([w]), np.array([w])+self.wires])
            rows = ind.reshape(-1, 1)
            cols = ind.reshape(1, -1)
            self._state[1][rows, cols] = cov
            return

        # get the symplectic matrix
        S = self._operator_map[gate_name](*par)

        # expand the symplectic to act on the proper subsystem
        if len(wires) == 1:
            w = wires[0]
            S2 = np.identity(2*self.wires)

            ind = np.concatenate([np.array([w]), np.array([w])+self.wires])
            rows = ind.reshape(-1, 1)
            cols = ind.reshape(1, -1)
            S2[rows, cols] = S.copy()

        elif len(wires) == 2:
            S2 = np.identity(2*self.wires)
            w = np.array(wires)

            S2[w.reshape(-1, 1), w.reshape(1, -1)] = S[:2, :2].copy() #X
            S2[(w+self.wires).reshape(-1, 1), (w+self.wires).reshape(1, -1)] = S[2:, 2:].copy() #P
            S2[w.reshape(-1, 1), (w+self.wires).reshape(1, -1)] = S[:2, 2:].copy() #XP
            S2[(w+self.wires).reshape(-1, 1), w.reshape(1, -1)] = S[2:, :2].copy() #PX

        # apply symplectic matrix to the means vector
        means = S2 @ self._state[0]

        # apply symplectic matrix to the covariance matrix
        cov = S2 @ self._state[1] @ S2.T

        self._state = [means, cov]

    def expectation(self, observable, wires, par):
        if self.shots == 0:
            # exact expectation value
            ev, var = self.ev(observable, wires, par)
        else:
            # estimate the ev
            # use central limit theorem, sample normal distribution once, only ok if n_eval is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ev, var = self.ev(observable, wires, par)
            ev = np.random.normal(ev, np.sqrt(var / self.shots))

        return ev

    def ev(self, observable, wires, params):
        """Returns the expectation value and variance for a particular observable.

        Args:
            observable (str): name of the observable.
            wires (int or Sequence[int]): wires to be measured.
            params (Sequence): list of observable parameters.

        Returns:
            tuple: expected value and variance.
        """
        # pylint: disable=too-many-return-statements
        mu, cov = self.reduced_state(wires)

        # measurement/expectation value
        if observable == 'PhotonNumber':
            ex = (np.trace(cov) + mu.T @ mu)/(2*self.hbar) - 1/2
            var = (np.trace(cov @ cov) + 2*mu.T @ cov @ mu)/(2*self.hbar**2) - 1/4
            return ex, var

        if observable == 'X':
            return mu[0], cov[0, 0]

        if observable == 'P':
            return mu[1], cov[1, 1]

        if observable == 'Homodyne':
            rot = rotation(params[0])
            muphi = rot.T @ mu
            covphi = rot.T @ cov @ rot
            return muphi[0], covphi[0, 0]

        if observable == 'PolyXP':
            mu, cov = self._state
            Q = params[0]

            # HACK, we need access to the Poly instance in order to expand the matrix!
            op = qm.expectation.PolyXP(Q, wires=wires, do_queue=False)
            Q = op.heisenberg_obs(self.wires)

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

            modes = np.arange(2*self.wires).reshape(2, -1).T
            var -= np.sum([np.linalg.det(self.hbar*A[:, m][n]) for m in modes for n in modes])

            return ex, var

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        self._state = vacuum_state(self.wires, self.hbar)

    def reduced_state(self, wires):
        r""" Returns the vector of means and the covariance matrix of the specified wires.

        Args:
            wires (int of Sequence[int]): indices of the requested wires

        Returns:
            tuple (means, cov): where means is an array containing the vector of means,
            and cov is a square array containing the covariance matrix.
        """
        if wires == list(range(self.wires)):
            # reduced state is full state
            return self._state

        # reduce rho down to specified subsystems
        if isinstance(wires, int):
            wires = [wires]

        if np.any(np.array(wires) > self.wires):
            raise ValueError("The number of specified wires cannot "
                             "be larger than the number of subsystems.")

        ind = np.concatenate([np.array(wires), np.array(wires)+self.wires])
        rows = ind.reshape(-1, 1)
        cols = ind.reshape(1, -1)

        return self._state[0][ind], self._state[1][rows, cols]
