# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
The :code:`default.gaussian` plugin is meant to be used as a template for writing PennyLane
device plugins for new CV backends.

It implements the necessary :class:`~pennylane._device.Device` methods as well as all built-in
:mod:`continuous-variable Gaussian operations <pennylane.ops.cv>`, and provides a very simple simulation of a
Gaussian-based quantum circuit architecture.
"""
# pylint: disable=attribute-defined-outside-init,too-many-arguments
import math
import cmath
import numpy as np

from scipy.special import factorial as fac

import pennylane as qml
from pennylane import Device

# tolerance for numerical errors
tolerance = 1e-10


# ========================================================
#  auxillary functions
# ========================================================


def partitions(s, include_singles=True):
    """Partitions a sequence into all groupings of pairs and singles of elements.

    Args:
        s (sequence): the sequence to partition
        include_singles (bool): if False, only partitions into pairs
            is returned.

    Returns:
        tuple: returns a nested tuple, containing all partitions of the sequence.
    """
    # pylint: disable=too-many-branches
    if len(s) == 2:
        if include_singles:
            yield (s[0],), (s[1],)

        yield tuple(s),
    else:
        # pull off a single item and partition the rest
        if include_singles:
            if len(s) > 1:
                item_partition = (s[0],)
                rest = s[1:]
                rest_partitions = partitions(rest, include_singles)
                for p in rest_partitions:
                    yield ((item_partition),) + p
            else:
                yield tuple(s),

        # pull off a pair of items and partition the rest
        for idx1 in range(1, len(s)):
            item_partition = (s[0], s[idx1])
            rest = s[1:idx1] + s[idx1 + 1 :]
            rest_partitions = partitions(rest, include_singles)
            for p in rest_partitions:
                yield ((item_partition),) + p


def fock_prob(mu, cov, event, hbar=2.0):
    r"""Returns the probability of detection of a particular PNR detection event.

    For more details, see:

    * Kruse, R., Hamilton, C. S., Sansoni, L., Barkhofen, S., Silberhorn, C., & Jex, I.
      "A detailed study of Gaussian Boson Sampling." `arXiv:1801.07488. (2018).
      <https://arxiv.org/abs/1801.07488>`_

    * Hamilton, C. S., Kruse, R., Sansoni, L., Barkhofen, S., Silberhorn, C., & Jex, I.
      "Gaussian boson sampling." `Physical review letters, 119(17), 170501. (2017).
      <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.170501>`_

    Args:
        mu (array): length-:math:`2N` means vector
        cov (array): :math:`2N\times 2N` covariance matrix
        event (array): length-:math:`N` array of non-negative integers representing the
            PNR detection event of the multi-mode system.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        float: probability of detecting the event
    """
    # number of modes
    N = len(mu) // 2
    I = np.identity(N)

    # mean displacement of each mode
    alpha = (mu[:N] + 1j * mu[N:]) / math.sqrt(2 * hbar)
    # the expectation values (<a_1>, <a_2>,...,<a_N>, <a^\dagger_1>, ..., <a^\dagger_N>)
    beta = np.concatenate([alpha, alpha.conj()])

    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    # Q(alpha) = exp[-(alpha-beta).sigma_Q^{-1}.(alpha-beta)/2]/|sigma_Q|
    Q = np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * N)

    # inverse Q matrix
    Qinv = np.linalg.inv(Q)
    # 1/sqrt(|Q|)
    sqrt_Qdet = 1 / math.sqrt(np.linalg.det(Q).real)

    prefactor = cmath.exp(-beta @ Qinv @ beta.conj() / 2)

    if np.all(np.array(event) == 0):
        # all PNRs detect the vacuum state
        return (prefactor * sqrt_Qdet).real / np.prod(fac(event))

    # the matrix X_n = [[0, I_n], [I_n, 0]]
    O = np.zeros_like(I)
    X = np.block([[O, I], [I, O]])

    gamma = X @ Qinv.conj() @ beta

    # For each mode, repeat the mode number event[i] times
    ind = [i for sublist in [[idx] * j for idx, j in enumerate(event)] for i in sublist]
    # extend the indices for xp-ordering of the Gaussian state
    ind += [i + N for i in ind]

    if np.linalg.norm(beta) < tolerance:
        # state has no displacement
        part = partitions(ind, include_singles=False)
    else:
        part = partitions(ind, include_singles=True)

    # calculate Hamilton's A matrix: A = X.(I-Q^{-1})*
    A = X @ (np.identity(2 * N) - Qinv).conj()
    summation = np.sum([np.prod([gamma[i[0]] if len(i) == 1 else A[i] for i in p]) for p in part])

    return (prefactor * sqrt_Qdet * summation).real / np.prod(fac(event))


# ========================================================
#  parametrized gates
# ========================================================


def rotation(phi):
    """Rotation in the phase space.

    Args:
        phi (float): rotation parameter

    Returns:
        array: symplectic transformation matrix
    """
    return np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])


def displacement(state, wire, alpha, hbar=2):
    """Displacement in the phase space.

    Args:
        state (tuple): contains means vector and covariance matrix
        wire (int): wire that the displacement acts on
        alpha (float): complex displacement

    Returns:
        tuple: contains the vector of means and covariance matrix
    """
    mu = state[0]
    mu[wire] += alpha.real * math.sqrt(2 * hbar)
    mu[wire + len(mu) // 2] += alpha.imag * math.sqrt(2 * hbar)
    return mu, state[1]


def squeezing(r, phi):
    """Squeezing in the phase space.

    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter

    Returns:
        array: symplectic transformation matrix
    """
    cp = math.cos(phi)
    sp = math.sin(phi)
    ch = math.cosh(r)
    sh = math.sinh(r)
    return np.array([[ch - cp * sh, -sp * sh], [-sp * sh, ch + cp * sh]])


def quadratic_phase(s):
    """Quadratic phase shift.

    Args:
        s (float): gate parameter

    Returns:
        array: symplectic transformation matrix
    """
    return np.array([[1, 0], [s, 1]])


def beamsplitter(theta, phi):
    r"""Beamsplitter.

    Args:
        theta (float): transmittivity angle (:math:`t=\cos\theta`)
        phi (float): phase angle (:math:`r=e^{i\phi}\sin\theta`)

    Returns:
        array: symplectic transformation matrix
    """
    cp = math.cos(phi)
    sp = math.sin(phi)
    ct = math.cos(theta)
    st = math.sin(theta)

    S = np.array(
        [
            [ct, -cp * st, 0, -st * sp],
            [cp * st, ct, -st * sp, 0],
            [0, st * sp, ct, -cp * st],
            [st * sp, 0, cp * st, ct],
        ]
    )

    return S


def two_mode_squeezing(r, phi):

    """Two-mode squeezing.

    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter

    Returns:
        array: symplectic transformation matrix
    """
    cp = math.cos(phi)
    sp = math.sin(phi)
    ch = math.cosh(r)
    sh = math.sinh(r)

    S = np.array(
        [
            [ch, cp * sh, 0, sp * sh],
            [cp * sh, ch, sp * sh, 0],
            [0, sp * sh, ch, -cp * sh],
            [sp * sh, 0, -cp * sh, ch],
        ]
    )

    return S


def controlled_addition(s):
    """CX gate.

    Args:
        s (float): gate parameter

    Returns:
        array: symplectic transformation matrix
    """
    S = np.array([[1, 0, 0, 0], [s, 1, 0, 0], [0, 0, 1, -s], [0, 0, 0, 1]])

    return S


def controlled_phase(s):
    """CZ gate.

    Args:
        s (float): gate parameter

    Returns:
        array: symplectic transformation matrix
    """
    S = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, s, 1, 0], [s, 0, 0, 1]])

    return S


def interferometer(U):
    """Interferometer

    Args:
        U (array): unitary matrix

    Returns:
        array: symplectic transformation matrix
    """
    N = 2 * len(U)
    X = U.real
    Y = U.imag
    rows = np.arange(N).reshape(2, -1).T.flatten()
    S = np.vstack([np.hstack([X, -Y]), np.hstack([Y, X])])[:, rows][rows]

    return S


# ========================================================
#  Arbitrary states and operators
# ========================================================


def squeezed_cov(r, phi, hbar=2):
    r"""Returns the squeezed covariance matrix of a squeezed state.

    Args:
        r (float): the squeezing magnitude
        p (float): the squeezing phase :math:`\phi`
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
    Returns:
        array: the squeezed state
    """
    cov = np.array([[math.exp(-2 * r), 0], [0, math.exp(2 * r)]]) * hbar / 2

    R = rotation(phi / 2)

    return R @ cov @ R.T


def vacuum_state(wires, hbar=2.0):
    r"""Returns the vacuum state.

    Args:
        basis (str): Returns the vector of means and the covariance matrix
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
    Returns:
        array: the vacuum state
    """
    means = np.zeros((2 * wires))
    cov = np.identity(2 * wires) * hbar / 2
    state = [means, cov]
    return state


def coherent_state(a, phi=0, hbar=2.0):
    r"""Returns a coherent state.

    Args:
        a (complex) : the displacement
        phi (float): the phase
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
    Returns:
        array: the coherent state
    """
    alpha = a * cmath.exp(1j * phi)
    means = np.array([alpha.real, alpha.imag]) * math.sqrt(2 * hbar)
    cov = np.identity(2) * hbar / 2
    state = [means, cov]
    return state


def squeezed_state(r, phi, hbar=2.0):
    r"""Returns a squeezed state.

    Args:
        r (float): the squeezing magnitude
        phi (float): the squeezing phase :math:`\phi`
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        array: the squeezed state
    """
    means = np.zeros((2))
    state = [means, squeezed_cov(r, phi, hbar)]
    return state


def displaced_squeezed_state(a, phi_a, r, phi_r, hbar=2.0):
    r"""Returns a squeezed coherent state

    Args:
        a (real): the displacement magnitude
        phi_a (real): the displacement phase
        r (float): the squeezing magnitude
        phi_r (float): the squeezing phase :math:`\phi_r`
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        array: the squeezed coherent state
    """
    alpha = a * cmath.exp(1j * phi_a)
    means = np.array([alpha.real, alpha.imag]) * math.sqrt(2 * hbar)
    state = [means, squeezed_cov(r, phi_r, hbar)]
    return state


def thermal_state(nbar, hbar=2.0):
    r"""Returns a thermal state.

    Args:
        nbar (float): the mean photon number
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        array: the thermal state
    """
    means = np.zeros([2])
    state = [means, (2 * nbar + 1) * np.identity(2) * hbar / 2]
    return state


def gaussian_state(mu, cov, hbar=2.0):
    r"""Returns a Gaussian state.

    This is simply a bare wrapper function,
    since the means vector and covariance matrix
    can be passed via the parameters unchanged.

    Note that both the means vector and covariance
    matrix should be in :math:`(\x_1,\dots, \x_N, \p_1, \dots, \p_N)`
    ordering.

    Args:
        mu (array): vector means. Must be length-:math:`2N`,
            where N is the number of modes
        cov (array): covariance matrix. Must be dimension :math:`2N\times 2N`,
            where N is the number of modes
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        tuple: the mean and covariance matrix of the Gaussian state
    """
    # pylint: disable=unused-argument
    return mu, cov


def set_state(state, wire, mu, cov):
    r"""Inserts a single mode Gaussian into the
    state representation of the complete system.

    Args:
        state (tuple): contains means vector
            and covariance matrix of existing state
        wire (int): wire corresponding to the new Gaussian state
        mu (array): vector of means to insert
        cov (array): covariance matrix to insert

    Returns:
        tuple: contains the vector of means and covariance matrix.
    """
    mu0 = state[0]
    cov0 = state[1]
    N = len(mu0) // 2

    # insert the new state into the means vector
    mu0[[wire, wire + N]] = mu

    # insert the new state into the covariance matrix
    ind = np.concatenate([np.array([wire]), np.array([wire]) + N])
    rows = ind.reshape(-1, 1)
    cols = ind.reshape(1, -1)
    cov0[rows, cols] = cov

    return mu0, cov0


# ========================================================
#  expectations
# ========================================================


def photon_number(mu, cov, wires, params, total_wires, hbar=2.0):
    r"""Calculates the mean photon number for a given one-mode state.

    Args:
        mu (array): length-2 vector of means
        cov (array): :math:`2\times 2` covariance matrix
        wires (Sequence[int]): wires to calculate the expectation for
        params (None): no parameters are used for this expectation value
        total_wires (int): total number of wires in the system
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        tuple: contains the photon number expectation and variance
    """
    # pylint: disable=unused-argument
    ex = (np.trace(cov) + mu.T @ mu) / (2 * hbar) - 1 / 2
    var = (np.trace(cov @ cov) + 2 * mu.T @ cov @ mu) / (2 * hbar ** 2) - 1 / 4
    return ex, var


def homodyne(phi=None):
    """Function factory that returns the Homodyne expectation of a one mode state.

    Args:
        phi (float): the default phase space axis to perform the Homodyne measurement

    Returns:
        function: A function that accepts a single mode means vector, covariance matrix,
        and phase space angle phi, and returns the quadrature expectation
        value and variance.
    """
    if phi is not None:

        def _homodyne(mu, cov, wires, params, total_wires, hbar=2.0):
            """Arbitrary angle homodyne expectation."""
            # pylint: disable=unused-argument
            rot = rotation(phi)
            muphi = rot.T @ mu
            covphi = rot.T @ cov @ rot
            return muphi[0], covphi[0, 0]

        return _homodyne

    def _homodyne(mu, cov, wires, params, total_wires, hbar=2.0):
        """Arbitrary angle homodyne expectation."""
        # pylint: disable=unused-argument
        rot = rotation(params[0])
        muphi = rot.T @ mu
        covphi = rot.T @ cov @ rot
        return muphi[0], covphi[0, 0]

    return _homodyne


def poly_quad_expectations(mu, cov, wires, params, total_wires, hbar=2.0):
    r"""Calculates the expectation and variance for an arbitrary
    polynomial of quadrature operators.

    Args:
        mu (array): vector of means
        cov (array): covariance matrix
        wires (Sequence[int]): wires to calculate the expectation for
        params (array): a :math:`(2N+1)\times (2N+1)` array containing the linear
            and quadratic coefficients of the quadrature operators
            :math:`(\I, \x_0, \p_0, \x_1, \p_1,\dots)`
        total_wires (int): total number of wires in the system
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        tuple: the mean and variance of the quadrature-polynomial observable
    """
    Q = params[0]

    # HACK, we need access to the Poly instance in order to expand the matrix!
    # TODO: maybe we should make heisenberg_obs a class method or a static method to avoid this being a 'hack'?
    op = qml.ops.PolyXP(Q, wires=wires)
    Q = op.heisenberg_obs(total_wires)

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

    d2 = 2 * A @ mu + d
    k2 = mu.T @ A @ mu + mu.T @ d + k

    ex = np.trace(A @ cov) + k2
    var = 2 * np.trace(A @ cov @ A @ cov) + d2.T @ cov @ d2

    modes = np.arange(2 * total_wires).reshape(2, -1).T
    groenewald_correction = np.sum([np.linalg.det(hbar * A[:, m][n]) for m in modes for n in modes])
    var -= groenewald_correction

    return ex, var


def fock_expectation(mu, cov, wires, params, total_wires, hbar=2.0):
    r"""Calculates the expectation and variance of a Fock state probability.

    Args:
        mu (array): length-:math:`2N` vector of means
        cov (array): :math:`2N\times 2N` covariance matrix
        wires (Sequence[int]): wires to calculate the expectation for
        params (Sequence[int]): the Fock state to return the expectation value for
        total_wires (int): total number of wires in the system
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        tuple: the Fock state expectation and variance
    """
    # pylint: disable=unused-argument
    ex = fock_prob(mu, cov, params[0], hbar=hbar)

    # var[|n><n|] = E[|n><n|^2] -  E[|n><n|]^2 = E[|n><n|] -  E[|n><n|]^2
    var = ex - ex ** 2
    return ex, var


def identity(*_, **__):
    r"""Returns 1.

    Returns:
        tuple: the Fock state expectation and variance
    """
    return 1, 0


# ========================================================
#  device
# ========================================================


class DefaultGaussian(Device):
    r"""Default Gaussian device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values.
            If ``analytic == True``, then the number of shots is ignored
            in the calculation of expectation values and variances, and only controls the number
            of samples returned by ``sample``.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
    """
    name = "Default Gaussian PennyLane plugin"
    short_name = "default.gaussian"
    pennylane_requires = "0.9"
    version = "0.9.0"
    author = "Xanadu Inc."

    _capabilities = {"model": "cv"}

    _operation_map = {
        "Beamsplitter": beamsplitter,
        "ControlledAddition": controlled_addition,
        "ControlledPhase": controlled_phase,
        "Displacement": displacement,
        "QuadraticPhase": quadratic_phase,
        "Rotation": rotation,
        "Squeezing": squeezing,
        "TwoModeSqueezing": two_mode_squeezing,
        "CoherentState": coherent_state,
        "DisplacedSqueezedState": displaced_squeezed_state,
        "SqueezedState": squeezed_state,
        "ThermalState": thermal_state,
        "GaussianState": gaussian_state,
        "Interferometer": interferometer,
    }

    _observable_map = {
        "NumberOperator": photon_number,
        "X": homodyne(0),
        "P": homodyne(np.pi / 2),
        "QuadOperator": homodyne(None),
        "PolyXP": poly_quad_expectations,
        "FockStateProjector": fock_expectation,
        "Identity": identity,
    }

    _circuits = {}

    def __init__(self, wires, *, shots=1000, hbar=2, analytic=True):
        super().__init__(wires, shots)
        self.eng = None
        self.hbar = hbar
        self.analytic = analytic

        self.reset()

    def pre_apply(self):
        self.reset()

    def apply(self, operation, wires, par):
        if operation == "Displacement":
            self._state = displacement(self._state, wires[0], par[0] * cmath.exp(1j * par[1]))
            return  # we are done here

        if operation == "GaussianState":
            if wires != list(range(self.num_wires)):
                raise ValueError(
                    "GaussianState means vector or covariance matrix is "
                    "the incorrect size for the number of subsystems."
                )
            self._state = self._operation_map[operation](*par, hbar=self.hbar)
            return  # we are done here

        if "State" in operation:
            # set the new device state
            mu, cov = self._operation_map[operation](*par, hbar=self.hbar)
            # state preparations only act on at most 1 subsystem
            self._state = set_state(self._state, wires[0], mu, cov)
            return  # we are done here

        # get the symplectic matrix
        S = self._operation_map[operation](*par)

        # expand the symplectic to act on the proper subsystem
        S = self.expand(S, wires)

        # apply symplectic matrix to the means vector
        means = S @ self._state[0]
        # apply symplectic matrix to the covariance matrix
        cov = S @ self._state[1] @ S.T

        self._state = [means, cov]

    def expand(self, S, wires):
        r"""Expands a Symplectic matrix S to act on the entire subsystem.

        Args:
            S (array): a :math:`2M\times 2M` Symplectic matrix
            wires (Sequence[int]): the wires of the modes that S acts on

        Returns:
            array: the resulting :math:`2N\times 2N` Symplectic matrix
        """
        if self.num_wires == 1:
            # total number of wires is 1, simply return the matrix
            return S

        N = self.num_wires
        w = np.asarray(wires)

        if np.any(w < 0) or np.any(w >= N) or len(set(w)) != len(w):
            raise ValueError("Invalid target subsystems provided in 'wires' argument.")

        M = len(S) // 2
        S2 = np.identity(2 * N)

        if M != len(wires):
            raise ValueError("Incorrect number of subsystems for provided operation.")

        S2[w.reshape(-1, 1), w.reshape(1, -1)] = S[:M, :M].copy()  # XX
        S2[(w + N).reshape(-1, 1), (w + N).reshape(1, -1)] = S[M:, M:].copy()  # PP
        S2[w.reshape(-1, 1), (w + N).reshape(1, -1)] = S[:M, M:].copy()  # XP
        S2[(w + N).reshape(-1, 1), w.reshape(1, -1)] = S[M:, :M].copy()  # PX

        return S2

    def expval(self, observable, wires, par):
        if observable == "PolyXP":
            mu, cov = self._state
        else:
            mu, cov = self.reduced_state(wires)

        ev, var = self._observable_map[observable](
            mu, cov, wires, par, self.num_wires, hbar=self.hbar
        )

        if not self.analytic:
            # estimate the ev
            # use central limit theorem, sample normal distribution once, only ok if n_eval is large
            # (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ev = np.random.normal(ev, math.sqrt(var / self.shots))

        return ev

    def var(self, observable, wires, par):
        mu, cov = self.reduced_state(wires)
        _, var = self._observable_map[observable](
            mu, cov, wires, par, hbar=self.hbar, total_wires=self.num_wires
        )
        return var

    def sample(self, observable, wires, par):
        """Return a sample of an observable.

        .. note::

            The ``default.gaussian`` plugin only supports sampling
            from :class:`~.X`, :class:`~.P`, and :class:`~.QuadOperator`
            observables.

        Args:
            observable (str): name of the observable
            wires (Sequence[int]): subsystems the observable is to be measured on
            par (tuple): parameters for the observable

        Returns:
            array[float]: samples in an array of dimension ``(n, num_wires)``
        """
        if len(wires) != 1:
            raise ValueError("Only one mode can be measured in homodyne.")

        if observable == "X":
            phi = 0.0
        elif observable == "P":
            phi = np.pi / 2
        elif observable == "QuadOperator":
            phi = par[0]
        else:
            raise NotImplementedError(
                "default.gaussian does not support sampling {}".format(observable)
            )

        mu, cov = self.reduced_state(wires)
        rot = rotation(phi)

        muphi = rot.T @ mu
        covphi = rot.T @ cov @ rot

        stdphi = math.sqrt(covphi[0, 0])
        meanphi = muphi[0]
        return np.random.normal(meanphi, stdphi, self.shots)

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        self._state = vacuum_state(self.num_wires, self.hbar)

    def reduced_state(self, wires):
        r""" Returns the vector of means and the covariance matrix of the specified wires.

        Args:
            wires (int of Sequence[int]): indices of the requested wires

        Returns:
            tuple (means, cov): means is an array containing the vector of means,
            and cov is a square array containing the covariance matrix
        """
        if wires == list(range(self.num_wires)):
            # reduced state is full state
            return self._state

        # reduce rho down to specified subsystems
        if isinstance(wires, int):
            wires = [wires]

        if np.any(np.array(wires) > self.num_wires):
            raise ValueError(
                "The specified wires cannot " "be larger than the number of subsystems."
            )

        ind = np.concatenate([np.array(wires), np.array(wires) + self.num_wires])
        rows = ind.reshape(-1, 1)
        cols = ind.reshape(1, -1)

        return self._state[0][ind], self._state[1][rows, cols]

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())
