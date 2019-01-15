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
r"""
.. _cv_ops:

CV quantum operations
=====================

.. currentmodule:: pennylane.ops.cv

**Module name:** :mod:`pennylane.ops.cv`

This section contains the available built-in continuous-variable
quantum operations supported by PennyLane, as well as their conventions.

.. todo:: Add gradient recipes for Gaussian state preparations

.. todo::

   The gradient computation assumes all parameters are real (floats), some
   docstrings here allow complex or even array parameter values. This includes
   :class:`~.DisplacedSqueezedState` and :class:`~.CatState`.

   Possible solution: disallow such operations to depend on free parameters,
   this way they won't be differentiated.

.. note::

   For the Heisenberg matrix representation of CV operations, we use the ordering
   :math:`(\hat{\mathbb{1}}, \hat{x}, \hat{p})` for single modes
   and :math:`(\hat{\mathbb{1}}, \hat{x}_1, \hat{p}_2, \hat{x}_1,\hat{p}_2)` for two modes .

Gates
-----

.. autosummary::
    Rotation
    Squeezing
    Displacement
    Beamsplitter
    TwoModeSqueezing
    QuadraticPhase
    ControlledAddition
    ControlledPhase
    Kerr
    CrossKerr
    CubicPhase


State preparation
-----------------

.. autosummary::
    CoherentState
    SqueezedState
    DisplacedSqueezedState
    ThermalState
    GaussianState
    FockState
    FockStateVector
    FockDensityMatrix
    CatState


Code details
~~~~~~~~~~~~
"""
import numpy as np
import scipy as sp

from pennylane.operation import CVOperation


def _rotation(phi, bare=False):
    r"""Utility function, returns the Heisenberg transformation of a phase rotation gate.

    The transformation matrix returned is:

    .. math:: M = \begin{bmatrix}
        1 & 0 & 0\\
        0 & \cos\phi & -\sin\phi\\
        0 & \sin\phi & \cos\phi
        \end{bmatrix}

    Args:
        phi (float): rotation angle.
        bare (bool): if True, return a simple 2d rotation matrix

    Returns:
        array[float]: transformation matrix
    """
    c = np.cos(phi)
    s = np.sin(phi)
    temp = np.array([[c, -s], [s, c]])
    if bare:
        return temp
    return sp.linalg.block_diag(1, temp) # pylint: disable=no-member


class Rotation(CVOperation):
    r"""pennylane.Rotation(phi, wires)
    Phase space rotation

    .. math::
        R(\phi) = \exp\left(i \phi \ad \a\right)=\exp\left(i \frac{\phi}{2}
        \left(\frac{\x^2+  \p^2}{\hbar}-\I\right)\right).

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{dr}R(r) = \frac{1}{2} \left[R(\phi+\pi/2) - R(\phi-\pi/2)\right]`
    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
        1 & 0 & 0\\
        0 & \cos\phi & -\sin\phi\\
        0 & \sin\phi & \cos\phi
        \end{bmatrix}

    Args:
        phi (float): the rotation angle
    """
    num_wires = 1
    num_params = 1
    par_domain = 'R'
    grad_method = 'A'

    @staticmethod
    def _heisenberg_rep(p):
        return _rotation(p[0])


class Squeezing(CVOperation):
    r"""pennylane.Squeezing(r, phi, wires)
    Phase space squeezing.

    .. math::
        S(z) = \exp\left(\frac{1}{2}(z^* a^2 -z {a^\dagger}^2)\right).

    where :math:`z = r e^{i\phi}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{dr}S(r,\phi) = \frac{1}{2\sinh s} \left[S(r+s, \phi) - S(r-s, \phi)\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default).
    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
        1 & 0 & 0 \\
        0 & \cosh r - \cos\phi \sinh r & -\sin\phi\sinh r \\
        0 & -\sin\phi\sinh r & \cosh r+\cos\phi\sinh r
        \end{bmatrix}

    Args:
        r (float): squeezing amount
        phi (float): squeezing phase angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 2
    par_domain = 'R'
    grad_method = 'A'

    shift = 0.1
    grad_recipe = [(0.5/np.sinh(shift), shift), None]

    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1] / 2)
        return R @ np.diag([1, np.exp(-p[0]), np.exp(p[0])]) @ R.T


class Displacement(CVOperation):
    r"""pennylane.Displacement(r, phi, wires)
    Phase space displacement.

    .. math::
       D(r,\phi) = D(\alpha) = \exp(\alpha a^\dagger -\alpha^* a)
       = \exp\left(-i\sqrt{\frac{2}{\hbar}}(\re(\alpha) \hat{p} -\im(\alpha) \hat{x})/\right).

    where :math:`\alpha = r e^{i\phi}` has magnitude :math:`r\geq 0` and phase :math:`\phi`.
    The result of applying a displacement to the vacuum is a coherent state
    :math:`D(\alpha)\ket{0} = \ket{\alpha}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{dr}D(r,\phi) = \frac{1}{2s} \left[D(r+s, \phi) - D(r-s, \phi)\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default)
    * Heisenberg representation:

      .. math:: M = \begin{bmatrix} 1 & 0 & 0 \\ 2r\cos\phi & 1 & 0 \\ 2r\sin\phi & 0 & 1\end{bmatrix}

    Args:
        r (float): displacement magnitude :math:`r=|\alpha|`
        phi (float): phase angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 2
    par_domain = 'R'
    grad_method = 'A'

    shift = 0.1
    grad_recipe = [(0.5/shift, shift), None]

    @staticmethod
    def _heisenberg_rep(p):
        c = np.cos(p[1])
        s = np.sin(p[1])
        scale = 2  # sqrt(2 * hbar)
        return np.array([[1, 0, 0], [scale * c * p[0], 1, 0], [scale * s * p[0], 0, 1]])


class Beamsplitter(CVOperation):
    r"""pennylane.Beamsplitter(theta, phi, wires)
    Beamsplitter interaction.

    .. math::
        B(\theta,\phi) = \exp\left(\theta (e^{i \phi} a b^\dagger -e^{-i \phi}a^\dagger b) \right).

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{dr}B(r,\phi) = \frac{1}{2} \left[B(\theta+\pi/2, \phi) - B(\theta-\pi/2, \phi)\right]`
    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0\\
            0 & \cos\theta & 0 & -\cos\phi\sin\theta & -\sin\phi\sin\theta \\
            0 & 0 & \cos\theta & \sin\phi\sin\theta & -\cos\phi\sin\theta\\
            0 & \cos\phi\sin\theta & -\sin\phi\sin\theta & \cos\theta & 0\\
            0 & \sin\phi\sin\theta & \cos\phi\sin\theta & 0 & \cos\theta
        \end{bmatrix}

    Args:
        theta (float): Transmittivity angle :math:`\theta`. The transmission amplitude
            of the beamsplitter is :math:`t = \cos(\theta)`.
            The value :math:`\theta=\pi/4` gives the 50-50 beamsplitter.
        phi (float): Phase angle :math:`\phi`. The reflection amplitude of the
            beamsplitter is :math:`r = e^{i\phi}\sin(\theta)`.
            The value :math:`\phi = \pi/2` gives the symmetric beamsplitter.
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 2
    num_wires = 2
    par_domain = 'R'
    grad_method = 'A'

    # For the beamsplitter, both parameters are rotation-like
    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1], bare=True)
        c = np.cos(p[0])
        s = np.sin(p[0])
        U = c * np.eye(5)
        U[0, 0] = 1
        U[1:3, 3:5] = -s * R.T
        U[3:5, 1:3] = s * R
        return U


class TwoModeSqueezing(CVOperation):
    r"""pennylane.TwoModeSqueezing(r, phi, wires)
    Phase space two-mode squeezing.

    .. math::
        S_2(z) = \exp\left(z^* ab -z a^\dagger b^\dagger \right)
        = \exp\left(r (e^{-i\phi} ab -e^{i\phi} a^\dagger b^\dagger \right).

    where :math:`z = r e^{i\phi}`.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{dr}S_2(r,\phi) = \frac{1}{2\sinh s} \left[S_2(r+s, \phi) - S_2(r-s, \phi)\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default).

    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 \\
            0 & \cosh r & 0 & \sinh r \cos \phi & \sinh r \sin \phi\\
            0 & 0 & \cosh r & \sinh r \sin \phi & -\sinh r \cos \phi\\
            0 & \sinh r \cos \phi & \sinh r \sin \phi & \cosh r & 0\\
            0 & \sinh r \sin \phi & -\sinh r \cos \phi & 0 & \cosh r
        \end{bmatrix}

    Args:
        r (float): squeezing amount
        phi (float): squeezing phase angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 2
    num_wires = 2
    par_domain = 'R'

    grad_method = 'A'
    shift = 0.1
    grad_recipe = [(0.5/np.sinh(shift), shift), None]

    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1], bare=True)

        S = np.sinh(p[0]) * np.diag([1, -1])
        U = np.cosh(p[0]) * np.identity(5)

        U[0, 0] = 1
        U[1:3, 3:5] = S @ R.T
        U[3:5, 1:3] = S @ R.T
        return U


class QuadraticPhase(CVOperation):
    r"""pennylane.QuadraticPhase(s, wires)
    Quadratic phase shift.

    .. math::
        P(s) = e^{i \frac{s}{2} \hat{x}^2/\hbar}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)
    * Gradient recipe: :math:`\frac{d}{dr}S_2(r,\phi) = \frac{1}{2\sinh s} \left[S_2(r+s, \phi) - S_2(r-s, \phi)\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default).

    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            0 & s & 1 \\
        \end{bmatrix}

    Args:
        s (float): parameter
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = 'R'

    grad_method = 'A'
    shift = 0.1
    grad_recipe = [(0.5/shift, shift)]

    @staticmethod
    def _heisenberg_rep(p):
        U = np.identity(3)
        U[2, 1] = p[0]
        return U


class ControlledAddition(CVOperation):
    r"""pennylane.ControlledAddition(s, wires)
    Controlled addition operation.

    .. math::
           \text{CX}(s) = \int dx \ket{x}\bra{x} \otimes D\left({\frac{1}{\sqrt{2\hbar}}}s x\right)
           = e^{-i s \: \hat{x} \otimes \hat{p}/\hbar}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)
    * Gradient recipe: :math:`\frac{d}{dr}S_2(r,\phi) = \frac{1}{2\sinh s} \left[S_2(r+s, \phi) - S_2(r-s, \phi)\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default).

    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & -s \\
            0 & s & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 1
        \end{bmatrix}

    Args:
        s (float): addition multiplier
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 2
    num_params = 1
    par_domain = 'R'

    grad_method = 'A'
    shift = 0.1
    grad_recipe = [(0.5/shift, shift)]

    @staticmethod
    def _heisenberg_rep(p):
        U = np.identity(5)
        U[2, 4] = -p[0]
        U[3, 1] = p[0]
        return U


class ControlledPhase(CVOperation):
    r"""pennylane.ControlledPhase(s, wires)
    Controlled phase operation.

    .. math::
           \text{CX}(s) =  \iint dx dy \: e^{i sxy/\hbar} \ket{x,y}\bra{x,y}
           = e^{i s \: \hat{x} \otimes \hat{x}/\hbar}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)
    * Gradient recipe: :math:`\frac{d}{dr}S_2(r,\phi) = \frac{1}{2\sinh s} \left[S_2(r+s, \phi) - S_2(r-s, \phi)\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default).

    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 1 & s & 0 \\
            0 & 0 & 0 & 1 & 0 \\
            0 & s & 0 & 0 & 1
        \end{bmatrix}

    Args:
        s (float):  phase shift multiplier
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 2
    num_params = 1
    par_domain = 'R'

    grad_method = 'A'
    shift = 0.1
    grad_recipe = [(0.5/shift, shift)]

    @staticmethod
    def _heisenberg_rep(p):
        U = np.identity(5)
        U[2, 3] = p[0]
        U[4, 1] = p[0]
        return U


class Kerr(CVOperation):
    r"""pennylane.Kerr(kappa, wires)
    Kerr interaction.

    .. math::
        K(\kappa) = e^{i \kappa \hat{n}^2}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        kappa (float): parameter
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = 'R'
    grad_method = 'F'


class CrossKerr(CVOperation):
    r"""pennylane.CrossKerr(kappa, wires)
    Cross-Kerr interaction.

    .. math::
        CK(\kappa) = e^{i \kappa \hat{n}_1\hat{n}_2}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        kappa (float): parameter
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = 'R'
    grad_method = 'F'


class CubicPhase(CVOperation):
    r"""pennylane.CubicPhase(gamma, wires)
    Cubic phase shift.

    .. math::
        V(\gamma) = e^{i \frac{\gamma}{3} \hat{x}^3/\hbar}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        gamma (float): parameter
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = 'R'
    grad_method = 'F'


#=============================================================================
# State preparation
#=============================================================================

#TODO: put Heisenberg reps of state preparations in docstrings?

class CoherentState(CVOperation):
    r"""pennylane.CoherentState(a, phi, wires)
    Prepares a coherent state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: None (uses finite difference)

    Args:
        a (float): displacement magnitude :math:`r=|\alpha|`
        phi (float): phase angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 2
    par_domain = 'R'
    grad_method = 'F'


class SqueezedState(CVOperation):
    r"""pennylane.SqueezedState(r, phi, wires)
    Prepares a squeezed vacuum state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: None (uses finite difference)

    Args:
        r (float): squeezing magnitude
        phi (float): squeezing angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 2
    par_domain = 'R'
    grad_method = 'F'


class DisplacedSqueezedState(CVOperation):
    r"""pennylane.DisplacedSqueezedState(a, phi_a, r, phi_r, wires)
    Prepares a displaced squeezed vacuum state.

    A displaced squeezed state is prepared by squeezing a vacuum state, and
    then applying a displacement operator,

    .. math::
       \ket{\alpha,z} = D(\alpha)\ket{0,z} = D(\alpha)S(z)\ket{0},

    with the displacement parameter :math:`\alpha=ae^{i\phi_a}` and the squeezing parameter :math:`z=re^{i\phi_r}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Gradient recipe: None (uses finite difference)

    Args:
        a (float): displacement magnitude :math:`a=|\alpha|`
        phi_a (float): displacement angle :math:`\phi_a`
        r (float): squeezing magnitude :math:`r=|z|`
        phi_r (float): squeezing angle :math:`\phi_r`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 4
    par_domain = 'R'
    grad_method = 'F'


class ThermalState(CVOperation):
    r"""pennylane.ThermalState(nbar, wires)
    Prepares a thermal state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        nbar (float): mean thermal population of the mode
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 1
    par_domain = 'R'
    grad_method = 'F'


class GaussianState(CVOperation):
    r"""pennylane.GaussianState(r, V, wires)
    Prepare subsystems in a given Gaussian state.

    **Details:**

    * Number of wires: None (applied to the entire system)
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        r (array): a length :math:`2N` vector of means, of the
            form :math:`(\x_0,\dots,\x_{N-1},\p_0,\dots,\p_{N-1})`
        V (array): the :math:`2N\times 2N` (real and positive definite) covariance matrix
    """
    num_wires = 0
    num_params = 2
    par_domain = 'A'
    grad_method = 'F'


class FockState(CVOperation):
    r"""pennylane.FockState(n, wires)
    Prepares a single Fock state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (not differentiable)

    Args:
        n (int): Fock state to prepare
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 1
    par_domain = 'N'
    grad_method = None


class FockStateVector(CVOperation):
    r"""pennylane.FockStateVector(state, wires)
    Prepare subsystems using the given ket vector in the Fock basis.

    **Details:**

    * Number of wires: None (applied to the entire system)
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        state (array): a single ket vector, for single mode state preparation,
            or a multimode ket, with one array dimension per mode
    """
    num_wires = 0
    num_params = 1
    par_domain = 'A'
    grad_method = 'F'


class FockDensityMatrix(CVOperation):
    r"""pennylane.FockDensityMatrix(state, wires)
    Prepare subsystems using the given density matrix in the Fock basis.

    **Details:**

    * Number of wires: None (applied to the entire system)
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        state (array): a single mode matrix :math:`\rho_{ij}`, or
            a multimode tensor :math:`\rho_{ij,kl,\dots,mn}`, with two indices per mode
    """
    num_wires = 0
    num_params = 1
    par_domain = 'A'
    grad_method = 'F'


class CatState(CVOperation):
    r"""pennylane.CatState(a, phi, p, wires)
    Prepares a cat state.

    A cat state is the coherent superposition of two coherent states,

    .. math::
       \ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{ip\pi} \ket{-\alpha}),

    where :math:`\ket{\pm\alpha} = D(\pm\alpha)\ket{0}` are a coherent states with displacement
    parameters :math:`\pm\alpha=\pm ae^{i\phi}` and
    :math:`N = \sqrt{2 (1+\cos(p\pi)e^{-2|\alpha|^2})}` is the normalization factor.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Gradient recipe: None (uses finite difference)

    Args:
        a (float): displacement magnitude :math:`a=|\alpha|`
        phi (float): displacement angle :math:`\phi`
        p (float): parity, where :math:`p=0` corresponds to an even
            cat state, and :math:`p=1` an odd cat state.
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 2
    par_domain = 'R'
    grad_method = 'F'


all_ops = [
    Beamsplitter,
    ControlledAddition,
    ControlledPhase,
    Displacement,
    Kerr,
    CrossKerr,
    QuadraticPhase,
    Rotation,
    Squeezing,
    TwoModeSqueezing,
    CubicPhase,
    CatState,
    CoherentState,
    FockDensityMatrix,
    DisplacedSqueezedState,
    FockState,
    FockStateVector,
    SqueezedState,
    ThermalState,
    GaussianState
]


__all__ = [cls.__name__ for cls in all_ops]
