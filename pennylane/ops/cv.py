# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-arguments
r"""
This module contains the available built-in continuous-variable
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
"""
# As the qubit based ``decomposition``, ``_matrix``, ``diagonalizing_gates``
# abstract methods are not defined in the CV case, disabling the related check
# pylint: disable=abstract-method
import math

import numpy as np
from scipy.linalg import block_diag

from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation

from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import

_two_term_shift_rule = [[0.5, 1, np.pi / 2], [-0.5, 1, -np.pi / 2]]


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
    c = math.cos(phi)
    s = math.sin(phi)
    temp = np.array([[c, -s], [s, c]])
    if bare:
        return temp
    return block_diag(1, temp)  # pylint: disable=no-member


class Rotation(CVOperation):
    r"""
    Phase space rotation.

    .. math::
        R(\phi) = \exp\left(i \phi \ad \a\right)=\exp\left(i \frac{\phi}{2}
        \left(\frac{\x^2+  \p^2}{\hbar}-\I\right)\right).

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{dr}f(R(r)) = \frac{1}{2} \left[f(R(\phi+\pi/2)) - f(R(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R(r)`.
    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
        1 & 0 & 0\\
        0 & \cos\phi & -\sin\phi\\
        0 & \sin\phi & \cos\phi
        \end{bmatrix}

    Args:
        phi (float): the rotation angle
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = "A"
    grad_recipe = (_two_term_shift_rule,)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        return _rotation(p[0])

    def adjoint(self):
        return Rotation(-self.parameters[0], wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "R", cache=cache)


class Squeezing(CVOperation):
    r"""
    Phase space squeezing.

    .. math::
        S(z) = \exp\left(\frac{1}{2}(z^* \a^2 -z {\a^\dagger}^2)\right).

    where :math:`z = r e^{i\phi}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{dr}f(S(r,\phi)) = \frac{1}{2\sinh s} \left[f(S(r+s, \phi)) - f(S(r-s, \phi))\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`S(r,\phi)`.
    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
        1 & 0 & 0 \\
        0 & \cosh r - \cos\phi \sinh r & -\sin\phi\sinh r \\
        0 & -\sin\phi\sinh r & \cosh r+\cos\phi\sinh r
        \end{bmatrix}

    Args:
        r (float): squeezing amount
        phi (float): squeezing phase angle :math:`\phi`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 2
    num_wires = 1
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / math.sinh(shift)
    a = 1
    grad_recipe = ([[multiplier, a, shift], [-multiplier, a, -shift]], _two_term_shift_rule)

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1] / 2)
        return R @ np.diag([1, math.exp(-p[0]), math.exp(p[0])]) @ R.T

    def adjoint(self):
        r, phi = self.parameters
        new_phi = (phi + np.pi) % (2 * np.pi)
        return Squeezing(r, new_phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "S", cache=cache)


class Displacement(CVOperation):
    r"""
    Phase space displacement.

    .. math::
       D(a,\phi) = D(\alpha) = \exp(\alpha \ad -\alpha^* \a)
       = \exp\left(-i\sqrt{\frac{2}{\hbar}}(\re(\alpha) \hat{p} -\im(\alpha) \hat{x})\right).

    where :math:`\alpha = ae^{i\phi}` has magnitude :math:`a\geq 0` and phase :math:`\phi`.
    The result of applying a displacement to the vacuum is a coherent state
    :math:`D(\alpha)\ket{0} = \ket{\alpha}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{da}f(D(a,\phi)) = \frac{1}{2s} \left[f(D(a+s, \phi)) - f(D(a-s, \phi))\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`D(a,\phi)`.
    * Heisenberg representation:

      .. math:: M = \begin{bmatrix} 1 & 0 & 0 \\ 2a\cos\phi & 1 & 0 \\ 2a\sin\phi & 0 & 1\end{bmatrix}

    Args:
        a (float): displacement magnitude :math:`a=|\alpha|`
        phi (float): phase angle :math:`\phi`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 2
    num_wires = 1
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / shift
    a = 1
    grad_recipe = ([[multiplier, a, shift], [-multiplier, a, -shift]], _two_term_shift_rule)

    def __init__(self, a, phi, wires, id=None):
        super().__init__(a, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        c = math.cos(p[1])
        s = math.sin(p[1])
        scale = 2  # sqrt(2 * hbar)
        return np.array([[1, 0, 0], [scale * c * p[0], 1, 0], [scale * s * p[0], 0, 1]])

    def adjoint(self):
        a, phi = self.parameters
        new_phi = (phi + np.pi) % (2 * np.pi)
        return Displacement(a, new_phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "D", cache=cache)


class Beamsplitter(CVOperation):
    r"""
    Beamsplitter interaction.

    .. math::
        B(\theta,\phi) = \exp\left(\theta (e^{i \phi} \a \hat{b}^\dagger -e^{-i \phi}\ad \hat{b}) \right).

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{d \theta}f(B(\theta,\phi)) = \frac{1}{2} \left[f(B(\theta+\pi/2, \phi)) - f(B(\theta-\pi/2, \phi))\right]`
      where :math:`f` is an expectation value depending on :math:`B(\theta,\phi)`.
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
        wires (Sequence[Any]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 2
    num_wires = 2
    grad_method = "A"
    grad_recipe = (_two_term_shift_rule, _two_term_shift_rule)

    def __init__(self, theta, phi, wires, id=None):
        super().__init__(theta, phi, wires=wires, id=id)

    # For the beamsplitter, both parameters are rotation-like
    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1], bare=True)
        c = math.cos(p[0])
        s = math.sin(p[0])
        U = c * np.eye(5)
        U[0, 0] = 1
        U[1:3, 3:5] = -s * R.T
        U[3:5, 1:3] = s * R
        return U

    def adjoint(self):
        theta, phi = self.parameters
        return Beamsplitter(-theta, phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "BS", cache=cache)


class TwoModeSqueezing(CVOperation):
    r"""
    Phase space two-mode squeezing.

    .. math::
        S_2(z) = \exp\left(z^* \a \hat{b} -z \ad \hat{b}^\dagger \right)
        = \exp\left(r (e^{-i\phi} \a\hat{b} -e^{i\phi} \ad \hat{b}^\dagger \right).

    where :math:`z = r e^{i\phi}`.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{dr}f(S_2(r,\phi)) = \frac{1}{2\sinh s} \left[f(S_2(r+s, \phi)) - f(S_2(r-s, \phi))\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`S_2(r,\phi)`.

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
        wires (Sequence[Any]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 2
    num_wires = 2

    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / math.sinh(shift)
    a = 1
    grad_recipe = ([[multiplier, a, shift], [-multiplier, a, -shift]], _two_term_shift_rule)

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1], bare=True)

        S = math.sinh(p[0]) * np.diag([1, -1])
        U = math.cosh(p[0]) * np.identity(5)

        U[0, 0] = 1
        U[1:3, 3:5] = S @ R.T
        U[3:5, 1:3] = S @ R.T
        return U

    def adjoint(self):
        r, phi = self.parameters
        new_phi = (phi + np.pi) % (2 * np.pi)
        return TwoModeSqueezing(r, new_phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "S", cache=cache)


class QuadraticPhase(CVOperation):
    r"""
    Quadratic phase shift.

    .. math::
        P(s) = e^{i \frac{s}{2} \hat{x}^2/\hbar}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{ds}f(P(s)) = \frac{1}{2 a} \left[f(P(s+a)) - f(P(s-a))\right]`,
      where :math:`a` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`P(s)`.

    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            0 & s & 1 \\
        \end{bmatrix}

    Args:
        s (float): parameter
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1

    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / shift
    a = 1
    grad_recipe = ([[multiplier, a, shift], [-multiplier, a, -shift]],)

    def __init__(self, s, wires, id=None):
        super().__init__(s, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        U = np.identity(3)
        U[2, 1] = p[0]
        return U

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "P", cache=cache)


class ControlledAddition(CVOperation):
    r"""
    Controlled addition operation.

    .. math::
           \text{CX}(s) = \int dx \ket{x}\bra{x} \otimes D\left({\frac{1}{\sqrt{2\hbar}}}s x\right)
           = e^{-i s \: \hat{x} \otimes \hat{p}/\hbar}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{ds}f(\text{CX}(s)) = \frac{1}{2 a} \left[f(\text{CX}(s+a)) - f(\text{CX}(s-a))\right]`,
      where :math:`a` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`\text{CX}(s)`.

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
        wires (Sequence[Any]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 2
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / shift
    a = 1
    grad_recipe = ([[multiplier, a, shift], [-multiplier, a, -shift]],)

    def __init__(self, s, wires, id=None):
        super().__init__(s, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        U = np.identity(5)
        U[2, 4] = -p[0]
        U[3, 1] = p[0]
        return U

    def adjoint(self):
        return ControlledAddition(-self.parameters[0], wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "X", cache=cache)


class ControlledPhase(CVOperation):
    r"""
    Controlled phase operation.

    .. math::
           \text{CZ}(s) =  \iint dx dy \: e^{i sxy/\hbar} \ket{x,y}\bra{x,y}
           = e^{i s \: \hat{x} \otimes \hat{x}/\hbar}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{ds}f(\text{CZ}(s)) = \frac{1}{2 a} \left[f(\text{CZ}(s+a)) - f(\text{CZ}(s-a))\right]`,
      where :math:`a` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`\text{CZ}(s)`.

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
        wires (Sequence[Any]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 2
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / shift
    a = 1
    grad_recipe = ([[multiplier, a, shift], [-multiplier, a, -shift]],)

    def __init__(self, s, wires, id=None):
        super().__init__(s, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        U = np.identity(5)
        U[2, 3] = p[0]
        U[4, 1] = p[0]
        return U

    def adjoint(self):
        return ControlledPhase(-self.parameters[0], wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "Z", cache=cache)


class Kerr(CVOperation):
    r"""
    Kerr interaction.

    .. math::
        K(\kappa) = e^{i \kappa \hat{n}^2}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        kappa (float): parameter
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = "F"

    def __init__(self, kappa, wires, id=None):
        super().__init__(kappa, wires=wires, id=id)

    def adjoint(self):
        return Kerr(-self.parameters[0], wires=self.wires)


class CrossKerr(CVOperation):
    r"""
    Cross-Kerr interaction.

    .. math::
        CK(\kappa) = e^{i \kappa \hat{n}_1\hat{n}_2}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        kappa (float): parameter
        wires (Sequence[Any]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 2
    grad_method = "F"

    def __init__(self, kappa, wires, id=None):
        super().__init__(kappa, wires=wires, id=id)

    def adjoint(self):
        return CrossKerr(-self.parameters[0], wires=self.wires)


class CubicPhase(CVOperation):
    r"""
    Cubic phase shift.

    .. math::
        V(\gamma) = e^{i \frac{\gamma}{3} \hat{x}^3/\hbar}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        gamma (float): parameter
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = "F"

    def __init__(self, gamma, wires, id=None):
        super().__init__(gamma, wires=wires, id=id)

    def adjoint(self):
        return CubicPhase(-self.parameters[0], wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "V", cache=cache)


class InterferometerUnitary(CVOperation):
    r"""
    A linear interferometer transforming the bosonic operators according to
    the unitary matrix :math:`U`.

    .. note::

        This operation implements a **fixed** linear interferometer given a known
        unitary matrix.

        If you instead wish to parameterize the interferometer,
        and calculate the gradient/optimize with respect to these parameters,
        consider instead the :func:`pennylane.template.Interferometer` template,
        which constructs an interferometer from a combination of beamsplitters
        and rotation gates.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None
    * Heisenberg representation:

      .. math:: M = \begin{bmatrix}
        1 & 0\\
        0 & S\\
        \end{bmatrix}

    where :math:`S` is the Gaussian symplectic transformation representing the interferometer.

    Args:
        U (array): A shape ``(len(wires), len(wires))`` complex unitary matrix
        wires (Sequence[Any] or Any): the wires the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = AnyWires
    grad_method = None
    grad_recipe = None

    def __init__(self, U, wires, id=None):
        super().__init__(U, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        N = len(p[0])
        A = p[0].real
        B = p[0].imag

        rows = np.arange(2 * N).reshape(2, -1).T.flatten()
        S = np.vstack([np.hstack([A, -B]), np.hstack([B, A])])[:, rows][rows]

        M = np.eye(2 * N + 1)
        M[1 : 2 * N + 1, 1 : 2 * N + 1] = S
        return M

    def adjoint(self):
        U = self.parameters[0]
        return InterferometerUnitary(qml_math.T(qml_math.conj(U)), wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


# =============================================================================
# State preparation
# =============================================================================

# TODO: put Heisenberg reps of state preparations in docstrings?


class CoherentState(CVOperation):
    r"""
    Prepares a coherent state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: None (uses finite difference)

    Args:
        a (float): displacement magnitude :math:`r=|\alpha|`
        phi (float): phase angle :math:`\phi`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 2
    num_wires = 1
    grad_method = "F"

    def __init__(self, a, phi, wires, id=None):
        super().__init__(a, phi, wires=wires, id=id)


class SqueezedState(CVOperation):
    r"""
    Prepares a squeezed vacuum state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: None (uses finite difference)

    Args:
        r (float): squeezing magnitude
        phi (float): squeezing angle :math:`\phi`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 2
    num_wires = 1
    grad_method = "F"

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)


class DisplacedSqueezedState(CVOperation):
    r"""
    Prepares a displaced squeezed vacuum state.

    A displaced squeezed state is prepared by squeezing a vacuum state, and
    then applying a displacement operator,

    .. math::
       \ket{\alpha,z} = D(\alpha)\ket{0,z} = D(\alpha)S(z)\ket{0},

    with the displacement parameter :math:`\alpha=ae^{i\phi_a}` and the squeezing parameter :math:`z=re^{i\phi_r}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 4
    * Gradient recipe: None (uses finite difference)

    Args:
        a (float): displacement magnitude :math:`a=|\alpha|`
        phi_a (float): displacement angle :math:`\phi_a`
        r (float): squeezing magnitude :math:`r=|z|`
        phi_r (float): squeezing angle :math:`\phi_r`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 4
    num_wires = 1
    grad_method = "F"

    def __init__(self, a, phi_a, r, phi_r, wires, id=None):
        super().__init__(a, phi_a, r, phi_r, wires=wires, id=id)


class ThermalState(CVOperation):
    r"""
    Prepares a thermal state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        nbar (float): mean thermal population of the mode
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = "F"

    def __init__(self, nbar, wires, id=None):
        super().__init__(nbar, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "Thermal", cache=cache)


class GaussianState(CVOperation):
    r"""
    Prepare subsystems in a given Gaussian state.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 2
    * Gradient recipe: None

    Args:
        V (array): the :math:`2N\times 2N` (real and positive definite) covariance matrix
        r (array): a length :math:`2N` vector of means, of the
            form :math:`(\x_0,\dots,\x_{N-1},\p_0,\dots,\p_{N-1})`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 2
    num_wires = AnyWires
    grad_method = "F"

    def __init__(self, V, r, wires, id=None):
        super().__init__(V, r, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "Gaussian", cache=cache)


class FockState(CVOperation):
    r"""
    Prepares a single Fock state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (not differentiable)

    Args:
        n (int): Fock state to prepare
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = None

    def __init__(self, n, wires, id=None):
        super().__init__(n, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> qml.FockState(7, wires=0).label()
        '|7⟩'

        """
        if base_label is not None:
            if decimals is None:
                return base_label
            p = format(qml_math.asarray(self.parameters[0]), ".0f")
            return base_label + f"\n({p})"
        return f"|{qml_math.asarray(self.parameters[0])}⟩"


class FockStateVector(CVOperation):
    r"""
    Prepare subsystems using the given ket vector in the Fock basis.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        state (array): a single ket vector, for single mode state preparation,
            or a multimode ket, with one array dimension per mode
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    .. details::
        :title: Usage Details

        For a single mode with cutoff dimension :math:`N`, the input is a
        1-dimensional vector of length :math:`N`.

        .. code-block::

            dev_fock = qml.device("strawberryfields.fock", wires=4, cutoff_dim=4)

            state = np.array([0, 0, 1, 0])

            @qml.qnode(dev_fock)
            def circuit():
                qml.FockStateVector(state, wires=0)
                return qml.expval(qml.NumberOperator(wires=0))

        For multiple modes, the input is the tensor product of single mode
        kets. For example, given a set of :math:`M` single mode vectors of
        length :math:`N`, the input should have shape ``(N, ) * M``.

        .. code-block::

            used_wires = [0, 3]
            cutoff_dim = 5

            dev_fock = qml.device("strawberryfields.fock", wires=4, cutoff_dim=cutoff_dim)

            state_1 = np.array([0, 1, 0, 0, 0])
            state_2 = np.array([0, 0, 0, 1, 0])

            combined_state = np.kron(state_1, state_2).reshape(
                (cutoff_dim, ) * len(used_wires)
            )

            @qml.qnode(dev_fock)
            def circuit():
                qml.FockStateVector(combined_state, wires=used_wires)
                return qml.expval(qml.NumberOperator(wires=0))

    """

    num_params = 1
    num_wires = AnyWires
    grad_method = "F"

    def __init__(self, state, wires, id=None):
        super().__init__(state, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> qml.FockStateVector([1,2,3], wires=(0,1,2)).label()
        '|123⟩'

        """
        if base_label is not None:
            return base_label
        basis_string = "".join(str(int(i)) for i in self.parameters[0])
        return f"|{basis_string}⟩"


class FockDensityMatrix(CVOperation):
    r"""
    Prepare subsystems using the given density matrix in the Fock basis.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        state (array): a single mode matrix :math:`\rho_{ij}`, or
            a multimode tensor :math:`\rho_{ij,kl,\dots,mn}`, with two indices per mode
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = AnyWires
    grad_method = "F"

    def __init__(self, state, wires, id=None):
        super().__init__(state, wires=wires, id=id)


class CatState(CVOperation):
    r"""
    Prepares a cat state.

    A cat state is the coherent superposition of two coherent states,

    .. math::
       \ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{ip\pi} \ket{-\alpha}),

    where :math:`\ket{\pm\alpha} = D(\pm\alpha)\ket{0}` are coherent states with displacement
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
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 3
    num_wires = 1
    grad_method = "F"

    def __init__(self, a, phi, p, wires, id=None):
        super().__init__(a, phi, p, wires=wires, id=id)


# =============================================================================
# Observables
# =============================================================================


class NumberOperator(CVObservable):
    r"""
    The photon number observable :math:`\langle \hat{n}\rangle`.

    The number operator is defined as
    :math:`\hat{n} = \a^\dagger \a = \frac{1}{2\hbar}(\x^2 +\p^2) -\I/2`.

    When used with the :func:`~pennylane.expval` function, the mean
    photon number :math:`\braket{\hat{n}}` is returned.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Observable order: 2nd order in the quadrature operators
    * Heisenberg representation:

      .. math:: M = \frac{1}{2\hbar}\begin{bmatrix}
            -\hbar & 0 & 0\\
            0 & 1 & 0\\
            0 & 0 & 1
        \end{bmatrix}

    Args:
        wires (Sequence[Any] or Any): the wire the operation acts on
    """

    num_params = 0
    num_wires = 1

    ev_order = 2

    def __init__(self, wires):
        super().__init__(wires=wires)

    @staticmethod
    def _heisenberg_rep(p):
        hbar = 2
        return np.diag([-0.5, 0.5 / hbar, 0.5 / hbar])

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "n"


class TensorN(CVObservable):
    r"""
    The tensor product of the :class:`~.NumberOperator` acting on different wires.

    If a single wire is defined, returns a :class:`~.NumberOperator` instance for convenient gradient computations.

    When used with the :func:`~pennylane.expval` function, the expectation value
    :math:`\langle \hat{n}_{i_0} \hat{n}_{i_1}\dots \hat{n}_{i_{N-1}}\rangle`
    for a (sub)set of modes :math:`[i_0, i_1, \dots, i_{N-1}]` of the system is
    returned.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 0

    Args:
        wires (Sequence[Any] or Any): the wire the operation acts on

    .. details::
        :title: Usage Details

        Example for multiple modes:

        >>> cv_obs = qml.TensorN(wires=[0, 1])
        >>> cv_obs
        TensorN(wires=[0, 1])
        >>> cv_obs.ev_order is None
        True

        Example for a single mode (yields a :class:`~.NumberOperator`):

        >>> cv_obs = qml.TensorN(wires=[1])
        >>> cv_obs
        NumberOperator(wires=[1])
        >>> cv_obs.ev_order
        2
    """

    num_params = 0
    num_wires = AnyWires
    ev_order = None

    def __init__(self, wires):
        super().__init__(wires=wires)

    def __new__(cls, wires=None):
        # Custom definition for __new__ needed such that a NumberOperator can
        # be returned when a single mode is defined

        if wires is not None and (isinstance(wires, int) or len(wires) == 1):
            return NumberOperator(wires=wires)

        return super().__new__(cls)

    def label(self, decimals=None, base_label=None, cache=None):
        if base_label is not None:
            return base_label
        return "⊗".join("n" for _ in self.wires)


class QuadX(CVObservable):
    r"""
    The position quadrature observable :math:`\hat{x}`.

    When used with the :func:`~pennylane.expval` function, the position expectation
    value :math:`\braket{\hat{x}}` is returned. This corresponds to
    the mean displacement in the phase space along the :math:`x` axis.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Observable order: 1st order in the quadrature operators
    * Heisenberg representation:

      .. math:: d = [0, 1, 0]

    Args:
        wires (Sequence[Any] or Any): the wire the operation acts on
    """

    num_params = 0
    num_wires = 1

    ev_order = 1

    def __init__(self, wires):
        super().__init__(wires=wires)

    @staticmethod
    def _heisenberg_rep(p):
        return np.array([0, 1, 0])


class QuadP(CVObservable):
    r"""
    The momentum quadrature observable :math:`\hat{p}`.

    When used with the :func:`~pennylane.expval` function, the momentum expectation
    value :math:`\braket{\hat{p}}` is returned. This corresponds to
    the mean displacement in the phase space along the :math:`p` axis.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Observable order: 1st order in the quadrature operators
    * Heisenberg representation:

      .. math:: d = [0, 0, 1]

    Args:
        wires (Sequence[Any] or Any): the wire the operation acts on
    """

    num_params = 0
    num_wires = 1

    ev_order = 1

    def __init__(self, wires):
        super().__init__(wires=wires)

    @staticmethod
    def _heisenberg_rep(p):
        return np.array([0, 0, 1])


class QuadOperator(CVObservable):
    r"""
    The generalized quadrature observable :math:`\x_\phi = \x cos\phi+\p\sin\phi`.

    When used with the :func:`~pennylane.expval` function, the expectation
    value :math:`\braket{\hat{\x_\phi}}` is returned. This corresponds to
    the mean displacement in the phase space along axis at angle :math:`\phi`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Observable order: 1st order in the quadrature operators
    * Heisenberg representation:

      .. math:: d = [0, \cos\phi, \sin\phi]

    Args:
        phi (float): axis in the phase space at which to calculate
            the generalized quadrature observable
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1

    grad_method = "A"
    ev_order = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        phi = p[0]
        return np.array([0, math.cos(phi), math.sin(phi)])  # TODO check

    def label(self, decimals=None, base_label=None, cache=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> op = qml.QuadOperator(1.234, wires=0)
        >>> op.label()
        'cos(φ)x\n+sin(φ)p'
        >>> op.label(decimals=2)
        'cos(1.23)x\n+sin(1.23)p'
        >>> op.label(base_label="Quad", decimals=2)
        'Quad\n(1.23)'

        """

        if base_label is not None:
            return super().label(decimals=decimals, base_label=base_label, cache=cache)

        if decimals is None:
            p = "φ"
        else:
            p = format(qml_math.array(self.parameters[0]), f".{decimals}f")
        return f"cos({p})x\n+sin({p})p"


class PolyXP(CVObservable):
    r"""
    An arbitrary second-order polynomial observable.

    Represents an arbitrary observable :math:`P(\x,\p)` that is a second order
    polynomial in the basis :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)`.

    For first-order observables the representation is a real vector
    :math:`\mathbf{d}` such that :math:`P(\x,\p) = \mathbf{d}^T \mathbf{r}`.

    For second-order observables the representation is a real symmetric
    matrix :math:`A` such that :math:`P(\x,\p) = \mathbf{r}^T A \mathbf{r}`.

    Used for evaluating arbitrary order-2 CV expectation values of
    :class:`~.pennylane.tape.CVParamShiftTape`.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Observable order: 2nd order in the quadrature operators
    * Heisenberg representation: :math:`A`

    Args:
        q (array[float]): expansion coefficients
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    """

    num_params = 1
    num_wires = AnyWires

    grad_method = "F"
    ev_order = 2

    def __init__(self, q, wires, id=None):
        super().__init__(q, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        return p[0]


class FockStateProjector(CVObservable):
    r"""
    The number state observable :math:`\ket{n}\bra{n}`.

    Represents the non-Gaussian number state observable

    .. math:: \ket{n}\bra{n} = \ket{n_0, n_1, \dots, n_P}\bra{n_0, n_1, \dots, n_P}

    where :math:`n_i` is the occupation number of the :math:`i` th wire.

    The expectation of this observable is

    .. math::
        E[\ket{n}\bra{n}] = \text{Tr}(\ket{n}\bra{n}\rho)
        = \text{Tr}(\braketT{n}{\rho}{n})
        = \braketT{n}{\rho}{n}

    corresponding to the probability of measuring the quantum state in the state
    :math:`\ket{n}=\ket{n_0, n_1, \dots, n_P}`.

    .. note::

        If ``expval(FockStateProjector)`` is applied to a subset of wires,
        the unaffected wires are traced out prior to the expectation value
        calculation.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Observable order: None (non-Gaussian)

    Args:
        n (array): Array of non-negative integers representing the number state
            observable :math:`\ket{n}\bra{n}=\ket{n_0, n_1, \dots, n_P}\bra{n_0, n_1, \dots, n_P}`.

            For example, to return the observable :math:`\ket{0,4,2}\bra{0,4,2}` acting on
            wires 0, 1, and 3 of a QNode, you would call ``FockStateProjector(np.array([0, 4, 2], wires=[0, 1, 3]))``.

            Note that ``len(n)==len(wires)``, and that ``len(n)`` cannot exceed the
            total number of wires in the QNode.
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = AnyWires

    grad_method = None
    ev_order = None

    def __init__(self, n, wires, id=None):
        super().__init__(n, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> qml.FockStateProjector([1,2,3], wires=(0,1,2)).label()
        '|123⟩⟨123|'

        """

        if base_label is not None:
            return super().label(decimals=decimals, base_label=base_label, cache=cache)

        basis_string = "".join(str(int(i)) for i in self.parameters[0])
        return f"|{basis_string}⟩⟨{basis_string}|"


__ops__ = {
    "Identity",
    "Snapshot",
    "Beamsplitter",
    "ControlledAddition",
    "ControlledPhase",
    "Displacement",
    "Kerr",
    "CrossKerr",
    "QuadraticPhase",
    "Rotation",
    "Squeezing",
    "TwoModeSqueezing",
    "CubicPhase",
    "InterferometerUnitary",
    "CatState",
    "CoherentState",
    "FockDensityMatrix",
    "DisplacedSqueezedState",
    "FockState",
    "FockStateVector",
    "SqueezedState",
    "ThermalState",
    "GaussianState",
}


__obs__ = {
    "QuadOperator",
    "NumberOperator",
    "TensorN",
    "QuadP",
    "QuadX",
    "PolyXP",
    "FockStateProjector",
}


__all__ = list(__ops__ | __obs__)
