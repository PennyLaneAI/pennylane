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
"""This module contains a beamsplitter Operation"""
from .operation import Operation


class Rotation(Operation):
    r"""Continuous-variable phase space rotation.

    .. math::
        R(\phi) = \exp\left(i \phi \ad \a\right)=\exp\left(i \frac{\phi}{2}
        \left(\frac{\x^2+  \p^2}{\hbar}-\I\right)\right)

    Args:
        phi (float): the rotation angle.
        wires (int): the subsystem the Operation acts on.
    """

    def __init__(self, phi, wires):
        super().__init__('Rotation', [phi], wires)


class Displacement(Operation):
    r"""Continuous-variable phase space displacement.

    .. math::
       D(\alpha) = \exp(\alpha a^\dagger -\alpha^* a)
       = \exp\left(-i\sqrt{2}(\re(\alpha) \hat{p} -\im(\alpha) \hat{x})/\sqrt{\hbar}\right)

    where :math:`\alpha = r e^{i\phi}` has magnitude :math:`r\geq 0` and phase :math:`\phi`.

    The gate is parameterized so that a user can specify a single complex number :math:`a=\alpha`
    or use the polar form :math:`a = r, \phi` and still get the same result.

    Args:
        a (complex): displacement parameter :math:`\alpha`
        phi (float): extra (optional) phase angle :math:`\phi`
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, a, phi, wires):
        super().__init__('Displacement', [a, phi], wires)


class Squeezing(Operation):
    r"""Continuous-variable phase space squeezing.

    This Operation can act on a single mode,

    .. math::
        S(z) = \exp\left(\frac{1}{2}(z^* a^2 -z {a^\dagger}^2)\right)

    or on two modes,

    .. math::
        S_2(z) = \exp\left(z^* ab -z a^\dagger b^\dagger \right)
        = \exp\left(r (e^{-i\phi} ab -e^{i\phi} a^\dagger b^\dagger \right)

    where :math:`z = r e^{i\phi}`.

    Args:
        r (float): squeezing amount
        phi (float): squeezing phase angle :math:`\phi`
        wires (int or seq[int]): the subsystem(s) the Operation acts on.
    """
    def __init__(self, r, phi, wires):
        if isinstance(wires, int) or len(wires) == 1:
            super().__init__('Squeezing', [r, phi], wires)
        elif len(wires) == 2:
            super().__init__('TwoModeSqueezing', [r, phi], wires)


class QuadraticPhase(Operation):
    r"""Continuous-variable quadratic phase shift.

    .. math::
        P(s) = e^{i \frac{s}{2} \hat{x}^2/\hbar}

    Args:
        s (float): parameter
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, s, wires):
        super().__init__('QuadraticPhase', [s], wires)


class CubicPhase(Operation):
    r"""Continuous-variable cubic phase shift.

    .. math::
        V(\gamma) = e^{i \frac{\gamma}{3} \hat{x}^3/\hbar}

    Args:
        gamma (float): parameter
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, gamma, wires):
        super().__init__('CubicPhase', [gamma], wires)


class Kerr(Operation):
    r"""Continuous-variable Kerr interaction.

    This Operation can act on a single mode,

    .. math::
        K(\kappa) = e^{i \kappa \hat{n}^2}

    or on two modes, to perform the Cross-Kerr interaction:

    .. math::
        CK(\kappa) = e^{i \kappa \hat{n}_1\hat{n}_2}

    Args:
        kappa (float): parameter
        wires (int or seq[int]): the subsystem(s) the Operation acts on.
    """
    def __init__(self, kappa, wires):
        if isinstance(wires, int) or len(wires) == 1:
            super().__init__('Kerr', [kappa], wires)
        elif len(wires) == 2:
            super().__init__('CrossKerr', [kappa], wires)


class Beamsplitter(Operation):
    r"""Continuous-variable beamsplitter interaction.

    .. math::
        B(\theta,\phi) = \exp\left(\theta (e^{i \phi} a^\dagger b -e^{-i \phi}a b^\dagger) \right)

    Args:
        theta (float): Transmittivity angle :math:`\theta`. The transmission amplitude
            of the beamsplitter is :math:`t = \cos(\theta)`.
            The value :math:`\theta=\pi/4` gives the 50-50 beamsplitter.
        phi (float): Phase angle :math:`\phi`. The reflection amplitude of the
            beamsplitter is :math:`r = e^{i\phi}\sin(\theta)`.
            The value :math:`\phi = \pi/2` gives the symmetric beamsplitter.
        wires (seq[int]): sequence of two subsystems the Operation acts on.
    """
    def __init__(self, theta, phi, wires):
        super().__init__('Beamsplitter', [theta, phi], wires)


class ControlledAddition(Operation):
    r"""Continuous-variable controlled addition Operation.

    .. math::
           \text{CX}(s) = \int dx \ket{x}\bra{x} \otimes D\left({\frac{1}{\sqrt{2\hbar}}}s x\right)
           = e^{-i s \: \hat{x} \otimes \hat{p}/\hbar}

    Args:
        s (float): addition multiplier
        wires (seq[int]): sequence of two subsystems the Operation acts on.
    """
    def __init__(self, s, wires):
        super().__init__('ControlledAddition', [s], wires)


class ControlledPhase(Operation):
    r"""Continuous-variable controlled phase Operation.

    .. math::
           \text{CX}(s) =  \iint dx dy \: e^{i sxy/\hbar} \ket{x,y}\bra{x,y}
           = e^{i s \: \hat{x} \otimes \hat{x}/\hbar}

    Args:
        s (float):  phase shift multiplier
        wires (seq[int]): sequence of two subsystems the Operation acts on.
    """
    def __init__(self, s, wires):
        super().__init__('ControlledPhase', [s], wires)


#=============================================================================
# State preparation
#=============================================================================


class CoherentState(Operation):
    r"""Prepares a coherent state.

    Args:
        a (complex): displacement parameter :math:`\alpha`
        phi (float): phase angle :math:`\phi`
        wires (int): subsystem the Operation acts on.
    """
    def __init__(self, a, phi, wires):
        super().__init__('CoherentState', [a, phi], wires)


class SqueezedState(Operation):
    r"""Prepares a squeezed vacuum state.

    Args:
        r (float): squeezing magnitude
        phi (float): squeezing angle :math:`\phi`
        wires (int): subsystem the Operation acts on.
    """
    def __init__(self, r, phi, wires):
        super().__init__('SqueezedState', [r, phi], wires)


class DisplacedSqueezedState(Operation):
    r"""Prepares a displaced squeezed vacuum state.

    A displaced squeezed state is prepared by squeezing a vacuum state, and
    then applying a displacement operator.

    .. math::
       \ket{\alpha,z} = D(\alpha)\ket{0,z} = D(\alpha)S(z)\ket{0},

    where the squeezing parameter :math:`z=re^{i\phi}`.

    Args:
        alpha (complex): displacement parameter
        r (float): squeezing magnitude
        phi (float): squeezing angle :math:`\phi`
        wires (int): subsystem the Operation acts on.
    """
    def __init__(self, alpha, r, phi, wires):
        super().__init__('DisplacedSqueezedState', [alpha, r, phi], wires)


class FockState(Operation):
    r"""Prepares a single Fock state.

    Args:
        n (int): Fock state to prepare
        wires (int): subsystem the Operation acts on.
    """
    def __init__(self, n, wires):
        super().__init__('FockState', [n], wires, par_domain='N')


class ThermalState(Operation):
    r"""Prepares a thermal state.

    Args:
        n (float): mean thermal population of the mode
        wires (int): subsystem the Operation acts on.
    """
    def __init__(self, n, wires):
        super().__init__('ThermalState', [n], wires)


class CatState(Operation):
    r"""Prepares a cat state.

    A cat state is the coherent superposition of two coherent states,

    .. math::
       \ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{i\phi} \ket{-\alpha}),

    where :math:`N = \sqrt{2 (1+\cos(\phi)e^{-2|\alpha|^2})}` is the normalization factor.

    Args:
        alpha (complex): displacement parameter
        p (float): parity, where :math:`\phi=p\pi`. ``p=0`` corresponds to an even
            cat state, and ``p=1`` an odd cat state.
        wires (int): subsystem the Operation acts on.
    """
    def __init__(self, a, p, wires):
        super().__init__('CatState', [a, p], wires)


class FockStateVector(Operation):
    r"""Prepare subsystems using the given ket vector in the Fock basis.

    Args:
        state (array): a single ket vector, for single mode state preparation,
            or a multimode ket, with one array dimension per mode.
        wires (int or seq[int]): subsystem(s) the Operation acts on.
    """
    def __init__(self, state, wires):
        super().__init__('FockStateVector', [state], wires)


class FockDensityMatrix(Operation):
    r"""Prepare subsystems using the given density matrix in the Fock basis.

    Args:
        state (array): a single mode two-dimensional matrix :math:`\rho_{ij}`, or
            a multimode tensor :math:`\rho_{ij,kl,\dots,mn}`, with two indices per mode.
        wires (int or seq[int]): subsystem(s) the Operation acts on.
    """
    def __init__(self, state, wires):
        super().__init__('FockDensityMatrix', [state], wires)


class GaussianState(Operation):
    r"""Prepare subsystems in a given Gaussian state.

    Args:
        r (array): a length :math:`2N` vector of means, of the
            form :math:`(\x_0,\dots,\x_{N-1},\p_0,\dots,\p_{N-1})`.
        V (array): the :math:`2N\times 2N` (real and positive definite) covariance matrix.
        wires (int or seq[int]): subsystem(s) the Operation acts on.
    """
    def __init__(self, r, V, wires):
        super().__init__('GaussianState', [r, V], wires)
