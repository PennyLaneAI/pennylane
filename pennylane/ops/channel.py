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
"""
This module contains the available built-in noisy
quantum channels supported by PennyLane, as well as their conventions.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access

import numpy as np

from pennylane.operation import Channel


class AmplitudeDamping(Channel):
    r"""AmplitudeDamping(gamma, wires)
    Amplitude damping channel in the Kraus representation.

    Interaction with the environment can lead to changes in the state populations of a qubit.
    This is the phenomenon behind scattering, dissipation, attenuation, and spontaneous emission.
    It can be modelled by the amplitude damping channel, with the following Kraus matrices:

    .. math::
        K_1 = \begin{bmatrix}
                0 & \sqrt{\gamma}  \\
                0 & 0
                \end{bmatrix}

    .. math::
        K_2 = \begin{bmatrix}
                1 & 0 \\
                0 & \sqrt{1-\gamma}
                \end{bmatrix}

    where :math:`\gamma` is the amplitude damping probability.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        gamma (float): amplitude damping probability :math:`\gamma`
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "F"

    @classmethod
    def _kraus_matrices(cls, *params):
        gamma = params[0]
        K1 = np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
        K2 = np.diag([1, np.sqrt(1 - gamma)])
        return [K1, K2]


class GeneralizedAmplitudeDamping(Channel):
    r"""GeneralizedAmplitudeDamping(gamma, p, wires)
    Generalized amplitude damping channel in the Kraus representation.

    This channel models the exchange of energy between a qubit and its environment
    at finite temperatures. :math:`\gamma` is the probability of damping and
    :math:`p` is the probability of system being excited by the environment.

    .. math::
        K_1 = \sqrt{p} \begin{bmatrix}
                1 & 0 \\
                0 & \sqrt{1-\gamma}
                \end{bmatrix}

    .. math::
        K_2 = \sqrt{p}\begin{bmatrix}
                0 & \sqrt{\gamma}  \\
                0 & 0
                \end{bmatrix}

    .. math::
        K_3 = \sqrt{1-p}\begin{bmatrix}
                \sqrt{1-\gamma} & 0 \\
                0 & 1
                \end{bmatrix}

    .. math::
        K_4 = \sqrt{1-p}\begin{bmatrix}
                0 & 0 \\
                \sqrt{\gamma} & 0
                \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2

    Args:
        gamma (float): amplitude damping probability :math:`\gamma`
        p (float): excitation probability :math:`p`
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 2
    num_wires = 1
    par_domain = "R"
    grad_method = "F"

    @classmethod
    def _kraus_matrices(cls, *params):
        gamma, p = params
        K1 = np.sqrt(p) * np.diag([1, np.sqrt(1 - gamma)])
        K2 = np.sqrt(p) * np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
        K3 = np.sqrt(1 - p) * np.diag([np.sqrt(1 - gamma), 1])
        K4 = np.sqrt(1 - p) * np.sqrt(gamma) * np.array([[0, 0], [1, 0]])
        return [K1, K2, K3, K4]

class PhaseDamping(Channel):
    r"""PhaseDamping(gamma, wires)
    Phase damping channel in the Kraus representation.

    Interaction with the environment can lead to loss of quantum information changes without any
    loss in energy. This can be modelled by the phase damping channel, with
    the following Kraus matrices:

    .. math::
        K_1 = \begin{bmatrix}
                0 & 0  \\
                0 & \sqrt{\gamma}
                \end{bmatrix}

    .. math::
        K_2 = \begin{bmatrix}
                1 & 0 \\
                0 & \sqrt{1-\gamma}
                \end{bmatrix}

    where :math:`\gamma` is the phase damping probability.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        gamma (float): phase damping probability :math:`\gamma`
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "F"

    @classmethod
    def _kraus_matrices(cls, *params):
        gamma = params[0]
        K1 = np.diag([0, np.sqrt(gamma)])
        K2 = np.diag([1, np.sqrt(1-gamma)])
        return [K1, K2]

class DepolarizingChannel(Channel):
    r"""DepolarizingChannel(p, wires)
    Symmetrically depolarizing channel in the Kraus representation.

    With probability :math:`p`, qubit is equally depolarized in all Pauli directions. This can
    be modelled by the following Kraus matrices:

    .. math::
        K_1 = \sqrt{1-p} \begin{bmatrix}
                1 & 0 \\
                0 & 1
                \end{bmatrix}

    .. math::
        K_2 = \sqrt{p/3}\begin{bmatrix}
                0 & 1  \\
                1 & 0
                \end{bmatrix}

    .. math::
        K_3 = \sqrt{p/3}\begin{bmatrix}
                0 & -i \\
                i & 0
                \end{bmatrix}

    .. math::
        K_4 = \sqrt{p/3}\begin{bmatrix}
                1 & 0 \\
                0 & -1
                \end{bmatrix}


    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        p (float): Each Pauli gate is applied with probability :math:`\frac{p}{3}`
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "F"

    @classmethod
    def _kraus_matrices(cls, *params):
        p = params[0]
        K1 = np.sqrt(1-p) * np.eye(2)
        K2 = np.sqrt(p/3) * np.array([[0, 1], [1, 0]])
        K3 = np.sqrt(p/3) * np.array([[0, -1j], [1j, 0]])
        K4 = np.sqrt(p/3) * np.array([[1, 0], [0, -1]])
        return [K1, K2, K3, K4]


ops = {"AmplitudeDamping", "GeneralizedAmplitudeDamping", "PhaseDamping", "DepolarizingChannel"}

__all__ = list(ops)
