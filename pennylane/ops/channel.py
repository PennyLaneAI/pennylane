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

import numpy as np

from pennylane.operation import Channel


class AmplitudeDamping(Channel):
    r"""AmplitudeDamping(gamma, wires)
    Single-qubit amplitude damping error channel.

    Interaction with the environment can lead to changes in the state populations of a qubit.
    This is the phenomenon behind scattering, dissipation, attenuation, and spontaneous emission.
    It can be modelled by the amplitude damping channel, with the following Kraus matrices:

    .. math::
        K_0 = \begin{bmatrix}
                1 & 0 \\
                0 & \sqrt{1-\gamma}
                \end{bmatrix}
    .. math::
        K_1 = \begin{bmatrix}
                0 & \sqrt{\gamma}  \\
                0 & 0
                \end{bmatrix}

    where :math:`\gamma \in [0, 1]` is the amplitude damping probability.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        gamma (float): amplitude damping probability
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "F"

    @classmethod
    def _kraus_matrices(cls, *params):
        gamma = params[0]
        K0 = np.diag([1, np.sqrt(1 - gamma)])
        K1 = np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
        return [K0, K1]


class GeneralizedAmplitudeDamping(Channel):
    r"""GeneralizedAmplitudeDamping(gamma, p, wires)
    Single-qubit generalized amplitude damping error channel.

    This channel models the exchange of energy between a qubit and its environment
    at finite temperatures, with the following Kraus matrices:

    .. math::
        K_0 = \sqrt{p} \begin{bmatrix}
                1 & 0 \\
                0 & \sqrt{1-\gamma}
                \end{bmatrix}

    .. math::
        K_1 = \sqrt{p}\begin{bmatrix}
                0 & \sqrt{\gamma}  \\
                0 & 0
                \end{bmatrix}

    .. math::
        K_2 = \sqrt{1-p}\begin{bmatrix}
                \sqrt{1-\gamma} & 0 \\
                0 & 1
                \end{bmatrix}

    .. math::
        K_3 = \sqrt{1-p}\begin{bmatrix}
                0 & 0 \\
                \sqrt{\gamma} & 0
                \end{bmatrix}

    where :math:`\gamma \in [0, 1]` is the probability of damping and :math:`p \in [0, 1]`
    is the probability of the system being excited by the environment.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2

    Args:
        gamma (float): amplitude damping probability
        p (float): excitation probability
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 2
    num_wires = 1
    par_domain = "R"
    grad_method = "F"

    @classmethod
    def _kraus_matrices(cls, *params):
        gamma, p = params
        K0 = np.sqrt(p) * np.diag([1, np.sqrt(1 - gamma)])
        K1 = np.sqrt(p) * np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
        K2 = np.sqrt(1 - p) * np.diag([np.sqrt(1 - gamma), 1])
        K3 = np.sqrt(1 - p) * np.sqrt(gamma) * np.array([[0, 0], [1, 0]])
        return [K0, K1, K2, K3]


class PhaseDamping(Channel):
    r"""PhaseDamping(gamma, wires)
    Single-qubit phase damping error channel.

    Interaction with the environment can lead to loss of quantum information changes without any
    changes in qubit excitations. This can be modelled by the phase damping channel, with
    the following Kraus matrices:

    .. math::
        K_0 = \begin{bmatrix}
                1 & 0 \\
                0 & \sqrt{1-\gamma}
                \end{bmatrix}
    .. math::

        K_1 = \begin{bmatrix}
                0 & 0  \\
                0 & \sqrt{\gamma}
                \end{bmatrix}

    where :math:`\gamma \in [0, 1]` is the phase damping probability.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        gamma (float): phase damping probability
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "F"

    @classmethod
    def _kraus_matrices(cls, *params):
        gamma = params[0]
        K0 = np.diag([1, np.sqrt(1 - gamma)])
        K1 = np.diag([0, np.sqrt(gamma)])
        return [K0, K1]


class DepolarizingChannel(Channel):
    r"""DepolarizingChannel(p, wires)
    Single-qubit symmetrically depolarizing error channel.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \sqrt{1-p} \begin{bmatrix}
                1 & 0 \\
                0 & 1
                \end{bmatrix}

    .. math::
        K_1 = \sqrt{p/3}\begin{bmatrix}
                0 & 1  \\
                1 & 0
                \end{bmatrix}

    .. math::
        K_2 = \sqrt{p/3}\begin{bmatrix}
                0 & -i \\
                i & 0
                \end{bmatrix}

    .. math::
        K_3 = \sqrt{p/3}\begin{bmatrix}
                1 & 0 \\
                0 & -1
                \end{bmatrix}

    where :math:`p \in [0, 1]` is the depolarization probability and is equally
    divided in the application of all Pauli operations.

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
        K0 = np.sqrt(1 - p) * np.eye(2)
        K1 = np.sqrt(p / 3) * np.array([[0, 1], [1, 0]])
        K2 = np.sqrt(p / 3) * np.array([[0, -1j], [1j, 0]])
        K3 = np.sqrt(p / 3) * np.array([[1, 0], [0, -1]])
        return [K0, K1, K2, K3]


__qubit_channels__ = {
    "AmplitudeDamping",
    "GeneralizedAmplitudeDamping",
    "PhaseDamping",
    "DepolarizingChannel",
}

__all__ = list(__qubit_channels__)
