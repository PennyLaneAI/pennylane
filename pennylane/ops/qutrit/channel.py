# Copyright 2024 Xanadu Quantum Technologies Inc.

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
"""
This module contains the available built-in noisy
quantum channels supported by PennyLane, as well as their conventions.
"""
from pennylane import math
from pennylane.operation import Channel


class QutritDepolarizingChannel(Channel):
    r"""
    Single-qutrit symmetrically depolarizing error channel.
    This channel is modelled by the following Kraus matrices:
    where :math:`p \in [0, 1]` is the depolarization probability and is equally
    divided in the application of all Pauli operations.
    .. note::
        Multiple equivalent definitions of the Kraus operators :math:`\{K_0 \ldots K_3\}` exist in
        the literature [`1 <https://michaelnielsen.org/qcqi/>`_] (Eqs. 8.102-103). Here, we adopt the
        one from Eq. 8.103, which is also presented in [`2 <http://theory.caltech.edu/~preskill/ph219/chap3_15.pdf>`_] (Eq. 3.85).  # TODO change sources
        For this definition, please make a note of the following:
        * For :math:`p = 0`, the channel will be an Identity channel, i.e., a noise-free channel.
        * For :math:`p = \frac{3}{4}`, the channel will be a fully depolarizing channel.
        * For :math:`p = 1`, the channel will be a uniform Pauli error channel.
    **Details:**
    * Number of wires: 1
    * Number of parameters: 1
    Args:
        p (float): Each Pauli gate is applied with probability :math:`\frac{p}{3}`
        wires (Sequence[int] or int): the wire the channel acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)  # TODO

    def __init__(self, p, wires, id=None):
        super().__init__(p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p):  # pylint:disable=arguments-differ
        r"""Kraus matrices representing the depolarizing channel.
        Args:
            p (float): each Pauli gate is applied with probability :math:`\frac{p}{3}`
        Returns:
            list (array): list of Kraus matrices
        **Example**
        >>> qml.DepolarizingChannel.compute_kraus_matrices(0.5)
        [array([[0.70710678, 0.        ], [0.        , 0.70710678]]),
         array([[0.        , 0.40824829], [0.40824829, 0.        ]]),
         array([[0.+0.j        , 0.-0.40824829j], [0.+0.40824829j, 0.+0.j        ]]),
         array([[ 0.40824829,  0.        ], [ 0.        , -0.40824829]])]
        """
        if not math.is_abstract(p) and not 0.0 <= p <= 1.0:
            raise ValueError("p must be in the interval [0,1]")

        if math.get_interface(p) == "tensorflow":
            p = math.cast_like(p, 1j)

        Z0 = math.eye(3)
        Z1 = math.diag([1, math.exp(2j * math.pi / 3), math.exp(4j * math.pi / 3)])
        Z2 = math.diag([1, math.exp(4j * math.pi / 3), math.exp(8j * math.pi / 3)])

        X0 = math.eye(3)
        X1 = math.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        X2 = math.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        Ks = [
            math.sqrt(1 - (8 * p / 9) + math.eps) * math.convert_like(math.eye(2, dtype=complex), p)
        ]

        for i, Z in enumerate((Z0, Z1, Z2)):
            for j, X in enumerate((X0, X1, X2)):
                if i == 0 and j == 0:
                    continue
                Ks.append(
                    math.sqrt(p / 9 + math.eps)
                    * math.convert_like(math.array(X @ Z, dtype=complex), p)
                )

        return Ks


class QutritAmplitudeDampingChannel(Channel):
    r"""
    Single-qutrit amplitude damping error channel.
    Interaction with the environment can lead to changes in the state populations of a qubit.
    This is the phenomenon behind scattering, dissipation, attenuation, and spontaneous emission.
    It can be modelled by the amplitude damping channel, with the following Kraus matrices:
    .. math::
        K_0 = \begin{bmatrix}
                1 & 0 & 0\\
                0 & \sqrt{1-\gamma_1} & 0 \\
                0 & 0 & \sqrt{1-\gamma_2}
                \end{bmatrix}
    .. math::
        K_1 = \begin{bmatrix}
                0 & \sqrt{\gamma_1} & 0 \\
                0 & 0 & 0 \\
                0 & 0 & 0
                \end{bmatrix}
    .. math::
        K_2 = \begin{bmatrix}
                0 & 0 & \sqrt{\gamma_2} \\
                0 & 0 & 0 \\
                0 & 0 & 0
                \end{bmatrix}
    where :math:`\gamma \in [0, 1]` is the amplitude damping probability.
    **Details:**
    * Number of wires: 1
    * Number of parameters: 1
    Args:
        gamma (float): amplitude damping probability
        wires (Sequence[int] or int): the wire the channel acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = "F"

    def __init__(self, gamma_1, gamma_2, wires, id=None):
        super().__init__(gamma_1, gamma_2, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(gamma_1, gamma_2):  # pylint:disable=arguments-differ
        """Kraus matrices representing the AmplitudeDamping channel.
        Args:
            gamma_1 (float): amplitude damping probability #TODO
            gamma_2 (float): amplitude damping probability
        Returns:
            list(array): list of Kraus matrices
        **Example**
        >>> qml.QutritAmplitudeDampingChannel.compute_kraus_matrices(0.25, 0.25) #TODO
        [
        array([ [1.        , 0.        , 0.        ],
                [0.        , 0.70710678, 0.        ],
                [0.        , 0.        , 0.8660254 ]]),
        array([ [0.        , 0.70710678, 0.        ],
                [0.        , 0.        , 0.        ],
                [0.        , 0.        , 0.        ]]),
        array([ [0.        , 0.        , 0.5       ],
                [0.        , 0.        , 0.        ],
                [0.        , 0.        , 0.        ]])
        ]
        """
        if type(gamma_1) != type(gamma_2):
            raise ValueError("p1, p2, and p3 should be of the same type")

        if not math.is_abstract(gamma_1):
            for gamma in (gamma_1, gamma_2):
                if not 0.0 <= gamma <= 1.0:
                    raise ValueError("Each probability must be in the interval [0,1]")
            if not 0.0 <= gamma_1 + gamma_2 <= 1.0:
                raise ValueError("Sum of probabilities must be in the interval [0,1]")

        K0 = math.diag([1, math.sqrt(1 - gamma_1 + math.eps), math.sqrt(1 - gamma_2 + math.eps)])
        K1 = math.sqrt(gamma_1 + math.eps) * math.convert_like(
            math.cast_like(math.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), gamma_1), gamma_1
        )
        K2 = math.sqrt(gamma_2 + math.eps) * math.convert_like(
            math.cast_like(math.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]), gamma_2), gamma_2
        )
        return [K0, K1, K2]


__qutrit_channels__ = {
    "QutritDepolarizingChannel",
    "QutritAmplitudeDampingChannel",
}

__all__ = list(__qutrit_channels__)
