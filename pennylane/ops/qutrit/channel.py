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
