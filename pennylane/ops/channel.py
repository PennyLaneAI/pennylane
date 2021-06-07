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
"""
This module contains the available built-in noisy
quantum channels supported by PennyLane, as well as their conventions.
"""
import numpy as np

from pennylane.operation import AnyWires, Channel


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

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between [0,1].")

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

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between [0,1].")

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between [0,1].")

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

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between [0,1].")

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
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    @classmethod
    def _kraus_matrices(cls, *params):
        p = params[0]

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between [0,1]")

        K0 = np.sqrt(1 - p) * np.eye(2)
        K1 = np.sqrt(p / 3) * np.array([[0, 1], [1, 0]])
        K2 = np.sqrt(p / 3) * np.array([[0, -1j], [1j, 0]])
        K3 = np.sqrt(p / 3) * np.array([[1, 0], [0, -1]])
        return [K0, K1, K2, K3]


class BitFlip(Channel):
    r"""BitFlip(p, wires)
    Single-qubit bit flip (Pauli :math:`X`) error channel.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \sqrt{1-p} \begin{bmatrix}
                1 & 0 \\
                0 & 1
                \end{bmatrix}

    .. math::
        K_1 = \sqrt{p}\begin{bmatrix}
                0 & 1  \\
                1 & 0
                \end{bmatrix}

    where :math:`p \in [0, 1]` is the probability of a bit flip (Pauli :math:`X` error).

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        p (float): The probability that a bit flip error occurs.
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    @classmethod
    def _kraus_matrices(cls, *params):
        p = params[0]

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between [0,1]")

        K0 = np.sqrt(1 - p) * np.eye(2)
        K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]])
        return [K0, K1]


class ResetError(Channel):
    r"""ResetError(p_0, p_1, wires)
    Single-qubit Reset error channel.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \sqrt{1-p_0-p_1} \begin{bmatrix}
                1 & 0 \\
                0 & 1
                \end{bmatrix}

    .. math::
        K_1 = \sqrt{p_0}\begin{bmatrix}
                1 & 0  \\
                0 & 0
                \end{bmatrix}

    .. math::
        K_2 = \sqrt{p_0}\begin{bmatrix}
                0 & 1  \\
                0 & 0
                \end{bmatrix}

    .. math::
        K_3 = \sqrt{p_1}\begin{bmatrix}
                0 & 0  \\
                1 & 0
                \end{bmatrix}

    .. math::
        K_4 = \sqrt{p_1}\begin{bmatrix}
                0 & 0  \\
                0 & 1
                \end{bmatrix}

    where :math:`p_0 \in [0, 1]` is the probability of a reset to 0,
    and :math:`p_1 \in [0, 1]` is the probability of a reset to 1 error.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2

    Args:
        p_0 (float): The probability that a reset to 0 error occurs.
        p_1 (float): The probability that a reset to 1 error occurs.
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 2
    num_wires = 1
    par_domain = "R"
    grad_method = "F"

    @classmethod
    def _kraus_matrices(cls, *params):
        p_0, p_1 = params[0], params[1]

        if not 0.0 <= p_0 <= 1.0:
            raise ValueError("p_0 must be between [0,1]")

        if not 0.0 <= p_1 <= 1.0:
            raise ValueError("p_1 must be between [0,1]")

        if not 0.0 <= p_0 + p_1 <= 1.0:
            raise ValueError("p_0 + p_1 must be between [0,1]")

        K0 = np.sqrt(1 - p_0 - p_1) * np.eye(2)
        K1 = np.sqrt(p_0) * np.array([[1, 0], [0, 0]])
        K2 = np.sqrt(p_0) * np.array([[0, 1], [0, 0]])
        K3 = np.sqrt(p_1) * np.array([[0, 0], [1, 0]])
        K4 = np.sqrt(p_1) * np.array([[0, 0], [0, 1]])
        return [K0, K1, K2, K3, K4]


class PhaseFlip(Channel):
    r"""PhaseFlip(p, wires)
    Single-qubit bit flip (Pauli :math:`Z`) error channel.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \sqrt{1-p} \begin{bmatrix}
                1 & 0 \\
                0 & 1
                \end{bmatrix}

    .. math::
        K_1 = \sqrt{p}\begin{bmatrix}
                1 & 0  \\
                0 & -1
                \end{bmatrix}

    where :math:`p \in [0, 1]` is the probability of a phase flip (Pauli :math:`Z`) error.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        p (float): The probability that a phase flip error occurs.
        wires (Sequence[int] or int): the wire the channel acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    @classmethod
    def _kraus_matrices(cls, *params):
        p = params[0]

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between [0,1]")

        K0 = np.sqrt(1 - p) * np.eye(2)
        K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]])
        return [K0, K1]


class QubitChannel(Channel):
    r"""QubitChannel(K_list, wires)
    Apply an arbitrary fixed quantum channel.

    Kraus matrices that represent the fixed channel are provided
    as a list of NumPy arrays.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        K_list (list[array[complex]]): List of Kraus matrices
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "L"
    grad_method = None

    def __init__(self, *params, wires=None, do_queue=True):
        super().__init__(*params, wires=wires, do_queue=do_queue)
        K_list = params[0]

        # check all Kraus matrices are square matrices
        if not all(K.shape[0] == K.shape[1] for K in K_list):
            raise ValueError(
                "Only channels with the same input and output Hilbert space dimensions can be applied."
            )

        # check all Kraus matrices have the same shape
        if not all(K.shape == K_list[0].shape for K in K_list):
            raise ValueError("All Kraus matrices must have the same shape.")

        # check the dimension of all Kraus matrices are valid
        if not all(K.ndim == 2 for K in K_list):
            raise ValueError(
                "Dimension of all Kraus matrices must be (2**num_wires, 2**num_wires)."
            )

        # check that the channel represents a trace-preserving map
        K_arr = np.array(K_list)
        Kraus_sum = np.einsum("ajk,ajl->kl", K_arr.conj(), K_arr)
        if not np.allclose(Kraus_sum, np.eye(K_list[0].shape[0])):
            raise ValueError("Only trace preserving channels can be applied.")

    @classmethod
    def _kraus_matrices(cls, *params):
        K_list = params[0]
        return K_list


__qubit_channels__ = {
    "AmplitudeDamping",
    "GeneralizedAmplitudeDamping",
    "PhaseDamping",
    "DepolarizingChannel",
    "BitFlip",
    "PhaseFlip",
    "ResetError",
    "QubitChannel",
}

__all__ = list(__qubit_channels__)
