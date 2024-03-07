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
"""
This module contains the available built-in noisy
quantum channels supported by PennyLane, as well as their conventions.
"""
import warnings

from pennylane import math as np
from pennylane.operation import AnyWires, Channel


class AmplitudeDamping(Channel):
    r"""
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
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = "F"

    def __init__(self, gamma, wires, id=None):
        super().__init__(gamma, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(gamma):  # pylint:disable=arguments-differ
        """Kraus matrices representing the AmplitudeDamping channel.

        Args:
            gamma (float): amplitude damping probability

        Returns:
            list(array): list of Kraus matrices

        **Example**

        >>> qml.AmplitudeDamping.compute_kraus_matrices(0.5)
        [array([[1., 0.], [0., 0.70710678]]),
         array([[0., 0.70710678], [0., 0.]])]
        """
        if not np.is_abstract(gamma) and not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in the interval [0,1].")

        K0 = np.diag([1, np.sqrt(1 - gamma + np.eps)])
        K1 = np.sqrt(gamma + np.eps) * np.convert_like(
            np.cast_like(np.array([[0, 1], [0, 0]]), gamma), gamma
        )
        return [K0, K1]


class GeneralizedAmplitudeDamping(Channel):
    r"""
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
        id (str or None): String representing the operation (optional)
    """

    num_params = 2
    num_wires = 1
    grad_method = "F"

    def __init__(self, gamma, p, wires, id=None):
        super().__init__(gamma, p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(gamma, p):  # pylint:disable=arguments-differ
        """Kraus matrices representing the GeneralizedAmplitudeDamping channel.

        Args:
            gamma (float): amplitude damping probability
            p (float): excitation probability

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.GeneralizedAmplitudeDamping.compute_kraus_matrices(0.3, 0.6)
        [array([[0.77459667, 0.        ], [0.        , 0.64807407]]),
         array([[0.        , 0.42426407], [0.        , 0.        ]]),
         array([[0.52915026, 0.        ], [0.        , 0.63245553]]),
         array([[0.        , 0.        ], [0.34641016, 0.        ]])]
        """
        if not np.is_abstract(gamma) and not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in the interval [0,1].")

        if not np.is_abstract(p) and not 0.0 <= p <= 1.0:
            raise ValueError("p must be in the interval [0,1].")

        K0 = np.sqrt(p + np.eps) * np.diag([1, np.sqrt(1 - gamma + np.eps)])
        K1 = (
            np.sqrt(p + np.eps)
            * np.sqrt(gamma)
            * np.convert_like(np.cast_like(np.array([[0, 1], [0, 0]]), gamma), gamma)
        )
        K2 = np.sqrt(1 - p + np.eps) * np.diag([np.sqrt(1 - gamma + np.eps), 1])
        K3 = (
            np.sqrt(1 - p + np.eps)
            * np.sqrt(gamma)
            * np.convert_like(np.cast_like(np.array([[0, 0], [1, 0]]), gamma), gamma)
        )
        return [K0, K1, K2, K3]


class PhaseDamping(Channel):
    r"""
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
    grad_method = "F"

    def __init__(self, gamma, wires, id=None):
        super().__init__(gamma, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(gamma):  # pylint:disable=arguments-differ
        """Kraus matrices representing the PhaseDamping channel.

        Args:
            gamma (float): phase damping probability

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.PhaseDamping.compute_kraus_matrices(0.5)
        [array([[1.        , 0.        ], [0.        , 0.70710678]]),
         array([[0.        , 0.        ], [0.        , 0.70710678]])]
        """
        if not np.is_abstract(gamma) and not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in the interval [0,1].")

        K0 = np.diag([1, np.sqrt(1 - gamma + np.eps)])
        K1 = np.diag([0, np.sqrt(gamma + np.eps)])
        return [K0, K1]


class DepolarizingChannel(Channel):
    r"""
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

    .. note::

        Multiple equivalent definitions of the Kraus operators :math:`\{K_0 \ldots K_3\}` exist in
        the literature [`1 <https://michaelnielsen.org/qcqi/>`_] (Eqs. 8.102-103). Here, we adopt the
        one from Eq. 8.103, which is also presented in [`2 <http://theory.caltech.edu/~preskill/ph219/chap3_15.pdf>`_] (Eq. 3.85).
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
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

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
        if not np.is_abstract(p) and not 0.0 <= p <= 1.0:
            raise ValueError("p must be in the interval [0,1]")

        if np.get_interface(p) == "tensorflow":
            p = np.cast_like(p, 1j)

        K0 = np.sqrt(1 - p + np.eps) * np.convert_like(np.eye(2, dtype=complex), p)
        K1 = np.sqrt(p / 3 + np.eps) * np.convert_like(np.array([[0, 1], [1, 0]], dtype=complex), p)
        K2 = np.sqrt(p / 3 + np.eps) * np.convert_like(
            np.array([[0, -1j], [1j, 0]], dtype=complex), p
        )
        K3 = np.sqrt(p / 3 + np.eps) * np.convert_like(
            np.array([[1, 0], [0, -1]], dtype=complex), p
        )
        return [K0, K1, K2, K3]


class BitFlip(Channel):
    r"""
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
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    def __init__(self, p, wires, id=None):
        super().__init__(p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p):  # pylint:disable=arguments-differ
        """Kraus matrices representing the BitFlip channel.

        Args:
            p (float): probability that a bit flip error occurs

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.BitFlip.compute_kraus_matrices(0.5)
        [array([[0.70710678, 0.        ], [0.        , 0.70710678]]),
         array([[0.        , 0.70710678], [0.70710678, 0.        ]])]
        """
        if not np.is_abstract(p) and not 0.0 <= p <= 1.0:
            raise ValueError("p must be in the interval [0,1]")

        K0 = np.sqrt(1 - p + np.eps) * np.convert_like(np.cast_like(np.eye(2), p), p)
        K1 = np.sqrt(p + np.eps) * np.convert_like(np.cast_like(np.array([[0, 1], [1, 0]]), p), p)
        return [K0, K1]


class ResetError(Channel):
    r"""
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
        id (str or None): String representing the operation (optional)
    """

    num_params = 2
    num_wires = 1
    grad_method = "F"

    def __init__(self, p0, p1, wires, id=None):
        super().__init__(p0, p1, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p_0, p_1):  # pylint:disable=arguments-differ
        """Kraus matrices representing the ResetError channel.

        Args:
            p_0 (float): probability that a reset to 0 error occurs
            p_1 (float): probability that a reset to 1 error occurs

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.ResetError.compute_kraus_matrices(0.2, 0.3)
        [array([[0.70710678, 0.        ], [0.        , 0.70710678]]),
         array([[0.4472136, 0.       ], [0.       , 0.       ]]),
         array([[0.       , 0.4472136], [0.       , 0.       ]]),
         array([[0.        , 0.        ], [0.54772256, 0.        ]]),
         array([[0.        , 0.        ], [0.        , 0.54772256]])]
        """
        if not np.is_abstract(p_0) and not 0.0 <= p_0 <= 1.0:
            raise ValueError("p_0 must be in the interval [0,1]")

        if not np.is_abstract(p_1) and not 0.0 <= p_1 <= 1.0:
            raise ValueError("p_1 must be in the interval [0,1]")

        if not np.is_abstract(p_0 + p_1) and not 0.0 <= p_0 + p_1 <= 1.0:
            raise ValueError("p_0 + p_1 must be in the interval [0,1]")

        interface = np.get_interface(p_0, p_1)
        p_0, p_1 = np.coerce([p_0, p_1], like=interface)
        K0 = np.sqrt(1 - p_0 - p_1 + np.eps) * np.convert_like(np.cast_like(np.eye(2), p_0), p_0)
        K1 = np.sqrt(p_0 + np.eps) * np.convert_like(
            np.cast_like(np.array([[1, 0], [0, 0]]), p_0), p_0
        )
        K2 = np.sqrt(p_0 + np.eps) * np.convert_like(
            np.cast_like(np.array([[0, 1], [0, 0]]), p_0), p_0
        )
        K3 = np.sqrt(p_1 + np.eps) * np.convert_like(
            np.cast_like(np.array([[0, 0], [1, 0]]), p_0), p_0
        )
        K4 = np.sqrt(p_1 + np.eps) * np.convert_like(
            np.cast_like(np.array([[0, 0], [0, 1]]), p_0), p_0
        )

        return [K0, K1, K2, K3, K4]


class PauliError(Channel):
    r"""
    Pauli operator error channel for an arbitrary number of qubits.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \sqrt{1-p} * I

    .. math::
        K_1 = \sqrt{p} * (K_{w0} \otimes K_{w1} \otimes \dots K_{wn})

    Where :math:`I` is the Identity,
    and :math:`\otimes` denotes the Kronecker Product,
    and :math:`K_{wi}` denotes the Kraus matrix corresponding to the operator acting on wire :math:`wi`,
    and :math:`p` denotes the probability with which the channel is applied.

    .. warning::

        The size of the Kraus matrices for PauliError scale exponentially
        with the number of wires, the channel acts on. Simulations with
        PauliError can result in a significant increase in memory and
        computational usage. Use with caution!

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 3

    Args:
        operators (str): The Pauli operators acting on the specified (groups of) wires
        p (float): The probability of the operator being applied
        wires (Sequence[int] or int): The wires the channel acts on
        id (str or None): String representing the operation (optional)

    **Example:**

    >>> pe = PauliError("X", 0.5, wires=0)
    >>> km = pe.kraus_matrices()
    >>> km[0]
    array([[0.70710678, 0.        ],
           [0.        , 0.70710678]])
    >>> km[1]
        array([[0.        , 0.70710678],
               [0.70710678, 0.        ]])
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 2
    """int: Number of trainable parameters that the operator depends on."""

    def __init__(self, operators, p, wires=None, id=None):
        super().__init__(operators, p, wires=wires, id=id)

        # check if the specified operators are legal
        if not set(operators).issubset({"X", "Y", "Z"}):
            raise ValueError("The specified operators need to be either of 'X', 'Y' or 'Z'")

        # check if probabilities are legal
        if not np.is_abstract(p) and not 0.0 <= p <= 1.0:
            raise ValueError("p must be in the interval [0,1]")

        # check if the number of operators matches the number of wires
        if len(self.wires) != len(operators):
            raise ValueError("The number of operators must match the number of wires")

        nq = len(self.wires)

        if nq > 20:
            warnings.warn(
                f"The resulting Kronecker matrices will have dimensions {2**(nq)} x {2**(nq)}.\nThis equals {2**nq*2**nq*8/1024**3} GB of physical memory for each matrix."
            )

    @staticmethod
    def compute_kraus_matrices(operators, p):  # pylint:disable=arguments-differ
        """Kraus matrices representing the PauliError channel.

        Args:
            operators (str): the Pauli operators acting on the specified (groups of) wires
            p (float): probability of the operator being applied

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.PauliError.compute_kraus_matrices("X", 0.5)
        [array([[0.70710678, 0.        ], [0.        , 0.70710678]]),
         array([[0.        , 0.70710678], [0.70710678, 0.        ]])]
        """
        nq = len(operators)

        # K0 is sqrt(1-p) * Identity
        K0 = np.sqrt(1 - p + np.eps) * np.convert_like(np.cast_like(np.eye(2**nq), p), p)

        interface = np.get_interface(p)
        if interface == "tensorflow" or "Y" in operators:
            if interface == "numpy":
                p = (1 + 0j) * p
            else:
                p = np.cast_like(p, 1j)

        ops = {
            "X": np.convert_like(np.cast_like(np.array([[0, 1], [1, 0]]), p), p),
            "Y": np.convert_like(np.cast_like(np.array([[0, -1j], [1j, 0]]), p), p),
            "Z": np.convert_like(np.cast_like(np.array([[1, 0], [0, -1]]), p), p),
        }

        # K1 is composed by Kraus matrices of operators
        K1 = np.sqrt(p + np.eps) * np.convert_like(np.cast_like(np.eye(1), p), p)
        for op in operators[::-1]:
            K1 = np.multi_dispatch()(np.kron)(ops[op], K1)

        return [K0, K1]


class PhaseFlip(Channel):
    r"""
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
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    def __init__(self, p, wires, id=None):
        super().__init__(p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p):  # pylint:disable=arguments-differ
        """Kraus matrices representing the PhaseFlip channel.

        Args:
            p (float): the probability that a phase flip error occurs

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.PhaseFlip.compute_kraus_matrices(0.5)
        [array([[0.70710678, 0.        ], [0.        , 0.70710678]]),
         array([[ 0.70710678,  0.        ], [ 0.        , -0.70710678]])]
        """
        if not np.is_abstract(p) and not 0.0 <= p <= 1.0:
            raise ValueError("p must be in the interval [0,1]")

        K0 = np.sqrt(1 - p + np.eps) * np.convert_like(np.cast_like(np.eye(2), p), p)
        K1 = np.sqrt(p + np.eps) * np.convert_like(np.cast_like(np.diag([1, -1]), p), p)
        return [K0, K1]


class QubitChannel(Channel):
    r"""
    Apply an arbitrary fixed quantum channel.

    Kraus matrices that represent the fixed channel are provided
    as a list of NumPy arrays.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        K_list (list[array[complex]]): list of Kraus matrices
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, K_list, wires=None, id=None):
        super().__init__(*K_list, wires=wires, id=id)

        # check all Kraus matrices are square matrices
        if any(K.shape[0] != K.shape[1] for K in K_list):
            raise ValueError(
                "Only channels with the same input and output Hilbert space dimensions can be applied."
            )

        # check all Kraus matrices have the same shape
        if any(K.shape != K_list[0].shape for K in K_list):
            raise ValueError("All Kraus matrices must have the same shape.")

        # check the dimension of all Kraus matrices are valid
        if any(K.ndim != 2 for K in K_list):
            raise ValueError(
                "Dimension of all Kraus matrices must be (2**num_wires, 2**num_wires)."
            )

        # check that the channel represents a trace-preserving map
        if not any(np.is_abstract(K) for K in K_list):
            K_arr = np.array(K_list)
            Kraus_sum = np.einsum("ajk,ajl->kl", K_arr.conj(), K_arr)
            if not np.allclose(Kraus_sum, np.eye(K_list[0].shape[0])):
                raise ValueError("Only trace preserving channels can be applied.")

    def _flatten(self):
        return (self.data,), (self.wires, ())

    @staticmethod
    def compute_kraus_matrices(*kraus_matrices):  # pylint:disable=arguments-differ
        """Kraus matrices representing the QubitChannel channel.

        Args:
            *K_list (list[array[complex]]): list of Kraus matrices

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> K_list = qml.PhaseFlip(0.5, wires=0).kraus_matrices()
        >>> res = qml.QubitChannel.compute_kraus_matrices(K_list)
        >>> all(np.allclose(r, k) for r, k  in zip(res, K_list))
        True
        """
        return list(kraus_matrices)


class ThermalRelaxationError(Channel):
    r"""
    Thermal relaxation error channel.

    This channel is modelled by the following Kraus matrices:

    Case :math:`T_2 \leq T_1`:

    .. math::
        K_0 = \sqrt{1 - p_z - p_{r0} - p_{r1}} \begin{bmatrix}
                1 & 0 \\
                0 & 1
                \end{bmatrix}

    .. math::
        K_1 = \sqrt{p_z}\begin{bmatrix}
                1 & 0  \\
                0 & -1
                \end{bmatrix}

    .. math::
        K_2 = \sqrt{p_{r0}}\begin{bmatrix}
                1 & 0  \\
                0 & 0
                \end{bmatrix}

    .. math::
        K_3 = \sqrt{p_{r0}}\begin{bmatrix}
                0 & 1  \\
                0 & 0
                \end{bmatrix}

    .. math::
        K_4 = \sqrt{p_{r1}}\begin{bmatrix}
                0 & 0  \\
                1 & 0
                \end{bmatrix}

    .. math::
        K_5 = \sqrt{p_{r1}}\begin{bmatrix}
                0 & 0  \\
                0 & 1
                \end{bmatrix}

    where :math:`p_{r0} \in [0, 1]` is the probability of a reset to 0, :math:`p_{r1} \in [0, 1]` is the probability of
    a reset to 1 error, :math:`p_z \in [0, 1]` is the probability of a phase flip (Pauli :math:`Z`) error.

    Case :math:`T_2 > T_1`:
    The Choi matrix is given by

    .. math::
        \Lambda = \begin{bmatrix}
                        1 - p_e * p_{reset} & 0 & 0 & eT_2 \\
                        0 & p_e * p_{reset} & 0 & 0 \\
                        0 & 0 & (1 - p_e) * p_{reset} & 0 \\
                        eT_2 & 0 & 0 & 1 - (1 - p_e) * p_{reset}
                        \end{bmatrix}

    .. math::
        K_N = \sqrt{\lambda} \Phi(\nu_{\lambda})

    where :math:`\lambda` are the eigenvalues of the Choi matrix, :math:`\nu_{\lambda}` are the eigenvectors of
    the choi_matrix, and :math:`\Phi(x)` is a isomorphism from :math:`\mathbb{C}^{n^2}`
    to :math:`\mathbb{C}^{n \times n}` with column-major order mapping.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 4

    Args:
        pe (float): exited state population. Must be between ``0`` and ``1``
        t1 (float): the :math:`T_1` relaxation constant
        t2 (float): the :math:`T_2` dephasing constant. Must be less than :math:`2 T_1`
        tg (float): the gate time for relaxation error
        wires (Sequence[int] or int): the wire the channel acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 4
    num_wires = 1
    grad_method = "F"

    def __init__(self, pe, t1, t2, tq, wires, id=None):
        super().__init__(pe, t1, t2, tq, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(pe, t1, t2, tg):  # pylint:disable=arguments-differ
        """Kraus matrices representing the ThermalRelaxationError channel.

        Args:
            pe (float): exited state population. Must be between ``0`` and ``1``
            t1 (float): the :math:`T_1` relaxation constant
            t2 (float): The :math:`T_2` dephasing constant. Must be less than :math:`2 T_1`
            tg (float): the gate time for relaxation error

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.ThermalRelaxationError.compute_kraus_matrices(0.1, 1.2, 1.3, 0.1)
        [array([[0.        , 0.        ], [0.08941789, 0.        ]]),
         array([[0.        , 0.26825366], [0.        , 0.        ]]),
         array([[-0.12718544,  0.        ], [ 0.        ,  0.13165421]]),
         array([[0.98784022, 0.        ], [0.        , 0.95430977]])]
        """
        if not np.is_abstract(pe) and not 0.0 <= pe <= 1.0:
            raise ValueError("pe must be between 0 and 1.")
        if not np.is_abstract(tg) and tg < 0:
            raise ValueError(f"Invalid gate_time tg ({tg} < 0)")
        if not np.is_abstract(t1) and t1 <= 0:
            raise ValueError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
        if not np.is_abstract(t2) and t2 <= 0:
            raise ValueError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
        if not np.is_abstract(t2 - 2 * t1) and t2 - 2 * t1 > 0:
            raise ValueError("Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")
        # T1 relaxation rate
        eT1 = np.exp(-tg / t1)
        p_reset = 1 - eT1
        # T2 dephasing rate
        eT2 = np.exp(-tg / t2)

        def kraus_ops_small_t2():
            pz = (1 - p_reset) * (1 - eT2 / eT1) / 2
            pr0 = (1 - pe) * p_reset
            pr1 = pe * p_reset
            pid = 1 - pz - pr0 - pr1

            K0 = np.sqrt(pid + np.eps) * np.eye(2)
            K1 = np.sqrt(pz + np.eps) * np.array([[1, 0], [0, -1]])
            K2 = np.sqrt(pr0 + np.eps) * np.array([[1, 0], [0, 0]])
            K3 = np.sqrt(pr0 + np.eps) * np.array([[0, 1], [0, 0]])
            K4 = np.sqrt(pr1 + np.eps) * np.array([[0, 0], [1, 0]])
            K5 = np.sqrt(pr1 + np.eps) * np.array([[0, 0], [0, 1]])

            return [K0, K1, K2, K3, K4, K5]

        def kraus_ops_large_t2():
            e0 = p_reset * pe
            v0 = np.array([[0, 0], [1, 0]])
            K0 = np.sqrt(e0 + np.eps) * v0
            e1 = -p_reset * pe + p_reset
            v1 = np.array([[0, 1], [0, 0]])
            K1 = np.sqrt(e1 + np.eps) * v1
            base = sum(
                (
                    4 * eT2**2,
                    4 * p_reset**2 * pe**2,
                    -4 * p_reset**2 * pe,
                    p_reset**2,
                    np.eps,
                )
            )
            common_term = np.sqrt(base)
            e2 = 1 - p_reset / 2 - common_term / 2
            term2 = 2 * eT2 / (2 * p_reset * pe - p_reset - common_term)
            v2 = (term2 * np.array([[1, 0], [0, 0]]) + np.array([[0, 0], [0, 1]])) / np.sqrt(
                term2**2 + 1
            )
            K2 = np.sqrt(e2 + np.eps) * v2
            term3 = 2 * eT2 / (2 * p_reset * pe - p_reset + common_term)
            e3 = 1 - p_reset / 2 + common_term / 2
            v3 = (term3 * np.array([[1, 0], [0, 0]]) + np.array([[0, 0], [0, 1]])) / np.sqrt(
                term3**2 + 1
            )
            K3 = np.sqrt(e3 + np.eps) * v3
            K4 = np.cast_like(np.zeros((2, 2)), K1)
            K5 = np.cast_like(np.zeros((2, 2)), K1)

            return [K0, K1, K2, K3, K4, K5]

        K = np.cond(t2 <= t1, kraus_ops_small_t2, kraus_ops_large_t2, ())
        return K


__qubit_channels__ = {
    "AmplitudeDamping",
    "GeneralizedAmplitudeDamping",
    "PhaseDamping",
    "DepolarizingChannel",
    "BitFlip",
    "PhaseFlip",
    "PauliError",
    "ResetError",
    "QubitChannel",
    "ThermalRelaxationError",
}

__all__ = list(__qubit_channels__)
