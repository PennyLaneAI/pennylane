# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the functions needed for estimating the number of logical qubits and
non-Clifford gates for quantum algorithms in first quantization using a plane-wave basis.
"""
# pylint: disable=no-self-use disable=too-many-arguments disable=too-many-instance-attributes
import numpy as np
from scipy import integrate

from pennylane.operation import AnyWires, Operation


class FirstQuantization(Operation):
    r"""Estimate the number of non-Clifford gates and logical qubits for a quantum phase estimation
    algorithm in first quantization using a plane-wave basis.

    To estimate the gate and qubit costs for implementing this method, the number of plane waves,
    the number of electrons and the lattice vectors need to be defined. The costs can then be
    computed using the functions :func:`~.pennylane.resource.FirstQuantization.gate_cost` and
    :func:`~.pennylane.resource.FirstQuantization.qubit_cost` with a target error that has the default
    value of 0.0016 Ha (chemical accuracy). Atomic units are used throughout the class.

    Args:
        n (int): number of plane waves
        eta (int): number of electrons
        omega (float): unit cell volume
        error (float): target error in the algorithm
        charge (int): total electric charge of the system
        br (int): number of bits for ancilla qubit rotation
        vectors (array[float]): lattice vectors

    **Example**

    >>> n = 100000
    >>> eta = 156
    >>> vectors = np.array([[10.46219511,  0.00000000,  0.00000000],
    ...                     [ 0.00000000, 10.46219511,  0.00000000],
    ...                     [ 0.00000000,  0.00000000, 10.46219511]])
    >>> algo = FirstQuantization(n, eta, vectors=vectors)
    >>> print(algo.lamb,  # the 1-Norm of the Hamiltonian
    >>>       algo.gates, # estimated number of non-Clifford gates
    >>>       algo.qubits # estimated number of logical qubits
    >>>       )
    649912.4804278888 1.1e+13 4416

    .. details::
        :title: Theory

        Following `PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_
        , the target algorithm error, :math:`\epsilon`, is distributed among four different sources
        of error using Eq. (131)

        .. math::
            \epsilon^2 \geq \epsilon_{qpe}^2 + (\epsilon_{\mathcal{M}} + \epsilon_R + \epsilon_T)^2,

        where :math:`\epsilon_{qpe}` is the quantum phase estimation error and
        :math:`\epsilon_{\mathcal{M}}`, :math:`\epsilon_R`, and :math:`\epsilon_T` are defined in
        Eqs. (132-134).

        Here, we fix :math:`\epsilon_{\mathcal{M}} = \epsilon_R = \epsilon_T = \alpha \epsilon` with
        a default value of :math:`\alpha = 0.01` and obtain

        .. math::
            \epsilon_{qpe} = \sqrt{\epsilon^2 [1 - (3 \alpha)^2]}.

        Note that the user only needs to define the target algorithm error :math:`\epsilon`. The
        error distribution takes place inside the functions.
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(
        self,
        n,
        eta,
        omega=None,
        error=0.0016,
        charge=0,
        br=7,
        vectors=None,
    ):
        self.n = n
        self.eta = eta
        self.omega = omega
        self.error = error
        self.charge = charge
        self.br = br
        self.vectors = vectors
        self.cubic = True

        if omega is None and vectors is None:
            raise ValueError("The lattice vectors must be provided.")

        if omega is not None and vectors is not None:
            raise ValueError(
                "The lattice vectors and the unit cell volume should not be provided"
                " together. Please provide only one of them."
            )

        if self.vectors is not None:
            self.omega = np.abs(np.sum((np.cross(vectors[0], vectors[1]) * vectors[2])))
            recip_vectors = (  # taken from Eq. (35) of arXiv:2302.07981v1 (2023)
                2
                * np.pi
                / self.omega
                * np.array([np.cross(vectors[i], vectors[j]) for i, j in [(1, 2), (2, 0), (0, 1)]])
            )
            bbt = np.matrix(recip_vectors) @ np.matrix(recip_vectors).T
            self.cubic = np.linalg.norm(bbt - (recip_vectors**2).max() * np.identity(3)) < 1e-6

        self.lamb = self.norm(
            self.n, self.eta, self.omega, self.error, self.br, self.charge, self.cubic, self.vectors
        )

        self.gates = self.gate_cost(
            self.n, self.eta, self.omega, self.error, self.br, self.charge, self.cubic, self.vectors
        )

        self.qubits = self.qubit_cost(
            self.n, self.eta, self.omega, self.error, self.br, self.charge, self.cubic, self.vectors
        )

        super().__init__(wires=range(self.qubits))

    def _flatten(self):
        return (self.n, self.eta), (
            ("omega", self.omega),
            ("error", self.error),
            ("charge", self.charge),
            ("br", self.br),
            ("vectors", self.vectors),
        )

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, **dict(metadata))

    @staticmethod
    def success_prob(n, br):
        r"""Return the probability of success for state preparation.

        The expression for computing the probability of success is taken from Eqs. (59, 60) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

        Args:
            n (int): number of basis states to create an equal superposition for state preparation
            br (int): number of bits for ancilla qubit rotation

        Returns:
            float: probability of success for state preparation

        **Example**

        >>> n = 3
        >>> br = 8
        >>> success_prob(n, br)
        0.9999928850303523
        """
        if n <= 0:
            raise ValueError("The number of plane waves must be a positive number.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        c = n / 2 ** np.ceil(np.log2(n))
        d = 2 * np.pi / 2**br

        theta = d * np.round((1 / d) * np.arcsin(np.sqrt(1 / (4 * c))))

        p = c * ((1 + (2 - 4 * c) * np.sin(theta) ** 2) ** 2 + np.sin(2 * theta) ** 2)

        return p

    @staticmethod
    def norm(n, eta, omega, error, br=7, charge=0, cubic=True, vectors=None):
        r"""Return the 1-norm of a first-quantized Hamiltonian in the plane-wave basis.

        The expressions needed for computing the norm are taken from
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].
        The norm is computed assuming that amplitude ampliﬁcation is performed.

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            cubic (bool): True if the unit cell is cubic
            vectors (array[float]): lattice vectors

        Returns:
            float: 1-norm of a first-quantized Hamiltonian in the plane-wave basis

        **Example**

        >>> n = 10000
        >>> eta = 156
        >>> omega = 1145.166
        >>> error = 0.001
        >>> norm(n, eta, omega, error)
        281053.75612801575

        .. details::
            :title: Theory

            To compute the norm, for numerical convenience, we use the following modified
            expressions to obtain parameters that contain a sum over
            :math:`\frac{1}{\left \| \nu \right \|^k}` where :math:`\nu` denotes an element of the
            set of reciprocal lattice vectors, :math:`G_0`, and
            :math:`k \in \left \{ 1, 2 \right \}`.

            For :math:`\lambda_{\nu}` defined in Eq. (25) of
            `PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_
            as

            .. math::

                \lambda_{\nu} = \sum_{\nu \in G_0} \frac{1}{\left \| \nu \right \|^2},

            we follow Eq. (F6) of
            `PRX 8, 011044 (2018) <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.011044>`_
            and use

            .. math::

                \lambda_{\nu} = 4\pi \left ( \frac{\sqrt{3}}{2} N^{1/3} - 1 \right) + 3 - \frac{3}{N^{1/3}}
                + 3 \int_{x=1}^{N^{1/3}} \int_{y=1}^{N^{1/3}} \frac{1}{x^2 + y^2} dydx.

            We also need to compute :math:`\lambda^{\alpha}_{\nu}` defined in Eq. (123) of
            `PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_

            .. math::

                \lambda^{\alpha}_{\nu} = \alpha \sum_{\nu \in G_0} \frac{\left \lceil
                \mathcal{M}(2^{\mu - 2}) / \left \| \nu \right \|^2 \right \rceil}{\mathcal{M}
                2^{2\mu - 4}},

            which we compute here, for :math:`\alpha = 1`, as

            .. math::

                \lambda^{1}_{\nu} = \lambda_{\nu} + \epsilon_l,

            where :math:`\epsilon_l` is simply defined as the difference of
            :math:`\lambda^{1}_{\nu}` and :math:`\lambda_{\nu}`. We follow Eq. (113) of
            `PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_
            to derive an upper bound for its absolute value:

            .. math::

                |\epsilon_l| \le \frac{4}{2^{n_m}} (7 \times 2^{n_p + 1} + 9 n_p - 11 - 3 \times 2^{-n_p}),

            where :math:`\mathcal{M} = 2^{n_m}` and :math:`n_m` is defined in Eq. (132) of
            `PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_.
            Finally, for :math:`p_{\nu}` defined in Eq. (128) of
            `PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_

            .. math::

                p_{\nu} = \sum_{\mu = 2}^{n_p + 1} \sum_{\nu \in B_{\mu}} \frac{\left \lceil M(2^{\mu-2}
                / \left \| \nu \right \|)^2 \right \rceil}{M 2^{2\mu} 2^{n_{\mu} + 1}},

            we use the upper bound from Eq. (29) in
            `arXiv:1807.09802v2 <https://arxiv.org/abs/1807.09802v2>`_ which gives
            :math:`p_{\nu} = 0.2398`.
        """
        if n <= 0:
            raise ValueError("The number of plane waves must be a positive number.")

        if eta <= 0 or not isinstance(eta, (int, np.integer)):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        if not cubic:
            return FirstQuantization._norm_noncubic(n, eta, error, br, charge, vectors)[0]

        l_z = eta + charge

        # target error in the qubitization of U+V which we set to be 0.01 of the algorithm error
        error_uv = 0.01 * error

        # taken from Eq. (22) of PRX Quantum 2, 040332 (2021)
        n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))

        n0 = n ** (1 / 3)
        lambda_nu = (  # expression is taken from Eq. (F6) of PRX 8, 011044 (2018)
            4 * np.pi * (np.sqrt(3) * n ** (1 / 3) / 2 - 1)
            + 3
            - 3 / n ** (1 / 3)
            + 3 * integrate.nquad(lambda x, y: 1 / (x**2 + y**2), [[1, n0], [1, n0]])[0]
        )
        n_m = int(
            np.log2(  # taken from Eq. (132) of PRX Quantum 2, 040332 (2021)
                (2 * eta)
                / (error_uv * np.pi * omega ** (1 / 3))
                * (eta - 1 + 2 * l_z)
                * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
            )
        )
        # computed using Eq. (113) of PRX Quantum 2, 040332 (2021)
        lambda_nu_1 = lambda_nu + 4 / 2**n_m * (
            7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p)
        )

        p_nu = 0.2398  # approximation from Eq. (29) in arxiv:1807.09802
        p_nu_amp = (
            np.sin(3 * np.arcsin(np.sqrt(p_nu))) ** 2
        )  # Eq. (129), PRX Quantum 2, 040332 (2021)

        # lambda_u and lambda_v are taken from Eq. (25) of PRX Quantum 2, 040332 (2021)
        lambda_u = eta * l_z * lambda_nu / (np.pi * omega ** (1 / 3))
        lambda_v = eta * (eta - 1) * lambda_nu / (2 * np.pi * omega ** (1 / 3))

        # taken from Eq. (71) of PRX Quantum 2, 040332 (2021)
        lambda_t_p = 6 * eta * np.pi**2 / omega ** (2 / 3) * 2 ** (2 * n_p - 2)

        # lambda_u_1 and lambda_v_1 are taken from Eq. (124) of PRX Quantum 2, 040332 (2021)
        lambda_u_1 = lambda_u * lambda_nu_1 / lambda_nu
        lambda_v_1 = lambda_v * lambda_nu_1 / lambda_nu

        # taken from Eq. (63) of PRX Quantum 2, 040332 (2021)
        p_eq = (
            FirstQuantization.success_prob(3, 8)
            * FirstQuantization.success_prob(3 * eta + 2 * charge, br)
            * FirstQuantization.success_prob(eta, br) ** 2
        )

        # final lambda value is computed from Eq. (126) of PRX Quantum 2, 040332 (2021)
        lambda_a = lambda_t_p + lambda_u_1 + lambda_v_1
        lambda_b = (lambda_u_1 + lambda_v_1 / (1 - 1 / eta)) / p_nu_amp

        return np.maximum(lambda_a, lambda_b) / p_eq

    @staticmethod
    def _cost_qrom(lz):
        r"""Return the minimum number of Toffoli gates needed for erasing the output of a QROM.

        Args:
            lz (int): sum of the atomic numbers

        Returns:
            int: the minimum cost of erasing the output of a QROM

        **Example**

        >>> lz = 100
        >>> _cost_qrom(lz)
        21
        """
        if lz <= 0 or not isinstance(lz, (int, np.integer)):
            raise ValueError("The sum of the atomic numbers must be a positive integer.")

        k_f = np.floor(np.log2(lz) / 2)
        k_c = np.ceil(np.log2(lz) / 2)

        cost_f = int(2**k_f + np.ceil(2 ** (-1 * k_f) * lz))
        cost_c = int(2**k_c + np.ceil(2 ** (-1 * k_c) * lz))

        return min(cost_f, cost_c)

    @staticmethod
    def unitary_cost(n, eta, omega, error, br=7, charge=0):
        r"""Return the number of Toffoli gates needed to implement the qubitization unitary
        operator.

        The expression for computing the cost is taken from Eq. (125) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system

        Returns:
            int: the number of Toffoli gates needed to implement the qubitization unitary operator

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 169.69608
        >>> error = 0.01
        >>> unitary_cost(n, eta, omega, error)
        17033
        """
        if n <= 0:
            raise ValueError("The number of plane waves must be a positive number.")

        if eta <= 0 or not isinstance(eta, (int, np.integer)):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        lamb = FirstQuantization.norm(n, eta, omega, error, br, charge)
        alpha = 0.01
        l_z = eta + charge
        l_nu = 2 * np.pi * n ** (2 / 3)

        # defined in the third and second paragraphs of page 15 of PRX Quantum 2, 040332 (2021)
        n_eta = np.ceil(np.log2(eta))
        n_etaz = np.ceil(np.log2(eta + 2 * l_z))

        # n_p is taken from Eq. (22)
        n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))

        # errors in Eqs. (132-134) are set to be 0.01 of the algorithm error
        error_t = alpha * error
        error_r = alpha * error
        error_m = alpha * error

        # parameters taken from Eqs. (132-134) of PRX Quantum 2, 040332 (2021)
        n_t = int(np.log2(np.pi * lamb / error_t))  # Eq. (134)
        n_r = int(np.log2((eta * l_z * l_nu) / (error_r * omega ** (1 / 3))))  # Eq. (133)
        n_m = int(
            np.log2(  # Eq. (132)
                (2 * eta)
                / (error_m * np.pi * omega ** (1 / 3))
                * (eta - 1 + 2 * l_z)
                * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
            )
        )

        e_r = FirstQuantization._cost_qrom(l_z)

        # taken from Eq. (125)
        cost = 2 * (n_t + 4 * n_etaz + 2 * br - 12) + 14 * n_eta + 8 * br - 36
        cost += 3 * (3 * n_p**2 + 15 * n_p - 7 + 4 * n_m * (n_p + 1))
        cost += l_z + e_r + 2 * (2 * n_p + 2 * br - 7) + 12 * eta * n_p
        cost += 5 * (n_p - 1) + 2 + 24 * n_p + 6 * n_p * n_r + 18
        cost += n_etaz + 2 * n_eta + 6 * n_p + n_m + 16

        return int(np.ceil(cost))

    @staticmethod
    def estimation_cost(n, eta, omega, error, br=7, charge=0, cubic=True, vectors=None):
        r"""Return the number of calls to the unitary needed to achieve the desired error in quantum
        phase estimation.

        The expression for computing the cost is taken from Eq. (125) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            cubic (bool): True if the unit cell is cubic
            vectors (array[float]): lattice vectors

        Returns:
            int: number of calls to unitary

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 1145.166
        >>> error = 0.01
        >>> estimation_cost(n, eta, omega, error)
        102133985
        """
        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        lamb = FirstQuantization.norm(
            n, eta, omega, error, br=br, charge=charge, cubic=cubic, vectors=vectors
        )

        if cubic:
            alpha = 0.01
            # qpe_error obtained to satisfy inequality (131)
            error_qpe = np.sqrt(error**2 * (1 - (3 * alpha) ** 2))
        else:
            alpha = 0.0248759298
            # qpe_error obtained to satisfy inequality (131)
            error_qpe = np.sqrt(error**2 * (1 - (4 * alpha) ** 2))

        return int(np.ceil(np.pi * lamb / (2 * error_qpe)))

    @staticmethod
    def gate_cost(n, eta, omega, error, br=7, charge=0, cubic=True, vectors=None):
        r"""Return the total number of Toffoli gates needed to implement the first quantization
        algorithm.

        The expression for computing the cost is taken from Eq. (125) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            cubic (bool): True if the unit cell is cubic
            vectors (array[float]): lattice vectors

        Returns:
            int: the number of Toffoli gates needed to implement the first quantization algorithm

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 169.69608
        >>> error = 0.01
        >>> gate_cost(n, eta, omega, error)
        3676557345574
        """
        if n <= 0:
            raise ValueError("The number of plane waves must be a positive number.")

        if eta <= 0 or not isinstance(eta, (int, np.integer)):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        e_cost = FirstQuantization.estimation_cost(
            n, eta, omega, error, br=br, charge=charge, cubic=cubic, vectors=vectors
        )

        if cubic:
            u_cost = FirstQuantization.unitary_cost(n, eta, omega, error, br, charge)
        else:
            u_cost = FirstQuantization._unitary_cost_noncubic(n, eta, error, br, charge, vectors)

        return e_cost * u_cost

    @staticmethod
    def qubit_cost(n, eta, omega, error, br=7, charge=0, cubic=True, vectors=None):
        r"""Return the number of logical qubits needed to implement the first quantization
        algorithm.

        The expression for computing the cost is taken from Eq. (101) of
        [`arXiv:2204.11890v1 <https://arxiv.org/abs/2204.11890v1>`_].

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            cubic (bool): True if the unit cell is cubic
            vectors (array[float]): lattice vectors

        Returns:
            int: number of logical qubits needed to implement the first quantization algorithm

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 169.69608
        >>> error = 0.01
        >>> qubit_cost(n, eta, omega, error)
        4377
        """
        if n <= 0:
            raise ValueError("The number of plane waves must be a positive number.")

        if eta <= 0 or not isinstance(eta, (int, np.integer)):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        if not cubic:
            return FirstQuantization._qubit_cost_noncubic(n, eta, error, br, charge, vectors)

        lamb = FirstQuantization.norm(n, eta, omega, error, br=br, charge=charge)
        alpha = 0.01
        l_z = eta + charge
        l_nu = 2 * np.pi * n ** (2 / 3)

        # n_p is taken from Eq. (22) of PRX Quantum 2, 040332 (2021)
        n_p = np.ceil(np.log2(n ** (1 / 3) + 1))

        # errors in Eqs. (132-134) of PRX Quantum 2, 040332 (2021),
        # set to 0.01 of the algorithm error
        error_t = alpha * error
        error_r = alpha * error
        error_m = alpha * error

        # parameters taken from Eqs. (132-134) of PRX Quantum 2, 040332 (2021)
        n_t = int(np.log2(np.pi * lamb / error_t))  # Eq. (134)
        n_r = int(np.log2((eta * l_z * l_nu) / (error_r * omega ** (1 / 3))))  # Eq. (133)
        n_m = int(
            np.log2(  # Eq. (132)
                (2 * eta)
                / (error_m * np.pi * omega ** (1 / 3))
                * (eta - 1 + 2 * l_z)
                * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
            )
        )

        alpha = 0.01
        # qpe_error obtained to satisfy inequality (131) of PRX Quantum 2, 040332 (2021)
        error_qpe = np.sqrt(error**2 * (1 - (3 * alpha) ** 2))

        # the expression for computing the cost is taken from Eq. (101) of arXiv:2204.11890v1
        qubits = 3 * eta * n_p + 4 * n_m * n_p + 12 * n_p
        qubits += 2 * (np.ceil(np.log2(np.ceil(np.pi * lamb / (2 * error_qpe))))) + 5 * n_m
        qubits += 2 * np.ceil(np.log2(eta)) + 3 * n_p**2 + np.ceil(np.log2(eta + 2 * l_z))
        qubits += np.maximum(5 * n_p + 1, 5 * n_r - 4) + np.maximum(n_t, n_r + 1) + 33

        return int(np.ceil(qubits))

    @staticmethod
    def _norm_noncubic(
        n,
        eta,
        error,
        br,
        charge,
        vectors,
    ):
        r"""Return the 1-norm of a first-quantized Hamiltonian in the plane-wave basis
        for non-cubic systems.

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            vectors (array[float]): lattice vectors

        Returns:
            float: 1-norm of a first-quantized Hamiltonian in the plane-wave basis
        """
        omega = np.abs(np.sum((np.cross(vectors[0], vectors[1]) * vectors[2])))

        recip_vectors = (  # taken from Eq. (35) of arXiv:2302.07981v1 (2023)
            2
            * np.pi
            / omega
            * np.array([np.cross(vectors[i], vectors[j]) for i, j in [(1, 2), (2, 0), (0, 1)]])
        )

        bbt = np.matrix(recip_vectors) @ np.matrix(recip_vectors).T

        orthogonal = (
            np.linalg.norm(bbt - np.array([np.max(b**2) for b in recip_vectors]) * np.identity(3))
            < 1e-6
        )

        l_z = eta + charge

        alpha = 0.0248759298  # optimal value for lower resource estimates
        p_th = 0.95  # optimal value for lower resource estimates

        error_uv = alpha * error

        # taken from Eq. (22) of PRX Quantum 2, 040332 (2021)
        n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))

        n0 = n ** (1 / 3)

        # defined in Eq. (F3) of arXiv:2302.07981v1 (2023)
        bmin = np.min(np.linalg.svd(recip_vectors)[1])

        n_m = int(
            np.ceil(
                np.log2(  # taken from Eq. (132) of PRX Quantum 2, 040332 (2021) with
                    # modifications taken from arXiv:2302.07981v1 (2023)
                    (8 * np.pi * eta)
                    / (error_uv * omega * bmin**2)
                    * (eta - 1 + 2 * l_z)
                    * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
                )
            )
        )

        lambda_nu = (  # expression is taken from Eq. (F6) of PRX 8, 011044 (2018)
            # bmin**2 is needed for non-cubic systems, from Eq. (F3) of arXiv:2302.07981v1 (2023)
            4 * np.pi * (np.sqrt(3) * n ** (1 / 3) / 2 - 1)
            + 3
            - 3 / n ** (1 / 3)
            + 3 * integrate.nquad(lambda x, y: 1 / (x**2 + y**2), [[1, n0], [1, n0]])[0]
        ) / bmin**2

        # computed using error term derived in Eq. (113) of PRX Quantum 2, 040332 (2021)
        # bmin**2 is needed for non-cubic systems, from Eq. (F3) of arXiv:2302.07981v1 (2023)
        lambda_nu_1 = lambda_nu + 4 / (2**n_m * bmin**2) * (
            7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p)
        )

        # taken from Eq. (I7) of arXiv:2302.07981v1 (2023)
        p_nu = lambda_nu_1 * bmin**2 / 2 ** (n_p + 6)

        # amplitude amplification steps, taken from Eq. (117) of PRX Quantum 2, 040332 (2021)
        p_nu_amp = 0
        aa_steps = 0
        n_steps = 30  # maximum number of amplitude amplification steps
        for i in reversed(range(n_steps)):
            probability = (np.sin((2 * i + 1) * np.arcsin(np.sqrt(p_nu)))) ** 2
            if probability > p_th:
                aa_steps = i
                p_nu_amp = probability

        # lambda_u and lambda_v are taken from Eq. (25) of PRX Quantum 2, 040332 (2021) with
        # modifications taken from arXiv:2302.07981v1 (2023)
        lambda_u = 4 * np.pi * eta * l_z * lambda_nu / omega
        lambda_v = 2 * np.pi * eta * (eta - 1) * lambda_nu / omega

        # lambda_u_1 and lambda_v_1 are taken from Eq. (124) of PRX Quantum 2, 040332 (2021)
        lambda_u_1 = lambda_u * lambda_nu_1 / lambda_nu
        lambda_v_1 = lambda_v * lambda_nu_1 / lambda_nu

        b_mat = np.matrix(recip_vectors)
        abs_sum = np.abs(b_mat @ b_mat.T).flatten().sum()

        # taken from Appendix I.1 of arXiv:2302.07981v1 (2023)
        if orthogonal:
            lambda_t_p = abs_sum * eta * 2 ** (2 * n_p - 2) / 4
        else:
            lambda_t_p = abs_sum * eta * 2 ** (2 * n_p - 2) / 2

        # taken from Eq. (63) of PRX Quantum 2, 040332 (2021) with
        # modifications taken from arXiv:2302.07981v1 (2023)
        p_eq = (
            FirstQuantization.success_prob(3 * eta + 2 * charge, br)
            * FirstQuantization.success_prob(eta, br) ** 2
        )

        # final lambda value is computed from case checking derived in Section E and Appendix D of
        # PRX Quantum 2, 040332 (2021)
        if p_nu * lambda_t_p >= (1 - p_nu) * (lambda_u_1 + lambda_v_1):
            raise ValueError("The computed 1-norm is zero.")  # return 0.0, 0.0
        if p_nu_amp * lambda_t_p >= (1 - p_nu_amp) * (lambda_u_1 + lambda_v_1):
            return (lambda_t_p + lambda_u_1 + lambda_v_1) / p_eq, aa_steps
        return ((lambda_u_1 + lambda_v_1 / (1 - 1 / eta)) / p_nu_amp) / p_eq, aa_steps

    @staticmethod
    def _qubit_cost_noncubic(n, eta, error, br, charge, vectors):
        r"""Return the number of logical qubits needed to implement the first quantization
        algorithm for non-cubic systems.

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            vectors (array[float]): lattice vectors

        Returns:
            int: number of logical qubits needed to implement the first quantization algorithm
        """

        lambda_total, _ = FirstQuantization._norm_noncubic(
            n,
            eta,
            error,
            br,
            charge,
            vectors,
        )

        omega = np.abs(np.sum((np.cross(vectors[0], vectors[1]) * vectors[2])))

        recip_vectors = (  # taken from Eq. (35) of arXiv:2302.07981v1 (2023)
            2
            * np.pi
            / omega
            * np.array([np.cross(vectors[i], vectors[j]) for i, j in [(1, 2), (2, 0), (0, 1)]])
        )

        l_z = eta + charge
        l_nu = 2 * np.pi * n ** (2 / 3)

        # taken from Eq. (22) of PRX Quantum 2, 040332 (2021)
        n_p = np.ceil(np.log2(n ** (1 / 3) + 1))

        # defined in Eq. (F3) of arXiv:2302.07981v1 (2023)
        bmin = np.min(np.linalg.svd(recip_vectors)[1])

        alpha = 0.0248759298  # optimal value for lower resource estimates

        # errors derived according to the strategy in Appendix M.1 of arXiv:2302.07981v1 (2023)
        error_t, error_r, error_m, error_b = [alpha * error] * 4

        # parameters taken from PRX Quantum 2, 040332 (2021)
        n_t = int(np.ceil(np.log2(np.pi * lambda_total / error_t)))  # Eq. (134)
        n_r = int(np.ceil(np.log2((eta * l_z * l_nu) / (error_r * omega ** (1 / 3)))))  # Eq. (133)
        n_m = int(  # taken from Eq. (J13) arXiv:2302.07981v1 (2023)
            np.ceil(
                np.log2(
                    (8 * np.pi * eta)
                    / (error_m * omega * bmin**2)
                    * (eta - 1 + 2 * l_z)
                    * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
                )
            )
        )

        # qpe error obtained to satisfy a modification of
        # inequality Eq. (131) of PRX Quantum 2, 040332 (2021) including error_b
        n_errors = 4
        error_qpe = np.sqrt(error**2 * (1 - (n_errors * alpha) ** 2))

        # adapted from equations (L1, L2) in Appendix L of arXiv:2302.07981v1 (2023)
        clean_temp_H_cost = max([5 * n_r - 4, 5 * n_p + 1]) + max([5, n_m + 3 * n_p])
        reflection_cost = (
            np.ceil(np.log2(eta + 2 * l_z)) + 2 * np.ceil(np.log2(eta)) + 6 * n_p + n_m + 16 + 3
        )
        clean_temp_cost = max([clean_temp_H_cost, reflection_cost])

        # the expression for computing the cost is taken from Appendix C of
        # PRX Quantum 2, 040332 (2021) and adapting to non-cubic using Appendix L of
        # arXiv:2302.07981v1 (2023)
        clean_cost = 3 * eta * n_p
        clean_cost += np.ceil(np.log2(np.ceil(np.pi * lambda_total / (2 * error_qpe))))
        clean_cost += 1 + 1 + np.ceil(np.log2(eta + 2 * l_z)) + 3 + 3
        clean_cost += 2 * np.ceil(np.log2(eta)) + 5 + 3 * (n_p + 1)
        clean_cost += n_p + n_m + 3 * n_p + 2 + 2 * n_p + 1 + 1 + 2 + 2 * n_p + 6 + 1

        clean_cost += clean_temp_cost

        # taken from Eq. (J7) of arXiv:2302.07981v1 (2023)
        n_b = np.ceil(
            np.log2(
                4
                * np.pi
                * eta
                * 2 ** (2 * n_p - 2)
                * np.abs(np.matrix(recip_vectors) @ np.matrix(recip_vectors).T).flatten().sum()
                / error_b
            )
        )
        clean_cost += np.max([n_r + 1, n_t, n_b]) + 6 + n_m + 1

        return int(np.ceil(clean_cost))

    @staticmethod
    def _unitary_cost_noncubic(n, eta, error, br, charge, vectors):
        r"""Return the number of Toffoli gates needed to implement the qubitization unitary
        operator for non-cubic systems.

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            vectors (array[float]): lattice vectors

        Returns:
            int: the number of Toffoli gates needed to implement the qubitization unitary operator
        """

        lambda_total, aa_steps = FirstQuantization._norm_noncubic(
            n,
            eta,
            error,
            br,
            charge,
            vectors,
        )

        omega = np.abs(np.sum((np.cross(vectors[0], vectors[1]) * vectors[2])))

        recip_vectors = (  # taken from Eq. (35) of arXiv:2302.07981v1 (2023)
            2
            * np.pi
            / omega
            * np.array([np.cross(vectors[i], vectors[j]) for i, j in [(1, 2), (2, 0), (0, 1)]])
        )

        n_tof = 500  # optimal value for lower resource estimates

        alpha = 0.0248759298  # optimal value for lower resource estimates
        l_z = eta + charge
        l_nu = 2 * np.pi * n ** (2 / 3)

        # defined in the third and second paragraphs of page 15 of PRX Quantum 2, 040332 (2021)
        n_eta = np.ceil(np.log2(eta))
        n_etaz = np.ceil(np.log2(eta + 2 * l_z))

        # taken from Eq. (22) of PRX Quantum 2, 040332 (2021)
        n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))

        # errors in Eqs. (132-134) of PRX Quantum 2, 040332 (2021)
        error_t, error_r, error_m = [alpha * error] * 3

        # parameters taken from PRX Quantum 2, 040332 (2021)
        n_t = int(np.ceil(np.log2(np.pi * lambda_total / error_t)))  # Eq. (134)
        n_r = int(np.ceil(np.log2((eta * l_z * l_nu) / (error_r * omega ** (1 / 3)))))  # Eq. (133)

        # defined in Eq. (F3) of arXiv:2302.07981v1 (2023)
        bmin = np.min(np.linalg.svd(recip_vectors)[1])

        # equivalent to Eq. (J13) of arXiv:2302.07981v1 (2023)
        n_m = int(
            np.ceil(
                np.log2(
                    (8 * np.pi * eta)
                    / (error_m * omega * bmin**2)
                    * (eta - 1 + 2 * l_z)
                    * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
                )
            )
        )

        e_r = FirstQuantization._cost_qrom(l_z)

        # taken from Eq. (125) of PRX Quantum 2, 040332 (2021)
        cost = 2 * (n_t + 4 * n_etaz + 2 * br - 12) + 14 * n_eta + 8 * br - 36
        cost += (2 * aa_steps + 1) * (3 * n_p**2 + 15 * n_p - 7 + 4 * n_m * (n_p + 1))
        cost += l_z + e_r + 2 * (2 * n_p + 2 * br - 7) + 12 * eta * n_p
        cost += 5 * (n_p - 1) + 2 + 24 * n_p + 6 * n_p * n_r + 18
        cost += n_etaz + 2 * n_eta + 6 * n_p + n_m + 16

        # taken from Eq. (J7) of arXiv:2302.07981v1 (2023)
        error_b = alpha * error
        n_b = np.ceil(
            np.log2(
                2
                * np.pi
                * eta
                * 2 ** (2 * n_p - 2)
                * np.abs(np.matrix(recip_vectors) @ np.matrix(recip_vectors).T).flatten().sum()
                / error_b
            )
        )

        n_dirty = FirstQuantization._qubit_cost_noncubic(n, eta, error, br, charge, vectors)
        ms_cost = FirstQuantization._momentum_state_qrom(n_p, n_m, n_dirty, n_tof, kappa=1)[0]

        # adapted Eq. (125) of PRX Quantum 2, 040332 (2021) to the noncubic case using Appendix K of
        # arXiv:2302.07981v1 (2023)
        cost -= 2 * (3 * 2 + 2 * br - 9)
        cost += 2 * (2 * (2 * (2 ** (4 + 1) - 1) + (n_b - 3) * 4 + 2**4 + (n_p - 2)))
        cost += 8
        cost -= (2 * aa_steps + 1) * (3 * n_p**2 + 15 * n_p - 7 + 4 * n_m * (n_p + 1))
        cost += (2 * aa_steps + 1) * ms_cost

        return int(np.ceil(cost))

    @staticmethod
    def _momentum_state_qrom(n_p, n_m, n_dirty, n_tof, kappa):
        r"""Returns the Toffoli cost for preparing the momentum state superposition.

        Derived from Section D.1 item (6) and Appendix K.1.f of arXiv:2302.07981v1 (2023)"""

        x = 2 ** (3 * n_p)

        beta_dirty = max([np.floor(n_dirty / n_m), 1])
        beta_parallel = max([np.floor(n_tof / kappa), 1])

        if n_tof == 1:
            beta_gate = max([np.floor(np.sqrt(2 * x / (3 * n_m))), 1])
            beta = np.min([beta_dirty, beta_gate])
            ms_cost_qrom = 2 * np.ceil(x / beta) + 3 * n_m * beta
        else:
            beta_gate = max([np.floor(2 * x / (3 * n_m / kappa) * np.log(2)), 1])
            beta = np.min([beta_dirty, beta_gate, beta_parallel])
            ms_cost_qrom = 2 * np.ceil(x / beta) + 3 * np.ceil(n_m / kappa) * np.ceil(np.log2(beta))

        ms_cost = 2 * ms_cost_qrom + n_m + 8 * (n_p - 1) + 6 * n_p + 2 + 2 * n_p + n_m + 2

        return ms_cost, beta
