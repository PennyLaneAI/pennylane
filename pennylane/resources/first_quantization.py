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
# pylint: disable=too-many-arguments
from scipy import integrate
from pennylane import numpy as np
from pennylane.operation import Operation, AnyWires


class FirstQuantization(Operation):
    """Contains the functionality for estimating the number of non-Clifford gates and logical qubits
    for quantum algorithms in first quantization using a plane-wave basis.

    Args:
        n (int): number of basis states
        eta (int): number of electrons
        omega (float): unit cell volume
        error (float): target error in the algorithm
        charge (int): total electric charge of the system
        br (int): number of bits for ancilla qubit rotation
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(
        self,
        n,
        eta,
        omega,
        error=0.0016,
        charge=0,
        br=7,
    ):
        self.n = n
        self.eta = eta
        self.omega = omega
        self.error = error
        self.charge = charge
        self.br = br

        self.lamb = self.norm(self.n, self.eta, self.omega, self.error, self.br, self.charge)

        self.gates = self.gate_cost(
            self.n, self.eta, self.omega, self.error, self.lamb, self.br, self.charge
        )
        self.qubits = self.qubit_cost(
            self.n, self.eta, self.omega, self.error, self.lamb, self.charge
        )

        super().__init__(wires=range(self.qubits))

    def success_prob(self, n, br):
        r"""Return the probability of success for state preparation.

        The expression for computing the probability of success is taken from Eqs. (59, 60) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

        Args:
            n (int): number of plane waves
            br (int): number of bits for ancilla qubit rotation

        Returns:
            float: probability of success for state preparation

        **Example**

        >>> success_prob(10000, 7)
        0.9998814293823286
        """
        if n <= 0 or not isinstance(n, int):
            raise ValueError("The number of planewaves must be a positive integer.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        c = n / 2 ** np.ceil(np.log2(n))
        d = 2 * np.pi / 2**br

        theta = d * np.round((1 / d) * np.arcsin(np.sqrt(1 / (4 * c))))

        p = c * ((1 + (2 - 4 * c) * np.sin(theta) ** 2) ** 2 + np.sin(2 * theta) ** 2)

        return p

    def norm(self, n, eta, omega, error, br=7, charge=0):
        r"""Return the 1-norm of a first-quantized Hamiltonian in the plane-wave basis.

        The expressions needed for computing the norm are taken from
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].
        The norm is computed assuming that amplitude ampliﬁcation is performed.

        Args:
            n (int): number of basis states
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system

        Returns:
            float: 1-norm of a first-quantized Hamiltonian in the plane-wave basis

        **Example**

        >>> n = 10000
        >>> eta = 156
        >>> omega = 1145.166
        >>> error = 0.001
        >>> norm(n, eta, omega, error)
        1254385.059691027

        .. details::
            :title: Theory

            For numerical convenience, we use the following modified expressions for computing
            parameters that contain a sum over the elements, :math:`\nu`, of the set of reciprocal
            lattice vectors, :math:`G_0`.

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

                \epsilon_l = \frac{4}{2^{n_m}} (7 \times 2^{n_p + 1} + 9 n_p - 11 - 3 \times 2^{-n_p}),

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
        if n <= 0 or not isinstance(n, int):
            raise ValueError("The number of planewaves must be a positive integer.")

        if eta <= 0 or not isinstance(eta, int):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

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
            7 * 2 ** (n_p + 1) + 9 * n_p - 11 - 3 * 2 ** (-1 * n_p)
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
            self.success_prob(3, 8)
            * self.success_prob(3 * eta + 2 * charge, br)
            * self.success_prob(eta, br) ** 2
        )

        # final lambda value is computed from Eq. (126) of PRX Quantum 2, 040332 (2021)
        lambda_a = lambda_t_p + lambda_u_1 + lambda_v_1
        lambda_b = (lambda_u_1 + lambda_v_1 / (1 - 1 / eta)) / p_nu_amp

        return np.maximum(lambda_a, lambda_b) / p_eq

    def _cost_qrom(self, lz):
        r"""Return the minimum number of Toffoli gates needed for erasing the output of a QROM.

        Args:
            lz (int): sum of the atomic numbers

        Returns:
            int: the minimum cost of erasing the output of a QROM

        **Example**

        >>> _cost_qrom_min(100)
        21
        """
        if lz <= 0 or not isinstance(lz, int):
            raise ValueError("The sum of the atomic numbers must be a positive integer.")

        k_f = np.floor(np.log2(lz) / 2)
        k_c = np.ceil(np.log2(lz) / 2)

        cost_f = int(2**k_f + np.ceil(2 ** (-1 * k_f) * lz))
        cost_c = int(2**k_c + np.ceil(2 ** (-1 * k_c) * lz))

        return min(cost_f, cost_c)

    def unitary_cost(self, n, eta, omega, error, lamb, br=7, charge=0):
        r"""Return the number of Toffoli gates needed to implement the qubitization unitary
        operator.

        The expression for computing the cost is taken from Eq. (125) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            lamb (float): 1-norm of the Hamiltonian
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system

        Returns:
            int: the number of Toffoli gates needed to implement the qubitization unitary operator

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 169.69608
        >>> error = 0.01
        >>> lamb = 5128920.595980267
        >>> unitary_cost(n, eta, omega, error, lamb)
        12819
        """
        if n <= 0 or not isinstance(n, int):
            raise ValueError("The number of planewaves must be a positive integer.")

        if eta <= 0 or not isinstance(eta, int):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if lamb <= 0.0:
            raise ValueError("The 1-norm must be greater than zero.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        alpha = 0.01
        l_z = eta + charge
        l_nu = 2 * np.pi * n ** (2 / 3)

        # n_eta and n_etaz are defined in the third and second paragraphs of page 040332-15
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

        e_r = self._cost_qrom(l_z)

        # taken from Eq. (125)
        cost = 2 * (n_t + 4 * n_etaz + 2 * br - 12) + 14 * n_eta + 8 * br - 36
        cost += 3 * n_p**2 + 15 * n_p - 7 + 4 * n_m * (n_p + 1)
        cost += l_z + e_r + 2 * (2 * n_p + 2 * br - 7) + 12 * eta * n_p
        cost += 5 * (n_p - 1) + 2 + 24 * n_p + 6 * n_p * n_r + 18
        cost += n_etaz + 2 * n_eta + 6 * n_p + n_m + 16

        return int(np.ceil(cost))

    def estimation_cost(self, lamb, error):
        r"""Return the number of calls to the unitary needed to achieve the desired error in quantum
        phase estimation.

        The expression for computing the cost is taken from Eq. (125) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

        Args:
            lamb (float): 1-norm of a second-quantized Hamiltonian
            error (float): target error in the algorithm

        Returns:
            int: number of calls to unitary

        **Example**

        >>> cost = estimation_cost(72.49779513025341, 0.001)
        >>> print(cost)
        113880
        """
        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if lamb <= 0.0:
            raise ValueError("The 1-norm must be greater than zero.")

        alpha = 0.01
        # qpe_error obtained to satisfy inequality (131)
        error_qpe = np.sqrt(error**2 * (1 - (3 * alpha) ** 2))

        return int(np.ceil(np.pi * lamb / (2 * error_qpe)))

    def gate_cost(self, n, eta, omega, error, lamb, br=7, charge=0):
        r"""Return the total number of Toffoli gates needed to implement the first quantization
        algorithm.

        The expression for computing the cost is taken from Eq. (125) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            lamb (float): 1-norm of the Hamiltonian
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system

        Returns:
            int: the number of Toffoli gates needed to implement the first quantization algorithm

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 169.69608
        >>> error = 0.01
        >>> lamb = 5128920.595980267
        >>> gate_cost(n, eta, omega, error, lamb)
        10327614069516

        .. details::
            :title: Theory

            The target algorithm error, :math:`\epsilon`, is distributed among four different
            sources of error following Eq. (131) of
            `PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_
            such that

            .. math::
                \epsilon^2 \geq \epsilon_{qpe}^2 + (\epsilon_{\mathcal{M}} + \epsilon_R + \epsilon_T)^2,

            where :math:`\epsilon_{qpe}` is the quantum phase estimation error and
            :math:`\epsilon_{\mathcal{M}}`, :math:`\epsilon_R`, and :math:`\epsilon_T` are defined
            in Eqs. (132-134) of
            `PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_.

            Here, we assume :math:`\epsilon_{\mathcal{M}} = \epsilon_R = \epsilon_T = \alpha \epsilon`
            with a default value of :math:`\alpha = 0.01` and obtain

            .. math::
                \epsilon_{qpe} = \sqrt{\epsilon^2 [1 - (3 \alpha)^2]}.

            Note that the user only needs to define the target algorithm error :math:`\epsilon`. The
            error distribution takes place inside the functions.
        """
        if n <= 0 or not isinstance(n, int):
            raise ValueError("The number of planewaves must be a positive integer.")

        if eta <= 0 or not isinstance(eta, int):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if lamb <= 0.0:
            raise ValueError("The 1-norm must be greater than zero.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        e_cost = self.estimation_cost(lamb, error)
        u_cost = self.unitary_cost(n, eta, omega, error, lamb, br, charge)

        return e_cost * u_cost

    def qubit_cost(self, n, eta, omega, error, lamb, charge=0):
        r"""Return the number of ancilla qubits needed to implement the first quantization
        algorithm.

        The expression for computing the cost is taken from Eq. (101) of
        [`arXiv:2204.11890v1 <https://arxiv.org/abs/2204.11890v1>`_].

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            lamb (float): 1-norm of the Hamiltonian
            charge (int): total electric charge of the system

        Returns:
            int: the number of ancilla qubits needed to implement the first quantization algorithm

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 169.69608
        >>> error = 0.01
        >>> lamb = 5128920.595980267
        >>> qubit_cost(n, eta, omega, error, lamb)
        4238
        """
        if n <= 0 or not isinstance(n, int):
            raise ValueError("The number of planewaves must be a positive integer.")

        if eta <= 0 or not isinstance(eta, int):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if lamb <= 0.0:
            raise ValueError("The 1-norm must be greater than zero.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

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

        # the expression for computing the cost is taken from Eq. (101) of arXiv:2204.11890v1
        qubits = 3 * eta * n_p + 4 * n_m * n_p + 12 * n_p
        qubits += 2 * (np.ceil(np.log2(np.ceil(np.pi * lamb / (2 * error))))) + 5 * n_m
        qubits += 2 * np.ceil(np.log2(eta)) + 3 * n_p**2 + np.ceil(np.log2(eta + 2 * l_z))
        qubits += np.maximum(5 * n_p + 1, 5 * n_r - 4) + np.maximum(n_t, n_r + 1) + 33

        return int(np.ceil(qubits))
