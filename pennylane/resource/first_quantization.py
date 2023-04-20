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
import numpy
from scipy import integrate

# from pennylane import numpy as np
import numpy as np
import inspect
from pennylane.operation import AnyWires, Operation
from itertools import product

angs2bohr = 1.8897259886


class FirstQuantization(Operation):
    r"""Estimate the number of non-Clifford gates and logical qubits for a quantum phase estimation
    algorithm in first quantization using a plane-wave basis.

    To estimate the gate and qubit costs for implementing this method, the number of plane waves,
    the number of electrons and the unit cell volume need to be defined. The costs can then be
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

    **Example**

    >>> n = 100000
    >>> eta = 156
    >>> omega = 1145.166
    >>> algo = FirstQuantization(n, eta, omega)
    >>> print(algo.lambda_total,  # the 1-Norm of the Hamiltonian
    >>>       algo.gates, # estimated number of non-Clifford gates
    >>>       algo.qubits # estimated number of logical qubits
    >>>       )
    649912.4801542697 1.10e+13 4416

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
        omega,
        vec_a=None,
        charge=0,
        error=0.0016,
        br=8,
        n_tof=500,
        n_dirty=None,
        # if given, it should be above a minimum, otherwise we get error as QROM cannot be implemented
        p_th=0.95,
        exact=False,
    ):
        self.n = n
        self.eta = eta
        self.omega = omega
        self.vec_a = vec_a
        self.charge = charge
        self.error = error
        self.br = br
        self.n_tof = n_tof
        self.n_dirty = n_dirty
        self.p_th = p_th
        self.exact = exact

        if vec_a is not None:
            self.omega = np.abs(np.sum((np.cross(vec_a[0], vec_a[1]) * vec_a[2])))

            self.recip_bv = (
                2
                * np.pi
                / self.omega
                * np.array([np.cross(vec_a[i], vec_a[j]) for i, j in [(1, 2), (2, 0), (0, 1)]])
            )

            bbt = np.matrix(self.recip_bv) @ np.matrix(self.recip_bv).T

            self.cubic = np.linalg.norm(bbt - (self.recip_bv**2).max() * np.identity(3)) < 1e-6

            self.orthogonal = (
                np.linalg.norm(
                    bbt - np.array([np.max(b**2) for b in self.recip_bv]) * np.identity(3)
                )
                < 1e-6
            )
        else:
            self.cubic = True
            self.orthogonal = True
            self.recip_bv = None

        if self.cubic:
            self.bmin = 2 * np.pi / omega ** (1 / 3)
        else:
            self.bmin = np.min(np.linalg.svd(self.recip_bv)[1])

        self.lambda_total, self.aa_steps = self.norm(
            self.n,
            self.eta,
            self.omega,
            self.error,
            self.br,
            self.charge,
            self.cubic,
            self.orthogonal,
            self.exact,
            self.p_th,
            self.bmin,
            self.recip_bv,
        )

        self.n_dirty = self._clean_cost(
            self.n,
            self.eta,
            self.omega,
            self.error,
            self.br,
            self.charge,
            self.cubic,
            self.exact,
            self.p_th,
            self.bmin,
            self.recip_bv,
            self.n_dirty,
            self.n_tof,
            self.lambda_total,
        )

        self.gates = self.gate_cost(
            self.n,
            self.eta,
            self.omega,
            self.error,
            self.br,
            self.charge,
            self.cubic,
            self.exact,
            self.p_th,
            self.bmin,
            self.recip_bv,
            self.n_dirty,
            self.n_tof,
            self.lambda_total,
            self.aa_steps,
        )

        self.qubits = self.qubit_cost(
            self.n,
            self.eta,
            self.omega,
            self.error,
            self.br,
            self.charge,
            self.cubic,
            self.exact,
            self.p_th,
            self.bmin,
            self.recip_bv,
            self.n_dirty,
            self.n_tof,
            self.lambda_total,
        )

        super().__init__(wires=range(self.qubits))

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
    def norm(
        n,
        eta,
        omega,
        error,
        br=8,
        charge=0,
        cubic=True,
        orthogonal=True,
        exact=False,
        p_th=0.95,
        bmin=None,
        recip_bv=None,
    ):
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

        if eta <= 0 or not isinstance(eta, (int, numpy.integer)):
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
        alpha = 0.0248759298  # same as the pseudopotential/AE code
        # target error in the qubitization of U+V
        error_uv = alpha * error

        # taken from Eq. (22) of PRX Quantum 2, 040332 (2021)
        n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))

        n0 = n ** (1 / 3)

        bmin = 2 * np.pi / omega ** (1 / 3) if cubic else bmin

        n_m = int(
            np.ceil(
                np.log2(  # taken from Eq. (132) of PRX Quantum 2, 040332 (2021)
                    (8 * np.pi * eta)
                    / (error_uv * omega * bmin**2)
                    * (eta - 1 + 2 * l_z)
                    * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
                )
            )
        )

        if not cubic and exact and n_p <= 6:
            lambda_nu = 0
            b_mus = {}
            for j in range(2, n_p + 3):
                b_mus[j] = []
            for nu in product(range(-(2 ** (n_p)), 2 ** (n_p) + 1), repeat=3):
                nu = np.array(nu)
                if list(nu) != [0, 0, 0]:
                    mu = int(np.floor(np.log2(np.max(abs(nu))))) + 2
                    b_mus[mu].append(nu)
            for mu in range(2, (n_p + 2)):
                for nu in b_mus[mu]:
                    gnu_norm = np.linalg.norm(np.sum(nu * recip_bv, axis=0))
                    lambda_nu += np.ceil(2**n_m * (bmin * 2 ** (mu - 2) / gnu_norm) ** 2) / (
                        2**n_m * (bmin * 2 ** (mu - 2)) ** 2
                    )
            lambda_nu_1 = lambda_nu

        else:

            lambda_nu = (  # expression is taken from Eq. (F6) of PRX 8, 011044 (2018)
                4 * np.pi * (np.sqrt(3) * n ** (1 / 3) / 2 - 1)
                + 3
                - 3 / n ** (1 / 3)
                + 3 * integrate.nquad(lambda x, y: 1 / (x**2 + y**2), [[1, n0], [1, n0]])[0]
            ) / bmin**2

            lambda_nu_1 = lambda_nu + 4 / (2**n_m * bmin**2) * (
                7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p)
            )

        p_nu = lambda_nu_1 * bmin**2 / 2 ** (n_p + 6)

        # compute_AA_steps
        probability_amplified = 0
        index = 0  # number of amplitude amplifications
        for i in range(29, -1, -1):
            probability = (np.sin((2 * i + 1) * np.arcsin(np.sqrt(p_nu)))) ** 2  # Eq (I6)
            if probability > p_th:
                index = i
                probability_amplified = probability
        p_nu_amp, aa_steps = probability_amplified, index

        # lambda_u and lambda_v are taken from Eq. (25) of PRX Quantum 2, 040332 (2021)
        lambda_u = 4 * np.pi * eta * l_z * lambda_nu / omega
        lambda_v = 2 * np.pi * eta * (eta - 1) * lambda_nu / omega

        # lambda_u_1 and lambda_v_1 are taken from Eq. (124) of PRX Quantum 2, 040332 (2021)
        lambda_u_1 = lambda_u * lambda_nu_1 / lambda_nu
        lambda_v_1 = lambda_v * lambda_nu_1 / lambda_nu

        # taken from Eq. (71) of PRX Quantum 2, 040332 (2021)
        abs_sum = 3 * bmin**2
        if not cubic:
            b_mat = np.matrix(recip_bv)
            abs_sum = np.abs(b_mat @ b_mat.T).flatten().sum()

        if cubic or orthogonal:
            lambda_t_p = abs_sum * eta * 2 ** (2 * n_p - 2) / 4
        else:
            lambda_t_p = abs_sum * eta * 2 ** (2 * n_p - 2) / 2

        # taken from Eq. (63) of PRX Quantum 2, 040332 (2021)
        p_eq = (
            FirstQuantization.success_prob(3 * eta + 2 * charge, br)
            * FirstQuantization.success_prob(eta, br) ** 2
        )
        if cubic:
            p_eq *= FirstQuantization.success_prob(3, 8)

        # final lambda value is computed from Eq. (126) of PRX Quantum 2, 040332 (2021)
        if p_nu * lambda_t_p >= (1 - p_nu) * (lambda_u_1 + lambda_v_1):
            return 0.0, 0.0
        elif p_nu_amp * lambda_t_p >= (1 - p_nu_amp) * (lambda_u_1 + lambda_v_1):
            return (lambda_t_p + lambda_u_1 + lambda_v_1) / p_eq, aa_steps
        else:
            return ((lambda_u_1 + lambda_v_1 / (1 - 1 / eta)) / p_nu_amp) / p_eq, aa_steps

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
        if lz <= 0 or not isinstance(lz, (int, numpy.integer)):
            raise ValueError("The sum of the atomic numbers must be a positive integer.")

        k_f = np.floor(np.log2(lz) / 2)
        k_c = np.ceil(np.log2(lz) / 2)

        cost_f = int(2**k_f + np.ceil(2 ** (-1 * k_f) * lz))
        cost_c = int(2**k_c + np.ceil(2 ** (-1 * k_c) * lz))

        return min(cost_f, cost_c)

    @staticmethod
    def _momentum_state_qrom(n_p, n_m, n_dirty, n_tof, kappa):

        x = 2 ** (3 * n_p)

        beta_dirty = np.floor(n_dirty / n_m)
        beta_parallel = np.floor(n_tof / kappa)

        if n_tof == 1:
            beta_gate = np.floor(np.sqrt(2 * x / (3 * n_m)))
            beta = np.min([beta_dirty, beta_gate])
            ms_cost_qrom = 2 * np.ceil(x / beta) + 3 * n_m * beta
        else:
            beta_gate = np.floor(2 * x / (3 * n_m / kappa) * np.log(2))
            beta = np.min([beta_dirty, beta_gate, beta_parallel])
            ms_cost_qrom = 2 * np.ceil(x / beta) + 3 * np.ceil(n_m / kappa) * np.ceil(np.log2(beta))

        ms_cost = 2 * ms_cost_qrom + n_m + 8 * (n_p - 1) + 6 * n_p + 2 + 2 * n_p + n_m + 2

        return ms_cost, beta

    @staticmethod
    def estimation_cost(
        n,
        eta,
        omega,
        error,
        cubic=True,
        lambda_total=None,
    ):
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

        Returns:
            int: number of calls to unitary

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 1145.166
        >>> error = 0.01
        >>> estimation_cost(n, eta, omega, error, cubic, lambda_total)
        102133985
        """
        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        alpha = 0.0248759298
        # qpe_error obtained to satisfy inequality (131)
        n_errors = 3 if cubic else 4
        error_qpe = np.sqrt(error**2 * (1 - (n_errors * alpha) ** 2))
        return int(np.ceil(np.pi * lambda_total / (2 * error_qpe)))

    @staticmethod
    def gate_cost(
        n,
        eta,
        omega,
        error,
        br=8,
        charge=0,
        cubic=True,
        exact=False,
        p_th=0.95,
        bmin=None,
        recip_bv=None,
        n_dirty=None,
        n_tof=500,
        lambda_total=None,
        aa_steps=None,
    ):
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

        if eta <= 0 or not isinstance(eta, (int, numpy.integer)):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        e_cost = FirstQuantization.estimation_cost(n, eta, omega, error, cubic, lambda_total)

        u_cost = FirstQuantization.unitary_cost(
            n,
            eta,
            omega,
            error,
            br,
            charge,
            cubic,
            bmin,
            recip_bv,
            lambda_total,
            aa_steps,
            n_dirty,
            n_tof,
        )

        return e_cost * u_cost

    @staticmethod
    def _clean_cost(
        n,
        eta,
        omega,
        error,
        br=8,
        charge=0,
        cubic=True,
        exact=False,
        p_th=0.95,
        bmin=None,
        recip_bv=None,
        n_dirty=None,
        n_tof=500,
        lambda_total=None,
    ):
        r"""Return the number of clean logical qubits needed to ...."""
        if n <= 0:
            raise ValueError("The number of plane waves must be a positive number.")

        if eta <= 0 or not isinstance(eta, (int, numpy.integer)):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        l_z = eta + charge
        l_nu = 2 * np.pi * n ** (2 / 3)

        # n_p is taken from Eq. (22) of PRX Quantum 2, 040332 (2021)
        n_p = np.ceil(np.log2(n ** (1 / 3) + 1))

        # errors in Eqs. (132-134) of PRX Quantum 2, 040332 (2021)
        alpha = 0.0248759298  # optimal value for lower resource estimates
        error_t, error_r, error_m, error_b = [alpha * error] * 4

        # parameters taken from Eqs. (132-134) of PRX Quantum 2, 040332 (2021)
        n_t = int(np.ceil(np.log2(np.pi * lambda_total / error_t)))  # Eq. (134)
        n_r = int(np.ceil(np.log2((eta * l_z * l_nu) / (error_r * omega ** (1 / 3)))))  # Eq. (133)
        n_m = int(  # Eq. (132)
            np.ceil(
                np.log2(
                    (8 * np.pi * eta)
                    / (error_m * omega * bmin**2)
                    * (eta - 1 + 2 * l_z)
                    * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
                )
            )
        )

        # qpe_error obtained to satisfy inequality (131) of PRX Quantum 2, 040332 (2021)
        n_errors = 3 if cubic else 4
        error_qpe = np.sqrt(error**2 * (1 - (n_errors * alpha) ** 2))

        clean_temp_H_cost = max([5 * n_r - 4, 5 * n_p + 1]) + max([5, n_m + 3 * n_p])
        __name = (
            np.ceil(np.log2(eta + 2 * l_z))
            + 2 * np.ceil(np.log2(eta))
            + 6 * n_p
            + n_m
            + 16
            + (0 if cubic else 3)
        )
        clean_temp_cost = max([clean_temp_H_cost, __name])

        # the expression for computing the cost is taken from Eq. (101) of arXiv:2204.11890v1
        clean_cost = 3 * eta * n_p
        clean_cost += np.ceil(np.log2(np.ceil(np.pi * lambda_total / (2 * error_qpe))))
        clean_cost += 1 + 1 + np.ceil(np.log2(eta + 2 * l_z)) + 3 + 3
        clean_cost += 2 * np.ceil(np.log2(eta)) + 5 + 3 * (n_p + 1)
        clean_cost += n_p + n_m + 3 * n_p + 2 + 2 * n_p + 1 + 1 + 2 + 2 * n_p + 6 + 1

        clean_cost += clean_temp_cost

        if cubic:
            clean_cost += np.max([n_r + 1, n_t])
            clean_cost += 3 * n_p**2 + n_p + 1 + 4 * n_m * (n_p + 1) + 4
        else:
            n_b = np.ceil(
                np.log2(
                    2
                    * np.pi
                    * eta
                    * 2 ** (2 * n_p - 2)
                    * np.abs(np.matrix(recip_bv) @ np.matrix(recip_bv).T).flatten().sum()
                    / error_b
                )
            )
            clean_cost += np.max([n_r + 1, n_t, n_b]) + 6 + n_m + 1

        return clean_cost

    @staticmethod
    def qubit_cost(
        n,
        eta,
        omega,
        error,
        br=8,
        charge=0,
        cubic=True,
        exact=False,
        p_th=0.95,
        bmin=None,
        recip_bv=None,
        n_dirty=None,
        n_tof=500,
        lambda_total=None,
    ):
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

        if eta <= 0 or not isinstance(eta, (int, numpy.integer)):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        l_z = eta + charge

        # n_p is taken from Eq. (22) of PRX Quantum 2, 040332 (2021)
        n_p = np.ceil(np.log2(n ** (1 / 3) + 1))

        # errors in Eqs. (132-134) of PRX Quantum 2, 040332 (2021)
        alpha = 0.0248759298  # optimal value for lower resource estimates
        error_m = alpha * error

        n_m = int(  # Eq. (132)
            np.ceil(
                np.log2(
                    (8 * np.pi * eta)
                    / (error_m * omega * bmin**2)
                    * (eta - 1 + 2 * l_z)
                    * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
                )
            )
        )

        logical_clean_qubits = FirstQuantization._clean_cost(
            n,
            eta,
            omega,
            error,
            br,
            charge,
            cubic,
            exact,
            p_th,
            bmin,
            recip_bv,
            n_dirty,
            n_tof,
            lambda_total,
        )

        if n_dirty is None or cubic:
            return int(np.ceil(logical_clean_qubits))

        beta = FirstQuantization._momentum_state_qrom(n_p, n_m, n_dirty, n_tof, kappa=1)[1]
        qubits = max([logical_clean_qubits, beta * (n_m + 1)])

        return int(np.ceil(qubits))

    @staticmethod
    def unitary_cost(
        n,
        eta,
        omega,
        error,
        br=8,
        charge=0,
        cubic=True,
        bmin=None,
        recip_bv=None,
        lambda_total=None,
        aa_steps=None,
        n_dirty=None,
        n_tof=500,
    ):
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

        if eta <= 0 or not isinstance(eta, (int, numpy.integer)):
            raise ValueError("The number of electrons must be a positive integer.")

        if omega <= 0:
            raise ValueError("The unit cell volume must be a positive number.")

        if error <= 0.0:
            raise ValueError("The target error must be greater than zero.")

        if br <= 0 or not isinstance(br, int):
            raise ValueError("br must be a positive integer.")

        if not isinstance(charge, int):
            raise ValueError("system charge must be an integer.")

        alpha = 0.0248759298
        l_z = eta + charge
        l_nu = 2 * np.pi * n ** (2 / 3)

        # n_eta and n_etaz are defined in the third and second paragraphs of page 040332-15
        n_eta = np.ceil(np.log2(eta))
        n_etaz = np.ceil(np.log2(eta + 2 * l_z))

        # n_p is taken from Eq. (22)
        n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))

        # errors in Eqs. (132-134) are set to be 0.0248759298 of the algorithm error
        error_t, error_r, error_m, error_b = [alpha * error] * 4

        # parameters taken from Eqs. (132-134) of PRX Quantum 2, 040332 (2021)
        n_t = int(np.ceil(np.log2(np.pi * lambda_total / error_t)))  # Eq. (134)
        n_r = int(np.ceil(np.log2((eta * l_z * l_nu) / (error_r * omega ** (1 / 3)))))  # Eq. (133)

        bmin = 2 * np.pi / omega ** (1 / 3) if cubic else bmin
        n_m = int(
            np.ceil(
                np.log2(  # Eq. (132)
                    (8 * np.pi * eta)
                    / (error_m * omega * bmin**2)
                    * (eta - 1 + 2 * l_z)
                    * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
                )
            )
        )

        e_r = FirstQuantization._cost_qrom(l_z)

        # taken from Eq. (125)
        cost = 2 * (n_t + 4 * n_etaz + 2 * br - 12) + 14 * n_eta + 8 * br - 36
        cost += (2 * aa_steps + 1) * (3 * n_p**2 + 15 * n_p - 7 + 4 * n_m * (n_p + 1))
        cost += l_z + e_r + 2 * (2 * n_p + 2 * br - 7) + 12 * eta * n_p
        cost += 5 * (n_p - 1) + 2 + 24 * n_p + 6 * n_p * n_r + 18
        cost += n_etaz + 2 * n_eta + 6 * n_p + n_m + 16

        if not cubic:
            n_b = np.ceil(
                np.log2(
                    2
                    * np.pi
                    * eta
                    * 2 ** (2 * n_p - 2)
                    * np.abs(np.matrix(recip_bv) @ np.matrix(recip_bv).T).flatten().sum()
                    / error_b
                )
            )
            # this and the next corrects the cost of PREP and PREP^dagger for T
            cost -= 2 * (3 * 2 + 2 * br - 9)
            cost += 2 * (2 * (2 * (2 ** (4 + 1) - 1) + (n_b - 3) * 4 + 2**4 + (n_p - 2)))
            cost += 8  # 5 to compute the flag qubit for T being prepared, uncomputation without Toffolis, and 3 for the ROT cost

            ms_cost = FirstQuantization._momentum_state_qrom(n_p, n_m, n_dirty, n_tof, kappa=1)[0]
            cost -= (2 * aa_steps + 1) * (3 * n_p**2 + 15 * n_p - 7 + 4 * n_m * (n_p + 1))
            cost += (2 * aa_steps + 1) * ms_cost

        return int(np.ceil(cost))
