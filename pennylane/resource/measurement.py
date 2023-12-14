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
This module contains the functions needed for estimating the number of measurements and the error
for computing expectation values.
"""
import numpy as np


def estimate_shots(coeffs, variances=None, error=0.0016):
    r"""Estimate the number of measurements required to compute an expectation value with a target
    error.

    See also :func:`estimate_error`.

    Args:
        coeffs (list[tensor_like]): list of coefficient groups
        variances (list[float]): variances of the Pauli word groups
        error (float): target error in computing the expectation value

    Returns:
        int: the number of measurements

    **Example**

    >>> coeffs = [np.array([-0.32707061, 0.7896887]), np.array([0.18121046])]
    >>> qml.resource.estimate_shots(coeffs)
    419218

    .. details::
        :title: Theory

        An estimation for the number of measurements :math:`M` required to predict the expectation
        value of an observable :math:`H = \sum_i A_i`, with :math:`A = \sum_j c_j O_j` representing
        a linear combination of Pauli words, can be obtained following Eq. (34) of
        [`PRX Quantum 2, 040320 (2021) <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040320>`_]
        as

        .. math::

            M = \frac{\left ( \sum_i \sqrt{\text{Var}(A_i)} \right )^2}{\epsilon^2},

        with :math:`\epsilon` and :math:`\text{Var}(A)` denoting the target error in computing
        :math:`\left \langle H \right \rangle` and the variance in computing
        :math:`\left \langle A \right \rangle`, respectively. It has been shown in Eq. (10) of
        [`arXiv:2201.01471v3 <https://arxiv.org/abs/2201.01471v3>`_] that
        the variances can be computed from the covariances between the Pauli words as

        .. math::

            \text{Var}(A_i) = \sum_{jk} c_j c_k \text{Cov}(O_j, O_k),

        where

        .. math::

            \text{Cov}(O_j, O_k) = \left \langle O_j O_k \right \rangle - \left \langle O_j \right \rangle \left \langle O_k \right \rangle.

        The values of :math:`\text{Cov}(O_j, O_k)` are not known a priori and should be either
        computed from affordable classical methods, such as the configuration interaction with
        singles and doubles (CISD), or approximated with other methods. If the variances are not
        provided to the function as input, they will be approximated following Eqs. (6-7) of
        [`Phys. Rev. Research 4, 033154, 2022 <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.033154>`_]
        by assuming
        :math:`\text{Cov}(O_j, O_k) =0` for :math:`j \neq k` and using :math:`\text{Var}(O_i) \leq 1`
        from

        .. math::

            \text{Var}(O_i) = \left \langle O_i^2 \right \rangle - \left \langle O_i \right \rangle^2 = 1 - \left \langle O_i \right \rangle^2.

        This approximation gives

        .. math::

            M \approx \frac{\left ( \sum_i \sqrt{\sum_j c_{ij}^2} \right )^2}{\epsilon^2},

        where :math:`i` and :math:`j` run over the observable groups and the Pauli words inside the
        group, respectively.
    """
    if variances:
        return int(np.ceil(np.sum(np.sqrt(variances)) ** 2 / error**2))

    group_sum = [np.sum(coeff**2) for coeff in coeffs]
    return int(np.ceil(np.sum(np.sqrt(group_sum)) ** 2 / error**2))


def estimate_error(coeffs, variances=None, shots=1000):
    r"""Estimate the error in computing an expectation value with a given number of measurements.

    See also :func:`estimate_shots`.

    Args:
        coeffs (list[tensor_like]): list of coefficient groups
        variances (list[float]): variances of the Pauli word groups
        shots (int): the number of measurements

    Returns:
        float: target error in computing the expectation value

    **Example**

    >>> coeffs = [np.array([-0.32707061, 0.7896887]), np.array([0.18121046])]
    >>> qml.resource.estimate_error(coeffs, shots=100000)
    0.00327597

    .. details::
        :title: Theory

        An estimation for the error :math:`\epsilon` in predicting the expectation
        value of an observable :math:`H = \sum_i A_i` with :math:`A = \sum_j c_j O_j` representing a
        linear combination of Pauli words can be obtained following Eq. (34) of
        [`PRX Quantum 2, 040320 (2021) <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040320>`_]
        as

        .. math::

            \epsilon = \frac{\sum_i \sqrt{\text{Var}(A_i)}}{\sqrt{M}},

        with :math:`M` and :math:`\text{Var}(A)` denoting the number of measurements and the
        variance in computing :math:`\left \langle A \right \rangle`, respectively. It has been
        shown in Eq. (10) of
        [`arXiv:2201.01471v3 <https://arxiv.org/abs/2201.01471v3>`_] that the variances can be
        computed from the covariances between the Pauli words as

        .. math::

            \text{Var}(A_i) = \sum_{jk} c_j c_k \text{Cov}(O_j, O_k),

        where

        .. math::

            \text{Cov}(O_j, O_k) = \left \langle O_j O_k \right \rangle - \left \langle O_j \right \rangle \left \langle O_k \right \rangle.

        The values of :math:`\text{Cov}(O_j, O_k)` are not known a priori and should be either
        computed from affordable classical methods, such as the configuration interaction with
        singles and doubles (CISD), or approximated with other methods. If the variances are not
        provided to the function as input, they will be approximated following Eqs. (6-7) of
        [`Phys. Rev. Research 4, 033154, 2022 <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.033154>`_]
        by assuming
        :math:`\text{Cov}(O_j, O_k) =0` for :math:`j \neq k` and using :math:`\text{Var}(O_i) \leq 1`
        from

        .. math::

            \text{Var}(O_i) = \left \langle O_i^2 \right \rangle - \left \langle O_i \right \rangle^2 = 1 - \left \langle O_i \right \rangle^2.

        This approximation gives

        .. math::

            \epsilon \approx \frac{\sum_i \sqrt{\sum_j c_{ij}^2}}{\sqrt{M}},

        where :math:`i` and :math:`j` run over the observable groups and the Pauli words inside the
        group, respectively.
    """
    if variances:
        return np.sum(np.sqrt(variances)) / np.sqrt(shots)

    group_sum = [np.sum(coeff**2) for coeff in coeffs]
    return np.sum(np.sqrt(group_sum)) / np.sqrt(shots)
