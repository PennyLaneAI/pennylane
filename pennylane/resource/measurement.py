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
from pennylane import numpy as np


def estimate_samples(coeffs, ops, error=0.0016, variances=None):
    r"""Estimate the number of measurements required to compute an expectation value with a target
    error.

    Args:
        coeffs (list[tensor_like]): list of coefficient groups
        ops (list[list[Observable]]): list of Pauli word groups
        error (float): target error in computing the expectation value
        variances (list[float]): variances of the Pauli word groups


    .. details::
        :title: Theory

        An estimation for the number of measurements :math:`M` required to predict the expectation
        value of an observable :math:`H = \sum_i A_i` with :math:`A = \sum_j c_j O_j` representing a
        linear combination of Pauli words can be obtained as

        .. math::

            M = \frac{\left ( \sum_i \sqrt{\text{Var}(A_i)} \right )^2}{\epsilon^2},

        with :math:`\epsilon` and :math:`\text{Var}(A)` denoting the target error in computing
        :math:`\left \langle H \right \rangle` and the variance in computing
        :math:`\left \langle A \right \rangle`, respectively. It has been shown by Yen et al. that
        the variances can be computed from the covariances between the Pauli words as

        .. math::

            \text{Var}(A_i) = \sum_{jk} c_j c_k \text{Cov}(O_j, O_k),

        where

        .. math::

            \text{Cov}(O_j, O_k) = \left \langle O_j O_k \right \rangle - \left \langle O_j \right \rangle \left \langle O_k \right \rangle.

        The values of :math:`\text{Cov}(O_j, O_k)` are not known a priori and should be either
        computed from affordable classical methods, such as the configuration interaction with
        singles and doubles (CISD), or approximated with other methods. If the variances are not
        provided to the function as input, they will be approximated by assuming
        :math:`\text{Cov}(O_j, O_k) =0` for :math:`j \neq k` and using :math:`\text{Var}(O_i) \leq 1`
        from

        .. math::

            \text{Var}(O_i) = \left \langle O_i^2 \right \rangle - \left \langle O_i \right \rangle^2 = 1 - \left \langle O_i \right \rangle^2.

        This approximation gives

        .. math::

            M \approx \frac{\left ( \sum_i \sqrt{\sum_j c_j^2} \right )^2}{\epsilon^2},

        where :math:`i` and :math:`j` run over the observable groups and the Pauli words inside the
        group, respectively.


    """
    if not variances:
        variances = np.ones(len(ops))

    n_measurements = (np.sum(abs(coeffs) * variances**0.5) / error) ** 2

    return int(n_measurements)
