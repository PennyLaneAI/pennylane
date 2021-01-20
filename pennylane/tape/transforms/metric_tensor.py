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
This module contains a function that computes the metric tensor of a tape.
"""
import itertools

import numpy as np

import pennylane as qml
from pennylane import math


def cov_matrix(prob, obs, diag_approx=False):
    """Calculate the covariance matrix of a list of commuting observables, given
    the joint probability distribution of the system in the shared eigenbasis.

    .. note::
        This method only works for **commuting observables.**
        The probability distribution must be rotated into the shared
        eigenbasis of the list of observables.

    Args:
        prob (tensor_like): probability distribution
        obs (list[Observable]): a list of observables for which
            to compute the covariance matrix for

    Returns:
        tensor_like: the covariance matrix of size ``(len(obs), len(obs))``
    """
    diag = []

    # diagonal variances
    for i, o in enumerate(obs):
        l = math.cast(o.eigvals, dtype=np.float64)
        p = math.marginal_prob(prob, o.wires.labels)

        res = math.dot(l ** 2, p) - (math.dot(l, p)) ** 2
        diag.append(res)

    diag = math.diag(diag)

    if diag_approx:
        return diag

    for i, j in itertools.combinations(range(len(obs)), r=2):
        o1 = obs[i]
        o2 = obs[j]

        l1 = math.cast(o1.eigvals, dtype=np.float64)
        l2 = math.cast(o2.eigvals, dtype=np.float64)
        l12 = math.cast(np.kron(l1, l2), dtype=np.float64)

        p1 = qml.math.marginal_prob(prob, o1.wires)
        p2 = qml.math.marginal_prob(prob, o2.wires)
        p12 = qml.math.marginal_prob(prob, o1.wires + o2.wires)

        res = math.dot(l12, p12) - math.dot(l1, p1) * math.dot(l2, p2)

        diag = math.scatter_element_add(diag, [i, j], res)
        diag = math.scatter_element_add(diag, [j, i], res)

    return diag
