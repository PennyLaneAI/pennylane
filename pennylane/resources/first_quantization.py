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

from pennylane import numpy as np


def success_prob(n, br):
    r"""Return the probability of success for state preparation.

    The expression for computing the probability of success is taken from
    [`arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_].

    Args:
        n (int): number of basis states
        br (int): number of bits for ancilla qubit rotation

    Returns:
        float: probability of success for state preparation

    **Example**

    >>> success_prob(10000, 7)
    0.9998814293823286
    """
    c = n / 2 ** np.ceil(np.log2(n))
    d = 2 * np.pi / 2**br

    theta = d * np.round((1 / d) * np.arcsin(np.sqrt(1 / (4 * c))))

    p = c * ((1 + (2 - 4 * c) * np.sin(theta) ** 2) ** 2 + np.sin(2 * theta) ** 2)

    return p
