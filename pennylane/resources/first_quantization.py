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
from functools import partial
from scipy.optimize import minimize_scalar
from pennylane import numpy as np


def _cost_qrom(k, lz):
    r"""Return the number of Toffoli gates needed for erasing the output of a QROM.
    ￼
    The expression for computing the cost is taken from
    [`arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_].

    Args:
        k (int): parameter taken to be a power of 2
        lz (int): sum of the atomic numbers of nuclei

    Returns:
        int: the cost of erasing the output of a QROM

    **Example**

    >>> _cost_qrom(4, 100)
    23
    """
    cost = 2**k + np.ceil(2 ** (-k) * lz)

    return int(cost)


def _cost_qrom_min(lz):
    r"""Return the minimum number of Toffoli gates needed for erasing the output of a QROM.
    ￼
    The expression for computing the cost is taken from
    [`arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_].

    Args:
        lz (int): sum of the atomic numbers of nuclei

    Returns:
        int: the minimum cost of erasing the output of a QROM

    **Example**

    >>> _cost_qrom_min(100)
    21
    """
    n = minimize_scalar(partial(_cost_qrom, lz=lz)).x

    cost = min(_cost_qrom(np.floor(n), lz), _cost_qrom(np.ceil(n), lz))

    return cost
