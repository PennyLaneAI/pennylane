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
Contains the ``QuantumMonteCarlo`` template.
"""
import itertools

import numpy as np

from pennylane.templates.decorator import template


def probs_to_unitary(probs):
    """Calculates the unitary matrix corresponding to an input probability distribution.

    For a given distribution :math:`p_{i}`, this function returns the unitary :math:`U` that
    transforms the :math:`|0\rangle` state as

    .. math::

        U |0\rangle = \sum_{i} \sqrt{p_{i}} |i\rangle,

    so that measuring the resulting state in the computational basis will give the state
    :math:`|i\rangle` with probability :math:`p_{i}`. Note that the returned unitary matrix is
    real and hence an orthogonal matrix.

    Args:
        probs (array): input probability distribution as a flat array

    Returns:
        array: corresponding unitary

    Raises:
        ValueError: if the input array is not flat or does not correspond to a probability
            distribution

    **Example:**

    >>> p = np.ones(4) / 4
    >>> probs_to_unitary(p)
    array([[ 0.5       ,  0.5       ,  0.5       ,  0.5       ],
           [ 0.5       , -0.83333333,  0.16666667,  0.16666667],
           [ 0.5       ,  0.16666667, -0.83333333,  0.16666667],
           [ 0.5       ,  0.16666667,  0.16666667, -0.83333333]])
    """
    if isinstance(probs, np.ndarray) and probs.ndim != 1:
        raise ValueError("The probability distribution must be specified as a flat array")
    if not np.allclose(sum(probs), 1) or min(probs) < 0:
        raise ValueError("A valid probability distribution of non-negative numbers that sum to one"
                         "must be input")

    dim = len(probs)
    unitary = np.zeros((dim, dim))

    unitary[:, 0] = np.sqrt(probs)
    unitary = np.linalg.qr(unitary)[0]

    # The QR decomposition introduces a phase of -1. We remove this so that we are preparing
    # sqrt(p_{i}) rather than -sqrt(p_{i}). Even though both options are valid, it may be surprising
    # to prepare the negative version.
    unitary *= -1

    return unitary

def func_to_unitary(func, xs):
    """TODO"""

    dim = np.prod([len(x) for x in xs])
    unitary = np.zeros((2 * dim, 2 * dim))

    for i, args in itertools.product(*reversed(xs)):
        f = func(*args)

        if not 0 <= f <= 1:
            raise ValueError("func must be bounded within the interval [0, 1]")

        unitary[i, i] = np.sqrt(1 - f)
        unitary[i + dim, i] = np.sqrt(f)
        unitary[i, i + dim] = np.sqrt(f)
        unitary[i + dim, i + dim] = - np.sqrt(1 - f)

    return unitary




@template
def QuantumMonteCarlo(distributions, random_variable, target_wires, estimation_wires):
    """TODO
    """
    ...