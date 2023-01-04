# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This file contains the definition of the dot function, which computes the dot product between
a vector and a list of operators.
"""
from collections import defaultdict
from typing import Sequence

import pennylane as qml
from pennylane.operation import Operator


def dot(coeffs: Sequence[float], ops: Sequence[Operator], pauli=False):
    r"""Returns the dot product between the ``coeffs`` vector and the ``ops`` list of operators.

    This function returns the following linear combination: :math:`\sum_{k=0}^{N-1} c_k O_k`, where
    :math:`c_k` and :math:`O_k` are the elements inside the ``coeffs`` and ``ops`` arguments respectively.

    Args:
        coeffs (Sequence[float]): sequence containing the coefficients of the linear combination
        ops (Sequence[Operator]): sequence containing the operators of the linear combination
        pauli (bool, optional): If True, a :class:`pennylane.pauli.pauli_arithmetic.PauliSentence`
            operator is used to represent the linear combination. If False, a :class:`Sum` operator
            is returned. Defaults to False.

    Raises:
        ValueError: if the number of coefficients and operators does not match or if they are empty

    Returns:
        .Sum | .Operator: operator describing the linear combination

    **Example**

    >>> coeffs = np.array([1.1, 2.2])
    >>> ops = [qml.PauliX(0), qml.PauliY(0)]
    >>> qml.ops.dot(coeffs, ops)
    (1.1*(PauliX(wires=[0]))) + (2.2*(PauliY(wires=[0])))
    >>> qml.ops.dot(coeffs, ops, pauli=True)
    1.1 * X(0)
    + 2.2 * Y(0)
    """
    if qml.math.shape(coeffs)[0] != len(ops):
        raise ValueError("Number of coefficients and operators does not match.")
    if len(coeffs) == 0 and len(ops) == 0:
        raise ValueError("Cannot compute the dot product of an empty sequence.")

    pauli_words = defaultdict(lambda: 0)
    for coeff, op in zip(coeffs, ops):
        sentence = qml.pauli.pauli_sentence(op)
        for pw in sentence:
            pauli_words[pw] += sentence[pw] * coeff

    pauli_sentence = qml.pauli.PauliSentence(pauli_words)

    return pauli_sentence if pauli else pauli_sentence.operation()
