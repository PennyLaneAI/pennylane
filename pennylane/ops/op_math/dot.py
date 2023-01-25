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

from .parametrized_hamiltonian import ParametrizedHamiltonian


def dot(coeffs: Sequence[float], ops: Sequence[Operator], pauli=False):
    r"""Returns the dot product between the ``coeffs`` vector and the ``ops`` list of operators.

    This function returns the following linear combination: :math:`\sum_{k} c_k O_k`, where
    :math:`c_k` and :math:`O_k` are the elements inside the ``coeffs`` and ``ops`` arguments, respectively.

    Args:
        coeffs (Sequence[float]): sequence containing the coefficients of the linear combination
        ops (Sequence[Operator]): sequence containing the operators of the linear combination
        pauli (bool, optional): If ``True``, a :class:`~.PauliSentence`
            operator is used to represent the linear combination. If False, a :class:`Sum` operator
            is returned. Defaults to ``False``.

    Raises:
        ValueError: if the number of coefficients and operators does not match or if they are empty

    Returns:
        Sum or Operator: operator describing the linear combination

    **Example**

    >>> coeffs = np.array([1.1, 2.2])
    >>> ops = [qml.PauliX(0), qml.PauliY(0)]
    >>> qml.ops.dot(coeffs, ops)
    (1.1*(PauliX(wires=[0]))) + (2.2*(PauliY(wires=[0])))
    >>> qml.ops.dot(coeffs, ops, pauli=True)
    1.1 * X(0)
    + 2.2 * Y(0)

    ``pauli=True`` can be used to construct a more efficient, simplified version of the operator.
    Note that it returns a :class:`~.PauliSentence`, which is not an :class:`~.Operator`. This
    specialized representation can be converted to an operator:

    >>> qml.ops.dot([1, 2], [qml.PauliX(0), qml.PauliX(0)], pauli=True).operation()
    3.0*(PauliX(wires=[0]))

    Using ``pauli=True`` and then converting the result to an :class:`~.Operator` is much faster
    than using ``pauli=False``, but it only works for pauli words
    (see :func:`~.is_pauli_word`).
    """
    if len(coeffs) != len(ops):
        raise ValueError("Number of coefficients and operators does not match.")
    if len(coeffs) == 0 and len(ops) == 0:
        raise ValueError("Cannot compute the dot product of an empty sequence.")

    if any(callable(c) for c in coeffs):
        return ParametrizedHamiltonian(coeffs, ops)

    if pauli:
        return _pauli_dot(coeffs, ops)

    operands = []
    for coeff, op in zip(coeffs, ops):
        operands.append(qml.s_prod(coeff, op))
    return qml.op_sum(*operands) if len(operands) > 1 else operands[0]


def _pauli_dot(coeffs: Sequence[float], ops: Sequence[Operator]):
    pauli_words = defaultdict(lambda: 0)
    for coeff, op in zip(coeffs, ops):
        sentence = qml.pauli.pauli_sentence(op)
        for pw in sentence:
            pauli_words[pw] += sentence[pw] * coeff

    return qml.pauli.PauliSentence(pauli_words)
