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
r"""
Contains the ``ArbitraryUnitary`` template.
"""
import numpy as np
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.templates.utils import check_wires, check_shape, get_shape

_PAULIS = ["I", "X", "Y", "Z"]


def _tuple_to_word(index_tuple):
    return "".join([_PAULIS[i] for i in index_tuple])

def _n_k_gray_code(n, k, start=0):
    """Iterates over a full n-ary Gray code with k digits.

    Args:
        n (int): Base of the Gray code. Needs to be greater than one.
        k (int): Number of digits of the Gray code. Needs to be greater than zero.
        start (int, optional): Optional start of the Gray code. The generated code
            will be shorter as the code does not wrap. Defaults to 0.
    """
    for i in range(start, n ** k):
        codeword = [0] * k

        base_repesentation = []
        val = i

        for j in range(k):
            base_repesentation.append(val % n)
            val //= n

        shift = 0
        for j in reversed(range(k)):
            codeword[j] = (base_repesentation[j] + shift) % n
            shift += (n - codeword[j])

        yield codeword


def _all_pauli_words_but_identity(num_wires):
    # Start at 1 to ignore identity
    yield from (_tuple_to_word(idx_tuple) for idx_tuple in _n_k_gray_code(4, num_wires, start=1))


@template
def ArbitraryUnitary(angles, wires):
    """Implements an arbitrary unitary on the specified wires.

    An arbitrary unitary on :math:`n` wires is parametrized by :math:`4^n - 1`
    independent real parameters. This templates uses Pauli word rotations to
    parametrize the unitary.

    Args:
        angles (array[float]): The angles of the Pauli word rotations, needs to have length :math:`4^n - 1`
            where :math:`n` is the number of wires the template acts upon.
        wires (List[int]): The wires on which the arbitrary unitary acts.
    """
    wires = check_wires(wires)

    n_wires = len(wires)
    expected_shape = (4 ** n_wires - 1,)
    check_shape(
        angles,
        expected_shape,
        msg="'angles' must be of shape {}; got {}." "".format(expected_shape, get_shape(angles)),
    )

    for i, pauli_word in enumerate(_all_pauli_words_but_identity(len(wires))):
        qml.PauliRot(angles[i], pauli_word, wires=wires)
