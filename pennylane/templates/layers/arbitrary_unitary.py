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


def _all_pauli_words_but_identity(num_wires):
    # TODO: Replace this with a 4-ary Gray code. This will make only one letter
    # change between each Pauli word and thus requires less gates. We would still
    # need some gate fusion logic for this to take effect.
    index_tuples = np.ndindex(tuple([4] * num_wires))

    # The first index represents the all-identity-word which we skip
    next(index_tuples)

    yield from (_tuple_to_word(idx_tuple) for idx_tuple in index_tuples)


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
