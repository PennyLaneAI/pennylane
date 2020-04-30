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
Contains the ``ArbitraryStatePreparation`` template.
"""
import functools
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.templates.utils import check_wires, check_shape, get_shape


@functools.lru_cache()
def _state_preparation_pauli_words(num_wires):
    if num_wires == 1:
        return ["X", "Y"]

    sub_pauli_words = _state_preparation_pauli_words(num_wires - 1)
    sub_id = "I" * (num_wires - 1)

    single_qubit_words = ["X" + sub_id, "Y" + sub_id]
    multi_qubit_words = list(map(lambda word: "I" + word, sub_pauli_words)) + list(
        map(lambda word: "X" + word, sub_pauli_words)
    )

    return single_qubit_words + multi_qubit_words


@template
def ArbitraryStatePreparation(angles, wires):
    """Implements an arbitrary state preparation on the specified wires.

    An arbitrary state on :math:`n` wires is parametrized by :math:`2^{n+1} - 2`
    independent real parameters. This templates uses Pauli word rotations to
    parametrize the unitary.

    Args:
        angles (array[float]): The angles of the Pauli word rotations, needs to have length :math:`2^n - 2`
            where :math:`n` is the number of wires the template acts upon.
        wires (List[int]): The wires on which the arbitrary unitary acts.
    """
    wires = check_wires(wires)

    n_wires = len(wires)
    expected_shape = (2 ** (n_wires + 1) - 2,)
    check_shape(
        angles,
        expected_shape,
        msg="'angles' must be of shape {}; got {}." "".format(expected_shape, get_shape(angles)),
    )

    for i, pauli_word in enumerate(_state_preparation_pauli_words(len(wires))):
        qml.PauliRot(angles[i], pauli_word, wires=wires)
