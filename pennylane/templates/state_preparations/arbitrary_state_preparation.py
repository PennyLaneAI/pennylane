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
from pennylane.templates.utils import check_shape, get_shape
from pennylane.wires import Wires


def _preprocess(weights, wires):
    """Validate and pre-process inputs as follows:

    * Check the shape of the weights tensor.

    Args:
        weights (tensor_like): trainable parameters of the template
        wires (Wires): wires that template acts on
    """

    if qml.tape_mode_active():

        shape = qml.math.shape(weights)
        if shape != (2 ** (len(wires) + 1) - 2,):
            raise ValueError(
                f"Weights tensor must be of shape {(2 ** (len(wires) + 1) - 2,)}; got {shape}."
            )

    else:

        expected_shape = (2 ** (len(wires) + 1) - 2,)
        check_shape(
            weights,
            expected_shape,
            msg="Weights tensor must be of shape {}; got {}.".format(
                expected_shape, get_shape(weights)
            ),
        )


@functools.lru_cache()
def _state_preparation_pauli_words(num_wires):
    """Pauli words necessary for a state preparation.

    Args:
        num_wires (int): Number of wires of the state preparation

    Returns:
        List[str]: List of all necessary Pauli words for the state preparation
    """
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
def ArbitraryStatePreparation(weights, wires):
    """Implements an arbitrary state preparation on the specified wires.

    An arbitrary state on :math:`n` wires is parametrized by :math:`2^{n+1} - 2`
    independent real parameters. This templates uses Pauli word rotations to
    parametrize the unitary.

    **Example**

    ArbitraryStatePreparation can be used to train state preparations,
    for example using a circuit with some measurement observable ``H``:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def vqe(weights):
            qml.ArbitraryStatePreparations(weights, wires=[0, 1, 2, 3])

            return qml.expval(qml.Hermitian(H, wires=[0, 1, 2, 3]))

    Args:
        weights (tensor_like): The angles of the Pauli word rotations, needs to have length :math:`2^(n+1) - 2`
            where :math:`n` is the number of wires the template acts upon.
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
    """

    wires = Wires(wires)
    _preprocess(weights, wires)

    for i, pauli_word in enumerate(_state_preparation_pauli_words(len(wires))):
        qml.PauliRot(weights[i], pauli_word, wires=wires)
