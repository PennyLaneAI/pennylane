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
r"""
Contains the ArbitraryStatePreparation template.
"""
# pylint: disable=trailing-comma-tuple
import functools
import pennylane as qml
from pennylane.operation import Operation, AnyWires


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


class ArbitraryStatePreparation(Operation):
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
            qml.ArbitraryStatePreparation(weights, wires=[0, 1, 2, 3])

            return qml.expval(qml.Hermitian(H, wires=[0, 1, 2, 3]))

    The shape of the weights parameter can be computed as follows:

    .. code-block:: python

        shape = qml.ArbitraryStatePreparation.shape(n_wires=4)


    Args:
        weights (tensor_like): Angles of the Pauli word rotations. Needs to have length :math:`2^{n+1} - 2`
            where :math:`n` is the number of wires the template acts upon.
        wires (Iterable): wires that the template acts on
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, do_queue=True, id=None):

        shape = qml.math.shape(weights)
        if shape != (2 ** (len(wires) + 1) - 2,):
            raise ValueError(
                f"Weights tensor must be of shape {(2 ** (len(wires) + 1) - 2,)}; got {shape}."
            )

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    def expand(self):

        with qml.tape.QuantumTape() as tape:
            for i, pauli_word in enumerate(_state_preparation_pauli_words(len(self.wires))):
                qml.PauliRot(self.parameters[0][i], pauli_word, wires=self.wires)

        return tape

    @staticmethod
    def shape(n_wires):
        r"""Returns the required shape for the weight tensor.

        Args:
                n_wires (int): number of wires

        Returns:
            tuple[int]: shape
        """
        return (2 ** (n_wires + 1) - 2,)
