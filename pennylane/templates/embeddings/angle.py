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
Contains the ``AngleEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.ops import RX, RY, RZ
from pennylane.operation import Operation, AnyWires

ROT = {"X": RX, "Y": RY, "Z": RZ}


class AngleEmbedding(Operation):
    r"""
    Encodes :math:`N` features into the rotation angles of :math:`n` qubits, where :math:`N \leq n`.

    The rotations can be chosen as either :class:`~pennylane.ops.RX`, :class:`~pennylane.ops.RY`
    or :class:`~pennylane.ops.RZ` gates, as defined by the ``rotation`` parameter:

    * ``rotation='X'`` uses the features as angles of RX rotations

    * ``rotation='Y'`` uses the features as angles of RY rotations

    * ``rotation='Z'`` uses the features as angles of RZ rotations

    The length of ``features`` has to be smaller or equal to the number of qubits. If there are fewer entries in
    ``features`` than rotations, the circuit does not apply the remaining rotation gates.

    Args:
        features (tensor_like): input tensor of shape ``(N,)``, where N is the number of input features to embed,
            with :math:`N\leq n`
        wires (Iterable): wires that the template acts on
        rotation (str): type of rotations used

    Example:

        Angle embedding encodes the features by using the specified rotation operation.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(feature_vector):
                qml.AngleEmbedding(features=feature_vector, wires=range(3), rotation='Z')
                qml.Hadamard(0)
                return qml.probs(wires=range(3))

            X = [1,2,3]

        Here, we have also used rotation angles :class:`RZ`. If not specified, :class:`RX` is used as default.
        The resulting circuit is:

        >>> print(qml.draw(circuit)(X))
            0: ──RZ(1)──H──╭┤ Probs
            1: ──RZ(2)─────├┤ Probs
            2: ──RZ(3)─────╰┤ Probs

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, features, wires, rotation="X", do_queue=True, id=None):

        if rotation not in ROT:
            raise ValueError(f"Rotation option {rotation} not recognized.")
        self.rotation = ROT[rotation]

        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        wires = wires[:n_features]
        super().__init__(features, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    def expand(self):

        features = self.parameters[0]
        batched = len(qml.math.shape(features)) > 1

        features = features.T if batched else features

        with qml.tape.QuantumTape() as tape:

            for i in range(len(self.wires)):
                self.rotation(features[i], wires=self.wires[i])

        return tape
