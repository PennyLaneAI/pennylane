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
        wires (Any or Iterable[Any]): wires that the template acts on
        rotation (str): type of rotations used
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

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

        >>> print(qml.draw(circuit, expansion_strategy="device")(X))
        0: ──RZ(1.00)──H─┤ ╭Probs
        1: ──RZ(2.00)────┤ ├Probs
        2: ──RZ(3.00)────┤ ╰Probs

    """

    num_wires = AnyWires
    grad_method = None

    def _flatten(self):
        hyperparameters = (("rotation", self._rotation),)
        return self.data, (self.wires, hyperparameters)

    def __repr__(self):
        return f"AngleEmbedding({self.data[0]}, wires={self.wires.tolist()}, rotation={self._rotation})"

    def __init__(self, features, wires, rotation="X", id=None):
        if rotation not in ROT:
            raise ValueError(f"Rotation option {rotation} not recognized.")

        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        self._rotation = rotation
        self._hyperparameters = {"rotation": ROT[rotation]}

        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (1,)

    @staticmethod
    def compute_decomposition(features, wires, rotation):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.AngleEmbedding.decomposition`.

        Args:
            features (tensor_like): input tensor of dimension ``(len(wires),)``
            wires (Any or Iterable[Any]): wires that the operator acts on
            rotation (.Operator): rotation gate class

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> features = torch.tensor([1., 2.])
        >>> qml.AngleEmbedding.compute_decomposition(features, wires=["a", "b"], rotation=qml.RX)
        [RX(tensor(1.), wires=['a']),
         RX(tensor(2.), wires=['b'])]
        """
        batched = qml.math.ndim(features) > 1
        # We will iterate over the first axis of `features` together with iterating over the wires.
        # If the leading dimension is a batch dimension, exchange the wire and batching axes.
        features = qml.math.T(features) if batched else features

        return [rotation(features[i], wires=wires[i]) for i in range(len(wires))]
