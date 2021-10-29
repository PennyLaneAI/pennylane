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
from pennylane import numpy as np

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
    """

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"
    grad_method = None

    def __init__(self, features, wires, rotation="X", do_queue=True, id=None):

        if rotation not in ROT:
            raise ValueError(f"Rotation option {rotation} not recognized.")
        self.rotation = ROT[rotation]

        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]

        features = self._preprocess(features, wires)

        wires = wires[:n_features]
        super().__init__(features, wires=wires, do_queue=do_queue, id=id)

    def expand(self):
        features = np.array(self.parameters[0]).T
        with qml.tape.QuantumTape() as tape:
            for i in range(len(self.wires)):
                self.rotation(features[i], wires=self.wires[i])

        return tape

    @staticmethod
    def _preprocess(features, wires):
        """Validate and pre-process inputs as follows:

        * If features is batched, the processing that follows is applied to each feature set in the batch.
        * Check that the features tensor is one-dimensional.
        """

        # check if features is batched
        batched = len(qml.math.shape(features)) > 1

        features_batch = features if batched else [features]

        # apply pre-processing to each features tensor in the batch
        for i, feature_set in enumerate(features_batch):
            shape = qml.math.shape(feature_set)

            # check shape
            if len(shape) != 1:
                raise ValueError(f"Features must be a one-dimensional tensor; got shape {shape}.")

            n_features = shape[0]
            if n_features > len(wires):
                raise ValueError(
                    f"Features must be of length {len(wires)} or less; got length {n_features}."
                )
            features_batch[i] = qml.math.cast(feature_set, np.complex128)

        return features_batch if batched else features_batch[0]
