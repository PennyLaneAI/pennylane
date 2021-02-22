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
Contains the ``AngleEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.templates import broadcast
from pennylane.wires import Wires


def _preprocess(features, wires):
    """Validate and pre-process inputs as follows:

    * Check that the features tensor is one-dimensional.
    * Check that the first dimension of the features tensor
      has length :math:`n` or less, where :math:`n` is the number of qubits.

    Args:
        features (tensor_like): input features to pre-process
        wires (Wires): wires that template acts on

    Returns:
        int: number of features
    """
    shape = qml.math.shape(features)

    if len(shape) != 1:
        raise ValueError(f"Features must be a one-dimensional tensor; got shape {shape}.")

    n_features = shape[0]
    if n_features > len(wires):
        raise ValueError(
            f"Features must be of length {len(wires)} or less; got length {n_features}."
        )
    return n_features


@template
def AngleEmbedding(features, wires, rotation="X"):
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
        features (array): input array of shape ``(N,)``, where N is the number of input features to embed,
            with :math:`N\leq n`
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        rotation (str): type of rotations used

    """

    wires = Wires(wires)
    n_features = _preprocess(features, wires)
    wires = wires[:n_features]

    if rotation == "X":
        broadcast(unitary=qml.RX, pattern="single", wires=wires, parameters=features)

    elif rotation == "Y":
        broadcast(unitary=qml.RY, pattern="single", wires=wires, parameters=features)

    elif rotation == "Z":
        broadcast(unitary=qml.RZ, pattern="single", wires=wires, parameters=features)

    else:
        raise ValueError(f"Rotation option {rotation} not recognized.")
