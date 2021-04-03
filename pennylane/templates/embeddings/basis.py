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
Contains the ``BasisEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.wires import Wires


def _preprocess(features, wires):
    """Validate and pre-process inputs as follows:

    * Check that the features tensor is one-dimensional.
    * Check that the first dimension of the features tensor
      has length :math:`n`, where :math:`n` is the number of qubits.
    * Check that the entries of the features tensor are zeros and ones.

    Args:
        features (tensor_like): input features to pre-process
        wires (Wires): wires that template acts on

    Returns:
        array: numpy array representation of the features tensor
    """
    shape = qml.math.shape(features)

    if len(shape) != 1:
        raise ValueError(f"Features must be one-dimensional; got shape {shape}.")

    n_features = shape[0]
    if n_features != len(wires):
        raise ValueError(f"Features must be of length {len(wires)}; got length {n_features}.")

    features = list(qml.math.toarray(features))

    if not set(features).issubset({0, 1}):
        raise ValueError(f"Basis state must only consist of 0s and 1s; got {features}")

    return features


@template
def BasisEmbedding(features, wires):
    r"""Encodes :math:`n` binary features into a basis state of :math:`n` qubits.

    For example, for ``features=np.array([0, 1, 0])``, the quantum system will be
    prepared in state :math:`|010 \rangle`.

    .. warning::

        ``BasisEmbedding`` calls a circuit whose architecture depends on the binary features.
        The ``features`` argument is therefore not differentiable when using the template, and
        gradients with respect to the argument cannot be computed by PennyLane.

    Args:
        features (array): binary input array of shape ``(n, )``
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.

    Raises:
        ValueError: if inputs do not have the correct format
    """

    wires = Wires(wires)

    features = _preprocess(features, wires)

    for wire, bit in zip(wires, features):
        if bit == 1:
            qml.PauliX(wire)
