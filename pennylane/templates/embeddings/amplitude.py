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
Contains the AmplitudeEmbedding template.
"""
from pennylane.ops import StatePrep


# pylint: disable=too-many-arguments
class AmplitudeEmbedding(StatePrep):
    r"""Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.

    By setting ``pad_with`` to a real or complex number, ``features`` is automatically padded to dimension
    :math:`2^n` where :math:`n` is the number of qubits used in the embedding.

    To represent a valid quantum state vector, the L2-norm of ``features`` must be one.
    The argument ``normalize`` can be set to ``True`` to automatically normalize the features.

    If both automatic padding and normalization are used, padding is executed *before* normalizing.

    Args:
        features (tensor_like): input tensor of dimension ``(2^len(wires),)``, or less if `pad_with` is specified
        wires (Any or Iterable[Any]): wires that the template acts on
        pad_with (float or complex): if not None, the input is padded with this constant to size :math:`2^n`
        normalize (bool): whether to automatically normalize the features
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
        validate_norm (bool): whether to validate the norm of the input state

    Example:

        Amplitude embedding encodes a normalized :math:`2^n`-dimensional feature vector into the state
        of :math:`n` qubits:

        .. code-block:: python

            import pennylane as qml

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(f=None):
                qml.AmplitudeEmbedding(features=f, wires=range(2))
                return qml.state()

            state = circuit(f=[1/2, 1/2, 1/2, 1/2])

        The final state of the device is - up to a global phase - equivalent to the input passed to the circuit:

        >>> state
        array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])

        **Differentiating with respect to the features**

        Due to non-trivial classical processing to construct the state preparation circuit,
        the features argument is in general **not differentiable**.

        **Normalization**

        The template will raise an error if the feature input is not normalized.
        One can set ``normalize=True`` to automatically normalize it:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(f=None):
                qml.AmplitudeEmbedding(features=f, wires=range(2), normalize=True)
                return qml.state()

            state = circuit(f=[15, 15, 15, 15])

        >>> state
        array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])

        **Padding**

        If the dimension of the feature vector is smaller than the number of amplitudes,
        one can automatically pad it with a constant for the missing dimensions using the ``pad_with`` option:

        .. code-block:: python

            from math import sqrt

            @qml.qnode(dev)
            def circuit(f=None):
                qml.AmplitudeEmbedding(features=f, wires=range(2), pad_with=0.)
                return qml.state()

            state = circuit(f=[1/sqrt(2), 1/sqrt(2)])

        >>> state # doctest: +SKIP
        array([0.7071+0.j, 0.7071+0.j, 0.    +0.j, 0.    +0.j])

    """

    def __init__(
        self, features, wires, *, pad_with=None, normalize=False, id=None, validate_norm=True
    ):
        super().__init__(
            features,
            wires=wires,
            pad_with=pad_with,
            normalize=normalize,
            validate_norm=validate_norm,
            id=id,
        )
