# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Embeddings
==========

**Module name:** :mod:`pennylane.templates.embeddings`

.. currentmodule:: pennylane.templates.embeddings

This module provides quantum circuit architectures that can serve as an embedding of inputs
(represented by the gate parameters) into a quantum state (see also Schuld & Killoran 2019
:cite:`schuld2019`).

Provided embeddings
--------------------

For qubit architectures:
************************

.. autosummary::

    AngleEmbedding
    AmplitudeEmbedding
    BasisEmbedding

For continuous-variable architectures:
**************************************

.. autosummary::

    SqueezingEmbedding
    DisplacementEmbedding

Code details
^^^^^^^^^^^^
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane import RX, RY, RZ, BasisState, PauliX, Squeezing, Displacement


def AngleEmbedding(features, n_wires, rotation='X'):
    """
    Uses the entries of `features` as rotation angles of qubits.

    The details of the strategy are defined by the `rotation` parameter:
     * `rotation = 'X'` uses the features to chronologically apply Pauli-X rotations to qubits
     * `rotation = 'Y'` uses the features to chronologically apply Pauli-Y rotations to qubits
     * `rotation = 'Z'` uses the features to chronologically apply Pauli-Z rotations to qubits
     * `rotation = 'XY'` performs the 'X' strategy using the first `n_wires` features, and the 'Y' strategy using the
        remaining qubits
     * `rotation = 'XY'` performs the 'X' strategy using the first `n_wires` features, the 'Y' strategy using the
        next `n_wires` features, and the 'Z' strategy using the remaining features

    If there are fewer entries in `features` than rotations prescribed by the strategy, the circuit does not apply the
    remaining rotation gates.
    If there are fewer rotations than entries in `features`, the circuit will not use the remaining features.

    Args:
        features (array): Input array of shape (N, ), where N is the number of input features to embed
        n_wires (int): Number of qubits in the circuit
        rotation (str): Strategy of embedding

    """
    if rotation == 'XYZ':
        n_ops = min(len(features), 3 * n_wires)

        for op in range(n_ops):
            if op < n_wires:
                RX(features[op], wires=op)
            elif op < 2*n_wires:
                RY(features[op], wires=op)
            else:
                RZ(features[op], wires=op)

    if rotation == 'XY':
        n_ops = min(len(features), 2 * n_wires)

        for op in range(n_ops):
            if op < n_wires:
                RX(features[op], wires=op)
            else:
                RY(features[op], wires=op)

    else:
        n_ops = min(len(features), n_wires)

        if rotation == 'X':
            for op in range(n_ops):
                RX(features[op], wires=op)

        elif rotation == 'Y':
            for op in range(n_ops):
                RY(features[op], wires=op)

        elif rotation == 'Z':
            for op in range(n_ops):
                RZ(features[op], wires=op)


def AmplitudeEmbedding(features, n_wires, execution='hack'):
    """
    Prepares a quantum state whose amplitude vector is given by `features`.

    `features` has to be an array representing a 1-d vector of unit length and with 2**`n_qubits` entries.

    .. note::
        At this point only `execution='hack'` is implemented, which uses the `BasisState` method available
        to some simulator backends.

    Args:
        features (array): Input array of shape (N, ), where N is the number of input features to embed
        n_wires (int): Number of qubits in the circuit
        execution (str): Strategy of implementation.

    """

    if 2**n_wires != len(features):
        raise ValueError("AmplitudeEmbedding requires a feature vector of size 2**n_wires, got {}.".format(len(features)))

    if execution == 'hack':
        BasisState(features, wires=range(n_wires))

    # Todo: Implement circuit which prepares amplitude vector using gates


def BasisEmbedding(features, n_wires):
    """
    Prepares a quantum basis state equivalent to the binary string `features`.

    `features` has to be an array of `n_qubits` binary entries.

    Args:
        features (array): Binary sequence to encode
        n_wires (int): Number of qubits in the circuit
    """

    if n_wires != len(features):
        raise ValueError("BasisEmbedding requires a feature vector of size n_wires, got {}.".format(len(features)))

    entrytypes = sorted(set(features))
    if len(entrytypes) > 2:
        raise ValueError("BasisEmbedding requires a feature vector containing only two kinds of entries, got {}.".format(features))

    for i in range(n_wires):
        if features[i] == entrytypes[1]:
            PauliX(wires=i)


def SqueezingEmbedding(features, n_wires, execution='amplitude'):
    """
    Encodes the entries of `features` into the squeezing phases :math:`\phi` or amplitudes :math:`r` of the modes of
    a continuous-variable quantum state.

    The mathematical definition of the squeezing gate is given by the operator

    ..math::
            S(z) = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    `features` has to be an array of `n_wires` floats.

    Args:
        features (array): Binary sequence to encode
        n_wires (int): Number of qubits in the circuit
        execution (str): 'phase' encodes the input into the phase of single-mode squeezing, while
                         'amplitude' uses the amplitude.
    """

    if n_wires != len(features):
        raise ValueError("Squeezing Embedding requires a feature vector of size n_wires, got {}.".format(len(features)))

    for i in range(n_wires):
        if execution == 'amplitude':
            Squeezing(r=features[i], phi=0, wires=i)
        elif execution == 'phase':
            Squeezing(r=0, phi=features[i], wires=i)
        else:
            raise ValueError("Execution strategy {} not known. Has to be 'phase' or 'amplitude'.".format(execution))


def DisplacementEmbedding(features, n_wires, execution='amplitude'):
    """
    Encodes the entries of `features` into the displacement phases :math:`\phi` or amplitudes :math:`r` of the modes of
    a continuous-variable quantum state.

    The mathematical definition of the displacement gate is given by the operator

    ..math::
            D(\alpha) = \exp(r (e^{i\phi}\ad -e^{-i\phi}\a)),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    `features` has to be an array of `n_wires` floats.

    Args:
        features (array): Binary sequence to encode
        n_wires (int): Number of qubits in the circuit
        execution (str): 'phase' encodes the input into the phase of single-mode squeezing, while
                         'amplitude' uses the amplitude.
    """

    if n_wires != len(features):
        raise ValueError("Squeezing Embedding requires a feature vector of size n_wires, got {}.".format(len(features)))

    for i in range(n_wires):
        if execution == 'amplitude':
            Displacement(a=features[i], phi=0, wires=i)
        elif execution == 'phase':
            Displacement(a=0, phi=features[i], wires=i)
        else:
            raise ValueError("Execution strategy {} not known. Has to be 'phase' or 'amplitude'.".format(execution))
