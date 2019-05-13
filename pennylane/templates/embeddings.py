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

This module provides quantum circuit architectures that can embed features into a quantum state.
The features are associated with gate parameters (see also Schuld & Killoran 2019
:cite:`schuld2018quantum`).

Qubit architectures
-------------------

Angle embedding
***************

Encodes ``n`` features into the rotation angles of ``n`` qubits. The rotation can be Pauli-X, -Y or -Z.

.. autosummary::

    AngleEmbedding

Amplitude embedding
*******************

Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits. The absolute square of all features
has to add up to one.

.. note::

    This embedding is a wrapper for PennyLane`s :mod:`QubitStateVector()`, and only works with backends which
    implement this method.

.. autosummary::

    AmplitudeEmbedding

Basis embedding
***************

Encodes :math:`n` bits into a basis state of :math:`n` qubits.

.. note::

    This embedding is a wrapper for PennyLane`s :mod:`BasisState()`, and only works with backends which
    implement this method.

.. autosummary::

    BasisEmbedding

Continuous-variable architectures
---------------------------------

Squeezing embedding
*******************

Encodes :math:`M` features into the squeezing amplitude or phase of :math:`M` modes.


.. autosummary::

    SqueezingEmbedding


Displacement embedding
**********************

Encodes :math:`M` features into the displacement amplitude or phase of :math:`M` modes.

.. autosummary::

    DisplacementEmbedding

Code details
^^^^^^^^^^^^
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane import RX, RY, RZ, BasisState, Squeezing, Displacement, QubitStateVector
from collections.abc import Iterable


def AngleEmbedding(features, wires, rotation='X'):
    r"""
    Uses the entries of ``features`` as rotation angles of qubits.

    The details of the strategy are defined by the `rotation` parameter:
     * ``rotation = 'X'`` uses the features to chronologically apply Pauli-X rotations to qubits
     * ``rotation = 'Y'`` uses the features to chronologically apply Pauli-Y rotations to qubits
     * ``rotation = 'Z'`` uses the features to chronologically apply Pauli-Z rotations to qubits

    The length of ``features`` has to be smaller or equal to the number of qubits. If there are fewer entries in
    ``features`` than rotations, the circuit does not apply the remaining rotation gates.

    This embedding method can also be used to encode a binary sequence into a basis state. Choose ``rotation='X'``
    and features of a nonzero value of :math:`\pi /2` only where a qubit has to be prepared in state 1.

    Args:
        features (array): Input array of shape ``(N, )``, where N is the number of input features to embed
        wires (int): List of qubit indices for the qubits used for the embedding
        rotation (str): Strategy of embedding
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses, got {}.".format(wires))

    if len(features) > len(wires):
        raise ValueError("Number of features to embed cannot be larger than number of wires which is {}, "
                         "got {}.".format(len(wires), len(features)))
    if rotation == 'X':
        for f, w in zip(features, wires):
            RX(f, wires=w)

    elif rotation == 'Y':
        for f, w in zip(features, wires):
            RY(f, wires=w)

    elif rotation == 'Z':
        for f, w in zip(features, wires):
            RZ(f, wires=w)

    else:
        raise ValueError("Rotation has to be `X`, `Y` or `Z`, got {}.".format(rotation))


def BasisEmbedding(basis_state, wires):
    r"""Prepares a quantum state in the state ``basis_state``.

    For example, for ``basis_state=[0, 1, 0]``, the quantum system will be prepared in state :math:`|010 \rangle`.

    .. note::

        BasisEmbedding uses PennyLane's :class:`~.BasisState` and only works in conjunction with
        devices that implement this function.

    Args:
        features (array): Input array of shape ``(N, )``, where ``N`` is the number of input features to embed
        wires (int): List of qubit indices for the qubits used for the embedding
    """
    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses, got {}.".format(wires))

    if len(basis_state) > len(wires):
        raise ValueError("Number of bits to embed cannot be larger than number of wires which is {}, "
                         "got {}.".format(len(wires), len(basis_state)))
    BasisState(basis_state, wires=wires)


def AmplitudeEmbedding(features, wires):
    """
    Prepares a quantum state whose amplitude vector is given by ``features``.

    ``features`` has to be an array representing a 1-d vector of unit length and with 2**`n_wires` entries.

    .. note::

        AmplitudeEmbedding uses PennyLane's :class:``~.QubitStateVector`` and only works in conjunction with
        devices that implement this function.

    Args:
        features (array): Input array of shape ``(N, )``, where ``N`` is the number of input features to embed
        wires (int): List of qubit indices for the qubits used for the embedding
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses, got {}.".format(wires))

    if 2**len(wires) != len(features):
        raise ValueError("AmplitudeEmbedding requires a feature vector of size 2**n_qubits which is {}, "
                         "got {}.".format(2 ** len(wires), len(features)))

    QubitStateVector(features, wires=wires)


def SqueezingEmbedding(features, wires, execution='amplitude', c=0.1):
    r"""
    Encodes the entries of ``features`` into the squeezing phases :math:`\phi` or amplitudes :math:`r` of the modes of
    a continuous-variable quantum state.

    The mathematical definition of the squeezing gate is given by the operator

    ..math::

        S(z) = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an array of ``n_wires`` floats.

    Args:
        features (array): Binary sequence to encode
        wires (int): List of qumode indices for the qumodes used for the embedding
        execution (str): ``'phase'`` encodes the input into the phase of single-mode squeezing, while
            ``'amplitude'`` uses the amplitude.
        c (float): parameter setting the value of the phase of all squeezing gates if ``execution='amplitude'``, or the
            amplitude of all squeezing gates if ``execution='phase'`` to a constant.
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses, got {}.".format(wires))

    if len(wires) != len(features):
        raise ValueError("SqueezingEmbedding requires a feature vector of size n_wires which is {}"
                         ", got {}.".format(len(wires), len(features)))

    for f, w in zip(features, wires):
        if execution == 'amplitude':
            Squeezing(f, c, wires=w)
        elif execution == 'phase':
            Squeezing(c, f, wires=w)
        else:
            raise ValueError("Execution strategy {} not known. Has to be `phase` or `amplitude`.".format(execution))


def DisplacementEmbedding(features, wires, execution='amplitude', c=0.1):
    r"""
    Encodes the entries of ``features`` into the displacement phases :math:`\phi` or amplitudes :math:`r` of the modes of
    a continuous-variable quantum state.

    The mathematical definition of the displacement gate is given by the operator

    ..math::
            D(\alpha) = \exp(r (e^{i\phi}\ad -e^{-i\phi}\a)),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an array of ``n_wires`` floats.

    Args:
        features (array): Binary sequence to encode
        wires (int): List of qumode indices for the qumodes used for the embedding
        execution (str): ``'phase'`` encodes the input into the phase of single-mode squeezing, while
            ``'amplitude'`` uses the amplitude.
        c (float): parameter setting the value of the phase of all squeezing gates if ``execution='amplitude'``, or the
            amplitude of all squeezing gates if ``execution='phase'`` to a constant.
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses, got {}.".format(wires))

    if len(wires) != len(features):
        raise ValueError("DisplacementEmbedding requires a feature vector of size n_wires which is {}"
                         ", got {}.".format(len(wires), len(features)))

    for f, w in zip(features, wires):
        if execution == 'amplitude':
            Displacement(f, c, wires=w)
        elif execution == 'phase':
            Displacement(c, f, wires=w)
        else:
            raise ValueError("Execution strategy {} not known. Has to be `phase` or `amplitude`.".format(execution))
