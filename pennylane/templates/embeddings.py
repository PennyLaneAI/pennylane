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
The features are associated with gate parameters, implicitly mapping them into the Hilbert space of the quantum state
(see also Schuld & Killoran 2019 :cite:`schuld2018quantum`).

Qubit architectures
-------------------

.. autosummary::

    AngleEmbedding
    AmplitudeEmbedding
    BasisEmbedding

Continuous-variable architectures
---------------------------------

.. autosummary::

    SqueezingEmbedding
    DisplacementEmbedding

Code details
^^^^^^^^^^^^
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane import RX, RY, RZ, BasisState, Squeezing, Displacement, QubitStateVector
from collections.abc import Iterable


def AngleEmbedding(features, rotation='X', wires=None):
    r"""
    Encodes :math:`n` features into the rotation angles of :math:`n` qubits.

    The rotations can be chosen as either :class:`~.RX`, :class:`~.RY`
    or :class:`~.RZ` gates, as defined by the ``rotation`` parameter:

    * ``rotation='X'`` uses the features to chronologically apply RX rotations to qubits

    * ``rotation='Y'`` uses the features to chronologically apply RY rotations to qubits

    * ``rotation='Z'`` uses the features to chronologically apply RZ rotations to qubits

    The length of ``features`` has to be smaller or equal to the number of qubits. If there are fewer entries in
    ``features`` than rotations, the circuit does not apply the remaining rotation gates.

    This embedding method can also be used to encode a binary sequence into a basis state. Choose ``rotation='X'``
    and features of a nonzero value of :math:`\pi /2` only where a qubit has to be prepared in state 1.

    Args:
        features (array): Input array of shape ``(N, )``, where N is the number of input features to embed, with :math:`N\leq n`

    Keyword Args:
        rotation (str): Type of rotations used
        wires (Sequence[int]): sequence of qubit indices that the template acts on
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses, got {}.".format(wires))

    if len(features) > len(wires):
        raise ValueError("Number of features to embed cannot be larger than number of wires, which is {}; "
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


def AmplitudeEmbedding(features, wires=None):
    """
    Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.

    The absolute square of all elements in ``features`` has to add up to one.

    .. note::

        AmplitudeEmbedding uses PennyLane's :class:`~.QubitStateVector` and only works in conjunction with
        devices that implement this function.

    Args:
        features (array): Input array of shape ``(2**n, )``

    Keyword Args:
        wires (Sequence[int]): sequence of qubit indices that the template acts on
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses, got {}.".format(wires))

    if 2**len(wires) != len(features):
        raise ValueError("AmplitudeEmbedding requires a feature vector of size 2**n_qubits which is {}, "
                         "got {}.".format(2 ** len(wires), len(features)))

    QubitStateVector(features, wires=wires)


def BasisEmbedding(features, wires=None):
    r"""Encodes :math:`n` binary features into a basis state of :math:`n` qubits.

    For example, for ``features=[0, 1, 0]``, the quantum system will be prepared in state :math:`|010 \rangle`.

    .. note::

        BasisEmbedding uses PennyLane's :class:`~.BasisState` and only works in conjunction with
        devices that implement this function.

    Args:
        features (array): Binary input array of shape ``(n, )``

    Keyword Args:
        wires (Sequence[int]): sequence of qubit indices that the template acts on
    """
    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses, got {}.".format(wires))

    if len(features) > len(wires):
        raise ValueError("Number of bits to embed cannot be larger than number of wires which is {}, "
                         "got {}.".format(len(wires), len(features)))
    BasisState(features, wires=wires)


def SqueezingEmbedding(features, execution='amplitude', c=0.1, wires=None):
    r"""Encodes :math:`M` features into the squeezing amplitudes :math:`r \geq 0` or phases :math:`\phi \in [0, 2\pi)`
    of :math:`M` modes.

    The mathematical definition of the squeezing gate is given by the operator

    .. math::

        S(z) = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an array of ``len(wires)`` floats.

    Args:
        features (array): Binary sequence to encode

    Keyword Args:
        execution (str): ``'phase'`` encodes the input into the phase of single-mode squeezing, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all squeezing gates if ``execution='amplitude'``, or the
            amplitude of all squeezing gates if ``execution='phase'``
        wires (Sequence[int]): sequence of mode indices that the template acts on
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


def DisplacementEmbedding(features, execution='amplitude', c=0.1, wires=None):
    r"""Encodes :math:`M` features into the displacement amplitudes :math:`r` or phases :math:`\phi` of :math:`M` modes.

    The mathematical definition of the displacement gate is given by the operator

    .. math::
            D(\alpha) = \exp(r (e^{i\phi}\ad -e^{-i\phi}\a)),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an array of ``len(wires)`` floats.

    Args:
        features (array): Binary sequence to encode

    Keyword Args:
        execution (str): ``'phase'`` encodes the input into the phase of single-mode displacement, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all displacement gates if ``execution='amplitude'``, or
            the amplitude of all displacement gates if ``execution='phase'``
        wires (Sequence[int]): sequence of mode indices that the template acts on
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
