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

Angle embedding
***************

.. autosummary::

    AngleEmbedding

Amplitude embedding
*******************

.. autosummary::

    AmplitudeEmbedding

Basis embedding
***************

.. autosummary::

    BasisEmbedding

Continuous-variable architectures
---------------------------------

Squeezing embedding
*******************

.. autosummary::

    SqueezingEmbedding

Displacement embedding
**********************

.. autosummary::

    DisplacementEmbedding

Code details
^^^^^^^^^^^^
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections.abc import Iterable
from pennylane.ops import RX, RY, RZ, BasisState, Squeezing, Displacement, QubitStateVector


def AngleEmbedding(features, wires, rotation='X'):
    r"""
    Encodes :math:`N` features into the rotation angles of :math:`n` qubits, where :math:`N \leq n`.

    The rotations can be chosen as either :class:`~pennylane.ops.RX`, :class:`~pennylane.ops.RY`
    or :class:`~pennylane.ops.RZ` gates, as defined by the ``rotation`` parameter:

    * ``rotation='X'`` uses the features as angles of RX rotations

    * ``rotation='Y'`` uses the features as angles of RY rotations

    * ``rotation='Z'`` uses the features as angles of RZ rotations

    The length of ``features`` has to be smaller or equal to the number of qubits. If there are fewer entries in
    ``features`` than rotations, the circuit does not apply the remaining rotation gates.

    This embedding method can also be used to encode a binary sequence into a basis state. For example, to prepare
    basis state :math:`|0,1,1,0\rangle`, choose ``rotation='X'`` and use the
    feature vector :math:`[0, \pi/2, \pi/2, 0]`. Alternatively, one can use the :mod:`BasisEmbedding()` template.

    Args:
        features (array): Input array of shape ``(N,)``, where N is the number of input features to embed,
            with :math:`N\leq n`
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Keyword Args:
        rotation (str): Type of rotations used
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses; got {}.".format(wires))

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
        raise ValueError("Rotation has to be `X`, `Y` or `Z`; got {}.".format(rotation))


def AmplitudeEmbedding(features, wires):
    r"""Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.

    The absolute square of all elements in ``features`` has to add up to one.

    .. note::

        AmplitudeEmbedding uses PennyLane's :class:`~pennylane.ops.QubitStateVector` and only works in conjunction with
        devices that implement this function.

    Args:
        features (array): Input array of shape ``(2**n,)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses; got {}.".format(wires))

    if 2**len(wires) != len(features):
        raise ValueError("AmplitudeEmbedding requires a feature vector of size 2**len(wires), which is {}; "
                         "got {}.".format(2 ** len(wires), len(features)))

    QubitStateVector(features, wires=wires)


def BasisEmbedding(features, wires):
    r"""Encodes :math:`n` binary features into a basis state of :math:`n` qubits.

    For example, for ``features=[0, 1, 0]``, the quantum system will be prepared in state :math:`|010 \rangle`.

    .. note::

        BasisEmbedding uses PennyLane's :class:`~pennylane.ops.BasisState` and only works in conjunction with
        devices that implement this function.

    Args:
        features (array): Binary input array of shape ``(n, )``
        wires (Sequence[int]): sequence of qubit indices that the template acts on
    """
    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses; got {}.".format(wires))

    if len(features) > len(wires):
        raise ValueError("Number of bits to embed cannot be larger than number of wires, which is {}; "
                         "got {}.".format(len(wires), len(features)))
    BasisState(features, wires=wires)


def SqueezingEmbedding(features, wires, method='amplitude', c=0.1):
    r"""Encodes :math:`N` features into the squeezing amplitudes :math:`r \geq 0` or phases :math:`\phi \in [0, 2\pi)`
    of :math:`M` modes, where :math:`N\leq M`.

    The mathematical definition of the squeezing gate is given by the operator

    .. math::

        S(z) = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an array of at most ``len(wires)`` floats. If there are fewer entries in
    ``features`` than wires, the circuit does not apply the remaining squeezing gates.

    Args:
        features (array): Array of features of size (N,)
        wires (Sequence[int]): sequence of mode indices that the template acts on

    Keyword Args:
        method (str): ``'phase'`` encodes the input into the phase of single-mode squeezing, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all squeezing gates if ``execution='amplitude'``, or the
            amplitude of all squeezing gates if ``execution='phase'``
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses; got {}.".format(wires))

    if len(wires) < len(features):
        raise ValueError("Number of features to embed cannot be larger than number of wires, which is {}; "
                         "got {}.".format(len(wires), len(features)))

    for idx, f in enumerate(features):
        if method == 'amplitude':
            Squeezing(f, c, wires=wires[idx])
        elif method == 'phase':
            Squeezing(c, f, wires=wires[idx])
        else:
            raise ValueError("Execution method '{}' not known. Has to be 'phase' or 'amplitude'.".format(method))


def DisplacementEmbedding(features, wires, method='amplitude', c=0.1):
    r"""Encodes :math:`N` features into the displacement amplitudes :math:`r` or phases :math:`\phi` of :math:`M` modes,
     where :math:`N\leq M`.

    The mathematical definition of the displacement gate is given by the operator

    .. math::
            D(\alpha) = \exp(r (e^{i\phi}\ad -e^{-i\phi}\a)),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an array of at most ``len(wires)`` floats. If there are fewer entries in
    ``features`` than wires, the circuit does not apply the remaining displacement gates.

    Args:
        features (array): Array of features of size (N,)
        wires (Sequence[int]): sequence of mode indices that the template acts on

    Keyword Args:
        method (str): ``'phase'`` encodes the input into the phase of single-mode displacement, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all displacement gates if ``execution='amplitude'``, or
            the amplitude of all displacement gates if ``execution='phase'``
   """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires needs to be a list of wires that the embedding uses; got {}.".format(wires))

    if len(wires) < len(features):
        raise ValueError("Number of features to embed cannot be larger than number of wires, which is {}; "
                         "got {}.".format(len(wires), len(features)))

    for idx, f in enumerate(features):
        if method == 'amplitude':
            Displacement(f, c, wires=wires[idx])
        elif method == 'phase':
            Displacement(c, f, wires=wires[idx])
        else:
            raise ValueError("Execution method '{}' not known. Has to be 'phase' or 'amplitude'.".format(method))
