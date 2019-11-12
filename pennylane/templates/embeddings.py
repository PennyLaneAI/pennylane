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
Embeddings are templates that take features and encode them into a quantum state.
They can optionally be repeated, and may contain trainable parameters. Embeddings are typically
used at the beginning of a circuit.
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections.abc import Iterable
from pennylane.ops import RX, RY, RZ, BasisState, Squeezing, Displacement, QubitStateVector
from pennylane.variable import Variable
import numpy as np


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
        features (array): Input array of shape ``(N,)``, where N is the number of features
            to embed. ``N`` must be smaller or equal to the total number of wires.
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Keyword Args:
        rotation (str): Type of rotations used

    Raises:
        ValueError: if ``features`` or ``wires`` is invalid
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires must be passed as a list of integers; got {}.".format(wires))

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


def AmplitudeEmbedding(features, wires, pad=False, normalize=False):
    r"""Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.

    If the total number of features to embed are less than the :math:`2^n` available amplitudes,
    non-informative constants (zeros) can be padded to ``features``. To enable this, the argument
    ``pad`` should be set to ``True``.

    The L2-norm of ``features`` must be one. By default, AmplitudeEmbedding expects a normalized
    feature vector. The argument ``normalize`` can be set to ``True`` to automatically normalize it.

    .. note::

        AmplitudeEmbedding uses PennyLane's :class:`~pennylane.ops.QubitStateVector` and only works in conjunction with
        devices that implement this function.

    Args:
        features (array): input array of shape ``(2**n,)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on
        pad (Boolean): controls the activation of the padding option
        normalize (Boolean): controls the activation of automatic normalization

    Raises:
        ValueError: if ``features`` or ``wires`` is invalid
    """

    if isinstance(wires, int):
        wires = [wires]

    features = np.array(features)

    n_features = len(features)
    n_amplitudes = 2**len(wires)

    if n_amplitudes < n_features:
        raise ValueError("AmplitudeEmbedding requires the size of feature vector to be "
                         "smaller than or equal to 2**len(wires), which is {}; "
                         "got {}.".format(n_amplitudes, n_features))

    if pad and n_amplitudes >= n_features:
        features = np.pad(features, (0, n_amplitudes-n_features), 'constant')

    if not pad and n_amplitudes != n_features:
        raise ValueError("AmplitudeEmbedding must get a feature vector of size 2**len(wires), "
                         "which is {}; got {}. Use ``pad=True`` to automatically pad the "
                         "features with zeros.".format(n_amplitudes, n_features))

    # Get normalization
    norm = 0
    for f in features:
        if isinstance(f, Variable):
            norm += np.conj(f.val)*f.val
        else:
            norm += np.conj(f)*f

    if not np.isclose(norm, 1):
        if normalize:
            features = features/np.sqrt(norm)
        else:
            raise ValueError("AmplitudeEmbedding requires a normalized feature vector. "
                             "Set ``normalize=True`` to automatically normalize it.")

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

    Raises:
        ValueError: if ``features`` or ``wires`` is invalid
    """
    if not isinstance(wires, Iterable):
        raise ValueError("Wires must be passed as a list of integers; got {}.".format(wires))

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

    Raises:
        ValueError: if ``features`` or ``wires`` is invalid
    """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires must be passed as a list of integers; got {}.".format(wires))

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

    Raises:
        ValueError: if ``features`` or ``wires`` is invalid or if ``method`` is unknown
   """

    if not isinstance(wires, Iterable):
        raise ValueError("Wires must be passed as a list of integers; got {}.".format(wires))

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


embeddings = {"AngleEmbedding", "AmplitudeEmbedding", "BasisEmbedding", "SqueezingEmbedding", "DisplacementEmbedding"}

__all__ = list(embeddings)
