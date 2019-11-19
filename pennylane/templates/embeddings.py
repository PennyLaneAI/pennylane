# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
import numpy as np

from pennylane.ops import RX, RY, RZ, BasisState, Squeezing, Displacement, QubitStateVector
from pennylane.templates.utils import (_check_shape, _check_no_variable, _check_wires,
                                       _check_hyperp_is_in_options, _check_type)
from pennylane.variable import Variable


TOLERANCE = 1e-3


def AmplitudeEmbedding(features, wires, pad=None, normalize=False):
    r"""Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.

    If the total number of features to embed is less than the :math:`2^n` available amplitudes,
    non-informative constants (zeros) can be padded to ``features``. To enable this, the argument
    ``pad`` should be set to ``True``.

    The L2-norm of ``features`` must be one. By default, ``AmplitudeEmbedding`` expects a normalized
    feature vector. The argument ``normalize`` can be set to ``True`` to automatically normalize it.

    Args:
        features (array): input array of shape ``(2^n,)``
        wires (Sequence[int] or int): qubit indices that the template acts on
        pad (float or complex): if not None, the input is padded with this constant to size :math:`2^n`
        normalize (Boolean): controls the activation of automatic normalization

    Raises:
        ValueError: if inputs do not have the correct format
    """

    #############
    # Input checks
    _check_no_variable([pad, normalize], ['pad', 'normalize'])

    wires, n_wires = _check_wires(wires)

    n_ampl = 2**n_wires
    if pad is None:
        msg = "AmplitudeEmbedding must get a feature vector of size 2**len(wires), which is {}. Use 'pad' " \
               "argument for automated padding.".format(n_ampl)
        shp = _check_shape(features, (n_ampl,), msg=msg)
    else:
        msg = "AmplitudeEmbedding must get a feature vector of at least size 2**len(wires) = {}.".format(n_ampl)
        shp = _check_shape(features, (n_ampl,), msg=msg, bound='max')

    _check_type(pad, [float, complex, type(None)])
    _check_type(normalize, [bool])
    ###############

    # Pad
    n_feats = shp[0]
    if pad is not None and n_ampl > n_feats:
        features = np.pad(features, (0, n_ampl-n_feats), mode='constant', constant_values=pad)

    # Normalize
    if isinstance(features[0], Variable):
        feature_values = [s.val for s in features]
        norm = np.sum(np.abs(feature_values)**2)
    else:
        norm = np.sum(np.abs(features)**2)

    if not np.isclose(norm, 1.0, atol=TOLERANCE, rtol=0):
        if normalize or pad:
            features = features/np.sqrt(norm)
        else:
            raise ValueError("Vector of features has to be normalized to 1.0, got {}."
                             "Use 'normalization=True' to automatically normalize.".format(norm))

    QubitStateVector(features, wires=wires)


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

    Args:
        features (array): input array of shape ``(N,)``, where N is the number of input features to embed,
            with :math:`N\leq n`
        wires (Sequence[int] or int): qubit indices that the template acts on
        rotation (str): Type of rotations used

    Raises:
        ValueError: if inputs do not have the correct format
    """

    #############
    # Input checks
    _check_no_variable([rotation], ['rotation'])
    wires, n_wires = _check_wires(wires)

    msg = "AngleEmbedding cannot process more features than number of qubits {};" \
          "got {}.".format(n_wires, len(features))
    _check_shape(features, (n_wires,), bound='max', msg=msg)
    _check_type(rotation, [str])

    msg = "Rotation strategy {} not recognized.".format(rotation)
    _check_hyperp_is_in_options(rotation, ['X', 'Y', 'Z'], msg=msg)
    ###############

    if rotation == 'X':
        for f, w in zip(features, wires):
            RX(f, wires=w)
    elif rotation == 'Y':
        for f, w in zip(features, wires):
            RY(f, wires=w)
    elif rotation == 'Z':
        for f, w in zip(features, wires):
            RZ(f, wires=w)


def BasisEmbedding(features, wires):
    r"""Encodes :math:`n` binary features into a basis state of :math:`n` qubits.

    For example, for ``features=np.array([0, 1, 0])``, the quantum system will be
    prepared in state :math:`|010 \rangle`.

    .. note::

        BasisEmbedding uses PennyLane's :class:`~pennylane.ops.BasisState` and only works in conjunction with
        devices that implement this function.

    Args:
        features (array): binary input array of shape ``(n, )``
        wires (Sequence[int] or int): qubit indices that the template acts on

    Raises:
        ValueError: if inputs do not have the correct format
    """

    #############
    # Input checks
    wires, n_wires = _check_wires(wires)
    _check_shape(features, (n_wires,))

    # basis_state cannot be trainable
    msg = "The input features in BasisEmbedding influence the circuit architecture and can " \
          "therefore not be passed as a positional argument to the quantum node."
    _check_no_variable([features], ['features'], msg=msg)

    # basis_state is guaranteed to be a list
    if any([b not in [0, 1] for b in features]):
        raise ValueError("Basis state must only consist of 0s and 1s, got {}".format(features))
    ###############

    features = np.array(features)

    BasisState(features, wires=wires)


def SqueezingEmbedding(features, wires, method='amplitude', c=0.1):
    r"""Encodes :math:`N` features into the squeezing amplitudes :math:`r \geq 0` or phases :math:`\phi \in [0, 2\pi)`
    of :math:`M` modes, where :math:`N\leq M`.

    The mathematical definition of the squeezing gate is given by the operator

    .. math::

        S(z) = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an iterable of at most ``len(wires)`` floats. If there are fewer entries in
    ``features`` than wires, the circuit does not apply the remaining squeezing gates.

    Args:
        features (array): Array of features of size (N,)
        wires (Sequence[int]): sequence of mode indices that the template acts on
        method (str): ``'phase'`` encodes the input into the phase of single-mode squeezing, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all squeezing gates if ``execution='amplitude'``, or the
            amplitude of all squeezing gates if ``execution='phase'``

    Raises:
        ValueError: if inputs do not have the correct format
    """


    #############
    # Input checks
    _check_no_variable([method, c], ['method', 'c'])

    wires, n_wires = _check_wires(wires)

    msg = "SqueezingEmbedding cannot process more features than number of wires {};" \
          "got {}.".format(n_wires, len(features))
    _check_shape(features, (n_wires,), bound='max', msg=msg)

    msg = "Did not recognise parameter encoding method {}.".format(method)
    _check_hyperp_is_in_options(method, ['amplitude', 'phase'], msg=msg)
    #############

    for idx, f in enumerate(features):
        if method == 'amplitude':
            Squeezing(f, c, wires=wires[idx])
        elif method == 'phase':
            Squeezing(c, f, wires=wires[idx])


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
        method (str): ``'phase'`` encodes the input into the phase of single-mode displacement, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all displacement gates if ``execution='amplitude'``, or
            the amplitude of all displacement gates if ``execution='phase'``

    Raises:
        ValueError: if inputs do not have the correct format
   """

    #############
    # Input checks
    _check_no_variable([method, c], ['method', 'c'])

    wires, n_wires = _check_wires(wires)

    msg = "DisplacementEmbedding cannot process more features than number of wires {};" \
          "got {}.".format(n_wires, len(features))
    _check_shape(features, (n_wires,), bound='max', msg=msg)

    msg = "Did not recognise parameter encoding method {}.".format(method)
    _check_hyperp_is_in_options(method, ['amplitude', 'phase'], msg=msg)
    #############

    for idx, f in enumerate(features):
        if method == 'amplitude':
            Displacement(f, c, wires=wires[idx])
        elif method == 'phase':
            Displacement(c, f, wires=wires[idx])


embeddings = {"AngleEmbedding", "AmplitudeEmbedding", "BasisEmbedding", "SqueezingEmbedding", "DisplacementEmbedding"}

__all__ = list(embeddings)
