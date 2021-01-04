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
Contains the ``AmplitudeEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import warnings
import numpy as np

import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.ops import QubitStateVector
from pennylane.variable import Variable
from pennylane.wires import Wires
from pennylane.templates.utils import check_shape, get_shape, check_type

# tolerance for normalization
TOLERANCE = 1e-10


def _preprocess(features, wires, pad_with, normalize):
    """Validate and pre-process inputs as follows:

    * Check that the features tensor is one-dimensional.
    * If pad_with is None, check that the first dimension of the features tensor
      has length :math:`2^n` where :math:`n` is the number of qubits. Else check that the
      first dimension of the features tensor is not larger than :math:`2^n` and pad features with value if necessary.
    * If normalize is false, check that first dimension of features is normalised to one. Else, normalise the
      features tensor.

    Args:
        features (tensor_like): input features to pre-process
        wires (Wires): wires that template acts on
        pad_with (float): constant used to pad the features tensor to required dimension
        normalize (bool): whether or not to normalize the features vector

    Returns:
        tensor: pre-processed features
    """

    if qml.tape_mode_active():

        shape = qml.math.shape(features)

        # check shape
        if len(shape) != 1:
            raise ValueError(f"Features must be a one-dimensional tensor; got shape {shape}.")

        n_features = shape[0]
        if pad_with is None and n_features != 2 ** len(wires):
            raise ValueError(
                f"Features must be of length {2 ** len(wires)}; got length {n_features}. "
                f"Use the 'pad' argument for automated padding."
            )

        if pad_with is not None and n_features > 2 ** len(wires):
            raise ValueError(
                f"Features must be of length {2 ** len(wires)} or "
                f"smaller to be padded; got length {n_features}."
            )

        # pad
        if pad_with is not None and n_features < 2 ** len(wires):
            padding = [pad_with] * (2 ** len(wires) - n_features)
            features = qml.math.concatenate([features, padding], axis=0)

        # normalize
        norm = qml.math.sum(qml.math.abs(features) ** 2)

        if not qml.math.allclose(norm, 1.0, atol=TOLERANCE):
            if normalize or pad_with:
                features = features / np.sqrt(norm)
            else:
                raise ValueError(
                    f"Features must be a vector of length 1.0; got length {norm}."
                    "Use 'normalize=True' to automatically normalize."
                )

    # todo: delete if tape is only core
    else:
        n_amplitudes = 2 ** len(wires)
        expected_shape = (n_amplitudes,)

        if len(get_shape(features)) > 1:
            raise ValueError(
                f"Features must be a one-dimensional vector; got shape {get_shape(features)}."
            )

        if pad_with is None:
            shape = check_shape(
                features,
                expected_shape,
                msg="Features must be of length {}; got {}. Use the 'pad' "
                "argument for automated padding."
                "".format(expected_shape, get_shape(features)),
            )
        else:
            shape = check_shape(
                features,
                expected_shape,
                bound="max",
                msg="Features must be of length {} or smaller "
                "to be padded; got {}"
                "".format(expected_shape, get_shape(features)),
            )

        check_type(
            pad_with,
            [float, complex, type(None)],
            msg="'pad' must be a float or complex; got {}".format(pad_with),
        )
        check_type(normalize, [bool], msg="'normalize' must be a boolean; got {}".format(normalize))

        # pad
        n_features = shape[0]
        if pad_with is not None and n_amplitudes > n_features:
            features = np.pad(
                features, (0, n_amplitudes - n_features), mode="constant", constant_values=pad_with
            )

        # normalize
        if isinstance(features[0], Variable):
            feature_values = [s.val for s in features]
            norm = np.sum(np.abs(feature_values) ** 2)
        else:
            norm = np.sum(np.abs(features) ** 2)

        if not np.isclose(norm, 1.0, atol=TOLERANCE):
            if normalize or pad_with:
                features = features / np.sqrt(norm)
            else:
                raise ValueError(
                    "Features must be a vector of length 1.0; got length {}."
                    "Use 'normalize=True' to automatically normalize.".format(norm)
                )

        ###############

        features = np.array(features)

    return features


@template
def AmplitudeEmbedding(features, wires, pad_with=None, normalize=False, pad=None):
    r"""Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.

    By setting ``pad_with`` to a real or complex number, ``features`` is automatically padded to dimension
    :math:`2^n` where :math:`n` is the number of qubits used in the embedding.

    To represent a valid quantum state vector, the L2-norm of ``features`` must be one.
    The argument ``normalize`` can be set to ``True`` to automatically normalize the features.

    If both automatic padding and normalization are used, padding is executed *before* normalizing.

    .. note::

        On some devices, ``AmplitudeEmbedding`` must be the first operation of a quantum circuit.

    .. warning::

        At the moment, the ``features`` argument is **not differentiable** when using the template, and
        gradients with respect to the features cannot be computed by PennyLane.

    Args:
        features (tensor-like): input vector of length ``2^n``, or less if `pad_with` is specified
        wires (Iterable or :class:`.wires.Wires`): Wires that the template acts on.
            Accepts an iterable of numbers or strings, or a Wires object.
        pad_with (float or complex): if not None, the input is padded with this constant to size :math:`2^n`
        normalize (bool): whether to automatically normalize the features
        pad (float or complex): same as `pad`, to be deprecated

    Example:

        Amplitude embedding encodes a normalized :math:`2^n`-dimensional feature vector into the state
        of :math:`n` qubits:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import AmplitudeEmbedding

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(f=None):
                AmplitudeEmbedding(features=f, wires=range(2))
                return qml.expval(qml.PauliZ(0))

            circuit(f=[1/2, 1/2, 1/2, 1/2])

        The final state of the device is - up to a global phase - equivalent to the input passed to the circuit:

        >>> dev.state
        [0.5+0.j 0.5+0.j 0.5+0.j 0.5+0.j]

        **Differentiating with respect to the features**

        Due to non-trivial classical processing to construct the state preparation circuit,
        the features argument is **not always differentiable**.

        .. code-block:: python

            from pennylane import numpy as np

            @qml.qnode(dev)
            def circuit(f):
                AmplitudeEmbedding(features=f, wires=range(2))
                return qml.expval(qml.PauliZ(0))

        >>> g = qml.grad(circuit, argnum=0)
        >>> f = np.array([1, 1, 1, 1], requires_grad=True)
        >>> g(f)
        ValueError: Cannot differentiate wrt parameter(s) {0, 1, 2, 3}.

        **Normalization**

        The template will raise an error if the feature input is not normalized.
        One can set ``normalize=True`` to automatically normalize it:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(f=None):
                AmplitudeEmbedding(features=f, wires=range(2), normalize=True)
                return qml.expval(qml.PauliZ(0))

            circuit(f=[15, 15, 15, 15])

        >>> dev.state
        [0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j]

        **Padding**

        If the dimension of the feature vector is smaller than the number of amplitudes,
        one can automatically pad it with a constant for the missing dimensions using the ``pad_with`` option:

        .. code-block:: python

            from math import sqrt

            @qml.qnode(dev)
            def circuit(f=None):
                AmplitudeEmbedding(features=f, wires=range(2), pad_with=0.)
                return qml.expval(qml.PauliZ(0))

            circuit(f=[1/sqrt(2), 1/sqrt(2)])

        >>> dev.state
        [0.70710678 + 0.j, 0.70710678 + 0.j, 0.0 + 0.j, 0.0 + 0.j]

        **Operations before the embedding**

        On some devices, ``AmplitudeEmbedding`` must be the first operation in the quantum node.
        For example, ``'default.qubit'`` complains when running the following circuit:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(f=None):
                qml.Hadamard(wires=0)
                AmplitudeEmbedding(features=f, wires=range(2))
                return qml.expval(qml.PauliZ(0))


        >>> circuit(f=[1/2, 1/2, 1/2, 1/2])
        pennylane._device.DeviceError: Operation QubitStateVector cannot be used
        after other Operations have already been applied on a default.qubit device.

    """

    wires = Wires(wires)

    # pad is replaced with the more verbose pad_with
    if pad is not None:
        warnings.warn(
            "The pad argument will be replaced by the pad_with option in future versions of PennyLane.",
            PendingDeprecationWarning,
        )
        if pad_with is None:
            pad_with = pad

    features = _preprocess(features, wires, pad_with, normalize)

    QubitStateVector(features, wires=wires)
