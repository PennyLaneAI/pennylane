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
import math
import numpy as np
from pennylane.templates.decorator import template
from pennylane.ops import QubitStateVector
from pennylane.templates.utils import (
    check_shape,
    check_no_variable,
    check_wires,
    check_type,
    get_shape,
)
from pennylane.variable import Variable

# tolerance for normalization
TOLERANCE = 1e-10


@template
def AmplitudeEmbedding(features, wires, pad=None, normalize=False):
    r"""Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.

    By setting ``pad`` to a real or complex number, ``features`` is automatically padded to dimension
    :math:`2^n` where :math:`n` is the number of qubits used in the embedding.

    To represent a valid quantum state vector, the L2-norm of ``features`` must be one.
    The argument ``normalize`` can be set to ``True`` to automatically normalize the features.

    If both automatic padding and normalization are used, padding is executed *before* normalizing.

    .. note::

        On some devices, ``AmplitudeEmbedding`` must be the first operation of a quantum node.


    .. warning::

        ``AmplitudeEmbedding`` calls a circuit that involves non-trivial classical processing of the
        features. The ``features`` argument is therefore **not differentiable** when using the template, and
        gradients with respect to the features cannot be computed by PennyLane.

    Args:
        features (array): input array of shape ``(2^n,)``
        wires (Sequence[int] or int): :math:`n` qubit indices that the template acts on
        pad (float or complex): if not None, the input is padded with this constant to size :math:`2^n`
        normalize (Boolean): controls the activation of automatic normalization

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

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

        Checking the final state of the device, we find that it is equivalent to the input passed to the circuit:

        >>> dev._state
        [0.5+0.j 0.5+0.j 0.5+0.j 0.5+0.j]

        **Passing features as positional arguments to a quantum node**

        The ``features`` argument of ``AmplitudeEmbedding`` can in principle also be passed to the quantum node
        as a positional argument:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(f):
                AmplitudeEmbedding(features=f, wires=range(2))
                return qml.expval(qml.PauliZ(0))

        However, due to non-trivial classical processing to construct the state preparation circuit,
        the features argument is **not differentiable**.

        >>> g = qml.grad(circuit, argnum=0)
        >>> g([1,1,1,1])
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

        The re-normalized feature vector is encoded into the quantum state vector:

        >>> dev._state
        [0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j]

        **Padding**

        If the dimension of the feature vector is smaller than the number of amplitudes,
        one can automatically pad it with a constant for the missing dimensions using the ``pad`` option:

        .. code-block:: python

            from math import sqrt

            @qml.qnode(dev)
            def circuit(f=None):
                AmplitudeEmbedding(features=f, wires=range(2), pad=0.)
                return qml.expval(qml.PauliZ(0))

            circuit(f=[1/sqrt(2), 1/sqrt(2)])

        >>> dev._state
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

    #############
    # Input checks

    check_no_variable(pad, msg="'pad' cannot be differentiable")
    check_no_variable(normalize, msg="'normalize' cannot be differentiable")

    wires = check_wires(wires)

    n_amplitudes = 2 ** len(wires)
    expected_shape = (n_amplitudes,)
    if pad is None:
        shape = check_shape(
            features,
            expected_shape,
            msg="'features' must be of shape {}; got {}. Use the 'pad' "
            "argument for automated padding."
            "".format(expected_shape, get_shape(features)),
        )
    else:
        shape = check_shape(
            features,
            expected_shape,
            bound="max",
            msg="'features' must be of shape {} or smaller "
            "to be padded; got {}"
            "".format(expected_shape, get_shape(features)),
        )

    check_type(
        pad,
        [float, complex, type(None)],
        msg="'pad' must be a float or complex; got {}".format(pad),
    )
    check_type(normalize, [bool], msg="'normalize' must be a boolean; got {}".format(normalize))

    ###############

    #############
    # Preprocessing

    # pad
    n_features = shape[0]
    if pad is not None and n_amplitudes > n_features:
        features = np.pad(
            features, (0, n_amplitudes - n_features), mode="constant", constant_values=pad
        )

    # normalize
    if isinstance(features[0], Variable):
        feature_values = [s.val for s in features]
        norm = np.sum(np.abs(feature_values) ** 2)
    else:
        norm = np.sum(np.abs(features) ** 2)

    if not np.isclose(norm, 1.0, atol=TOLERANCE):
        if normalize or pad:
            features = features / math.sqrt(norm)
        else:
            raise ValueError(
                "'features' must be a vector of length 1.0; got length {}."
                "Use 'normalization=True' to automatically normalize.".format(norm)
            )

    ###############

    features = np.array(features)
    QubitStateVector(features, wires=wires)
