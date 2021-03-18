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
Contains the ``QAOAEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.ops import RX, RY, RZ, MultiRZ, Hadamard
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    check_shape,
    check_number_of_layers,
    get_shape,
)
from pennylane.wires import Wires


def _preprocess(features, wires, weights):
    """Validate and pre-process inputs as follows:

    * Check that the features tensor is one-dimensional.
    * Check that the first dimension of the features tensor
      has length :math:`n` or less, where :math:`n` is the number of qubits.
    * Check that the shape of the weights tensor is correct for the number of qubits.

    Args:
        features (tensor_like): input features to pre-process
        wires (Wires): wires that template acts on
        weights (tensor_like): weights of the embedding

    Returns:
        int: number of times that embedding is repeated
    """

    if qml.tape_mode_active():

        shape = qml.math.shape(features)

        if len(shape) != 1:
            raise ValueError(f"Features must be a one-dimensional tensor; got shape {shape}.")

        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        shape = qml.math.shape(weights)
        repeat = shape[0]

        if len(wires) == 1:
            if shape != (repeat, 1):
                raise ValueError(f"Weights tensor must be of shape {(repeat, 1)}; got {shape}")

        elif len(wires) == 2:
            if shape != (repeat, 3):
                raise ValueError(f"Weights tensor must be of shape {(repeat, 3)}; got {shape}")
        else:
            if shape != (repeat, 2 * len(wires)):
                raise ValueError(
                    f"Weights tensor must be of shape {(repeat, 2*len(wires))}; got {shape}"
                )

    else:

        expected_shape = (len(wires),)
        check_shape(
            features,
            expected_shape,
            bound="max",
            msg="Features must be of shape {} or smaller; got {}"
            "".format((len(wires),), get_shape(features)),
        )

        repeat = check_number_of_layers([weights])

        if len(wires) == 1:
            expected_shape = (repeat, 1)
            check_shape(
                weights,
                expected_shape,
                msg="Weights tensor must be of shape {}; got {}"
                "".format(expected_shape, get_shape(features)),
            )
        elif len(wires) == 2:
            expected_shape = (repeat, 3)
            check_shape(
                weights,
                expected_shape,
                msg="Weights tensor must be of shape {}; got {}"
                "".format(expected_shape, get_shape(features)),
            )
        else:
            expected_shape = (repeat, 2 * len(wires))
            check_shape(
                weights,
                expected_shape,
                msg="Weights tensor must be of shape {}; got {}"
                "".format(expected_shape, get_shape(features)),
            )

    return repeat


def qaoa_feature_encoding_hamiltonian(features, wires):
    """Implements the encoding Hamiltonian of the QAOA embedding.

    Args:
        features (tensor_like): array of features to encode
        wires (Wires): wires that the template acts on
    """

    try:
        # works for tensors
        n_features = features.shape[0]
    except AttributeError:
        # works for lists and tuples
        n_features = len(features)

    feature_encoding_wires = wires[:n_features]
    remaining_wires = wires[n_features:]

    broadcast(unitary=RX, pattern="single", wires=feature_encoding_wires, parameters=features)
    broadcast(unitary=Hadamard, pattern="single", wires=remaining_wires)


def qaoa_ising_hamiltonian(weights, wires, local_fields):
    """Implements the Ising-like Hamiltonian of the QAOA embedding.

    Args:
        weights (tensor_like): array of weights for one layer
        wires (Wires): qubit indices that the template acts on
        local_fields (str): gate implementing the local field
    """

    if len(wires) == 1:
        weights_zz = []
        weights_fields = weights

    elif len(wires) == 2:
        # for 2 wires the periodic boundary condition is dropped in broadcast's "ring" pattern
        # only feed in 1 parameter
        weights_zz = weights[:1]
        weights_fields = weights[1:]

    else:
        weights_zz = weights[: len(wires)]
        weights_fields = weights[len(wires) :]

    # zz couplings
    broadcast(unitary=MultiRZ, pattern="ring", wires=wires, parameters=weights_zz)
    # local fields
    broadcast(unitary=local_fields, pattern="single", wires=wires, parameters=weights_fields)


@template
def QAOAEmbedding(features, weights, wires, local_field="Y"):
    r"""
    Encodes :math:`N` features into :math:`n>N` qubits, using a layered, trainable quantum
    circuit that is inspired by the QAOA ansatz.

    A single layer applies two circuits or "Hamiltonians": The first encodes the features, and the second is
    a variational ansatz inspired by a 1-dimensional Ising model. The feature-encoding circuit associates features with
    the angles of :class:`RX` rotations. The Ising ansatz consists of trainable two-qubit ZZ interactions
    :math:`e^{-i \frac{\alpha}{2} \sigma_z \otimes \sigma_z}` (in PennyLane represented by the :class:`~.MultiRZ` gate),
    and trainable local fields :math:`e^{-i \frac{\beta}{2} \sigma_{\mu}}`, where :math:`\sigma_{\mu}`
    can be chosen to be :math:`\sigma_{x}`, :math:`\sigma_{y}` or :math:`\sigma_{z}`
    (default choice is :math:`\sigma_{y}` or the ``RY`` gate), and :math:`\alpha, \beta` are adjustable gate parameters.

    The number of features has to be smaller or equal to the number of qubits. If there are fewer features than
    qubits, the feature-encoding rotation is replaced by a Hadamard gate.

    The argument ``weights`` contains an array of the :math:`\alpha, \beta` parameters for each layer.
    The number of layers :math:`L` is derived from the first dimension of ``weights``, which has the following
    shape:

    * :math:`(L, 1)`, if the embedding acts on a single wire,
    * :math:`(L, 3)`, if the embedding acts on two wires,
    * :math:`(L, 2n)` else.

    After the :math:`L` th layer, another set of feature-encoding :class:`RX` gates is applied.

    This is an example for the full embedding circuit using 2 layers, 3 features, 4 wires, and ``RY`` local fields:

    |

    .. figure:: ../../_static/qaoa_layers.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    .. note::
        ``QAOAEmbedding`` supports gradient computations with respect to both the ``features`` and the ``weights``
        arguments. Note that trainable parameters need to be passed to the quantum node as positional arguments.

    Args:
        features (tensor_like): array of features to encode
        weights (tensor_like): array of weights
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        local_field (str): type of local field used, one of ``'X'``, ``'Y'``, or ``'Z'``

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        The QAOA embedding encodes an :math:`n`-dimensional feature vector into at most :math:`n` qubits. The
        embedding applies layers of a circuit, and each layer is defined by a set of weight parameters.

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import QAOAEmbedding

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(weights, f=None):
                QAOAEmbedding(features=f, weights=weights, wires=range(2))
                return qml.expval(qml.PauliZ(0))

            features = [1., 2.]
            layer1 = [0.1, -0.3, 1.5]
            layer2 = [3.1, 0.2, -2.8]
            weights = [layer1, layer2]

            print(circuit(weights, f=features))

        **Using parameter initialization functions**

        The initial weight parameters can alternatively be generated by utility functions from the
        ``pennylane.init`` module, for example using the function :func:`~.qaoa_embedding_normal`:

        .. code-block:: python

            from pennylane.init import qaoa_embedding_normal
            weights = qaoa_embedding_normal(n_layers=2, n_wires=2, mean=0, std=0.2)


        **Training the embedding**

        The embedding is typically trained with respect to a given cost. For example, one can train it to
        minimize the PauliZ expectation of the first qubit:

        .. code-block:: python

            o = GradientDescentOptimizer()
            for i in range(10):
                weights = o.step(lambda w : circuit(w, f=features), weights)
                print("Step ", i, " weights = ", weights)


        **Training the features**

        In principle, also the features are trainable, which means that gradients with respect to feature values
        can be computed. To train both weights and features, they need to be passed to the qnode as
        positional arguments. If the built-in optimizer is used, they have to be merged to one input:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit2(pars):
                weights = pars[0]
                features = pars[1]
                QAOAEmbedding(features=features, weights=weights, wires=range(2))
                return qml.expval(qml.PauliZ(0))


            features = [1., 2.]
            weights = [[0.1, -0.3, 1.5], [3.1, 0.2, -2.8]]
            pars = [weights, features]

            o = GradientDescentOptimizer()
            for i in range(10):
                pars = o.step(circuit2, pars)
                print("Step ", i, " weights = ", pars[0], " features = ", pars[1])

        **Local Fields**

        While by default, ``RY`` gates are used as local fields, one may also choose ``local_field='Z'`` or
        ``local_field='X'`` as hyperparameters of the embedding.

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(weights, f=None):
                QAOAEmbedding(features=f, weights=weights, wires=range(2), local_field='Z')
                return qml.expval(qml.PauliZ(0))

        Choosing ``'Z'`` fields implements a QAOAEmbedding where the second Hamiltonian is a
        1-dimensional Ising model.

    """
    wires = Wires(wires)
    repeat = _preprocess(features, wires, weights)

    if local_field == "Z":
        local_fields = RZ
    elif local_field == "X":
        local_fields = RX
    elif local_field == "Y":
        local_fields = RY
    else:
        raise ValueError(f"did not recognize local field {local_field}")

    for l in range(repeat):
        # apply alternating Hamiltonians
        qaoa_feature_encoding_hamiltonian(features, wires)
        qaoa_ising_hamiltonian(weights[l], wires, local_fields)

    # repeat the feature encoding once more at the end
    qaoa_feature_encoding_hamiltonian(features, wires)
