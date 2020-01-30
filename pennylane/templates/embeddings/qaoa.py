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
Contains the ``QAOAEmbedding`` template.
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.templates.decorator import template
from pennylane.ops import RX, RY, RZ, CNOT, Hadamard
from pennylane.templates.utils import (_check_shape,
                                       _check_wires,
                                       _check_is_in_options,
                                       _check_number_of_layers,
                                       _get_shape)


def qaoa_feature_encoding_hamiltonian(features, n_features, wires):
    """Implements the encoding Hamiltonian of the QAOA embedding.

    Args:
        features (array): array of features to encode
        n_features (int): number of features to encode
    """
    for idx, w in enumerate(wires):
        # Either feed in feature
        if idx < n_features:
            RX(features[idx], wires=w)
        # or a Hadamard
        else:
            Hadamard(wires=w)


def qaoa_ising_hamiltonian(weights, wires, local_fields, l):
    """Implements the Ising-like Hamiltonian of the QAOA embedding.

    Args:
        weights (array): array of weights
        wires (Sequence[int] or int): `n` qubit indices that the template acts on
        local_fields (str): gate implementing the local field
        l (int): layer index
    """
    # trainable "Ising" ansatz
    if len(wires) == 1:
        local_fields(weights[l][0], wires=wires[0])

    elif len(wires) == 2:
        # ZZ coupling
        CNOT(wires=[wires[0], wires[1]])
        RZ(2 * weights[l][0], wires=wires[0])
        CNOT(wires=[wires[0], wires[1]])

        # local fields
        for i, _ in enumerate(wires):
            local_fields(weights[l][i + 1], wires=wires[i])

    else:
        for i, _ in enumerate(wires):
            if i < len(wires) - 1:
                # ZZ coupling
                CNOT(wires=[wires[i], wires[i + 1]])
                RZ(2 * weights[l][i], wires=wires[i])
                CNOT(wires=[wires[i], wires[i + 1]])
            else:
                # ZZ coupling to enforce periodic boundary condition
                CNOT(wires=[wires[i], wires[0]])
                RZ(2 * weights[l][i], wires=wires[i])
                CNOT(wires=[wires[i], wires[0]])
        # local fields
        for i, _ in enumerate(wires):
            local_fields(weights[l][len(wires) + i], wires=wires[i])


@template
def QAOAEmbedding(features, weights, wires, local_field='Y'):
    r"""
    Encodes :math:`N` features into :math:`n>N` qubits, using a layered, trainable quantum
    circuit that is inspired by the QAOA ansatz.

    A single layer applies two circuits or "Hamiltonians": The first encodes the features, and the second is
    a variational ansatz inspired by a 1-dimensional Ising model. The feature-encoding circuit associates features with
    the angles of :class:`RX` rotations. The Ising ansatz consists of trainable two-qubit ZZ interactions
    :math:`e^{-i \alpha \sigma_z \otimes \sigma_z}`,
    and trainable local fields :math:`e^{-i \frac{\beta}{2} \sigma_{\mu}}`, where :math:`\sigma_{\mu}`
    can be chosen to be :math:`\sigma_{x}`, :math:`\sigma_{y}` or :math:`\sigma_{z}`
    (default choice is :math:`\sigma_{y}` or the ``RY`` gate), and :math:`\alpha, \beta` are adjustable gate parameters.

    The number of features has to be smaller or equal to the number of qubits. If there are fewer features than
    qubits, the feature-encoding rotation is replaced by a Hadamard gate.

    The argument ``weights`` contains an array of the :math:`\alpha, \beta` parameters for each layer.
    The number of layers :math:`L` is derived from the first dimension of ``weights``, which has the following
    shape:

    * :math:`(L, )`, if the embedding acts on a single wire,
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
        features (array): array of features to encode
        weights (array): array of weights
        wires (Sequence[int] or int): `n` qubit indices that the template acts on
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
    #############
    # Input checks

    wires = _check_wires(wires)

    expected_shape = (len(wires),)
    _check_shape(features, expected_shape, bound='max', msg="'features' must be of shape {} or smaller; got {}"
                                                            "".format((len(wires),), _get_shape(features)))

    _check_is_in_options(local_field, ['X', 'Y', 'Z'], msg="did not recognize option {} for 'local_field'"
                                                           "".format(local_field))

    repeat = _check_number_of_layers([weights])

    if len(wires) == 1:
        expected_shape = (repeat, 1)
        _check_shape(weights, expected_shape, msg="'weights' must be of shape {}; got {}"
                                                  "".format(expected_shape, _get_shape(features)))
    elif len(wires) == 2:
        expected_shape = (repeat, 3)
        _check_shape(weights, expected_shape, msg="'weights' must be of shape {}; got {}"
                                                  "".format(expected_shape, _get_shape(features)))
    else:
        expected_shape = (repeat, 2*len(wires))
        _check_shape(weights, expected_shape, msg="'weights' must be of shape {}; got {}"
                                                  "".format(expected_shape, _get_shape(features)))

    #####################

    n_features = _get_shape(features)[0]

    if local_field == 'Z':
        local_fields = RZ
    elif local_field == 'X':
        local_fields = RX
    else:
        local_fields = RY

    for l in range(repeat):
        # apply alternating Hamiltonians
        qaoa_feature_encoding_hamiltonian(features, n_features, wires)
        qaoa_ising_hamiltonian(weights, wires, local_fields, l)

    # repeat the feature encoding once more at the end
    qaoa_feature_encoding_hamiltonian(features, n_features, wires)
