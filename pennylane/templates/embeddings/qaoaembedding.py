# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the QAOAEmbedding template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access, consider-using-enumerate
import pennylane as qml
from pennylane.operation import Operation, AnyWires


def qaoa_feature_encoding_hamiltonian(features, wires):
    """Implements the encoding Hamiltonian of the QAOA embedding.

    Args:
        features (tensor_like): tensor of features to encode
        wires (Iterable): wires that the template acts on
    """
    n_features = qml.math.shape(features)[0]

    for i in range(n_features):
        qml.RX(features[i], wires=wires[i])
    for i in range(n_features, len(wires)):
        qml.Hadamard(wires=wires[i])


def qaoa_ising_hamiltonian(weights, wires, local_fields):
    """Implements the Ising-like Hamiltonian of the QAOA embedding.

    Args:
        weights (tensor_like): tensor of weights for one layer
        wires (Iterable): qubit indices that the template acts on
        local_fields (str): gate implementing the local field
    """

    if len(wires) == 1:
        local_fields(weights[0], wires=wires)

    elif len(wires) == 2:
        # deviation for 2 wires: we do not connect last to first qubit
        # with the entangling gates
        qml.MultiRZ(weights[0], wires=wires.subset([0, 1]))
        local_fields(weights[1], wires=wires[0:1])
        local_fields(weights[2], wires=wires[1:2])

    else:
        for i in range(len(wires)):
            qml.MultiRZ(weights[i], wires=wires.subset([i, i + 1], periodic_boundary=True))
        for i in range(len(wires)):
            local_fields(weights[len(wires) + i], wires=wires[i])


class QAOAEmbedding(Operation):
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
        features (tensor_like): tensor of features to encode
        weights (tensor_like): tensor of weights
        wires (Iterable): wires that the template acts on
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

        **Parameter shape**

        The shape of the weights argument can be computed by the static method
        :meth:`~.QAOAEmbedding.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = QAOAEmbedding.shape(n_layers=2, n_wires=2)
            weights = np.random.random(shape)

        **Training the embedding**

        The embedding is typically trained with respect to a given cost. For example, one can train it to
        minimize the PauliZ expectation of the first qubit:

        .. code-block:: python

            opt = qml.GradientDescentOptimizer()
            for i in range(10):
                weights = opt.step(lambda w : circuit(w, f=features), weights)
                print("Step ", i, " weights = ", weights)


        **Training the features**

        In principle, also the features are trainable, which means that gradients with respect to feature values
        can be computed. To train both weights and features, they need to be passed to the qnode as
        positional arguments. If the built-in optimizer is used, they have to be merged to one input:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit2(weights, features):
                QAOAEmbedding(features=features, weights=weights, wires=range(2))
                return qml.expval(qml.PauliZ(0))


            features = [1., 2.]
            weights = [[0.1, -0.3, 1.5], [3.1, 0.2, -2.8]]

            opt = qml.GradientDescentOptimizer()
            for i in range(10):
                weights, features = opt.step(circuit2, weights, features)
                print("Step ", i, "\n weights = ", weights, "\n features = ", features,"\n")

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

    num_params = 2
    num_wires = AnyWires
    par_domain = "A"

    def __init__(self, features, weights, wires, local_field="Y", do_queue=True, id=None):

        if local_field == "Z":
            self.local_field = qml.RZ
        elif local_field == "X":
            self.local_field = qml.RX
        elif local_field == "Y":
            self.local_field = qml.RY
        else:
            raise ValueError(f"did not recognize local field {local_field}")

        self._preprocess(features, weights, wires)
        super().__init__(features, weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        features = self.parameters[0]
        weights = self.parameters[1]

        # first dimension of the weights tensor determines
        # the number of layers
        repeat = qml.math.shape(weights)[0]

        with qml.tape.QuantumTape() as tape:

            for l in range(repeat):
                # apply alternating Hamiltonians
                qaoa_feature_encoding_hamiltonian(features, self.wires)
                qaoa_ising_hamiltonian(weights[l], self.wires, self.local_field)

            # repeat the feature encoding once more at the end
            qaoa_feature_encoding_hamiltonian(features, self.wires)

        return tape

    @staticmethod
    def _preprocess(features, weights, wires):
        """Validate and pre-process inputs as follows:

        * Check that the features tensor is one-dimensional.
        * Check that the first dimension of the features tensor
          has length :math:`n` or less, where :math:`n` is the number of qubits.
        * Check that the shape of the weights tensor is correct for the number of qubits.

        Args:
            features (tensor_like): feature tensor
            weights (tensor_like): weight tensor
        """

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

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns the shape of the weight tensor required for this template.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of qubits

        Returns:
            tuple[int]: shape
        """

        if n_wires == 1:
            return n_layers, 1

        if n_wires == 2:
            return n_layers, 3

        return n_layers, 2 * n_wires
