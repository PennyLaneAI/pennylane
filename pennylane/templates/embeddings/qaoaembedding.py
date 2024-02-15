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


class QAOAEmbedding(Operation):
    r"""
    Encodes :math:`N` features into :math:`n>N` qubits, using a layered, trainable quantum
    circuit that is inspired by the QAOA ansatz proposed by `Killoran et al. (2020) <https://arxiv.org/abs/2001.03622>`_.

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
        local_field (str, type): type of local field used, either one of ``'X'``, ``'Y'``, or ``'Z'`` or
            :class:`~.RX`, :class:`~.RY`, or :class:`~.RZ`.

    Raises:
        ValueError: if inputs do not have the correct format

    .. details::
        :title: Usage Details

        The QAOA embedding encodes an :math:`n`-dimensional feature vector into at most :math:`n` qubits. The
        embedding applies layers of a circuit, and each layer is defined by a set of weight parameters.

        .. code-block:: python

            import pennylane as qml

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(weights, f=None):
                qml.QAOAEmbedding(features=f, weights=weights, wires=range(2))
                return qml.expval(qml.Z(0))

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

            shape = qml.QAOAEmbedding.shape(n_layers=2, n_wires=2)
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
                qml.QAOAEmbedding(features=features, weights=weights, wires=range(2))
                return qml.expval(qml.Z(0))


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
                qml.QAOAEmbedding(features=f, weights=weights, wires=range(2), local_field='Z')
                return qml.expval(qml.Z(0))

        Choosing ``'Z'`` fields implements a QAOAEmbedding where the second Hamiltonian is a
        1-dimensional Ising model.

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, features, weights, wires, local_field="Y", id=None):
        if local_field == "Z":
            local_field = qml.RZ
        elif local_field == "X":
            local_field = qml.RX
        elif local_field == "Y":
            local_field = qml.RY
        elif not (
            isinstance(local_field, type) and issubclass(local_field, (qml.RX, qml.RY, qml.RZ))
        ):
            raise ValueError(f"did not recognize local field {local_field}")

        shape = qml.math.shape(features)

        if len(shape) not in {1, 2}:
            raise ValueError(
                "Features must be a one-dimensional tensor or two-dimensional with broadcasting;"
                f" got shape {shape}."
            )

        n_features = shape[-1]
        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        shape = qml.math.shape(weights)
        if len(shape) not in {2, 3}:
            raise ValueError(
                "Weights must be a two-dimensional tensor or three-dimensional with broadcasting;"
                f" got shape {shape}."
            )

        # Either the first or the second axis (with broadcasting) gives the number of repetitions
        repeat = shape[-2]

        if len(wires) == 1:
            exp_shape = (repeat, 1)
        elif len(wires) == 2:
            exp_shape = (repeat, 3)
        else:
            exp_shape = (repeat, 2 * len(wires))
        if shape[-2:] != exp_shape:
            raise ValueError(f"Weights tensor must be of shape {exp_shape}; got {shape}")

        self._hyperparameters = {"local_field": local_field}
        super().__init__(features, weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @property
    def ndim_params(self):
        # The feature vector has one dimension, the weights tensor has two.
        return (1, 2)

    @staticmethod
    def compute_decomposition(
        features, weights, wires, local_field
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.QAOAEmbedding.decomposition`.

        Args:
            features (tensor_like): tensor of features to encode
            weights (tensor_like): tensor of weights
            wires (Any or Iterable[Any]): wires that the template acts on
            local_field (type): type of :class:`~.Operator` for local field gate

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> features = torch.tensor([1., 2.])
        >>> weights = torch.tensor([[0.1, -0.3, 1.3], [0.9, -0.2, -2.1]])
        >>> qml.QAOAEmbedding.compute_decomposition(features, weights, wires=["a", "b"], local_field=qml.RY)
        [RX(tensor(1.), wires=['a']), RX(tensor(2.), wires=['b']),
        MultiRZ(tensor(0.1000), wires=['a', 'b']), RY(tensor(-0.3000), wires=['a']), RY(tensor(1.3000), wires=['b']),
        RX(tensor(1.), wires=['a']), RX(tensor(2.), wires=['b']),
        MultiRZ(tensor(0.9000), wires=['a', 'b']), RY(tensor(-0.2000), wires=['a']), RY(tensor(-2.1000), wires=['b']),
        RX(tensor(1.), wires=['a']), RX(tensor(2.), wires=['b'])]
        """
        wires = qml.wires.Wires(wires)
        # second to last dimension of the weights tensor determines
        # the number of layers
        repeat = qml.math.shape(weights)[-2]
        op_list = []
        n_features = qml.math.shape(features)[-1]
        if qml.math.ndim(features) > 1:
            # If the features are broadcasted, move the broadcasting axis to the last place
            # in order to propagate broadcasted parameters to the gates in the decomposition.
            features = qml.math.T(features)
        if qml.math.ndim(weights) > 2:
            # If the weights are broadcasted, move the broadcasting axis to the last place
            weights = qml.math.moveaxis(weights, 0, -1)

        for l in range(repeat):
            # ---- apply encoding Hamiltonian
            for i in range(n_features):
                op_list.append(qml.RX(features[i], wires=wires[i]))
            for i in range(n_features, len(wires)):
                op_list.append(qml.Hadamard(wires=wires[i]))

            # ---- apply weight Hamiltonian
            if len(wires) == 1:
                op_list.append(local_field(weights[l][0], wires=wires))

            elif len(wires) == 2:
                # deviation for 2 wires: we do not connect last to first qubit
                # with the entangling gates
                op_list.append(qml.MultiRZ(weights[l][0], wires=wires.subset([0, 1])))
                op_list.append(local_field(weights[l][1], wires=wires[0:1]))
                op_list.append(local_field(weights[l][2], wires=wires[1:2]))

            else:
                multirz_gates = (
                    qml.MultiRZ(weights[l][i], wires.subset([i, i + 1], periodic_boundary=True))
                    for i in range(len(wires))
                )
                op_list.extend(multirz_gates)
                local_field_gates = (
                    local_field(weights[l][len(wires) + i], wires=wires[i])
                    for i in range(len(wires))
                )
                op_list.extend(local_field_gates)

        # repeat the feature encoding once more at the end
        for i in range(n_features):
            op_list.append(qml.RX(features[i], wires=wires[i]))
        for i in range(n_features, len(wires)):
            op_list.append(qml.Hadamard(wires=wires[i]))

        return op_list

    @staticmethod
    def shape(n_layers, n_wires, n_broadcast=None):
        r"""Returns the shape of the weight tensor required for this template.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of qubits

        Returns:
            tuple[int]: shape
        """
        if n_wires == 1:
            wire_dim = 1
        elif n_wires == 2:
            wire_dim = 3
        else:
            wire_dim = 2 * n_wires

        if n_broadcast:
            return n_broadcast, n_layers, wire_dim

        return n_layers, wire_dim
