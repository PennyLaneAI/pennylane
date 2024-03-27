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
Contains the BasicEntanglerLayers template.
"""
# pylint: disable=consider-using-enumerate,too-many-arguments
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class BasicEntanglerLayers(Operation):
    r"""Layers consisting of one-parameter single-qubit rotations on each qubit, followed by a closed chain
    or *ring* of CNOT gates.

    The ring of CNOT gates connects every qubit with its neighbour,
    with the last qubit being considered as a neighbour to the first qubit.

    .. figure:: ../../_static/templates/layers/basic_entangler.png
        :align: center
        :width: 40%
        :target: javascript:void(0);

    The number of layers :math:`L` is determined by the first dimension of the argument ``weights``.
    When using a single wire, the template only applies the single
    qubit gates in each layer.

    .. note::

        This template follows the convention of dropping the entanglement between the last and the first
        qubit when using only two wires, so the entangler is not repeated on the same wires.
        In this case, only one CNOT gate is applied in each layer:

        .. figure:: ../../_static/templates/layers/basic_entangler_2wires.png
            :align: center
            :width: 30%
            :target: javascript:void(0);

    Args:
        weights (tensor_like): Weight tensor of shape ``(L, len(wires))``. Each weight is used as a parameter
            for the rotation.
        wires (Iterable): wires that the template acts on
        rotation (pennylane.ops.Operation): one-parameter single-qubit gate to use,
            if ``None``, :class:`~pennylane.ops.RX` is used as default

    Raises:
        ValueError: if inputs do not have the correct format

    .. details::
        :title: Usage Details

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml
            from math import pi

            n_wires = 3
            dev = qml.device('default.qubit', wires=n_wires)

            @qml.qnode(dev)
            def circuit(weights):
                qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
                return [qml.expval(qml.Z(i)) for i in range(n_wires)]

        >>> circuit([[pi, pi, pi]])
        [1., 1., -1.]

        **Parameter shape**

        The shape of the weights argument can be computed by the static method
        :meth:`~.BasicEntanglerLayers.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = qml.BasicEntanglerLayers.shape(n_layers=2, n_wires=2)
            weights = np.random.random(size=shape)

        **No periodic boundary for two wires**

        When using two wires, the convention is to drop the periodic boundary condition.
        This means that the connection from the second to the first wire is omitted.

        .. code-block:: python

            n_wires = 2
            dev = qml.device('default.qubit', wires=n_wires)

            @qml.qnode(dev)
            def circuit(weights):
                qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
                return [qml.expval(qml.Z(i)) for i in range(n_wires)]

        >>> circuit([[pi, pi]])
        [-1, 1]


        **Changing the rotation gate**

        Any single-qubit gate can be used as a rotation gate, as long as it only takes a single parameter. The default is the ``RX`` gate.

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(weights):
                qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires), rotation=qml.RZ)
                return [qml.expval(qml.Z(i)) for i in range(n_wires)]

        Accidentally using a gate that expects more parameters throws a
        ``ValueError: Wrong number of parameters``.
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires=None, rotation=None, id=None):
        # convert weights to numpy array if weights is list otherwise keep unchanged
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)

        shape = qml.math.shape(weights)
        if not (len(shape) == 3 or len(shape) == 2):  # 3 is when batching, 2 is no batching
            raise ValueError(
                f"Weights tensor must be 2-dimensional "
                f"or 3-dimensional if batching; got shape {shape}"
            )

        if shape[-1] != len(wires):
            # index with -1 since we may or may not have batching in first dimension
            raise ValueError(
                f"Weights tensor must have last dimension of length {len(wires)}; got {shape[-1]}"
            )

        self._hyperparameters = {"rotation": rotation or qml.RX}
        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights, wires, rotation):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.BasicEntanglerLayers.decomposition`.

        Args:
            weights (tensor_like): Weight tensor of shape ``(L, len(wires))``. Each weight is used as a parameter
                for the rotation.
            wires (Any or Iterable[Any]): wires that the operator acts on
            rotation (pennylane.ops.Operation): one-parameter single-qubit gate to use

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> weights = torch.tensor([[1.2, -0.4], [0.3, -0.2]])
        >>> qml.BasicEntanglerLayers.compute_decomposition(weights, wires=["a", "b"], rotation=qml.RX)
        [RX(tensor(1.2000), wires=['a']), RX(tensor(-0.4000), wires=['b']),
        CNOT(wires=['a', 'b']),
        RX(tensor(0.3000), wires=['a']), RX(tensor(-0.2000), wires=['b']),
        CNOT(wires=['a', 'b'])]
        """
        # first dimension of the weights tensor (second when batching) determines
        # the number of layers
        repeat = qml.math.shape(weights)[-2]

        op_list = []
        for layer in range(repeat):
            for i in range(len(wires)):
                op_list.append(rotation(weights[..., layer, i], wires=wires[i : i + 1]))

            if len(wires) == 2:
                op_list.append(qml.CNOT(wires=wires))

            elif len(wires) > 2:
                for i in range(len(wires)):
                    w = wires.subset([i, i + 1], periodic_boundary=True)
                    op_list.append(qml.CNOT(wires=w))

        return op_list

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns the shape of the weight tensor required for this template.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of qubits

        Returns:
            tuple[int]: shape
        """

        return n_layers, n_wires
