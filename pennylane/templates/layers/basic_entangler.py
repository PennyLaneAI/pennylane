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

    .. UsageDetails::

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import BasicEntanglerLayers
            from math import pi

            n_wires = 3
            dev = qml.device('default.qubit', wires=n_wires)

            @qml.qnode(dev)
            def circuit(weights):
                BasicEntanglerLayers(weights=weights, wires=range(n_wires))
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        >>> circuit([[pi, pi, pi]])
        [1., 1., -1.]

        **Parameter shape**

        The shape of the weights argument can be computed by the static method
        :meth:`~.BasicEntanglerLayers.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = BasicEntanglerLayers.shape(n_layers=2, n_wires=2)
            weights = np.random.random(size=shape)

        **No periodic boundary for two wires**

        When using two wires, the convention is to drop the periodic boundary condition.
        This means that the connection from the second to the first wire is omitted.

        .. code-block:: python

            n_wires = 2
            dev = qml.device('default.qubit', wires=n_wires)

            @qml.qnode(dev)
            def circuit(weights):
                BasicEntanglerLayers(weights=weights, wires=range(n_wires))
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        >>> circuit([[pi, pi]])
        [-1, 1]


        **Changing the rotation gate**

        Any single-qubit gate can be used as a rotation gate, as long as it only takes a single parameter. The default is the ``RX`` gate.

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(weights):
                BasicEntanglerLayers(weights=weights, wires=range(n_wires), rotation=qml.RZ)
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        Accidentally using a gate that expects more parameters throws a
        ``ValueError: Wrong number of parameters``.
    """

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(self, weights, wires=None, rotation=None, do_queue=True, id=None):

        self.rotation = rotation or qml.RX

        shape = qml.math.shape(weights)
        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")
        if shape[1] != len(wires):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(wires)}; got {shape[1]}"
            )

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        weights = self.parameters[0]

        # first dimension of the weights tensor determines
        # the number of layers
        repeat = qml.math.shape(weights)[0]

        with qml.tape.QuantumTape() as tape:

            for layer in range(repeat):
                for i in range(len(self.wires)):
                    self.rotation(weights[layer][i], wires=self.wires[i : i + 1])

                if len(self.wires) == 2:
                    qml.CNOT(wires=self.wires)

                elif len(self.wires) > 2:
                    for i in range(len(self.wires)):
                        w = self.wires.subset([i, i + 1], periodic_boundary=True)
                        qml.CNOT(wires=w)

        return tape

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
