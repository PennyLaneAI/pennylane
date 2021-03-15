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
Contains the ``BasicEntanglerLayers`` template.
"""
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.templates import broadcast
from pennylane.wires import Wires


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
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
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

        **Parameter initialization function**

        A random numpy weights array can be generated using the static methods
        `BasicEntanglerLayers.weights_normal` and `BasicEntanglerLayers.weights_uniform`.

        .. code-block:: python

            weights = BasicEntanglerLayers.weights_normal(n_layers=2, n_wires=2, mean=0, std=0.2)

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

    def __init__(self, weights, wires=None, rotation=None, do_queue=True):

        self.rotation = rotation or qml.RX

        super().__init__(weights, wires=wires, do_queue=do_queue)
        self._preprocess()

    def expand(self):

        weights = self.data[0]

        # first dimension of the weights tensor determines
        # the number of layers
        repeat = qml.math.shape(weights)[0]

        with qml.tape.QuantumTape() as tape:

            for layer in range(repeat):
                for i in range(len(self.wires)):
                    self.rotation(weights[layer, i], wires=self.wires[i:i+1])

                broadcast(unitary=qml.CNOT, pattern="ring", wires=self.wires)

        return tape

    def _preprocess(self):
        """Validate and pre-process inputs as follows:

        * Check the shape of the weights tensor, making sure that the second dimension
          has length :math:`n`, where :math:`n` is the number of qubits.

        Args:
            weights (tensor_like): trainable parameters of the template
            wires (Wires): wires that template acts on
        """
        shape = qml.math.shape(self.parameters[0])

        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")

        if shape[1] != len(self.wires):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(self.wires)}; got {shape[1]}"
            )

    @staticmethod
    def weights_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
        r"""Creates a standard numpy weights array whose entries are drawn from a normal
        distribution.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of qubits
            mean (float): mean of parameters
            std (float): standard deviation of parameters
            seed (int): seed used in sampling the parameters, makes function call deterministic

        Returns:
            array: weights array
        """
        if seed is not None:
            np.random.seed(seed)

        params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))

        return params

    @staticmethod
    def weights_uniform(n_layers, n_wires, low=0, high=2 * np.pi, seed=None):
        r"""Creates a standard numpy weights array whose entries are drawn from a uniform
        distribution.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of qubits
            low (float): minimum value of uniform distribution
            high (float): maximum value of uniform distribution
            seed (int): seed used in sampling the parameters, makes function call deterministic

        Returns:
            array: weights array
        """
        if seed is not None:
            np.random.seed(seed)

        params = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))

        return params
