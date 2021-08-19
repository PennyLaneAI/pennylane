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
Contains the RandomLayers template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class RandomLayers(Operation):
    r"""Layers of randomly chosen single qubit rotations and 2-qubit entangling gates, acting
    on randomly chosen qubits.

    .. warning::
        This template uses random number generation inside qnodes. Find more
        details about how to invoke the desired random behaviour in the "Usage Details" section below.

    The argument ``weights`` contains the weights for each layer. The number of layers :math:`L` is therefore derived
    from the first dimension of ``weights``.

    The two-qubit gates of type ``imprimitive`` and the rotations are distributed randomly in the circuit.
    The number of random rotations is derived from the second dimension of ``weights``. The number of
    two-qubit gates is determined by ``ratio_imprim``. For example, a ratio of ``0.3`` with ``30`` rotations
    will lead to the use of ``10`` two-qubit gates.

    .. note::
        If applied to one qubit only, this template will use no imprimitive gates.

    This is an example of two 4-qubit random layers with four Pauli-Y/Pauli-Z rotations :math:`R_y, R_z`,
    controlled-Z gates as imprimitives, as well as ``ratio_imprim=0.3``:

    .. figure:: ../../_static/layer_rnd.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    Args:
        weights (tensor_like): weight tensor of shape ``(L, k)``,
        wires (Iterable): wires that the template acts on
        ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation gates
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
        rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture, defaults to 42

    .. UsageDetails::

        **Default seed**

        ``RandomLayers`` always uses a seed to initialize the construction of a random circuit. This means
        that the template creates the same circuit every time it is called. If no seed is provided, the default
        seed of ``42`` is used.

        .. code-block:: python

            import pennylane as qml
            import numpy as np
            from pennylane.templates.layers import RandomLayers

            dev = qml.device("default.qubit", wires=2)
            weights = [[0.1, -2.1, 1.4]]

            @qml.qnode(dev)
            def circuit1(weights):
                RandomLayers(weights=weights, wires=range(2))
                return qml.expval(qml.PauliZ(0))

            @qml.qnode(dev)
            def circuit2(weights):
                RandomLayers(weights=weights, wires=range(2))
                return qml.expval(qml.PauliZ(0))

        >>> np.allclose(circuit1(weights), circuit2(weights))
        True

        You can verify this by drawing the circuits.

            >>> print(circuit1.draw())
            0: ─────────────────────╭X──╭X──RZ(1.4)──┤ ⟨Z⟩
            1: ──RX(0.1)──RX(-2.1)──╰C──╰C───────────┤

            >>> print(circuit2.draw())
            0: ─────────────────────╭X──╭X──RZ(1.4)──┤ ⟨Z⟩
            1: ──RX(0.1)──RX(-2.1)──╰C──╰C───────────┤


        **Changing the seed**

        To change the randomly generated circuit architecture, you have to change the seed passed to the template.
        For example, these two calls of ``RandomLayers`` *do not* create the same circuit:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit_9(weights):
                RandomLayers(weights=weights, wires=range(2), seed=9)
                return qml.expval(qml.PauliZ(0))

            @qml.qnode(dev)
            def circuit_12(weights):
                RandomLayers(weights=weights, wires=range(2), seed=12)
                return qml.expval(qml.PauliZ(0))

        >>> np.allclose(circuit_9(weights), circuit_12(weights))
        >>> False

        >>> print(circuit_9.draw())
        0: ──╭X──RX(0.1)────────────┤ ⟨Z⟩
        1: ──╰C──RY(-2.1)──RX(1.4)──┤

        >>> print(circuit_12.draw())
        0: ──╭X──RZ(0.1)───╭C──╭X───────────┤ ⟨Z⟩
        1: ──╰C──RX(-2.1)──╰X──╰C──RZ(1.4)──┤


        **Automatic creation of random circuits**

        To automate the process of creating different circuits with ``RandomLayers``,
        you can set ``seed=None`` to avoid specifying a seed. However, in this case care needs
        to be taken. In the default setting, a quantum node is **mutable**, which means that the quantum function is
        re-evaluated every time it is called. This means that the circuit is re-constructed from scratch
        each time you call the qnode:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit_rnd(weights):
                RandomLayers(weights=weights, wires=range(2), seed=None)
                return qml.expval(qml.PauliZ(0))

            first_call = circuit_rnd(weights)
            second_call = circuit_rnd(weights)

        >>> np.allclose(first_call, second_call)
        False

        This can be rectified by making the quantum node **immutable**.

        .. code-block:: python

            @qml.qnode(dev, mutable=False)
            def circuit_rnd(weights):
                RandomLayers(weights=weights, wires=range(2), seed=None)
                return qml.expval(qml.PauliZ(0))

            first_call = circuit_rnd(weights)
            second_call = circuit_rnd(weights)

        >>> np.allclose(first_call, second_call)
        True

        **Parameter shape**

        The expected shape for the weight tensor can be computed with the static method
        :meth:`~.RandomLayers.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = RandomLayers.shape(n_layers=2, n_rotations=3)
            weights = np.random.random(size=shape)
    """

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(
        self,
        weights,
        wires,
        ratio_imprim=0.3,
        imprimitive=None,
        rotations=None,
        seed=42,
        do_queue=True,
        id=None,
    ):

        self.seed = seed
        self.rotations = rotations or [qml.RX, qml.RY, qml.RZ]

        shape = qml.math.shape(weights)
        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")

        self.n_layers = shape[0]
        self.imprimitive = imprimitive or qml.CNOT
        self.ratio_imprimitive = ratio_imprim

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        shape = qml.math.shape(self.parameters[0])

        with qml.tape.QuantumTape() as tape:

            for l in range(self.n_layers):

                i = 0
                while i < shape[1]:
                    if np.random.random() > self.ratio_imprimitive:
                        # apply a random rotation gate to a random wire
                        gate = np.random.choice(self.rotations)
                        rnd_wire = self.wires.select_random(1)
                        gate(self.parameters[0][l, i], wires=rnd_wire)
                        i += 1

                    else:
                        # apply the entangler to two random wires
                        if len(self.wires) > 1:
                            rnd_wires = self.wires.select_random(2)
                            self.imprimitive(wires=rnd_wires)
        return tape

    @staticmethod
    def shape(n_layers, n_rotations):
        r"""Returns the expected shape of the weights tensor.

        Args:
            n_layers (int): number of layers
            n_rotations (int): number of rotations

        Returns:
            tuple[int]: shape
        """

        return n_layers, n_rotations
