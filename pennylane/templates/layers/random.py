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
        rotations (tuple[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture, defaults to 42

    .. details::
        :title: Usage Details

        **Default seed**

        ``RandomLayers`` always uses a seed to initialize the construction of a random circuit. This means
        that the template creates the same circuit every time it is called. If no seed is provided, the default
        seed of ``42`` is used.

        .. code-block:: python

            import pennylane as qml
            from pennylane import numpy as np

            dev = qml.device("default.qubit", wires=2)
            weights = np.array([[0.1, -2.1, 1.4]])

            @qml.qnode(dev)
            def circuit1(weights):
                qml.RandomLayers(weights=weights, wires=range(2))
                return qml.expval(qml.Z(0))

            @qml.qnode(dev)
            def circuit2(weights):
                qml.RandomLayers(weights=weights, wires=range(2))
                return qml.expval(qml.Z(0))

        >>> np.allclose(circuit1(weights), circuit2(weights))
        True

        You can verify this by drawing the circuits.

        >>> print(qml.draw(circuit1, expansion_strategy="device")(weights))
        0: ──────────────────────╭X─╭X──RZ(1.40)─┤  <Z>
        1: ──RX(0.10)──RX(-2.10)─╰●─╰●───────────┤

        >>> print(qml.draw(circuit2, expansion_strategy="device")(weights))
        0: ──────────────────────╭X─╭X──RZ(1.40)─┤  <Z>
        1: ──RX(0.10)──RX(-2.10)─╰●─╰●───────────┤


        **Changing the seed**

        To change the randomly generated circuit architecture, you have to change the seed passed to the template.
        For example, these two calls of ``RandomLayers`` *do not* create the same circuit:

        >>> @qml.qnode(dev)
        ... def circuit(weights, seed=None):
        ...     qml.RandomLayers(weights=weights, wires=range(2), seed=seed)
        ...     return qml.expval(qml.Z(0))
        >>> np.allclose(circuit(weights, seed=9), circuit(weights, seed=12))
        False
        >>>  print(qml.draw(circuit, expansion_strategy="device")(weights, seed=9))
        0: ─╭X──RX(0.10)────────────┤  <Z>
        1: ─╰●──RY(-2.10)──RX(1.40)─┤
        >>> print(qml.draw(circuit, expansion_strategy="device")(weights, seed=12))
        0: ─╭X──RZ(0.10)──╭●─╭X───────────┤  <Z>
        1: ─╰●──RX(-2.10)─╰X─╰●──RZ(1.40)─┤


        **Automatic creation of random circuits**

        To automate the process of creating different circuits with ``RandomLayers``,
        you can set ``seed=None`` to avoid specifying a seed. However, in this case care needs
        to be taken. In the default setting, a quantum node is **mutable**, which means that the quantum function is
        re-evaluated every time it is called. This means that the circuit is re-constructed from scratch
        each time you call the qnode:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit_rnd(weights):
                qml.RandomLayers(weights=weights, wires=range(2), seed=None)
                return qml.expval(qml.Z(0))

            first_call = circuit_rnd(weights)
            second_call = circuit_rnd(weights)

        >>> np.allclose(first_call, second_call)
        False

        This can be rectified by making the quantum node **immutable**.

        .. code-block:: python

            @qml.qnode(dev, mutable=False)
            def circuit_rnd(weights):
                qml.RandomLayers(weights=weights, wires=range(2), seed=None)
                return qml.expval(qml.Z(0))

            first_call = circuit_rnd(weights)
            second_call = circuit_rnd(weights)

        >>> np.allclose(first_call, second_call)
        True

        **Parameter shape**

        The expected shape for the weight tensor can be computed with the static method
        :meth:`~.RandomLayers.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = qml.RandomLayers.shape(n_layers=2, n_rotations=3)
            weights = np.random.random(size=shape)
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(
        self,
        weights,
        wires,
        ratio_imprim=0.3,
        imprimitive=None,
        rotations=None,
        seed=42,
        id=None,
    ):
        shape = qml.math.shape(weights)
        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")

        self._hyperparameters = {
            "ratio_imprim": ratio_imprim,
            "imprimitive": imprimitive or qml.CNOT,
            "rotations": tuple(rotations) if rotations else (qml.RX, qml.RY, qml.RZ),
            "seed": seed,
        }

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights, wires, ratio_imprim, imprimitive, rotations, seed
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.RandomLayers.decomposition`.

        Args:
            weights (tensor_like): weight tensor
            wires (Any or Iterable[Any]): wires that the operator acts on
            ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation gates
            imprimitive (pennylane.ops.Operation): two-qubit gate to use
            rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates.
            seed (int): seed to generate random architecture

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> weights = torch.tensor([[0.1, -2.1, 1.4]])
        >>> rotations=[qml.RY, qml.RX]
        >>> qml.RandomLayers.compute_decomposition(weights, wires=["a", "b"], ratio_imprim=0.3,
        ...                                         imprimitive=qml.CNOT, rotations=rotations, seed=42)
        [RY(tensor(0.1000), wires=['b']),
         RY(tensor(-2.1000), wires=['b']),
         CNOT(wires=['b', 'a']),
         CNOT(wires=['b', 'a']),
         RX(tensor(1.4000), wires=['a'])]
        """
        wires = qml.wires.Wires(wires)
        rng = np.random.default_rng(seed)

        shape = qml.math.shape(weights)
        n_layers = qml.math.shape(weights)[0]
        op_list = []

        for l in range(n_layers):
            i = 0
            while i < shape[1]:
                if rng.random() > ratio_imprim:
                    # apply a random rotation gate to a random wire
                    gate = rng.choice(rotations)
                    rnd_wire = wires.select_random(1, seed=rng)
                    op_list.append(gate(weights[l][i], wires=rnd_wire))
                    i += 1

                else:
                    # apply the entangler to two random wires
                    if len(wires) > 1:
                        rnd_wires = wires.select_random(2, seed=rng)
                        op_list.append(imprimitive(wires=rnd_wires))
        return op_list

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
