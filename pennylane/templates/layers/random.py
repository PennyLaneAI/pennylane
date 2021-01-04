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
Contains the ``RandomLayers`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.ops import CNOT, RX, RY, RZ
from pennylane.templates.utils import (
    check_shape,
    check_number_of_layers,
    get_shape,
)
from pennylane.wires import Wires


def _preprocess(weights):
    """Validate and pre-process inputs as follows:

    * Check that the weights tensor is 2-dimensional.

    Args:
        weights (tensor_like): trainable parameters of the template

    Returns:
        int: number of times that the ansatz is repeated
    """

    if qml.tape_mode_active():

        shape = qml.math.shape(weights)

        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")

        repeat = shape[0]

    else:
        repeat = check_number_of_layers([weights])
        n_rots = get_shape(weights)[1]

        expected_shape = (repeat, n_rots)
        check_shape(
            weights,
            expected_shape,
            msg="'weights' must be of shape {}; got {}"
            "".format(expected_shape, get_shape(weights)),
        )

    return repeat


def random_layer(weights, wires, ratio_imprim, imprimitive, rotations, seed):
    r"""A single random layer.

    Args:
        weights (tensor_like): tensor of weights of shape ``(k,)``
        wires (Wires): wires that the template acts on
        ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation gates
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
        rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture
    """
    if seed is not None:
        np.random.seed(seed)

    i = 0
    while i < len(weights):
        if np.random.random() > ratio_imprim:
            # Apply a random rotation gate to a random wire
            gate = np.random.choice(rotations)
            rnd_wire = wires.select_random(1)

            gate(weights[i], wires=rnd_wire)

            i += 1
        else:
            # Apply the imprimitive to two random wires
            if len(wires) > 1:
                rnd_wires = wires.select_random(2)
                imprimitive(wires=rnd_wires)


@template
def RandomLayers(weights, wires, ratio_imprim=0.3, imprimitive=CNOT, rotations=None, seed=42):
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
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation gates
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
        rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture, defaults to 42

    Raises:
        ValueError: if inputs do not have the correct format

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
        >>> True

        You can verify this by drawing the circuits.

            >>> print(circuit1.draw())
            >>>  0: ──RX(0.1)──RX(-2.1)──╭X──╭X───────────┤ ⟨Z⟩
            ...  1: ─────────────────────╰C──╰C──RZ(1.4)──┤

            >>> print(circuit2.draw())
            >>>  0: ──RX(0.1)──RX(-2.1)──╭X──╭X───────────┤ ⟨Z⟩
            ...  1: ─────────────────────╰C──╰C──RZ(1.4)──┤

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
        >>>  0: ──╭X──RY(-2.1)──RX(1.4)──┤ ⟨Z⟩
        ...  1: ──╰C──RX(0.1)────────────┤

        >>> print(circuit_12.draw())
        >>>  0: ──╭X──RX(-2.1)──╭C──╭X──RZ(1.4)──┤ ⟨Z⟩
        ...  1: ──╰C──RZ(0.1)───╰X──╰C───────────┤


        **Automatically creating random circuits**

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
        >>> False

        This can be rectified by making the quantum node **immutable**.

        .. code-block:: python

            @qml.qnode(dev, mutable=False)
            def circuit_rnd(weights):
                RandomLayers(weights=weights, wires=range(2), seed=None)
                return qml.expval(qml.PauliZ(0))

            first_call = circuit_rnd(weights)
            second_call = circuit_rnd(weights)

        >>> np.allclose(first_call, second_call)
        >>> True
    """
    if seed is not None:
        np.random.seed(seed)

    if rotations is None:
        rotations = [RX, RY, RZ]

    wires = Wires(wires)
    repeat = _preprocess(weights)

    for l in range(repeat):
        random_layer(
            weights=weights[l],
            wires=wires,
            ratio_imprim=ratio_imprim,
            imprimitive=imprimitive,
            rotations=rotations,
            seed=seed,
        )
