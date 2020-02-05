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
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
from pennylane.templates.decorator import template
from pennylane.ops import CNOT, RX, RY, RZ
from pennylane.templates.utils import (_check_shape,
                                       _check_no_variable,
                                       _check_wires,
                                       _check_type,
                                       _check_number_of_layers,
                                       _get_shape)


def random_layer(weights, wires, ratio_imprim, imprimitive, rotations, seed):
    r"""A single random layer.

    Args:
        weights (array[float]): array of weights of shape ``(k,)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on
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
            gate = np.random.choice(rotations)
            wire = np.random.choice(wires)
            gate(weights[i], wires=wire)
            i += 1
        else:
            if len(wires) > 1:
                on_wires = np.random.permutation(wires)[:2]
                on_wires = list(on_wires)
                imprimitive(wires=on_wires)


@template
def RandomLayers(weights, wires, ratio_imprim=0.3, imprimitive=CNOT, rotations=None, seed=42):
    r"""Layers of randomly chosen single qubit rotations and 2-qubit entangling gates, acting
    on randomly chosen qubits.

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

    .. note::
        Using the default seed (or any other fixed integer seed) generates one and the same circuit in every
        quantum node. To generate different circuit architectures, either use a different random seed, or use ``seed=None``
        together with the ``cache=False`` option when creating a quantum node.

    .. warning::
        When using a random number generator anywhere inside the quantum function without the ``cache=False`` option,
        a new random circuit architecture will be created every time the quantum node is evaluated.

    Args:
        weights (array[float]): array of weights of shape ``(L, k)``,
        wires (Sequence[int]): sequence of qubit indices that the template acts on
        ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation gates
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
        rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture

    Raises:
        ValueError: if inputs do not have the correct format
    """
    if seed is not None:
        np.random.seed(seed)

    if rotations is None:
        rotations = [RX, RY, RZ]

    #############
    # Input checks

    _check_no_variable(ratio_imprim, msg="'ratio_imprim' cannot be differentiable")
    _check_no_variable(imprimitive, msg="'imprimitive' cannot be differentiable")
    _check_no_variable(rotations, msg="'rotations' cannot be differentiable")
    _check_no_variable(seed, msg="'seed' cannot be differentiable")

    wires = _check_wires(wires)

    repeat = _check_number_of_layers([weights])
    n_rots = _get_shape(weights)[1]

    expected_shape = (repeat, n_rots)
    _check_shape(weights, expected_shape, msg="'weights' must be of shape {}; got {}"
                                              "".format(expected_shape, _get_shape(weights)))

    _check_type(ratio_imprim, [float, type(None)], msg="'ratio_imprim' must be a float; got {}".format(ratio_imprim))
    _check_type(n_rots, [int, type(None)], msg="'n_rots' must be an integer; got {}".format(n_rots))
    #TODO: Check that 'rotations' contains operations
    _check_type(rotations, [list, type(None)], msg="'rotations' must be a list of PennyLane operations; got {}"
                                                   "".format(rotations))
    _check_type(seed, [int, type(None)], msg="'seed' must be an integer; got {}.".format(seed))

    ###############

    for l in range(repeat):
        random_layer(weights=weights[l], wires=wires, ratio_imprim=ratio_imprim, imprimitive=imprimitive,
                     rotations=rotations, seed=seed)
