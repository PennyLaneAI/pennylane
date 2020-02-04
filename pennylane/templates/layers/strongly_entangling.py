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
Contains the ``StronglyEntanglingLayers`` template.
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.templates.decorator import template
from pennylane.ops import CNOT, Rot
from pennylane.templates.utils import (_check_shape,
                                       _check_no_variable,
                                       _check_wires,
                                       _check_type,
                                       _check_number_of_layers,
                                       _get_shape)


def strongly_entangling_layer(weights, wires, r, imprimitive):
    r"""A layer applying rotations on each qubit followed by cascades of 2-qubit entangling gates.

    Args:
        weights (array[float]): array of weights of shape ``(len(wires), 3)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on
        r (int): range of the imprimitive gates of this layer, defaults to 1
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
    """

    for i, wire in enumerate(wires):
        Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    n_wires = len(wires)
    if n_wires > 1:
        for i in range(n_wires):
            imprimitive(wires=[wires[i], wires[(i + r) % n_wires]])


@template
def StronglyEntanglingLayers(weights, wires, ranges=None, imprimitive=CNOT):
    r"""Layers consisting of single qubit rotations and entanglers, inspired by the circuit-centric classifier design
    `arXiv:1804.00633 <https://arxiv.org/abs/1804.00633>`_.

    The argument ``weights`` contains the weights for each layer. The number of layers :math:`L` is therefore derived
    from the first dimension of ``weights``.

    The 2-qubit gates, whose type is specified by the ``imprimitive`` argument,
    act chronologically on the :math:`M` wires, :math:`i = 1,...,M`. The second qubit of each gate is given by
    :math:`(i+r)\mod M`, where :math:`r` is a  hyperparameter called the *range*, and :math:`0 < r < M`.
    If applied to one qubit only, this template will use no imprimitive gates.

    This is an example of two 4-qubit strongly entangling layers (ranges :math:`r=1` and :math:`r=2`, respectively) with
    rotations :math:`R` and CNOTs as imprimitives:

    .. figure:: ../../_static/layer_sec.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    Args:

        weights (array[float]): array of weights of shape ``(:math:`L`, :math:`M`, 3)``
        wires (Sequence[int] or int): qubit indices that the template acts on
        ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer; if None
                                using :math:`r=l \mod M` for the :math:`l`th layer and :math:`M` wires.
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`

    Raises:
        ValueError: if inputs do not have the correct format
    """

    #############
    # Input checks

    _check_no_variable(ranges, msg="'ranges' cannot be differentiable")
    _check_no_variable(imprimitive, msg="'imprimitive' cannot be differentiable")

    wires = _check_wires(wires)

    repeat = _check_number_of_layers([weights])

    expected_shape = (repeat, len(wires), 3)
    _check_shape(weights, expected_shape, msg="'weights' must be of shape {}; got {}"
                                              "".format(expected_shape, _get_shape(weights)))

    if ranges is None:
        # tile ranges with iterations of range(1, n_wires)
        ranges = [(l % (len(wires)-1)) + 1 for l in range(repeat)]

    expected_shape = (repeat,)
    _check_shape(ranges, expected_shape, msg="'ranges' must be of shape {}; got {}"
                                             "".format(expected_shape, _get_shape(weights)))

    _check_type(ranges, [list], msg="'ranges' must be a list; got {}"
                                    "".format(ranges))
    for r in ranges:
        _check_type(r, [int], msg="'ranges' must be a list of integers; got {}"
                                  "".format(ranges))
    if any((r >= len(wires) or r == 0) for r in ranges):
        raise ValueError("the range for all layers needs to be smaller than the number of "
                         "qubits; got ranges {}.".format(ranges))

    ###############

    for l in range(repeat):

        strongly_entangling_layer(weights=weights[l], wires=wires, r=ranges[l], imprimitive=imprimitive)
