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
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.ops import CNOT, Rot
from pennylane.templates import broadcast
from pennylane.wires import Wires


def _preprocess(weights, wires, ranges):
    """Validate and pre-process inputs as follows:

    * Check the shape of the weights tensor.
    * If ranges is None, define a default.

    Args:
        weights (tensor_like): trainable parameters of the template
        wires (Wires): wires that template acts on
        ranges (Sequence[int]): range for each subsequent layer

    Returns:
        int, list[int]: number of times that the ansatz is repeated and preprocessed ranges
    """
    shape = qml.math.shape(weights)
    repeat = shape[0]

    if len(shape) != 3:
        raise ValueError(f"Weights tensor must be 3-dimensional; got shape {shape}")

    if shape[1] != len(wires):
        raise ValueError(
            f"Weights tensor must have second dimension of length {len(wires)}; got {shape[1]}"
        )

    if shape[2] != 3:
        raise ValueError(f"Weights tensor must have third dimension of length 3; got {shape[2]}")

    if len(wires) > 1:
        if ranges is None:
            # tile ranges with iterations of range(1, n_wires)
            ranges = [(l % (len(wires) - 1)) + 1 for l in range(repeat)]
    else:
        ranges = [0] * repeat

    return repeat, ranges


def strongly_entangling_layer(weights, wires, r, imprimitive):
    r"""A layer applying rotations on each qubit followed by cascades of 2-qubit entangling gates.

    Args:
        weights (tensor_like): weight tensor of shape ``(len(wires), 3)``
        wires (Wires): wires that the template acts on
        r (int): range of the imprimitive gates of this layer, defaults to 1
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
    """

    broadcast(unitary=Rot, pattern="single", wires=wires, parameters=weights)

    n_wires = len(wires)
    if n_wires > 1:
        for i in range(n_wires):
            act_on = wires.subset([i, i + r], periodic_boundary=True)
            imprimitive(wires=act_on)


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

        weights (tensor_like): weight tensor of shape ``(L, M, 3)``
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer; if None
                                using :math:`r=l \mod M` for the :math:`l`th layer and :math:`M` wires.
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`

    Raises:
        ValueError: if inputs do not have the correct format
    """
    wires = Wires(wires)
    repeat, ranges = _preprocess(weights, wires, ranges)

    for l in range(repeat):

        strongly_entangling_layer(
            weights=weights[l], wires=wires, r=ranges[l], imprimitive=imprimitive
        )
