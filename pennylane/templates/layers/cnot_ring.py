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
Contains the ``CnotRingLayers`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.templates.decorator import template
from pennylane.ops import CNOT, RX
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    _check_shape,
    _check_no_variable,
    _check_wires,
    _check_number_of_layers,
    _get_shape,
)


def cnot_ring_layer(weights, wires, rotation):
    r"""A layer applying a one-parameter single-qubit rotation on each qubit, followed by a chain of CNOT gates.

    Args:
        weights (array[float]): array of weights of shape ``(len(wires), 3)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on
        rotation (pennylane.ops.Operation): one-parameter single-qubit gate to use,
                                            defaults to :class:`~pennylane.ops.RX`
    """

    broadcast(unitary=rotation, pattern="single", wires=wires, parameters=weights)
    broadcast(unitary=CNOT, pattern="ring", wires=wires)


@template
def CnotRingLayers(weights, wires, rotation=None):
    r"""Layers consisting of one-parameter single-qubit rotations on each qubit, followed by a ring of CNOT gates.

    The ring of CNOT gates connects every qubit with its neighbour, whereas the last qubit is considered to be
    a neighbour of the first qubit.

    .. figure:: ../../_static/cnot_ring.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    The argument ``weights`` contains the weights for each layer. The number of layers :math:`L` is therefore derived
    from the first dimension of ``weights``.

    .. note::

        When using a single wire, the template only applies the single qubit gates:

        .. figure:: ../../_static/cnot_ring_1wire.png
            :align: center
            :width: 40%
            :target: javascript:void(0);

    .. note::

        For two wires, only one CNOT gate is applied in each layer:

        .. figure:: ../../_static/cnot_ring_2wires.png
            :align: center
            :width: 30%
            :target: javascript:void(0);

    Args:

        weights (array[float]): array of weights of shape ``(:math:`L`, :math:`M`, 3)``
        wires (Sequence[int] or int): qubit indices that the template acts on
        rotation (pennylane.ops.Operation): one-parameter single-qubit gate to use,
                                            defaults to :class:`~pennylane.ops.RX`
    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::


    """

    #############
    # Input checks

    if rotation is None:
        rotation = RX

    _check_no_variable(rotation, msg="'rotation' cannot be differentiable")

    wires = _check_wires(wires)

    repeat = _check_number_of_layers([weights])

    expected_shape = (repeat, len(wires))
    _check_shape(
        weights,
        expected_shape,
        msg="'weights' must be of shape {}; got {}" "".format(expected_shape, _get_shape(weights)),
    )

    ###############

    for layer in range(repeat):

        cnot_ring_layer(
            weights=weights[layer], wires=wires, rotation=rotation
        )
