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
Contains the ``CerezoTwoDesignLayers`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane import numpy as np
from pennylane.templates.decorator import template
from pennylane.ops import CZ, RY
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    _check_shape,
    _check_wires,
    _check_number_of_layers,
    _check_type,
    _get_shape,
)


@template
def CerezoTwoDesignLayers(first, even, odd, wires):
    r"""Layers consisting of a 2-design architecture of Pauli-Y rotations and controlled-Z entanglers
        proposed in `Cerezo et al. arXiv:2001.00550 <https://arxiv.org/abs/2001.00550>`_.

    A 2-design is a set of unitaries whose statistical properties are the same as sampling random unitaries
    with respect to the Haar measure up to the first 2 moments.

    The template starts with an initial block of single qubit Pauli-Y rotations, followed by :math:`L` layers of
    controlled-Z and Pauli-Z gates. Each layer consists of an "even" block whose entanglers start with the first qubit,
    and an "odd" block that starts with the second qubit.
    This is an example of two layers:

    .. figure:: ../../_static/templates/layers/cerez_two_design.png
        :align: center
        :width: 40%
        :target: javascript:void(0);

    The argument ``first`` contains the weights of the initial block, while ``even`` and ``odd`` are the weights of
    the respective layers. The number of layers :math:`L` is derived from the first dimensions of ``even`` and ``odd``,
    which must be the same.

    Args:
        first (array[float]): array of weights for the initial single qubit rotations, shape ``(len(wires),)``
        even (array[float]): array of weights for the even layers, shape ``(L, len(wires)//2, 3)``
        odd (array[float]): array of weights for the odd layers, shape ``(L, (len(wires)-1)//2, 3)``
        wires (Sequence[int] or int): qubit indices that the template acts on

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import CerezoTwoDesign
            from math import pi

            n_wires = 3
            dev = qml.device('default.qubit', wires=n_wires)

            @qml.qnode(dev)
            def circuit(weights):
                CerezoTwoDesign(first=first, even=even, odd=odd, wires=range(n_wires))
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        >>> circuit([[pi, pi, pi]])




    """

    #############
    # Input checks

    wires = _check_wires(wires)

    repeat = _check_number_of_layers([even, odd])

    _check_type(first, [list, np.ndarray],
                msg="'first' must be of type list or np.ndarray; got type {}".format(type(first)))
    _check_type(even, [list, np.ndarray],
                msg="'even' must be of type list or np.ndarray; got type {}".format(type(even)))
    _check_type(odd, [list, np.ndarray],
                msg="'odd' must be of type list or np.ndarray; got type {}".format(type(odd)))

    if isinstance(first, list):
        first = np.array(first)
    if isinstance(even, list):
        even = np.array(even)
    if isinstance(odd, list):
        odd = np.array(odd)

    expected_shape_even = (repeat, len(wires)//2, 3)
    _check_shape(
        even,
        expected_shape_even,
        msg="'even' must be of shape {}; got {}" "".format(expected_shape_even, _get_shape(even)),
    )

    expected_shape_odd = (repeat, (len(wires)-1)//2, 3)
    _check_shape(
        odd,
        expected_shape_odd,
        msg="'odd' must be of shape {}; got {}" "".format(expected_shape_odd, _get_shape(odd)),
    )

    expected_shape_first = (len(wires),)
    _check_shape(
        first,
        expected_shape_first,
        msg="'first' must be of shape {}; got {}" "".format(expected_shape_first, _get_shape(first)),
    )

    ###############

    # initial rotations
    broadcast(unitary=RY, pattern="single", wires=wires, parameters=first)

    # alternate layers
    for layer in range(repeat):

        # even layer
        broadcast(unitary=CZ, pattern="double", wires=wires, weights=even[layer, :, 0])
        broadcast(unitary=RY, pattern="single", wires=wires, weights=even[layer, :, 1:3].flatten())

        # odd layer
        broadcast(unitary=CZ, pattern="double_odd", wires=wires, weights=even[layer, :, 0])
        broadcast(unitary=RY, pattern="single", wires=wires[1:], weights=even[layer, :, 1:3].flatten())

