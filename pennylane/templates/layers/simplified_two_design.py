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
Contains the ``SimplifiedTwoDesign`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane import numpy as np
from pennylane.templates.decorator import template
from pennylane.ops import CZ, RY
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    check_shape,
    check_wires,
    check_number_of_layers,
    check_type,
    get_shape,
)


@template
def entangler(par1, par2, wires):
    """Implements a two qubit unitary consisting of a controlled-Z entangler and Pauli-Y rotations.

    Args:
         par1 (float or qml.Variable): parameter of first Pauli-Y rotation
         par2 (float or qml.Variable): parameter of second Pauli-Y rotation
         wires (list): two wire indices that unitary acts on
    """

    CZ(wires=wires)
    RY(par1, wires=wires[0])
    RY(par2, wires=wires[1])


@template
def SimplifiedTwoDesign(initial_layer_weights, weights, wires):
    r"""
    Layers consisting of a simplified 2-design architecture of Pauli-Y rotations and controlled-Z entanglers
    proposed in `Cerezo et al. (2020) <https://arxiv.org/abs/2001.00550>`_.

    A 2-design is an ensemble of unitaries whose statistical properties are the same as sampling random unitaries
    with respect to the Haar measure up to the first 2 moments.

    The template is not a strict 2-design, since
    it does not consist of universal 2-qubit gates as building blocks, but has been shown in
    `Cerezo et al. (2020) <https://arxiv.org/abs/2001.00550>`_ to exhibit important properties to study "barren plateaus"
    in quantum optimization landscapes.

    The template starts with an initial layer of single qubit Pauli-Y rotations, before the main
    :math:`L` layers are applied. The basic building block of the main layers are controlled-Z entanglers
    followed by a pair of Pauli-Y rotation gates (one for each wire).
    Each layer consists of an "even" part whose entanglers start with the first qubit,
    and an "odd" part that starts with the second qubit.

    This is an example of two layers, including the initial layer:

    .. figure:: ../../_static/templates/layers/simplified_two_design.png
        :align: center
        :width: 40%
        :target: javascript:void(0);

    |

    The argument ``initial_layer_weights`` contains the rotation angles of the initial layer of Pauli-Y rotations,
    while ``weights`` contains the pairs of Pauli-Y rotation angles of the respective layers. Each layer takes
    :math:`\lfloor M/2 \rfloor + \lfloor (M-1)/2 \rfloor = M-1` pairs of angles, where :math:`M` is the number of wires.
    The number of layers :math:`L` is derived from the first dimension of ``weights``.

    Args:
        initial_layer_weights (array[float]): array of weights for the initial rotation block, shape ``(M,)``
        weights (array[float]): array of rotation angles for the layers, shape ``(L, M-1, 2)``
        wires (Sequence[int] or int): qubit indices that the template acts on

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        template - here shown for two layers - is used inside a :class:`~.QNode`:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import SimplifiedTwoDesign
            from math import pi

            n_wires = 3
            dev = qml.device('default.qubit', wires=n_wires)

            @qml.qnode(dev)
            def circuit(init_weights, weights):
                SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(n_wires))
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

            init_weights = [pi, pi, pi]
            weights_layer1 = [[0., pi],
                              [0., pi]]
            weights_layer2 = [[pi, 0.],
                              [pi, 0.]]
            weights = [weights_layer1, weights_layer2]

            >>> circuit(init_weights, weights)
            [1., -1., 1.]

        **Parameter initialization function**

        The :mod:`~pennylane.init` module contains four parameter initialization functions:

        * ``simplified_two_design_initial_layer_normal``
        * ``simplified_two_design_initial_layer_uniform``
        * ``simplified_two_design_weights_normal``.
        * ``simplified_two_design_weights_uniform``.

        They can be used as follows:

        .. code-block:: python

            from pennylane.init import (simplified_two_design_initial_layer_normal,
                                        simplified_two_design_weights_normal)

            n_layers = 4
            init_weights = simplified_two_design_initial_layer_normal(n_wires)
            weights = simplified_two_design_weights_normal(n_layers, n_wires)

            >>> circuit(initial_layer_weights, weights)

    """

    #############
    # Input checks

    wires = check_wires(wires)

    repeat = check_number_of_layers([weights])

    check_type(
        initial_layer_weights,
        [list, np.ndarray],
        msg="'initial_layer_weights' must be of type list or np.ndarray; got type {}".format(
            type(initial_layer_weights)
        ),
    )
    check_type(
        weights,
        [list, np.ndarray],
        msg="'weights' must be of type list or np.ndarray; got type {}".format(type(weights)),
    )

    expected_shape_initial = (len(wires),)
    check_shape(
        initial_layer_weights,
        expected_shape_initial,
        msg="'initial_layer_weights' must be of shape {}; got {}"
        "".format(expected_shape_initial, get_shape(initial_layer_weights)),
    )

    if len(wires) in [0, 1]:
        expected_shape_weights = (0,)
    else:
        expected_shape_weights = (repeat, len(wires) - 1, 2)

    check_shape(
        weights,
        expected_shape_weights,
        msg="'weights' must be of shape {}; got {}"
        "".format(expected_shape_weights, get_shape(weights)),
    )

    ###############
    # initial rotations
    broadcast(unitary=RY, pattern="single", wires=wires, parameters=initial_layer_weights)

    # alternate layers
    for layer in range(repeat):

        # even layer
        weights_even = weights[layer][: len(wires) // 2]
        broadcast(unitary=entangler, pattern="double", wires=wires, parameters=weights_even)

        # odd layer
        weights_odd = weights[layer][len(wires) // 2 :]
        broadcast(unitary=entangler, pattern="double_odd", wires=wires, parameters=weights_odd)
