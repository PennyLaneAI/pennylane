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
Contains the ``BasicEntanglerLayers`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.ops import CNOT, RX
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    check_shape,
    check_number_of_layers,
    get_shape,
)
from pennylane.wires import Wires


def _preprocess(weights, wires):
    """Validate and pre-process inputs as follows:

    * Check the shape of the weights tensor, making sure that the second dimension
      has length :math:`n`, where :math:`n` is the number of qubits.

    Args:
        weights (tensor_like): trainable parameters of the template
        wires (Wires): wires that template acts on

    Returns:
        int: number of times that the ansatz is repeated
    """

    if qml.tape_mode_active():

        shape = qml.math.shape(weights)
        repeat = shape[0]

        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")

        if shape[1] != len(wires):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(wires)}; got {shape[1]}"
            )

    else:

        repeat = check_number_of_layers([weights])

        expected_shape = (repeat, len(wires))
        check_shape(
            weights,
            expected_shape,
            msg=f"Weights tensor must have second dimension of length {len(wires)}; got {get_shape(weights)[1]}",
        )

    return repeat


@template
def BasicEntanglerLayers(weights, wires, rotation=None):
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

        The :mod:`~pennylane.init` module has two parameter initialization functions, ``basic_entangler_layers_normal``
        and ``basic_entangler_layers_uniform``.

        .. code-block:: python

            from pennylane.init import basic_entangler_layers_normal

            n_layers = 4
            weights = basic_entangler_layers_normal(n_layers=n_layers, n_wires=n_wires)

            circuit(weights)


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

    if rotation is None:
        rotation = RX

    wires = Wires(wires)

    repeat = _preprocess(weights, wires)

    for layer in range(repeat):

        broadcast(unitary=rotation, pattern="single", wires=wires, parameters=weights[layer])
        broadcast(unitary=CNOT, pattern="ring", wires=wires)
