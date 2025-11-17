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
Contains the StronglyEntanglingLayers template.
"""
from functools import partial

# pylint: disable=too-many-arguments
from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.ops import CNOT, Rot
from pennylane.ops.op_math import cond

from ..subroutine import Subroutine


def _validation(weights, wires):
    _shape = math.shape(weights)[-3:]

    if _shape[1] != len(wires):
        raise ValueError(
            f"Weights tensor must have second dimension of length {len(wires)}; got {_shape[1]}"
        )

    if _shape[2] != 3:
        raise ValueError(f"Weights tensor must have third dimension of length 3; got {_shape[2]}")


def _setup_ranges(weights, wires, ranges):
    shape = math.shape(weights)[-3:]
    if ranges is None:
        if len(wires) > 1:
            # tile ranges with iterations of range(1, n_wires)
            return tuple((l % (len(wires) - 1)) + 1 for l in range(shape[0]))
        return (0,) * shape[0]
    ranges = tuple(ranges)
    if len(ranges) != shape[0]:
        raise ValueError(f"Range sequence must be of length {shape[0]}; got {len(ranges)}")
    for r in ranges:
        if r % len(wires) == 0:
            raise ValueError(
                f"Ranges must not be zero nor divisible by the number of wires; got {r}"
            )
    return ranges


def shape(n_layers, n_wires):
    r"""Returns the expected shape of the weights tensor.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of wires

    Returns:
        tuple[int]: shape
    """

    return n_layers, n_wires, 3


@partial(Subroutine, static_argnames={"ranges", "imprimitive"})
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

    .. note::
        The two-qubit gate used as the imprimitive or entangler must not depend on parameters.

    Args:

        weights (tensor_like): weight tensor of shape ``(L, M, 3)``
        wires (Iterable): wires that the template acts on
        ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer; if ``None``
                                using :math:`r=l \mod M` for the :math:`l` th layer and :math:`M` wires.
        imprimitive (type of pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`

    Example:

        There are multiple arguments that the user can use to customize the layer.

        The required arguments are ``weights`` and ``wires``.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=4)

            @qml.qnode(dev)
            def circuit(parameters):
                qml.StronglyEntanglingLayers(weights=parameters, wires=range(4))
                return qml.expval(qml.Z(0))

            shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
            rng = np.random.default_rng(12345)
            weights = rng.random(size=shape)

        The shape of the ``weights`` argument decides the number of layers.

        The resulting circuit is:

        >>> print(qml.draw(circuit, level="device")(weights))
        0: ──Rot(0.23,0.32,0.80)─╭●───────╭X──Rot(0.67,0.10,0.44)─╭●────╭X────┤  <Z>
        1: ──Rot(0.68,0.39,0.33)─╰X─╭●────│───Rot(0.89,0.70,0.33)─│──╭●─│──╭X─┤
        2: ──Rot(0.60,0.19,0.67)────╰X─╭●─│───Rot(0.73,0.22,0.08)─╰X─│──╰●─│──┤
        3: ──Rot(0.94,0.25,0.95)───────╰X─╰●──Rot(0.16,0.34,0.47)────╰X────╰●─┤

        The default two-qubit gate used is :class:`~pennylane.ops.CNOT`. This can be changed by using the ``imprimitive`` argument.

        The ``ranges`` argument takes an integer sequence where each element
        determines the range hyperparameter for each layer. This range hyperparameter
        is the difference of the wire indices representing the two qubits the
        ``imprimitive`` gate acts on. For example, for ``range=[2,3]`` the
        first layer will have a range parameter of ``2`` and the second layer will
        have a range parameter of ``3``.
        Assuming ``wires=[0, 1, 2, 3]`` and a range parameter of ``2``, there will be
        an imprimitive gate acting on:

        * qubits ``(0, 2)``;
        * qubits ``(1, 3)``;
        * qubits ``(2, 0)``;
        * qubits ``(3, 1)``.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=4)

            @qml.qnode(dev)
            def circuit(parameters):
                qml.StronglyEntanglingLayers(weights=parameters, wires=range(4), ranges=[2, 3], imprimitive=qml.ops.CZ)
                return qml.expval(qml.Z(0))

            shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
            rng = np.random.default_rng(12345)
            weights = rng.random(size=shape)

        The resulting circuit is:

        >>> print(qml.draw(circuit, level="device")(weights))
        0: ──Rot(0.23,0.32,0.80)─╭●────╭Z──Rot(0.67,0.10,0.44)──────────────────────╭●─╭Z───────┤  <Z>
        1: ──Rot(0.68,0.39,0.33)─│──╭●─│──╭Z────────────────────Rot(0.89,0.70,0.33)─│──╰●─╭Z────┤
        2: ──Rot(0.60,0.19,0.67)─╰Z─│──╰●─│─────────────────────Rot(0.73,0.22,0.08)─│─────╰●─╭Z─┤
        3: ──Rot(0.94,0.25,0.95)────╰Z────╰●────────────────────Rot(0.16,0.34,0.47)─╰Z───────╰●─┤

    .. details::
        :title: Usage Details

        **Parameter shape**

        The expected shape for the weight tensor can be computed with the static method
        :meth:`~.qml.StronglyEntanglingLayers.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
            weights = np.random.random(size=shape)

    """
    _validation(weights, wires)
    ranges = _setup_ranges(weights, wires, ranges)
    if capture.enabled():
        ranges = math.stack(ranges, like="jax")

    n_wires = len(wires)
    n_layers = weights.shape[0]

    @for_loop(n_layers)
    def layers(l):
        @for_loop(n_wires)
        def rot_loop(i):
            Rot(
                weights[l, i, 0],
                weights[l, i, 1],
                weights[l, i, 2],
                wires=wires[i],
            )

        def imprim_true():
            @for_loop(n_wires)
            def imprimitive_loop(i):
                if capture.enabled():
                    act_on = math.array([i, i + ranges[l]], like="jax") % n_wires
                else:
                    act_on = wires.subset([i, i + ranges[l]], periodic_boundary=True)
                imprimitive(wires=act_on)

            imprimitive_loop()  # pylint: disable=no-value-for-parameter

        def imprim_false():
            pass

        rot_loop()  # pylint: disable=no-value-for-parameter
        cond(n_wires > 1, imprim_true, imprim_false)()

    layers()  # pylint: disable=no-value-for-parameter


StronglyEntanglingLayers.shape = staticmethod(shape)
