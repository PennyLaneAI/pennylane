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
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class StronglyEntanglingLayers(Operation):
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
            weights = np.random.random(size=shape)

        The shape of the ``weights`` argument decides the number of layers.

        The resulting circuit is:

        >>> print(qml.draw(circuit, expansion_strategy="device")(weights))
        0: ──Rot(0.68,0.98,0.48)─╭●───────╭X──Rot(0.94,0.22,0.70)─╭●────╭X────┤  <Z>
        1: ──Rot(0.91,0.19,0.15)─╰X─╭●────│───Rot(0.50,0.20,0.63)─│──╭●─│──╭X─┤
        2: ──Rot(0.91,0.68,0.96)────╰X─╭●─│───Rot(0.14,0.05,0.16)─╰X─│──╰●─│──┤
        3: ──Rot(0.46,0.56,0.80)───────╰X─╰●──Rot(0.87,0.04,0.22)────╰X────╰●─┤

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
            weights = np.random.random(size=shape)

        The resulting circuit is:

        >>> print(qml.draw(circuit, expansion_strategy="device")(weights))
        0: ──Rot(0.99,0.17,0.12)─╭●────╭Z──Rot(0.02,0.94,0.57)──────────────────────╭●─╭Z───────┤  <Z>
        1: ──Rot(0.55,0.42,0.61)─│──╭●─│──╭Z────────────────────Rot(0.15,0.26,0.82)─│──╰●─╭Z────┤
        2: ──Rot(0.79,0.93,0.27)─╰Z─│──╰●─│─────────────────────Rot(0.73,0.01,0.44)─│─────╰●─╭Z─┤
        3: ──Rot(0.30,0.74,0.93)────╰Z────╰●────────────────────Rot(0.57,0.50,0.80)─╰Z───────╰●─┤

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

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, ranges=None, imprimitive=None, id=None):
        shape = qml.math.shape(weights)[-3:]

        if shape[1] != len(wires):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(wires)}; got {shape[1]}"
            )

        if shape[2] != 3:
            raise ValueError(
                f"Weights tensor must have third dimension of length 3; got {shape[2]}"
            )

        if ranges is None:
            if len(wires) > 1:
                # tile ranges with iterations of range(1, n_wires)
                ranges = tuple((l % (len(wires) - 1)) + 1 for l in range(shape[0]))
            else:
                ranges = (0,) * shape[0]
        else:
            ranges = tuple(ranges)
            if len(ranges) != shape[0]:
                raise ValueError(f"Range sequence must be of length {shape[0]}; got {len(ranges)}")
            for r in ranges:
                if r % len(wires) == 0:
                    raise ValueError(
                        f"Ranges must not be zero nor divisible by the number of wires; got {r}"
                    )

        self._hyperparameters = {"ranges": ranges, "imprimitive": imprimitive or qml.CNOT}

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights, wires, ranges, imprimitive
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.StronglyEntanglingLayers.decomposition`.

        Args:
            weights (tensor_like): weight tensor
            wires (Any or Iterable[Any]): wires that the operator acts on
            ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer
            imprimitive (pennylane.ops.Operation): two-qubit gate to use

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> weights = torch.tensor([[-0.2, 0.1, -0.4], [1.2, -2., -0.4]])
        >>> qml.StronglyEntanglingLayers.compute_decomposition(weights, wires=["a", "b"], ranges=[2], imprimitive=qml.CNOT)
        [Rot(tensor(-0.2000), tensor(0.1000), tensor(-0.4000), wires=['a']),
        Rot(tensor(1.2000), tensor(-2.), tensor(-0.4000), wires=['b']),
        CNOT(wires=['a', 'a']),
        CNOT(wires=['b', 'b'])]
        """
        n_layers = qml.math.shape(weights)[0]
        wires = qml.wires.Wires(wires)
        op_list = []

        for l in range(n_layers):
            for i in range(len(wires)):  # pylint: disable=consider-using-enumerate
                op_list.append(
                    qml.Rot(
                        weights[..., l, i, 0],
                        weights[..., l, i, 1],
                        weights[..., l, i, 2],
                        wires=wires[i],
                    )
                )

            if len(wires) > 1:
                for i in range(len(wires)):
                    act_on = wires.subset([i, i + ranges[l]], periodic_boundary=True)
                    op_list.append(imprimitive(wires=act_on))

        return op_list

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns the expected shape of the weights tensor.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            tuple[int]: shape
        """

        return n_layers, n_wires, 3
