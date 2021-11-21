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
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`

    Example:

        There are multiple arguments that the user can use to customize the layer.

        The required arguments are ``weights`` and ``wires``.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=4)

            @qml.qnode(dev)
            def circuit(parameters):
                qml.StronglyEntanglingLayers(weights=parameters, wires=range(4))
                return qml.expval(qml.PauliZ(0))

            shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
            weights = np.random.random(size=shape)

        The shape of the ``weights`` argument decides the number of layers.

        The resulting circuit is:

        >>> print(qml.draw(circuit)(weights))
            0: ──Rot(0.106, 0.0913, 0.483)──╭C───────────────────────────────────────────────────────────╭X──Rot(0.0691, 0.841, 0.624)──────╭C──────╭X──┤ ⟨Z⟩
            1: ──Rot(0.0911, 0.249, 0.181)──╰X──╭C───Rot(0.311, 0.692, 0.141)────────────────────────────│──────────────────────────────╭C──│───╭X──│───┤
            2: ──Rot(0.0597, 0.982, 0.594)──────╰X──╭C─────────────────────────Rot(0.547, 0.349, 0.276)──│──────────────────────────────│───╰X──│───╰C──┤
            3: ──Rot(0.765, 0.81, 0.99)─────────────╰X───────────────────────────────────────────────────╰C──Rot(0.627, 0.348, 0.476)───╰X──────╰C──────┤

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
                return qml.expval(qml.PauliZ(0))

            shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
            weights = np.random.random(size=shape)

        The resulting circuit is:

        >>> print(qml.draw(circuit)(weights))
            0: ──Rot(0.629, 0.345, 0.566)───────╭C──────╭Z──Rot(0.874, 0.0388, 0.922)──╭C──╭Z──────────┤ ⟨Z⟩
            1: ──Rot(0.0596, 0.927, 0.807)──╭C──│───╭Z──│───Rot(0.311, 0.644, 0.297)───│───╰C──╭Z──────┤
            2: ──Rot(0.161, 0.29, 0.498)────│───╰Z──│───╰C──Rot(0.96, 0.79, 0.819)─────│───────╰C──╭Z──┤
            3: ──Rot(0.589, 0.103, 0.108)───╰Z──────╰C──────Rot(0.869, 0.69, 0.0183)───╰Z──────────╰C──┤

    .. UsageDetails::

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

    def __init__(self, weights, wires, ranges=None, imprimitive=None, do_queue=True, id=None):

        shape = qml.math.shape(weights)[-3:]
        self.n_layers = shape[0]

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
                self.ranges = [(l % (len(wires) - 1)) + 1 for l in range(self.n_layers)]
            else:
                self.ranges = [0] * self.n_layers
        else:
            if len(ranges) != self.n_layers:
                raise ValueError(
                    f"Range sequence must be of length {self.n_layers}; got {len(ranges)}"
                )
            for r in ranges:
                if r % len(wires) == 0:
                    raise ValueError(
                        f"Ranges must not be zero nor divisible by the number of wires; got {r}"
                    )
            self.ranges = ranges

        self.imprimitive = imprimitive or qml.CNOT

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    def expand(self):

        with qml.tape.QuantumTape() as tape:

            for l in range(self.n_layers):

                for i in range(len(self.wires)):
                    qml.Rot(
                        self.parameters[0][..., l, i, 0],
                        self.parameters[0][..., l, i, 1],
                        self.parameters[0][..., l, i, 2],
                        wires=self.wires[i],
                    )

                if len(self.wires) > 1:
                    for i in range(len(self.wires)):
                        act_on = self.wires.subset([i, i + self.ranges[l]], periodic_boundary=True)
                        self.imprimitive(wires=act_on)

        return tape

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
