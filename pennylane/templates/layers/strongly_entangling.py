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
        ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer; if None
                                using :math:`r=l \mod M` for the :math:`l` th layer and :math:`M` wires.
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`

    .. UsageDetails::

        **Parameter shape**

        The expected shape for the weight tensor can be computed with the static method
        :meth:`~.SimplifiedTwoDesign.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
            weights = np.random.random(size=shape)

    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(self, weights, wires, ranges=None, imprimitive=None, do_queue=True, id=None):

        shape = qml.math.shape(weights)
        self.n_layers = shape[0]

        if len(shape) != 3:
            raise ValueError(f"Weights tensor must be 3-dimensional; got shape {shape}")

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

    def expand(self):

        with qml.tape.QuantumTape() as tape:

            for l in range(self.n_layers):

                for i in range(len(self.wires)):
                    qml.Rot(
                        self.parameters[0][l, i, 0],
                        self.parameters[0][l, i, 1],
                        self.parameters[0][l, i, 2],
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
