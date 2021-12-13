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
Contains the MPS template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import warnings
import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Operation, AnyWires


def compute_indices_MPS(wires, loc):
    r"""
    Generate a list containing the wires for each block.

    Args:
        wires (Iterable): wires that the template acts on
        loc (int): number of wires per block
    Returns:
        layers (array): array of wire indices or wire labels for each block
    """

    n_wires = len(wires)

    if loc % 2 != 0:
        raise AssertionError(f"loc must be an even integer; got {loc}")

    if loc < 2:
        raise ValueError(
            f"number of wires in each block must be larger than or equal to 2; got loc={loc}"
        )

    if n_wires < 2:
        raise ValueError(f"number of wires must be greater than or equal to 2; got {n_wires}")

    if loc > n_wires:
        raise ValueError(
            f"loc must be smaller than or equal to the number of wires; got loc = {loc} and number of wires = {n_wires}"
        )

    if n_wires % (loc / 2) > 0:
        warnings.warn(
            f"The number of wires should be a multiple of {int(loc/2)}; got {n_wires}"
        )

    layers = np.array(
        [
            [wires[idx] for idx in range(j, j + loc)]
            for j in range(0, len(wires) - int(len(wires) % (loc // 2)) - loc // 2, loc // 2)
        ]
    )
    return layers


class MPS(Operation):
    r"""Quantum circuit that broadcasts local gates, similar to the architecture in `arXiv:1803.11537 <https://arxiv.org/abs/1803.11537>`_.

    The argument ``block`` is a user-defined quantum function. ``block`` must include two arguments: ``block_weights`` and ``block_wires``.

    Args:
        wires (Iterable): wires that the template acts on
        loc (int): number of wires per block
        block (Callable): quantum circuit that composes a block
        n_params_block (int): the number of parameters in a block; equal to the number of elements in ``block_weights``
        weights (Sequence): list containing the weights for all blocks; weights should have one element per block

    .. UsageDetails::

        This example demonstrates the use of ``MPS`` for a simple block.

        .. code-block:: python

            import pennylane as qml
            import numpy as np

            def block(block_weights, block_wires):
                qml.CNOT(wires=[block_wires[0],block_wires[1]])
                qml.Rot(block_weights[0],block_weights[1],block_weights[2],wires=block_wires[0])
                qml.Rot(block_weights[3],block_weights[4],block_weights[5],wires=block_wires[1])

            n_wires = 4
            loc = 2
            n_params_block = 6
            template_weights = [[1,2,3,4,5,6],[3,4,5,6,7,8],[4,5,6,7,8,9]]

            dev= qml.device('default.qubit',wires=n_wires)
            @qml.qnode(dev)
            def circuit(weights):
                qml.MPS(wires = range(n_wires),loc=loc,block=block, n_params_block=n_params_block, weights=weights)
                return qml.expval(qml.PauliZ(wires=n_wires-1))

            >>> print(qml.draw(circuit,expansion_strategy='device')(template_weights))
            0: ──╭C──Rot(1, 2, 3)──────────────────────────────────────┤
            1: ──╰X──Rot(4, 5, 6)──╭C──Rot(3, 4, 5)────────────────────┤
            2: ────────────────────╰X──Rot(6, 7, 8)──╭C──Rot(4, 5, 6)──┤
            3: ──────────────────────────────────────╰X──Rot(7, 8, 9)──┤ ⟨Z⟩
    """

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(
        self,
        wires,
        loc,
        block,
        n_params_block,
        weights=None,
        do_queue=True,
        id=None,
    ):

        self.ind_gates = compute_indices_MPS(wires, loc)
        n_wires = len(wires)
        shape = qml.math.shape(weights)[-4:]  # (n_params_block, n_blocks)
        self.n_params_block = n_params_block
        self.n_blocks = int(n_wires / (loc / 2) - 1)
        self.block = block

        if weights is None:
            self.weights = np.random.rand(n_params_block, int(self.n_blocks))

        else:

            if shape[0] != self.n_blocks:
                raise ValueError(
                    f"Weights tensor must have first dimension of length {self.n_blocks}; got {shape[0]}"
                )
            if shape[-1] != self.n_params_block:
                raise ValueError(
                    f"Weights tensor must have last dimension of length {self.n_params_block}; got {shape[-1]}"
                )

            self.weights = weights

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        with qml.tape.QuantumTape() as tape:
            for idx, w in enumerate(self.ind_gates):
                self.block(block_weights=self.weights[idx][:], block_wires=w.tolist())

        return tape

    @staticmethod
    def shape(n_wires, loc, n_params_block):

        r"""Returns the expected shape of the weights tensor.
        Args:
            n_wires (int): number of wires the template acts on
            loc (int): number of wires per block
            n_params_block (int): number of parameters per block
        Returns:
            tuple[int]: expected shape of ``weights`` argument
        """
        if n_wires % (loc / 2) > 0:
            warnings.warn(
                f"The number of wires should be a multiple of loc/2 = {int(loc/2)}; got {n_wires}"
            )

        if loc > n_wires:
            raise ValueError(
                f"loc must be smaller than or equal to the number of wires; got loc = {loc} and number of wires = {n_wires}"
            )

        n_blocks = int(n_wires / (loc / 2) - 1)
        return n_blocks, n_params_block
