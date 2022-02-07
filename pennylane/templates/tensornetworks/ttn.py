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
"""
Contains the TTN template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import warnings
import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Operation, AnyWires


def compute_indices(wires, n_block_wires):
    """Generate a list containing the wires for each block.

    Args:
        wires (Iterable): wires that the template acts on
        n_block_wires (int): number of wires per block

    Returns:
        layers (array): array of wire labels for each block
    """

    n_wires = len(wires)

    if n_block_wires % 2 != 0:
        raise ValueError(f"n_block_wires must be an even integer; got {n_block_wires}")

    if n_block_wires < 2:
        raise ValueError(
            f"number of wires in each block must be larger than or equal to 2; got n_block_wires = {n_block_wires}"
        )

    if n_block_wires > n_wires:
        raise ValueError(
            f"n_block_wires must be smaller than or equal to the number of wires; "
            f"got n_block_wires = {n_block_wires} and number of wires = {n_wires}"
        )

    if not np.log2(n_wires / n_block_wires).is_integer():
        warnings.warn(
            f"The number of wires should be n_block_wires times 2^n; got n_wires/n_block_wires = {n_wires/n_block_wires}"
        )

    n_wires = 2 ** (int(np.log2(len(wires) / n_block_wires))) * n_block_wires
    n_layers = int(np.log2(n_wires // n_block_wires)) + 1

    layers = [
        [
            wires[i]
            for i in range(
                x + 2 ** (j - 1) * n_block_wires // 2 - n_block_wires // 2,
                x + n_block_wires // 2 + 2 ** (j - 1) * n_block_wires // 2 - n_block_wires // 2,
            )
        ]
        + [
            wires[i]
            for i in range(
                x
                + 2 ** (j - 1) * n_block_wires // 2
                + 2 ** (j - 1) * n_block_wires // 2
                - n_block_wires // 2,
                x
                + 2 ** (j - 1) * n_block_wires // 2
                + n_block_wires // 2
                + 2 ** (j - 1) * n_block_wires // 2
                - n_block_wires // 2,
            )
        ]
        for j in range(1, n_layers + 1)
        for x in range(0, n_wires - n_block_wires // 2, 2 ** (j - 1) * n_block_wires)
    ]

    return layers


class TTN(Operation):
    """The TTN template broadcasts an input circuit across many wires following the architecture of a tree tensor network.
    The result is similar to the architecture in `arXiv:1803.11537 <https://arxiv.org/abs/1803.11537>`_.

    The argument ``block`` is a user-defined quantum circuit. Each ``block`` may depend on a different set of parameters.
    These are passed as a list by the ``template_weights`` argument.

    For more details, see *Usage Details* below.

    Args:
        wires (Iterable): wires that the template acts on
        n_block_wires (int): number of wires per block
        block (Callable): quantum circuit that defines a block
        n_params_block (int): the number of parameters in a block
        template_weights (Sequence): list containing the weights for all blocks

    .. UsageDetails::

        In general, the block takes D parameters and **must** have the following signature:

        .. code-block:: python

            unitary(parameter1, parameter2, ... parameterD, wires)

        For a block with multiple parameters, ``n_params_block`` is equal to the number of parameters in ``block``.
        For a block with a single parameter, ``n_params_block`` is equal to the length of the parameter.

        To avoid ragged using arrays, all block parameters should have the same dimension.

        The length of the ``template_weights`` argument should match the number of blocks.
        The expected number of blocks can be obtained from ``qml.TTN.n_blocks(wires, n_block_wires)``.

        This example demonstrates the use of ``TTN`` for a simple block.

        .. code-block:: python

            import pennylane as qml
            import numpy as np

            def block(weights, wires):
                qml.CNOT(wires=[wires[0],wires[1]])
                qml.RY(weights[0], wires=wires[0])
                qml.RY(weights[1], wires=wires[1])

            n_wires = 4
            n_block_wires = 2
            n_params_block = 2
            n_blocks = qml.TTN.get_n_blocks(range(n_wires),n_block_wires)
            template_weights = [[0.1,-0.3]]*n_blocks

            dev= qml.device('default.qubit',wires=range(n_wires))
            @qml.qnode(dev)
            def circuit(template_weights):
                qml.TTN(range(n_wires),n_block_wires,block, n_params_block, template_weights)
                return qml.expval(qml.PauliZ(wires=n_wires-1))

        >>> print(qml.draw(circuit,expansion_strategy='device')(template_weights))
        0: ──╭C──RY(0.1)─────────────────┤
        1: ──╰X──RY(-0.3)──╭C──RY(0.1)───┤
        2: ──╭C──RY(0.1)───│─────────────┤
        3: ──╰X──RY(-0.3)──╰X──RY(-0.3)──┤ ⟨Z⟩
    """

    num_wires = AnyWires
    grad_method = None

    @property
    def num_params(self):
        return 1

    def __init__(
        self,
        wires,
        n_block_wires,
        block,
        n_params_block,
        template_weights=None,
        do_queue=True,
        id=None,
    ):

        self.ind_gates = compute_indices(wires, n_block_wires)
        n_wires = len(wires)
        shape = qml.math.shape(template_weights)  # (n_params_block, n_blocks)
        self.n_params_block = n_params_block
        self.n_blocks = 2 ** int(np.log2(n_wires / n_block_wires)) * 2 - 1
        self.block = block

        if shape == ():
            self.template_weights = np.random.rand(n_params_block, int(self.n_blocks))

        else:
            if shape[0] != self.n_blocks:
                raise ValueError(
                    f"Weights tensor must have first dimension of length {self.n_blocks}; got {shape[0]}"
                )
            if shape[-1] != self.n_params_block:
                raise ValueError(
                    f"Weights tensor must have last dimension of length {self.n_params_block}; got {shape[-1]}"
                )

        self.template_weights = template_weights

        super().__init__(template_weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):
        with qml.tape.QuantumTape() as tape:
            if self.block.__code__.co_argcount > 2:
                for idx, w in enumerate(self.ind_gates):
                    self.block(*self.template_weights[idx], wires=w)
            elif self.block.__code__.co_argcount == 2:
                for idx, w in enumerate(self.ind_gates):
                    self.block(self.template_weights[idx], wires=w)
            else:
                for idx, w in enumerate(self.ind_gates):
                    self.block(wires=w)

        return tape

    @staticmethod
    def get_n_blocks(wires, n_block_wires):
        """Returns the expected number of blocks for a set of wires and number of wires per block.
        Args:
            wires (Sequence): number of wires the template acts on
            n_block_wires (int): number of wires per block
        Returns:
            n_blocks (int): number of blocks; expected length of the template_weights argument
        """

        n_wires = len(wires)
        if not np.log2(n_wires / n_block_wires).is_integer():
            warnings.warn(
                f"The number of wires should be n_block_wires times 2^n; got n_wires/n_block_wires = {n_wires/n_block_wires}"
            )

        if n_block_wires > n_wires:
            raise ValueError(
                f"n_block_wires must be smaller than or equal to the number of wires; got n_block_wires = {n_block_wires} and number of wires = {n_wires}"
            )

        n_blocks = 2 ** int(np.log2(n_wires / n_block_wires)) * 2 - 1
        return n_blocks
