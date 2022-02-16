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
Contains the MPS template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import warnings
import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Operation, AnyWires


def compute_indices_MPS(wires, n_block_wires):
    """Generate a list containing the wires for each block.

    Args:
        wires (Iterable): wires that the template acts on
        n_block_wires (int): number of wires per block
    Returns:
        layers (array): array of wire indices or wire labels for each block
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
            f"n_block_wires must be smaller than or equal to the number of wires; got n_block_wires = {n_block_wires} and number of wires = {n_wires}"
        )

    if n_wires % (n_block_wires / 2) > 0:
        warnings.warn(
            f"The number of wires should be a multiple of {int(n_block_wires/2)}; got {n_wires}"
        )

    layers = np.array(
        [
            [wires[idx] for idx in range(j, j + n_block_wires)]
            for j in range(
                0,
                len(wires) - int(len(wires) % (n_block_wires // 2)) - n_block_wires // 2,
                n_block_wires // 2,
            )
        ]
    )
    return layers


class MPS(Operation):
    """The MPS template broadcasts an input circuit across many wires following the architecture of a Matrix Product State tensor network.
    The result is similar to the architecture in `arXiv:1803.11537 <https://arxiv.org/abs/1803.11537>`_.

    The argument ``block`` is a user-defined quantum circuit.``block`` should have two arguments: ``weights`` and ``wires``.
    For clarity, it is recommended to use a one-dimensional list or array for the block weights.

    Args:
        wires (Iterable): wires that the template acts on
        n_block_wires (int): number of wires per block
        block (Callable): quantum circuit that defines a block
        n_params_block (int): the number of parameters in a block; equal to the length of the ``weights`` argument in ``block``
        template_weights (Sequence): list containing the weights for all blocks

    .. note::

        The expected number of blocks can be obtained from ``qml.MPS.n_blocks(wires, n_block_wires)``.
        The length of ``template_weights`` argument should match the number of blocks.

    .. UsageDetails::

        This example demonstrates the use of ``MPS`` for a simple block.

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
            n_blocks = qml.MPS.get_n_blocks(range(n_wires),n_block_wires)
            template_weights = [[0.1,-0.3]]*n_blocks

            dev= qml.device('default.qubit',wires=range(n_wires))
            @qml.qnode(dev)
            def circuit(template_weights):
                qml.MPS(range(n_wires),n_block_wires,block, n_params_block, template_weights)
                return qml.expval(qml.PauliZ(wires=n_wires-1))

        >>> print(qml.draw(circuit,expansion_strategy='device')(template_weights))
        0: ──╭C──RY(0.1)───────────────────────────────┤
        1: ──╰X──RY(-0.3)──╭C──RY(0.1)─────────────────┤
        2: ────────────────╰X──RY(-0.3)──╭C──RY(0.1)───┤
        3: ──────────────────────────────╰X──RY(-0.3)──┤ ⟨Z⟩
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    num_wires = AnyWires
    par_domain = "A"

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
        ind_gates = compute_indices_MPS(wires, n_block_wires)
        n_wires = len(wires)
        n_blocks = int(n_wires / (n_block_wires / 2) - 1)

        if template_weights is None:
            template_weights = np.random.rand(n_params_block, int(n_blocks))

        else:
            shape = qml.math.shape(template_weights)[-4:]  # (n_params_block, n_blocks)
            if shape[0] != n_blocks:
                raise ValueError(
                    f"Weights tensor must have first dimension of length {n_blocks}; got {shape[0]}"
                )
            if shape[-1] != n_params_block:
                raise ValueError(
                    f"Weights tensor must have last dimension of length {n_params_block}; got {shape[-1]}"
                )

        self._hyperparameters = {"ind_gates": ind_gates, "block": block}
        super().__init__(template_weights, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(
        weights, wires, ind_gates, block
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.MPS.decomposition`.

        Args:
            weights (list[tensor_like]): list containing the weights for all blocks
            wires (Iterable): wires that the template acts on
            block (Callable): quantum circuit that defines a block
            ind_gates (array): array of wire indices

        Returns:
            list[.Operator]: decomposition of the operator
        """
        return [block(weights=weights[idx][:], wires=w.tolist()) for idx, w in enumerate(ind_gates)]

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
        if n_wires % (n_block_wires / 2) > 0:
            warnings.warn(
                f"The number of wires should be a multiple of {int(n_block_wires/2)}; got {n_wires}"
            )

        if n_block_wires > n_wires:
            raise ValueError(
                f"n_block_wires must be smaller than or equal to the number of wires; got n_block_wires = {n_block_wires} and number of wires = {n_wires}"
            )

        n_blocks = int(n_wires / (n_block_wires / 2) - 1)
        return n_blocks
