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
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


def compute_indices(wires, n_block_wires):
    """Generate a list containing the wires for each block.

    Args:
        wires (Iterable): wires that the template acts on
        n_block_wires (int): number of wires per block

    Returns:
        layers (tuple): array of wire labels for each block
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

    if not np.log2(n_wires / n_block_wires).is_integer():  # pylint:disable=no-member
        warnings.warn(
            f"The number of wires should be n_block_wires times 2^n; got n_wires/n_block_wires = {n_wires/n_block_wires}"
        )

    n_wires = 2 ** (int(np.log2(len(wires) / n_block_wires))) * n_block_wires
    n_layers = int(np.log2(n_wires // n_block_wires)) + 1

    half_block_wires = n_block_wires // 2

    block_wires = []
    for layer in range(n_layers):
        lower_shift = (2 ** (layer) - 1) * half_block_wires
        upper_shift = (2 ** (layer + 1) - 1) * half_block_wires

        step = 2**layer * n_block_wires
        for block_offset in range(0, n_wires - half_block_wires, step):
            wires1 = tuple(wires[block_offset + lower_shift + i] for i in range(half_block_wires))
            wires2 = tuple(wires[block_offset + upper_shift + i] for i in range(half_block_wires))
            block_wires.append(wires1 + wires2)

    return tuple(block_wires)


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

    .. details::
        :title: Usage Details

        In general, the block takes D parameters and **must** have the following signature:

        .. code-block:: python

            unitary(parameter1, parameter2, ... parameterD, wires)

        For a block with multiple parameters, ``n_params_block`` is equal to the number of parameters in ``block``.
        For a block with a single parameter, ``n_params_block`` is equal to the length of the parameter.

        To avoid using ragged arrays, all block parameters should have the same dimension.

        The length of the ``template_weights`` argument should match the number of blocks.
        The expected number of blocks can be obtained from ``qml.TTN.get_n_blocks(wires, n_block_wires)``.

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
                return qml.expval(qml.Z(n_wires-1))

        >>> print(qml.draw(circuit, expansion_strategy='device')(template_weights))
        0: ─╭●──RY(0.10)────────────────┤
        1: ─╰X──RY(-0.30)─╭●──RY(0.10)──┤
        2: ─╭●──RY(0.10)──│─────────────┤
        3: ─╰X──RY(-0.30)─╰X──RY(-0.30)─┤  <Z>

    """

    num_wires = AnyWires
    grad_method = None

    @property
    def num_params(self):
        return 1

    @classmethod
    def _unflatten(cls, data, metadata):
        new_op = cls.__new__(cls)
        new_op._hyperparameters = dict(metadata[1])
        Operation.__init__(new_op, data, wires=metadata[0])
        return new_op

    def __init__(
        self,
        wires,
        n_block_wires,
        block,
        n_params_block,
        template_weights=None,
        id=None,
    ):
        ind_gates = compute_indices(wires, n_block_wires)
        n_wires = len(wires)
        shape = qml.math.shape(template_weights)  # (n_params_block, n_blocks)
        n_blocks = 2 ** int(np.log2(n_wires / n_block_wires)) * 2 - 1

        if shape == ():
            template_weights = np.random.rand(n_params_block, int(n_blocks))

        else:
            if shape[0] != n_blocks:
                raise ValueError(
                    f"Weights tensor must have first dimension of length {n_blocks}; got {shape[0]}"
                )
            if shape[-1] != n_params_block:
                raise ValueError(
                    f"Weights tensor must have last dimension of length {n_params_block}; got {shape[-1]}"
                )

        self._hyperparameters = {"ind_gates": ind_gates, "block": block}

        super().__init__(template_weights, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(
        weights, wires, block, ind_gates
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.TTN.decomposition`.

        Args:
            weights (list[tensor_like]): list containing the weights for all blocks
            wires (Iterable): wires that the template acts on
            block (Callable): quantum circuit that defines a block
            ind_gates (array): array of wire indices

        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []
        block_gen = qml.tape.make_qscript(block)
        if block.__code__.co_argcount > 2:
            for idx, w in enumerate(ind_gates):
                op_list += block_gen(*weights[idx], wires=w)
        elif block.__code__.co_argcount == 2:
            for idx, w in enumerate(ind_gates):
                op_list += block_gen(weights[idx], wires=w)
        else:
            for w in ind_gates:
                op_list += block_gen(wires=w)

        return [qml.apply(op) for op in op_list] if qml.QueuingManager.recording() else op_list

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
        if not np.log2(n_wires / n_block_wires).is_integer():  # pylint:disable=no-member
            warnings.warn(
                f"The number of wires should be n_block_wires times 2^n; got n_wires/n_block_wires = {n_wires/n_block_wires}"
            )

        if n_block_wires > n_wires:
            raise ValueError(
                f"n_block_wires must be smaller than or equal to the number of wires; got n_block_wires = {n_block_wires} and number of wires = {n_wires}"
            )

        n_blocks = 2 ** int(np.log2(n_wires / n_block_wires)) * 2 - 1
        return n_blocks
