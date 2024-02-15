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
from pennylane.operation import Operation, AnyWires


def compute_indices_MPS(wires, n_block_wires, offset=None):
    r"""Generate a list containing the wires for each block.

    Args:
        wires (Iterable): wires that the template acts on
        n_block_wires (int): number of wires per block_gen
        offset (int): offset value for positioning the subsequent blocks relative to each other.
            If ``None``, it defaults to :math:`\text{offset} = \lfloor \text{n_block_wires}/2  \rfloor`,
            otherwise :math:`\text{offset} \in [1, \text{n_block_wires} - 1]`.

    Returns:
        layers (Tuple[Tuple]]): array of wire indices or wire labels for each block
    """

    n_wires = len(wires)

    if n_block_wires < 2:
        raise ValueError(
            f"The number of wires in each block must be larger than or equal to 2; got n_block_wires = {n_block_wires}"
        )

    if n_block_wires > n_wires:
        raise ValueError(
            f"n_block_wires must be smaller than or equal to the number of wires; got n_block_wires = {n_block_wires} and number of wires = {n_wires}"
        )

    if offset is None:
        offset = n_block_wires // 2

    if offset < 1 or offset > n_block_wires - 1:
        raise ValueError(
            f"Provided offset is outside the expected range; the expected range for n_block_wires = {n_block_wires} is range{1, n_block_wires - 1}"
        )

    n_step = offset
    n_layers = len(wires) - int(len(wires) % (n_block_wires // 2)) - n_step

    return tuple(
        tuple(wires[idx] for idx in range(j, j + n_block_wires))
        for j in range(
            0,
            n_layers,
            n_step,
        )
        if not j + n_block_wires > len(wires)
    )


class MPS(Operation):
    r"""The MPS template broadcasts an input circuit across many wires following the architecture of a Matrix Product State tensor network.
    The result is similar to the architecture in `arXiv:1803.11537 <https://arxiv.org/abs/1803.11537>`_.

    The keyword argument ``block`` is a user-defined quantum circuit that should accept two arguments: ``wires`` and ``weights``.
    The latter argument is optional in case the implementation of ``block`` doesn't require any weights. Any additional arguments
    should be provided using the ``kwargs``.

    Args:
        wires (Iterable): wires that the template acts on
        n_block_wires (int): number of wires per block
        block (Callable): quantum circuit that defines a block
        n_params_block (int): the number of parameters in a block; equal to the length of the ``weights`` argument in ``block``
        template_weights (Sequence): list containing the weights for all blocks
        offset (int): offset value for positioning the subsequent blocks relative to each other.
            If ``None``, it defaults to :math:`\text{offset} = \lfloor \text{n_block_wires}/2  \rfloor`,
            otherwise :math:`\text{offset} \in [1, \text{n_block_wires} - 1]`
        **kwargs: additional keyword arguments for implementing the ``block``

    .. note::

        The expected number of blocks can be obtained from ``qml.MPS.get_n_blocks(wires, n_block_wires, offset=0)``, and
        the length of ``template_weights`` argument should match the number of blocks. Whenever either ``n_block_wires``
        is odd or ``offset`` is not :math:`\lfloor \text{n_block_wires}/2  \rfloor`, the template deviates from the maximally
        unbalanced tree architecture described in `arXiv:1803.11537 <https://arxiv.org/abs/1803.11537>`_.

    .. details::
        :title: Usage Details

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
            template_weights = [[0.1, -0.3]] * n_blocks

            dev= qml.device('default.qubit',wires=range(n_wires))
            @qml.qnode(dev)
            def circuit(template_weights):
                qml.MPS(range(n_wires),n_block_wires,block, n_params_block, template_weights)
                return qml.expval(qml.Z(n_wires-1))

        >>> print(qml.draw(circuit, expansion_strategy='device')(template_weights))
        0: ─╭●──RY(0.10)──────────────────────────────┤
        1: ─╰X──RY(-0.30)─╭●──RY(0.10)────────────────┤
        2: ───────────────╰X──RY(-0.30)─╭●──RY(0.10)──┤
        3: ─────────────────────────────╰X──RY(-0.30)─┤  <Z>

        MPS can also be used with an ``offset`` argument that shifts the positioning the subsequent blocks from the default ``n_block_wires/2``.

        .. code-block:: python

            import pennylane as qml
            import numpy as np

            def block(wires):
                qml.MultiControlledX(wires=[wires[i] for i in range(len(wires))])

            n_wires = 8
            n_block_wires = 4
            n_params_block = 2

            dev= qml.device('default.qubit',wires=n_wires)
            @qml.qnode(dev)
            def circuit():
                qml.MPS(range(n_wires),n_block_wires, block, n_params_block, offset = 1)
                return qml.state()

        >>> print(qml.draw(circuit, expansion_strategy='device')())
        0: ─╭●─────────────┤  State
        1: ─├●─╭●──────────┤  State
        2: ─├●─├●─╭●───────┤  State
        3: ─╰X─├●─├●─╭●────┤  State
        4: ────╰X─├●─├●─╭●─┤  State
        5: ───────╰X─├●─├●─┤  State
        6: ──────────╰X─├●─┤  State
        7: ─────────────╰X─┤  State

    """

    num_wires = AnyWires
    par_domain = "A"

    @classmethod
    def _unflatten(cls, data, metadata):
        new_op = cls.__new__(cls)
        setattr(new_op, "_hyperparameters", dict(metadata[1]))
        setattr(new_op, "_weights", data[0] if len(data) else None)
        Operation.__init__(new_op, *data, wires=metadata[0])
        return new_op

    def __init__(
        self,
        wires,
        n_block_wires,
        block,
        n_params_block=0,
        template_weights=None,
        offset=None,
        id=None,
        **kwargs,
    ):
        ind_gates = compute_indices_MPS(wires, n_block_wires, offset)
        n_blocks = self.get_n_blocks(wires, n_block_wires, offset)

        if template_weights is not None:
            shape = qml.math.shape(template_weights)  # (n_blocks, n_params_block)
            if shape[0] != n_blocks:
                raise ValueError(
                    f"Weights tensor must have first dimension of length {n_blocks}; got {shape[0]}"
                )
            if shape[-1] != n_params_block:
                raise ValueError(
                    f"Weights tensor must have last dimension of length {n_params_block}; got {shape[-1]}"
                )

        self._weights = template_weights
        self._hyperparameters = {"ind_gates": ind_gates, "block": block, **kwargs}

        if self._weights is None:
            super().__init__(wires=wires, id=id)
        else:
            super().__init__(self._weights, wires=wires, id=id)

    @property
    def num_params(self):
        """int: Number of trainable parameters that the operator depends on."""
        return 0 if self._weights is None else 1

    @staticmethod
    def compute_decomposition(
        weights=None, wires=None, ind_gates=None, block=None, **kwargs
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.MPS.decomposition`.

        Args:
            weights (list[tensor_like]): list containing the weights for all blocks
            wires (Iterable): wires that the template acts on
            block (Callable): quantum circuit that defines a block
            ind_gates (array): array of wire indices
            **kwargs: additional keyword arguments for implementing the ``block``

        Returns:
            list[.Operator]: decomposition of the operator
        """
        decomp = []
        itrweights = iter([]) if weights is None else iter(weights)
        block_gen = qml.tape.make_qscript(block)
        for w in ind_gates:
            weight = next(itrweights, None)
            decomp += (
                block_gen(wires=w, **kwargs)
                if weight is None
                else block_gen(weights=weight, wires=w, **kwargs)
            )
        return [qml.apply(op) for op in decomp] if qml.QueuingManager.recording() else decomp

    @staticmethod
    def get_n_blocks(wires, n_block_wires, offset=None):
        r"""Returns the expected number of blocks for a set of wires and number of wires per block.

        Args:
            wires (Sequence): number of wires the template acts on
            n_block_wires (int): number of wires per block
            offset (int): offset value for positioning the subsequent blocks relative to each other.
                If ``None``, it defaults to :math:`\text{offset} = \lfloor \text{n_block_wires}/2  \rfloor`,
                otherwise :math:`\text{offset} \in [1, \text{n_block_wires} - 1]`.

        Returns:
            n_blocks (int): number of blocks; expected length of the template_weights argument
        """
        n_wires = len(wires)

        if offset is None and not n_block_wires % 2 and n_wires % (n_block_wires // 2) > 0:
            warnings.warn(
                f"The number of wires should be a multiple of {int(n_block_wires/2)}; got {n_wires}"
            )

        if n_block_wires > n_wires:
            raise ValueError(
                f"n_block_wires must be smaller than or equal to the number of wires; got n_block_wires = {n_block_wires} and number of wires = {n_wires}"
            )

        if offset is None:
            offset = n_block_wires // 2

        if offset < 1 or offset > n_block_wires - 1:
            raise ValueError(
                f"Provided offset is outside the expected range; the expected range for n_block_wires = {n_block_wires} is range{1, n_block_wires - 1}"
            )

        n_step = offset
        n_layers = n_wires - int(n_wires % (n_block_wires // 2)) - n_step

        return len([idx for idx in range(0, n_layers, n_step) if not idx + n_block_wires > n_wires])
