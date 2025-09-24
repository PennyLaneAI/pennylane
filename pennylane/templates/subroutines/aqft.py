# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This submodule contains the template for AQFT.
"""

import warnings

import numpy as np

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import SWAP, ControlledPhaseShift, Hadamard, PhaseShift, cond


class AQFT(Operation):
    r"""AQFT(order, wires)
    Apply an approximate quantum Fourier transform (AQFT).

    The `AQFT <https://arxiv.org/abs/1803.04933>`_ method helps to reduce the number of ``ControlledPhaseShift`` operations required
    for QFT by only using a maximum of ``order`` number of ``ControlledPhaseShift`` gates per qubit.

    .. seealso:: :class:`~.QFT`

    Args:
        order (int): the order of approximation
        wires (int or Iterable[Number, str]]): the wire(s) the operation acts on

    **Example**

    The approximate quantum Fourier transform is applied by specifying the corresponding wires and
    the order of approximation:

    .. code-block:: python

        wires = 3
        dev = qml.device('default.qubit', wires=wires)

        @qml.qnode(dev)
        def circuit_aqft():
            qml.X(0)
            qml.Hadamard(1)
            qml.AQFT(order=1,wires=range(wires))
            return qml.state()


    >>> circuit_aqft()
    array([ 0.5 +0.j  , -0.25-0.25j,  0.  +0.j  , -0.25+0.25j,  0.5 +0.j  ,
        -0.25-0.25j,  0.  +0.j  , -0.25+0.25j])


    .. details::
        :title: Usage Details

        **Order**

        The order of approximation must be a whole number less than :math:`n-1`
        where :math:`n` is the number of wires the operation is being applied on.
        This creates four cases for different ``order`` values:

        * ``order`` :math:`< 0`
            This will raise a ``ValueError``

        * ``order`` :math:`= 0`
            This will warn the user that only a Hadamard transform is being applied.

            .. code-block:: python

                @qml.qnode(qml.device('default.qubit'))
                def circ():
                    qml.AQFT(order=0, wires=range(6))
                    return qml.probs()

            The resulting circuit is:

            >>> print(qml.draw(circ, level='device')()) # doctest: +SKIP
            UserWarning: order=0, applying Hadamard transform warnings.warn("order=0, applying Hadamard transform")
            0: ──H─╭SWAP─────────────┤ ╭Probs
            1: ──H─│─────╭SWAP───────┤ ├Probs
            2: ──H─│─────│─────╭SWAP─┤ ├Probs
            3: ──H─│─────│─────╰SWAP─┤ ├Probs
            4: ──H─│─────╰SWAP───────┤ ├Probs
            5: ──H─╰SWAP─────────────┤ ╰Probs

        * :math:`0 <` ``order`` :math:`< n-1`
            This is the intended AQFT use case.

            .. code-block:: python

                @qml.qnode(qml.device('default.qubit'))
                def circ():
                    qml.AQFT(order=2, wires=range(4))
                    return qml.probs()

            The resulting circuit is:

            >>> print(qml.draw(circ, level='device')())
            0: ──H─╭Rϕ(1.57)─╭Rϕ(0.79)────────────────────────────────────────╭SWAP───────┤  Probs
            1: ────╰●────────│──────────H─╭Rϕ(1.57)─╭Rϕ(0.79)─────────────────│─────╭SWAP─┤  Probs
            2: ──────────────╰●───────────╰●────────│──────────H─╭Rϕ(1.57)────│─────╰SWAP─┤  Probs
            3: ─────────────────────────────────────╰●───────────╰●─────────H─╰SWAP───────┤  Probs

        * ``order`` :math:`\geq n-1`
            Using the QFT class is recommended in this case. The AQFT operation here is
            equivalent to QFT.

    """

    resource_keys = {"num_wires", "order"}

    def __init__(self, order, wires=None, id=None):
        n_wires = len(wires)

        if not isinstance(order, int):
            warnings.warn(f"The order must be an integer. Using order = {round(order)}")
            order = round(order)

        if order >= n_wires - 1:
            warnings.warn(
                f"The order ({order}) is >= to the number of wires - 1 ({n_wires-1}). Using the QFT class is recommended in this case."
            )
            order = n_wires - 1

        if order < 0:
            raise ValueError("Order can not be less than 0")

        if order == 0:
            warnings.warn("order=0, applying Hadamard transform")

        self.hyperparameters["order"] = order
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {"order": self.hyperparameters["order"], "num_wires": len(self.wires)}

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(wires, order):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.AQFT.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on
            order (int): order of approximation

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> qml.AQFT.compute_decomposition((0, 1, 2), order=1)
        [H(0), ControlledPhaseShift(1.57..., wires=Wires([1, 0])), H(1), ControlledPhaseShift(1.57..., wires=Wires([2, 1])), H(2), SWAP(wires=[0, 2])]

        """
        n_wires = len(wires)
        shifts = [2 * np.pi * 2**-i for i in range(2, n_wires + 1)]

        decomp_ops = []
        for i, wire in enumerate(wires):
            decomp_ops.append(Hadamard(wire))
            counter = 0

            for shift, control_wire in zip(shifts[: len(shifts) - i], wires[i + 1 :]):
                if counter >= order:
                    break

                op = ControlledPhaseShift(shift, wires=[control_wire, wire])
                decomp_ops.append(op)
                counter = counter + 1

        first_half_wires = wires[: n_wires // 2]
        last_half_wires = wires[-(n_wires // 2) :]

        for wire1, wire2 in zip(first_half_wires, reversed(last_half_wires)):
            swap = SWAP(wires=[wire1, wire2])
            decomp_ops.append(swap)

        return decomp_ops


def _AQFT_resources(num_wires, order):

    resources = {}

    resources[resource_rep(Hadamard)] = num_wires

    resources[
        controlled_resource_rep(
            PhaseShift,
            {},
            num_control_wires=1,
        )
    ] = sum(min(num_wires - 1 - i, order) for i in range(num_wires))

    resources[resource_rep(SWAP)] = num_wires // 2

    return dict(resources)


@register_resources(_AQFT_resources)
def _AQFT_decomposition(wires, order):
    n_wires = len(wires)
    shifts = [2 * np.pi * 2**-i for i in range(2, n_wires + 1)]

    if capture.enabled():
        shifts = math.array(shifts, like="jax")
        wires = math.array(wires, like="jax")

    @for_loop(len(wires))
    def wire_loop(i):
        wire = wires[i]
        Hadamard(wire)

        @for_loop(n_wires - 1 - i)
        def wires_limited_shift_loop(j):
            shift = shifts[j]
            control_wire = wires[i + 1 + j]

            ControlledPhaseShift(shift, wires=[control_wire, wire])

        @for_loop(order)
        def order_limited_shift_loop(j):
            shift = shifts[j]
            control_wire = wires[i + 1 + j]

            ControlledPhaseShift(shift, wires=[control_wire, wire])

        cond(n_wires - 1 - i < order, wires_limited_shift_loop, order_limited_shift_loop)()

    wire_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(len(wires) // 2)
    def half_wire_loop(k):
        wire1 = wires[k]
        wire2 = wires[-k - 1]
        SWAP(wires=[wire1, wire2])

    half_wire_loop()  # pylint: disable=no-value-for-parameter


add_decomps(AQFT, _AQFT_decomposition)
