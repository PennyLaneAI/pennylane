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

import pennylane as qml
from pennylane.operation import Operation


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

    .. code-block::

        wires = 3
        dev = qml.device('default.qubit', wires=wires)

        @qml.qnode(dev)
        def circuit_aqft():
            qml.X(0)
            qml.Hadamard(1)
            qml.AQFT(order=1,wires=range(wires))
            return qml.state()


    .. code-block:: pycon

        >>> circuit_aqft()
        [ 0.5 +0.j   -0.25-0.25j  0.  +0.j   -0.25+0.25j  0.5 +0.j   -0.25-0.25j   0.  +0.j   -0.25+0.25j]


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

            .. code-block::

                @qml.qnode(dev)
                def circ():
                    qml.AQFT(order=0, wires=range(6))
                    return qml.probs()

            The resulting circuit is:

            >>> print(qml.draw(circ, expansion_strategy='device')())
            UserWarning: order=0, applying Hadamard transform warnings.warn("order=0, applying Hadamard transform")
            0: ──H─╭SWAP─────────────┤ ╭Probs
            1: ──H─│─────╭SWAP───────┤ ├Probs
            2: ──H─│─────│─────╭SWAP─┤ ├Probs
            3: ──H─│─────│─────╰SWAP─┤ ├Probs
            4: ──H─│─────╰SWAP───────┤ ├Probs
            5: ──H─╰SWAP─────────────┤ ╰Probs

        * :math:`0 <` ``order`` :math:`< n-1`
            This is the intended AQFT use case.

            .. code-block::

                @qml.qnode(dev)
                def circ():
                    qml.AQFT(order=2, wires=range(4))
                    return qml.probs()

            The resulting circuit is:

            >>> print(qml.draw(circ, expansion_strategy='device')())
            0: ──H─╭Rϕ(1.57)─╭Rϕ(0.79)────────────────────────────────────────╭SWAP───────┤ ╭Probs
            1: ────╰●────────│──────────H─╭Rϕ(1.57)─╭Rϕ(0.79)─────────────────│─────╭SWAP─┤ ├Probs
            2: ──────────────╰●───────────╰●────────│──────────H─╭Rϕ(1.57)────│─────╰SWAP─┤ ├Probs
            3: ─────────────────────────────────────╰●───────────╰●─────────H─╰SWAP───────┤ ╰Probs

        * ``order`` :math:`\geq n-1`
            Using the QFT class is recommended in this case. The AQFT operation here is
            equivalent to QFT.

    """

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

        >>> qml.AQFT.compute_decomposition((0, 1, 2), 3, order=1)
        [Hadamard(wires=[0]), ControlledPhaseShift(1.5707963267948966, wires=[1, 0]), Hadamard(wires=[1]), ControlledPhaseShift(1.5707963267948966, wires=[2, 1]), Hadamard(wires=[2]), SWAP(wires=[0, 2])]

        """
        n_wires = len(wires)
        shifts = [2 * np.pi * 2**-i for i in range(2, n_wires + 1)]

        decomp_ops = []
        for i, wire in enumerate(wires):
            decomp_ops.append(qml.Hadamard(wire))
            counter = 0

            for shift, control_wire in zip(shifts[: len(shifts) - i], wires[i + 1 :]):
                if counter >= order:
                    break

                op = qml.ControlledPhaseShift(shift, wires=[control_wire, wire])
                decomp_ops.append(op)
                counter = counter + 1

        first_half_wires = wires[: n_wires // 2]
        last_half_wires = wires[-(n_wires // 2) :]

        for wire1, wire2 in zip(first_half_wires, reversed(last_half_wires)):
            swap = qml.SWAP(wires=[wire1, wire2])
            decomp_ops.append(swap)

        return decomp_ops
