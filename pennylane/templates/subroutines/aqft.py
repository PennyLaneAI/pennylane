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
This submodule contains the template for AQFT.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access

import warnings
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation


class AQFT(Operation):
    r"""AQFT(order, wires)
    Apply an approximate quantum Fourier transform (AQFT).

    The AQFT operation helps to reduce the number of `ControlledPhaseShift` operations required
    for QFT by only using a maximum of `order` number of `ControlledPhaseShift` gates per qubit.

    .. seealso:: :meth:`~.QFT`.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

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
        def circuit_aqft(basis_state):
            qml.BasisState(basis_state, wires=range(wires))
            qml.AQFT(order=1,wires=range(wires))
            return qml.state()


        circuit_aqft(np.array([1.0, 0.0, 0.0], requires_grad=False))

    .. details::
        :title: Usage Details

        **Order**

        The order of approximation must be a whole number less than :math:`n-1`
        where :math:`n` is the number of wires the operation is being applied on.
        This creates four cases for different `order` values:

        * `order` :math:`< 0`
            This will raise a `ValueError`

        * `order` :math:`= 0`
            This will warn the user that only a Hadamard transform is being applied.

            .. code-block::

                @qml.qnode(dev)
                def circ():
                    qml.AQFT(order=0, range(6))
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

        * :math:`0 < ``order` :math:`< n-1`
            This is the intended AQFT use case.

            .. code-block::

                @qml.qnode(dev)
                def circ():
                    qml.AQFT(order=2, range(4))
                    return qml.probs()

            The resulting circuit is:

            >>> print(qml.draw(circ, expansion_strategy='device')())
            0: ──H─╭Rϕ(1.57)─╭Rϕ(0.79)────────────────────────────────────────╭SWAP───────┤ ╭Probs
            1: ────╰●────────│──────────H─╭Rϕ(1.57)─╭Rϕ(0.79)─────────────────│─────╭SWAP─┤ ├Probs
            2: ──────────────╰●───────────╰●────────│──────────H─╭Rϕ(1.57)────│─────╰SWAP─┤ ├Probs
            3: ─────────────────────────────────────╰●───────────╰●─────────H─╰SWAP───────┤ ╰Probs

        * `order` :math:`>= n-1`
            Using the QFT class is recommended in this case. The AQFT operation here is
            equivalent to QFT.

    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, order, wires=None, id=None):
        wires = qml.wires.Wires(wires)

        self.hyperparameters["n_wires"] = len(wires)

        if not isinstance(order, int):
            warnings.warn(f"The order must be an integer. Using order = {round(order)}")
            order = round(order)

        if order >= self.hyperparameters["n_wires"] - 1:
            warnings.warn(
                f'The order ({order}) is >= to the number of wires - 1 ({self.hyperparameters["n_wires"]-1}). Using the QFT class is recommended in this case.'
            )
            order = self.hyperparameters["n_wires"] - 1

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
    def compute_decomposition(
        wires, n_wires, order
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.AQFT.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on
            n_wires (int): number of wires or ``len(wires)``
            order (int): order of approximation

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> qml.AQFT.compute_decomposition((0, 1, 2), 3, order=1)
        [Hadamard(wires=[0]), ControlledPhaseShift(1.5707963267948966, wires=[1, 0]), Hadamard(wires=[1]), ControlledPhaseShift(1.5707963267948966, wires=[2, 1]), Hadamard(wires=[2]), SWAP(wires=[0, 2])]

        """
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
