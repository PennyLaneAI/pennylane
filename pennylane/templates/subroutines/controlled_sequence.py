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
r"""
Contains the ControlledSequence template.
"""
from copy import copy
import pennylane as qml

from pennylane.operation import Operation
from pennylane.wires import Wires
from pennylane.ops.op_math.symbolicop import SymbolicOp


class ControlledSequence(SymbolicOp, Operation):
    r"""Creates a sequence of controlled gates raised to decreasing powers of 2. Can be used as
    a sub-block in building a `quantum phase estimation <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`__
    circuit.

    Given an :class:`~.Operator` and a list of control wires, this template creates a sequence of
    controlled gates, one for each control wire, with the base :class:`~.Operator` raised to
    decreasing powers of 2:

    .. figure:: ../../_static/templates/subroutines/big_ctrl.png
        :align: center
        :width: 40%
        :target: javascript:void(0);

    Args:
        base (Operator): the phase estimation unitary, specified as an :class:`~.Operator`
        control (Union[Wires, Sequence[int], or int]): the wires to be used for control

    Raises:
        ValueError: if the wires in ``control`` and wires on the ``base`` operator share a common
            element

    .. seealso:: :class:`~.QuantumPhaseEstimation`

    **Example**

    .. code-block:: python

        dev = qml.device("default.qubit", wires = 4)

        @qml.qnode(dev)
        def circuit():

            for i in range(3):
                qml.Hadamard(wires = i)

            qml.ControlledSequence(qml.RX(0.25, wires = 3), control = [0, 1, 2])

            qml.adjoint(qml.QFT)(wires = range(3))

            return qml.probs(wires = range(3))

    >>> print(circuit())
    [0.92059345 0.02637178 0.00729619 0.00423258 0.00360545 0.00423258 0.00729619 0.02637178]

    """

    def _flatten(self):
        return (self.base,), (self.control,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], control=metadata[0])

    def __init__(self, base, control, id=None):
        control_wires = Wires(control)

        if len(Wires.shared_wires([base.wires, control_wires])) != 0:
            raise ValueError("The control wires must be different from the base operation wires.")

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["base"] = base

        self._name = "ControlledSequence"

        super().__init__(base, id=id)

    @property
    def hash(self):
        return hash(
            (
                str(self.name),
                self.control,
                self.base.hash,
            )
        )

    @property
    def control(self):
        """The control wires for the sequence"""
        return self.hyperparameters["control_wires"]

    @property
    def control_wires(self):
        """The control wires for the sequence"""
        return self.hyperparameters["control_wires"]

    @property
    def wires(self):
        return self.control + self.base.wires

    @property
    def has_matrix(self):
        return False

    def __repr__(self):
        return f"ControlledSequence({self.base}, control={list(self.control)})"

    def map_wires(self, wire_map: dict):
        # pylint:disable=protected-access
        new_op = copy(self)
        new_op.hyperparameters["base"] = self.base.map_wires(wire_map=wire_map)
        new_op.hyperparameters["control_wires"] = Wires(
            [wire_map.get(wire, wire) for wire in self.control]
        )
        return new_op

    # pylint:disable=arguments-differ
    @staticmethod
    def compute_decomposition(*_, base=None, control_wires=None, lazy=False, **__):
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.CtrlSequence.decomposition`.

        Args:
            base (Operator): the operator that acts as the base for the sequence
            control_wires (Any or Iterable[Any]): the control wires for the sequence

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        .. code-block:: python

            dev = qml.device("default.qubit")
            op = qml.ControlledSequence(qml.RX(0.25, wires = 3), control = [0, 1, 2])

            @qml.qnode(dev)
            def circuit():
                op.decomposition()
                return qml.state()

        >>> print(qml.draw(circuit, wire_order=[0,1,2,3])())
        0: ─╭●─────────────────────────────────────┤  State
        1: ─│────────────╭●────────────────────────┤  State
        2: ─│────────────│────────────╭●───────────┤  State
        3: ─╰(RX(1.00))──╰(RX(0.50))──╰(RX(0.25))──┤  State

        To display the operators as powers of the base operator without further simplifcation,
        the `compute_decompostion` method can be used with `lazy=True`.

        .. code-block:: python

            dev = qml.device("default.qubit")
            op = qml.ControlledSequence(qml.RX(0.25, wires = 3), control = [0, 1, 2])

            @qml.qnode(dev)
            def circuit():
                op.compute_decomposition(base=op.base, control_wires=op.control, lazy=True)
                return qml.state()

        >>> print(qml.draw(circuit, wire_order=[0,1,2,3])())
        0: ─╭●─────────────────────────────────────┤  State
        1: ─│────────────╭●────────────────────────┤  State
        2: ─│────────────│────────────╭●───────────┤  State
        3: ─╰(RX(0.25))⁴─╰(RX(0.25))²─╰(RX(0.25))¹─┤  State

        """

        powers_of_two = [2**i for i in range(len(control_wires))]
        ops = []

        for z, ctrl_wire in zip(powers_of_two[::-1], control_wires):
            ops.append(qml.pow(qml.ctrl(base, control=ctrl_wire), z=z, lazy=lazy))

        return ops
