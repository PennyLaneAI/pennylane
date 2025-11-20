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

from pennylane.control_flow import for_loop
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    pow_resource_rep,
    register_resources,
)
from pennylane.operation import Operation
from pennylane.ops.op_math import SymbolicOp, ctrl
from pennylane.ops.op_math import pow as qml_pow
from pennylane.wires import Wires


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

    >>> print(circuit()) # doctest: +SKIP
    [0.9206 0.0264 0.0073 0.0042 0.0036 0.0042 0.0073 0.0264]

    """

    grad_method = None

    resource_keys = {"base_class", "base_params", "num_control_wires"}

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
    def resource_params(self) -> dict:
        params = {
            "base_class": self.hyperparameters["base"].__class__,
            "base_params": self.hyperparameters["base"].resource_params,
            "num_control_wires": len(self.hyperparameters["control_wires"]),
        }
        return params

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

        new_op = copy(self)
        new_op.hyperparameters["base"] = self.base.map_wires(wire_map=wire_map)
        new_op.hyperparameters["control_wires"] = Wires(
            [wire_map.get(wire, wire) for wire in self.control]
        )
        return new_op

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
        0: ─╭●────────────────────────────┤  State
        1: ─│─────────╭●──────────────────┤  State
        2: ─│─────────│─────────╭●────────┤  State
        3: ─╰RX(1.00)─╰RX(0.50)─╰RX(0.25)─┤  State

        To display the operators as powers of the base operator without further simplification,
        the `compute_decomposition` method can be used with `lazy=True`.

        .. code-block:: python

            dev = qml.device("default.qubit")
            op = qml.ControlledSequence(qml.RX(0.25, wires = 3), control = [0, 1, 2])

            @qml.qnode(dev)
            def circuit():
                op.compute_decomposition(base=op.base, control_wires=op.control, lazy=True)
                return qml.state()

        >>> print(qml.draw(circuit, wire_order=[0,1,2,3])())
        0: ─╭(RX(0.25))⁴───────────────────────────┤  State
        1: ─│────────────╭(RX(0.25))²──────────────┤  State
        2: ─│────────────│────────────╭(RX(0.25))¹─┤  State
        3: ─╰(RX(0.25))⁴─╰(RX(0.25))²─╰(RX(0.25))¹─┤  State

        """

        powers_of_two = [2**i for i in range(len(control_wires))]
        ops = []

        for z, ctrl_wire in zip(powers_of_two[::-1], control_wires):
            ops.append(qml_pow(ctrl(base, control=ctrl_wire), z=z, lazy=lazy))

        return ops


def _ctrl_seq_decomposition_resources(base_class, base_params, num_control_wires) -> dict:

    resources = {}

    powers_of_two = [2**i for i in range(num_control_wires)]

    for z in powers_of_two[::-1]:
        controlled_rep = controlled_resource_rep(base_class, base_params, 1)
        rep = pow_resource_rep(
            base_class=controlled_rep.op_type,
            base_params=controlled_rep.params,
            z=z,
        )
        resources[rep] = 1
    return resources


# pylint: disable=no-value-for-parameter
@register_resources(_ctrl_seq_decomposition_resources)
def _ctrl_seq_decomposition(*_, base=None, control_wires=None, **__):
    powers_of_two = [2**i for i in range(len(control_wires))]

    @for_loop(len(powers_of_two) - 1, -1, -1)
    def _powers_loop(i):
        j = len(powers_of_two) - 1 - i
        ctrl_wire = control_wires[j]
        z = powers_of_two[i]
        qml_pow(ctrl(base, control=ctrl_wire), z=z)

    _powers_loop()


add_decomps(ControlledSequence, _ctrl_seq_decomposition)
