# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the LeftQuantumComparator template for performing inequality test of two quantum registers."""

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, X
from pennylane.typing import AbstractWires, Wire
from pennylane.wires import Wires, WiresLike

from .temporary_and import Elbow


class LeftQuantumComparator(Operator2):
    r"""Perform an inequality test :math:`\lvert x\rangle\lvert y\rangle\lvert 0\rangle \mapsto \lvert x\rangle \lvert y\rangle\lvert x \leq y\rangle` between two states in separate quantum registers.

    This operator performs an inequality test between two quantum registers :math:`x` and
    :math:`y`, storing the result in a zeroed target qubit. The
    ``comparator`` argument can be one of four possible string values ``"<", "<=", ">", ">="`` to determine the type of inequality test. For example, choosing ``comparator="<"`` we have the following operation:

    .. math::

        \text{LeftQuantumComparator}_{<} \lvert x\rangle \lvert y\rangle \lvert 0\rangle = \lvert x\rangle \lvert y\rangle \lvert x < y\rangle

    The decomposition is defined as the left block in Figure 6 in Appendix E
    of `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_. Note that the decomposition uses auxiliary wires
    and in order to clean them, we must apply the adjoint of this operator via ``Adjoint(LeftQuantumComparator)``
    after using the target qubit, as shown in the example below.

    Args:
        x_wires (WiresLike): The wires that store the integer :math:`x`.
        y_wires (WiresLike): The wires that store the integer :math:`y`. The number of ``y_wires`` should be equal to
            the number of ``x_wires``.
        target_wire (WiresLike): The zeroed target wire that outputs the value of the inequality test.
        work_wires (WiresLike): The auxiliary wires to use for the addition.
            At least ``len(y_wires) - 1`` zeroed work wires should be provided. They are not returned in the zero state.
        comparator (str): The operator used in the inequality. The value could be '<', '<=', '>=' and '>'.

    **Example**

    In this example, we will use the ``LeftQuantumComparator``, generating the output on wire :math:`11`. After this,
    we will copy the result to wire :math:`12` using a ``CNOT`` gate, and then apply the ``adjoint(LeftQuantumComparator)``
    to clean up the auxiliary qubits used.

    .. code-block:: python

        import pennylane as qp

        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            x_wires = [0, 3, 6, 9]
            y_wires = [1, 4, 7, 10]
            work_wires = [2, 5, 8]
            qp.BasisState(qp.math.int_to_binary(3, 4), wires=x_wires)
            qp.BasisState(qp.math.int_to_binary(2, 4), wires=y_wires)
            qp.LeftQuantumComparator(x_wires, y_wires, 11, work_wires, ">=")
            qp.CNOT(wires=[11, 12])
            qp.adjoint(qp.LeftQuantumComparator(x_wires, y_wires, 11, work_wires, ">="))
            return qp.probs(wires=[12])

    >>> print(circuit())
    [0. 1.]
    """

    wire_argnames = ("x_wires", "y_wires", "target_wire", "work_wires")
    compilable_argnames = ("comparator",)
    arg_specs = {
        "x_wires": Wire[-1],
        "y_wires": Wire[-1],
        "target_wire": Wire[1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        target_wire: WiresLike,
        work_wires: WiresLike,
        comparator: str,
    ):  # pylint: disable=too-many-arguments

        if comparator not in ["<", "<=", ">=", ">"]:
            raise ValueError("Allowed values for 'comparator' are: '<', '<=', '>=' and '>'.")

        if isinstance(x_wires, AbstractWires):
            super().__init__(x_wires, y_wires, target_wire, work_wires, comparator)
            return

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        target_wire = Wires(target_wire)
        work_wires = Wires(work_wires)

        if len(work_wires) < len(y_wires) - 1:
            raise ValueError(f"At least {len(y_wires)-1} work_wires should be provided.")
        if len(x_wires) != len(y_wires):
            raise ValueError("The number of y_wires should be equal to the number of x_wires")
        if work_wires.intersection(target_wire):
            raise ValueError("None of the wires in work_wires should be the target wire.")
        if work_wires.intersection(x_wires):
            raise ValueError("None of the wires in work_wires should be included in x_wires.")
        if work_wires.intersection(y_wires):
            raise ValueError("None of the wires in work_wires should be included in y_wires.")
        if x_wires.intersection(target_wire):
            raise ValueError("None of the wires in x_wires should be the target wire.")
        if x_wires.intersection(y_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if y_wires.intersection(target_wire):
            raise ValueError("None of the wires in y_wires should be the target wire.")

        super().__init__(x_wires, y_wires, target_wire, work_wires, comparator)


def _left_quantum_comparator_resources(x_wires, y_wires, target_wire, work_wires, comparator):
    # pylint: disable=unused-argument
    num_y_wires = len(y_wires)
    resources = {
        Elbow: num_y_wires,
        CNOT: 2 + 5 * (num_y_wires - 1),
    }

    if comparator in [">=", "<="]:
        resources[X] = 1

    return resources


@register_resources(_left_quantum_comparator_resources, exact=True)
def _left_quantum_comparator(
    x_wires, y_wires, target_wire, work_wires, comparator, **_
):  # pylint: disable=too-many-arguments

    # revert to follow PL convention
    x_wires = x_wires[::-1]
    y_wires = y_wires[::-1]

    if comparator in ("<", ">="):
        x_wires, y_wires = y_wires, x_wires

    used_work_wires = Wires.all_wires([work_wires[: len(x_wires) - 1], target_wire])

    CNOT(wires=[x_wires[0], y_wires[0]])
    Elbow(wires=[x_wires[0], y_wires[0], used_work_wires[0]])
    CNOT(wires=[x_wires[0], y_wires[0]])

    if compiler.active() or capture.enabled():
        x_wires = math.array(x_wires, like="jax")
        y_wires = math.array(y_wires, like="jax")
        used_work_wires = math.array(used_work_wires, like="jax")

    # pylint: disable=no-value-for-parameter
    @for_loop(1, len(x_wires))
    def _loop(i):
        CNOT(wires=[x_wires[i], y_wires[i]])
        CNOT(wires=[x_wires[i], used_work_wires[i - 1]])
        Elbow(wires=[used_work_wires[i - 1], y_wires[i], used_work_wires[i]])
        CNOT(wires=[x_wires[i], used_work_wires[i - 1]])
        CNOT(wires=[used_work_wires[i - 1], used_work_wires[i]])
        CNOT(wires=[x_wires[i], y_wires[i]])

    _loop()

    if comparator in ("<=", ">="):
        X(target_wire)


add_decomps(LeftQuantumComparator, _left_quantum_comparator)
