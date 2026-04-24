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

from pennylane import capture, compiler, for_loop, math
from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.operation import Operation
from pennylane.ops import CNOT, X
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.templates.subroutines import Elbow
from pennylane.wires import Wires, WiresLike


class LeftQuantumComparator(Operation):
    r"""Perform an inequality test :math:`|x\rangle|y\rangle|0\rangle \mapsto |x\rangle |y\rangle|x \leq y\rangle` between two states in separate quantum registers.
    
    This operator performs an inequality test between two quantum registers :math:`x` and
    :math:`y`, storing the result in a zeroed target qubit. The
    ``comparator`` argument can be one of four possible string values ``"<", "<=", ">", ">="`` to determine the type of inequality test. For example, choosing ``comparator="<"`` we have the following operation:

    .. math::

        \text{LeftQuantumComparator}_{<} |x\rangle |y\rangle |0\rangle = |x\rangle |y\rangle |x < y\rangle

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


    .. code-block:: python

        import pennylane as qp
        from pennylane.labs.templates import LeftQuantumComparator

        dev = qp.device("lightning.qubit")

        @qp.qnode(dev, shots=1)
        def circuit(a, comparator, b):

            x_wires = [0, 3, 6, 9]
            y_wires = [1, 4, 7, 10]
            work_wires = [2, 5, 8]
            qp.BasisState(a, wires=x_wires)
            qp.BasisState(b, wires=y_wires)
            LeftQuantumComparator(x_wires, y_wires, 11, work_wires, comparator)

            # We copy the output in wire=12
            qp.CNOT(wires=[11, 12])

            # We clean the work wires used in LeftQuantumComparator
            qp.adjoint(LeftQuantumComparator(x_wires, y_wires, 11, work_wires, comparator))

            return qp.sample(wires=[12])

    .. code-block:: pycon

        output = circuit(3, ">=", 2)
        print(bool(output))
        True

    """

    grad_method = None

    resource_keys = {"num_y_wires", "comparator"}

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        target_wire: WiresLike,
        work_wires: WiresLike,
        comparator: str,
    ):  # pylint: disable=too-many-arguments

        target_wire = Wires(target_wire)
        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        work_wires = Wires(work_wires)

        if comparator not in ["<", "<=", ">=", ">"]:
            raise ValueError("Allowed values for 'comparator' are: '<', '<=', '>=' and '>'.")

        if len(work_wires) < len(y_wires) - 1:
            raise ValueError(f"At least {len(y_wires)-1} work_wires should be provided.")
        if work_wires.intersection(target_wire):
            raise ValueError("None of the wires in work_wires should be the target wire.")
        if work_wires.intersection(x_wires):
            raise ValueError("None of the wires in work_wires should be included in x_wires.")
        if work_wires.intersection(y_wires):
            raise ValueError("None of the wires in work_wires should be included in y_wires.")
        if len(x_wires) != len(y_wires):
            raise ValueError("The number of y_wires should be equal to the number of x_wires")
        if x_wires.intersection(target_wire):
            raise ValueError("None of the wires in x_wires should be the target wire.")
        if x_wires.intersection(y_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if y_wires.intersection(target_wire):
            raise ValueError("None of the wires in y_wires should be the target wire.")

        self.hyperparameters["target_wire"] = target_wire
        self.hyperparameters["x_wires"] = x_wires
        self.hyperparameters["y_wires"] = y_wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["comparator"] = comparator

        all_wires = [x_wires, y_wires, target_wire, work_wires]
        all_wires = Wires.all_wires(all_wires)
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        return {
            "num_y_wires": len(self.hyperparameters["y_wires"]),
            "comparator": self.hyperparameters["comparator"],
        }

    @property
    def num_params(self):
        return 0

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(**hyperparams_dict)

    def map_wires(self, wire_map: dict) -> "LeftQuantumComparator":
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["x_wires", "y_wires", "target_wire", "work_wires"]
        }

        return LeftQuantumComparator(**new_dict, comparator=self.hyperparameters["comparator"])

    def decomposition(self):
        r"""Representation of the operator as a product of other operators."""
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires, y_wires, target_wire, work_wires, comparator
    ):  # pylint: disable=arguments-differ, too-many-arguments
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (WiresLike): The wires that store the integer :math:`x`.
            y_wires (WiresLike): The wires that store the integer :math:`y`.
            target_wire (WiresLike): The wire that stores the value of the inequality test.
            work_wires (WiresLike): The auxiliary wires to use for the addition.
                At least ``len(y_wires) - 1`` work wires should be provided.
            comparator (str): The operator used in the inequality. The value could be '<', '<=', '>=' and '>'.

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        with AnnotatedQueue() as q:
            _left_quantum_comparator(
                x_wires, y_wires, target_wire, work_wires, comparator=comparator
            )

        if QueuingManager.recording():
            for o in q.queue:
                apply(o)

        return q.queue


def _left_quantum_comparator_resources(num_y_wires, comparator):

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
    if comparator in ("<=", ">="):
        X(target_wire)

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


add_decomps(LeftQuantumComparator, _left_quantum_comparator)
