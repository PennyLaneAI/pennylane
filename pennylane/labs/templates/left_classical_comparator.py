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
"""Contains the LeftClassicalComparator template for performing inequality test of two quantum registers."""

from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.operation import Operation
from pennylane.ops import CNOT, X
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.wires import Wires, WiresLike
from pennylane.templates.subroutines import Elbow
from pennylane import cond, for_loop
from pennylane import compiler, math, capture


class LeftClassicalComparator(Operation):
    r"""This operator performs an inequality test between a classical value :math:`L` and quantum registers :math:`x`,
    storing the result in a target qubit. Depending on the value of the
    ``op`` argument, the operator evaluates one of four possible relations:

    .. math::

        \text{LeftQuantumComparator}(op) |x\rangle |y\rangle |0\rangle =
        \begin{cases}
        |x\rangle |y\rangle |x < y\rangle & \text{if } op = 0 \\
        |x\rangle |y\rangle |x \leq y\rangle & \text{if } op = 1 \\
        |x\rangle |y\rangle |x \geq y\rangle & \text{if } op = 2 \\
        |x\rangle |y\rangle |x > y\rangle & \text{if } op = 3
        \end{cases}

    The decomposition is defined as the left block in Figure 6 in Appendix E
    of `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_. Note that the decomposition uses auxiliary wires
    and in order to clean them, we must apply the adjoint of this operator after using the target qubit.

    Args:
            x_wires (WiresLike): The wires that store the integer :math:`x`.
            y_wires (WiresLike): The wires that store the integer :math:`y`. The number of ``y_wires`` should be equal to
                the number of ``x_wires``.
            target_wire (WiresLike): The wire that stores the value of the inequality test.
            work_wires (WiresLike): The auxiliary wires to use for the addition.
                At least ``len(y_wires) - 1`` zeroed work wires should be provided. They are not returned in the zero state.
            op (str): The operator used in the inequality. The value could be '<', '<=', '>=' and '>'.


    **Example**


    .. code-block:: python

        import pennylane as qp
        from pennylane.labs.templates import LeftQuantumComparator

        dev = qp.device("lightning.qubit")

        @qp.qjit
        @qp.qnode(dev, shots=1)
        def circuit(a, b):

            op = 2
            qp.BasisState(a, wires=[0, 3, 6, 9])
            qp.BasisState(b, wires=[1, 4, 7, 10])
            LeftQuantumComparator([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], op)
            qp.CNOT([11, 12])
            qp.adjoint(
                lambda: LeftQuantumComparator([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], op)
            )()
            return qp.sample(wires=[12])

    .. code-block:: pycon

        >>> output = circuit(3, 2)
        >>> print(bool(output)) # 3 >= 2
        True

    """

    grad_method = None

    resource_keys = {"num_x_wires", "op"}

    def __init__(
        self,
        x_wires: WiresLike,
        L: int,
        target_wire: WiresLike,
        work_wires: WiresLike,
        op=None,
    ):  # pylint: disable=too-many-arguments

        target_wire = Wires(target_wire)
        x_wires = Wires(x_wires)
        work_wires = Wires(work_wires)

        if op not in ["<", "<=", ">=", ">"]:
            raise ValueError("Allowed values for 'op' are: '<', '<=', '>=' and '>'.")

        if len(work_wires) < len(x_wires) - 1:
            raise ValueError(f"At least {len(x_wires)-1} work_wires should be provided.")
        if work_wires.intersection(target_wire):
            raise ValueError("None of the wires in work_wires should be the target wire.")
        if work_wires.intersection(x_wires):
            raise ValueError("None of the wires in work_wires should be included in x_wires.")
        if x_wires.intersection(target_wire):
            raise ValueError("None of the wires in x_wires should be the target wire.")

        self.hyperparameters["target_wire"] = target_wire
        self.hyperparameters["x_wires"] = x_wires
        self.hyperparameters["L"] = L
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["op"] = op

        all_wires = [x_wires, target_wire, work_wires]
        all_wires = Wires.all_wires(all_wires)
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "op": self.hyperparameters["op"],
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

    def map_wires(self, wire_map: dict) -> "LeftClassicalComparator":
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["x_wires", "y_wires", "target_wire", "work_wires"]
        }

        return LeftClassicalComparator(**new_dict, L=self.hyperparameters["L"], op=self.hyperparameters["op"])

    def decomposition(self):
        r"""Representation of the operator as a product of other operators."""
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires, L, target_wire, work_wires, op=None
    ):  # pylint: disable=arguments-differ, too-many-arguments
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (WiresLike): The wires that store the integer :math:`x`.
            y_wires (WiresLike): The wires that store the integer :math:`y`.
            target_wire (WiresLike): The wire that stores the value of the inequality test.
            work_wires (WiresLike): The auxiliary wires to use for the addition.
                At least ``len(y_wires) - 1`` work wires should be provided.
            op (str): The operator used in the inequality. The value could be '<', '<=', '>=' and '>'.

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        with AnnotatedQueue() as q:
            _left_classical_comparator(x_wires, L, target_wire, work_wires, op=op)

        if QueuingManager.recording():
            for op in q.queue:
                apply(op)

        return q.queue


def _get_specific_bit(L, i):
    return (L >> i) & 1

def _left_classical_comparator_resources(num_x_wires, op):

    resources = {
        Elbow: num_x_wires,
        CNOT: 2 + 5 * (num_x_wires - 1),
    }

    if op in [">=", "<="]:
        resources[X] = 1

    return resources


@register_resources(_left_classical_comparator_resources, exact=True)
def _left_classical_comparator(
    x_wires, L, target_wire, work_wires, op, **_
):  # pylint: disable=too-many-arguments
    # op = ['<', '<=', '>=', '>']

    # revert to follow PL convention
    x_wires = x_wires[::-1]

    def _negate_output():
        X(wires=target_wire)

    @cond(math.logical_or(op == "<=", op == ">"))
    def _add(L):
        return L + 1

    @_add.otherwise
    def _add(L):
        return L

    L = _add(L)

    cond(op == ">", _negate_output)()
    cond(op == ">=", _negate_output)()
    used_work_wires = Wires.all_wires([work_wires[: len(x_wires) - 1], target_wire])

    bit = _get_specific_bit(L, 0)
    cond(bit, X)(wires=[x_wires[0]])
    cond(bit, CNOT)(wires=[x_wires[0], used_work_wires[0]])
    cond(bit, X)(wires=[x_wires[0]])

    if compiler.active() or capture.enabled():
        x_wires = math.array(x_wires, like="jax")
        used_work_wires = math.array(used_work_wires, like="jax")

    @for_loop(1, len(x_wires))
    def _loop(i):
        bit = _get_specific_bit(L, i)
        cond(bit, X)(wires=[x_wires[i]])
        cond(bit, X)(wires=[used_work_wires[i - 1]])
        Elbow(wires=[used_work_wires[i - 1], x_wires[i], used_work_wires[i]])
        cond(bit, X)(wires=[used_work_wires[i - 1]])
        CNOT(wires=[used_work_wires[i - 1], used_work_wires[i]])
        cond(bit, X)(wires=[x_wires[i]])

    _loop()


add_decomps(LeftClassicalComparator, _left_classical_comparator)
