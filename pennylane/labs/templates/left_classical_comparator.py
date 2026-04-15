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
    r"""This operator performs an inequality test between a quantum register :math:`x` and a
    classical integer :math:`L`, storing the result in a target qubit.

    Depending on the value of the ``op`` argument, the operator evaluates one of four
    possible relations:

    .. math::

        \text{LeftClassicalComparator}(op) |x\rangle |0\rangle =
        \begin{cases}
        |x\rangle |x < L\rangle & \text{if } op = \text{'<' } \\
        |x\rangle |x \leq L\rangle & \text{if } op = \text{'<='} \\
        |x\rangle |x \geq L\rangle & \text{if } op = \text{'>='} \\
        |x\rangle |x > L\rangle & \text{if } op = \text{'>' }
        \end{cases}

    The decomposition is based on the left block in Figure 6 in Appendix E
    of `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_, adapted for a classical
    constant. Note that the decomposition uses auxiliary wires and in order to clean them,
    one must apply the adjoint of this operator after using the target qubit.

    Args:
        x_wires (WiresLike): The wires that store the quantum integer :math:`x`.
        L (int): The classical integer to compare against.
        target_wire (WiresLike): The wire that stores the value of the inequality test.
        work_wires (WiresLike): The auxiliary wires to use for the comparison.
            At least ``len(x_wires) - 1`` zeroed work wires should be provided.
            They are not returned in the zero state.
        op (str): The operator used in the inequality. Possible values are:
            '<', '<=', '>=' and '>'.

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane.labs.templates import LeftClassicalComparator

        dev = qml.device("lightning.qubit", wires=6, shots = 1)

        @qml.qnode(dev)
        def circuit(x_val, L_val):

            qml.BasisState(x_val, wires=[0, 1, 2])

            LeftClassicalComparator(
                x_wires=[0, 1, 2],
                L=L_val,
                target_wire=3,
                work_wires=[4, 5],
                op='>='
            )
            return qml.sample(wires=3)

    .. code-block:: pycon

        >>> output = circuit(3, 2)
        >>> print(bool(output)) # 3 >= 2
        True

    """

    grad_method = None

    resource_keys = {"num_x_wires", "op", "L"}

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
            "L": self.hyperparameters["L"],
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
            for key in ["x_wires", "target_wire", "work_wires"]
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
            x_wires (WiresLike): The wires that store the quantum integer :math:`x`.
            L (int): The classical integer to compare against.
            target_wire (WiresLike): The wire that stores the value of the inequality test.
            work_wires (WiresLike): The auxiliary wires to use for the comparison.
                At least ``len(x_wires) - 1`` zeroed work wires should be provided.
                They are not returned in the zero state.
            op (str): The operator used in the inequality. Possible values are:
                '<', '<=', '>=' and '>'.

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
    # returns the i-th bit of the binary representation of L
    return (L >> i) & 1

def _left_classical_comparator_resources(num_x_wires, L, op):
    if op in ["<=", ">"]:
        L += 1

    resources = {
        Elbow: num_x_wires - 1,
        CNOT: 0,
        X: 0,
    }

    bit_0 = (L >> 0) & 1
    if bit_0:
        resources[X] += 2
        resources[CNOT] += 1

    for i in range(1, num_x_wires):
        bit_i = (L >> i) & 1
        resources[CNOT] += 1
        if bit_i:
            resources[X] += 4

    if op in [">", ">="]:
        resources[X] += 1

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
