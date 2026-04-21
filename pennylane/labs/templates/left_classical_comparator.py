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
"""Contains the LeftClassicalComparator template for performing inequality test of a quantum register and a classical integer."""

from pennylane import capture, compiler, cond, for_loop, math
from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.operation import Operation
from pennylane.ops import CNOT, X
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.templates.subroutines import Elbow
from pennylane.wires import Wires, WiresLike


class LeftClassicalComparator(Operation):
    r"""This operator performs an inequality test between a quantum register :math:`x` and a
    classical integer :math:`L`, storing the result in a target qubit.

    Depending on the value of the ``comparator`` argument, the operator evaluates one of four
    possible relations:

    .. math::

        \text{LeftClassicalComparator}_{<} |x\rangle |0\rangle = |x\rangle |x < L\rangle & \text{if } op = \text{'<' }


    The decomposition is based on the left block in Figure 6 in Appendix E
    of `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_, adapted for a classical
    constant. Note that the decomposition uses auxiliary wires and in order to clean them,
    one must apply the adjoint of this operator after using the target qubit.

    Args:
        x_wires (WiresLike): The wires that store the quantum integer :math:`x`.
        L (int): The classical integer to compare against. It must be smaller than :math:`2^{\text{len(x_wires)}}`.
        target_wire (WiresLike): The wire that stores the value of the inequality test.
        work_wires (WiresLike): The auxiliary wires to use for the comparison.
            At least ``len(x_wires) - 1`` zeroed work wires should be provided.
            They are not returned in the zero state.
        comparator (str): The operator used in the inequality. Possible values are:
            `'<'`, `'<='`, `'>='` and `'>'`.


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
                comparator='>='
            )
            return qml.sample(wires=3)

    .. code-block:: pycon

        >>> output = circuit(3, 2)
        >>> print(bool(output)) # 3 >= 2
        True

    """

    grad_method = None

    resource_keys = {"num_x_wires", "comparator", "L"}

    def __init__(
        self,
        x_wires: WiresLike,
        L: int,
        target_wire: WiresLike,
        work_wires: WiresLike,
        comparator: str,
    ):  # pylint: disable=too-many-arguments

        target_wire = Wires(target_wire)
        x_wires = Wires(x_wires)
        work_wires = Wires(work_wires)

        if comparator not in ["<", "<=", ">=", ">"]:
            raise ValueError("Allowed values for 'comparator' are: '<', '<=', '>=' and '>'.")

        if len(work_wires) < len(x_wires) - 1:
            raise ValueError(f"At least {len(x_wires)-1} work_wires should be provided.")
        if work_wires.intersection(target_wire):
            raise ValueError("None of the wires in work_wires should be the target wire.")
        if work_wires.intersection(x_wires):
            raise ValueError("None of the wires in work_wires should be included in x_wires.")
        if x_wires.intersection(target_wire):
            raise ValueError("None of the wires in x_wires should be the target wire.")
        if L >= 2 ** len(x_wires):
            raise ValueError("L must be less than 2**len(x_wires).")
        self.hyperparameters["target_wire"] = target_wire
        self.hyperparameters["x_wires"] = x_wires
        self.hyperparameters["L"] = L
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["comparator"] = comparator

        all_wires = [x_wires, target_wire, work_wires]
        all_wires = Wires.all_wires(all_wires)
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "L": self.hyperparameters["L"],
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

    def map_wires(self, wire_map: dict) -> "LeftClassicalComparator":
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["x_wires", "target_wire", "work_wires"]
        }

        return LeftClassicalComparator(
            **new_dict, L=self.hyperparameters["L"], comparator=self.hyperparameters["comparator"]
        )

    def decomposition(self):
        r"""Representation of the operator as a product of other operators."""
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires, L, target_wire, work_wires, comparator
    ):  # pylint: disable=arguments-differ, too-many-arguments
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (WiresLike): The wires that store the quantum integer :math:`x`.
            L (int): The classical integer to compare against. It must be smaller than :math:`2^{\text{len(x_wires)}}`.
            target_wire (WiresLike): The wire that stores the value of the inequality test.
            work_wires (WiresLike): The auxiliary wires to use for the comparison.
                At least ``len(x_wires) - 1`` zeroed work wires should be provided.
                They are not returned in the zero state.
            comparator (str): The operator used in the inequality. Possible values are:
                '<', '<=', '>=' and '>'.

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        with AnnotatedQueue() as q:
            _left_classical_comparator(x_wires, L, target_wire, work_wires, comparator=comparator)

        if QueuingManager.recording():
            for o in q.queue:
                apply(o)

        return q.queue


def _get_specific_bit(L, i):
    # returns the i-th bit of the binary representation of L in little endian convention.
    return (L >> i) & 1


def _left_classical_comparator_resources(num_x_wires, L, comparator):
    if comparator in ["<=", ">"]:
        L += 1

    resources = {
        Elbow: num_x_wires - 1,
        CNOT: 0,
        X: 0,
    }

    bit_0 = _get_specific_bit(L, 0)
    if bit_0:
        resources[X] += 2
        resources[CNOT] += 1

    resources[CNOT] += num_x_wires - 1
    resources[X] += 4*(L.bit_count() - L&1)

    if comparator in [">", ">="]:
        resources[X] += 1

    return resources


@register_resources(_left_classical_comparator_resources, exact=True)
def _left_classical_comparator(
    x_wires, L, target_wire, work_wires, comparator, **_
):  # pylint: disable=too-many-arguments

    # revert to follow PL convention
    x_wires = x_wires[::-1]

    def _negate_output():
        X(wires=target_wire)

    if comparator in ["<=", ">"]:
        L += 1

    cond(comparator == ">", _negate_output)()
    cond(comparator == ">=", _negate_output)()
    used_work_wires = Wires.all_wires([work_wires[: len(x_wires) - 1], target_wire])

    bit = _get_specific_bit(L, 0)
    cond(bit, X)(wires=[x_wires[0]])
    cond(bit, CNOT)(wires=[x_wires[0], used_work_wires[0]])
    cond(bit, X)(wires=[x_wires[0]])

    if compiler.active() or capture.enabled():
        x_wires = math.array(x_wires, like="jax")
        used_work_wires = math.array(used_work_wires, like="jax")

    # pylint: disable=no-value-for-parameter
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
