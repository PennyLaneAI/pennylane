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
"""Contains the LeftClassicalComparator template for performing an inequality
test of a quantum register and a classical integer."""

from pennylane import capture, compiler, cond, for_loop, math
from pennylane.core.operator import Operation
from pennylane.core.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.ops import CNOT, X
from pennylane.templates.subroutines import Elbow
from pennylane.wires import Wires, WiresLike


class LeftClassicalComparator(Operation):
    r"""This operator performs an inequality test between a quantum register :math:`\lvert x\rangle` and a
    classical integer :math:`L`, storing the result in a target qubit.

    The operator evaluates the following relation, given by the ``comparator="<"`` argument

    .. math::

        \text{LeftClassicalComparator}_{<} \lvert x\rangle \lvert 0\rangle = \lvert x\rangle \lvert x < L\rangle \text{ if comparator} = \text{'<'}

    The decomposition is based on the left block in Figure 6 in Appendix E
    of `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_, adapted for a classical constant.

    .. warning::

        Note that the decomposition uses auxiliary wires and in order to clean them,
        one must apply the adjoint of this operator after using the target qubit.

    Args:
        x_wires (WiresLike): The wires that store the quantum integer :math:`x`.
        L (int): The classical integer to compare against. It must be smaller than :math:`2^{n}`,
            where :math:`n` is the number of ``x_wires``.
        target_wire (WiresLike): The wire that stores the value of the inequality test.
        work_wires (WiresLike): The auxiliary wires to use for the comparison.
            At least ``len(x_wires) - 1`` zeroed work wires should be provided.
            They are not returned in the zero state.
        comparator (str): The operator used in the inequality. Possible values are:
            ``'<'``, ``'<='``, ``'>='`` and ``'>'``.

    **Example**

    .. code-block:: python

        import pennylane as qp
        from pennylane.labs.templates import LeftClassicalComparator

        dev = qp.device("lightning.qubit", wires=6, shots=1)

        @qp.qnode(dev)
        def circuit(x_val, L_val):
            qp.BasisState(x_val, wires=[0, 1, 2])

            LeftClassicalComparator(
                x_wires=[0, 1, 2],
                L=L_val,
                target_wire=3,
                work_wires=[4, 5],
                comparator='>='
            )
            return qp.sample(wires=3)

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
            raise ValueError(
                f"At least {len(x_wires) - 1} work_wires are required, but only "
                f"{len(work_wires)} were provided. (x_wires={list(x_wires)}, "
                f"work_wires={list(work_wires)})"
            )

        overlap = work_wires.intersection(target_wire)
        if overlap:
            raise ValueError(
                f"work_wires and target_wire must be disjoint, but share: {list(overlap)}. "
                f"(work_wires={list(work_wires)}, target_wire={list(target_wire)})"
            )

        overlap = work_wires.intersection(x_wires)
        if overlap:
            raise ValueError(
                f"work_wires and x_wires must be disjoint, but share: {list(overlap)}. "
                f"(work_wires={list(work_wires)}, x_wires={list(x_wires)})"
            )

        overlap = x_wires.intersection(target_wire)
        if overlap:
            raise ValueError(
                f"x_wires and target_wire must be disjoint, but share: {list(overlap)}. "
                f"(x_wires={list(x_wires)}, target_wire={list(target_wire)})"
            )
        if not math.is_abstract(L) and L >= 2 ** len(x_wires):
            raise ValueError(
                f"L must be less than 2**len(x_wires). Got {L=} and {2**len(x_wires)=}"
            )
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

        **Example**

        >>> dev = qp.device("lightning.qubit", wires=6)
        >>> @qp.qnode(dev)
        >>> @qp.decompose(gate_set={"TemporaryAND", 'CNOT', 'PauliX'})
        >>> def circuit(x_val, L_val):
        ...    qp.BasisState(x_val, wires=[0, 1, 2])
        ...    qp.LeftClassicalComparator(
        ...     x_wires=[0, 1, 2],
        ...     L=L_val,
        ...     target_wire=3,
        ...     work_wires=[4, 5],
        ...     comparator='<='
        ... )
        ... return qp.state()

        >>> print(qp.draw(circuit, wire_order = [2, 4, 1, 5, 0, 6, 3])(2,0))
        2: ──X─╭●──X──────────┤  State
        4: ────╰X─╭●─╭●───────┤  State
        1: ───────├●─│────────┤  State
        5: ───────╰⊕─╰X─╭●─╭●─┤  State
        0: ─────────────├●─│──┤  State
        3: ─────────────╰⊕─╰X─┤  State

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

    n = num_x_wires
    resources = {
        Elbow: n - 1,
        CNOT: n - 1,
        X: 0,
    }

    bit_0 = _get_specific_bit(L, 0)
    if bit_0:
        resources[X] += 2
        resources[CNOT] += 1

    # X gates from the conditional negations inside the loop: 4 per set bit in
    # positions ``1 .. n-1`` of ``L`` (each such iteration flips ``x[i]`` and
    # ``w[i-1]`` before and after the Elbow).
    mid_bits = (L & (2**n - 1)).bit_count() - bit_0
    resources[X] += 4 * mid_bits

    # Degenerate bound: the effective ``L`` equals ``2 ** n`` (only reachable for
    # ``"<="`` / ``">"`` with the original ``L == 2 ** n - 1``). The low ``n``
    # bits of ``L`` are all zero, so the core comparison computes the constant 0
    # (``x < 0``); bit ``n`` of ``L`` is set exactly in this case and a single
    # conditional ``X`` sets the correct "always true" result on the target wire.
    if _get_specific_bit(L, n):
        resources[X] += 1

    if comparator in [">", ">="]:
        resources[X] += 1

    return resources


@register_resources(_left_classical_comparator_resources, exact=True)
def _left_classical_comparator(x_wires, L, target_wire, work_wires, comparator, **_):
    x_wires = x_wires[::-1]

    def _negate_output():
        X(wires=target_wire)

    if comparator in ["<=", ">"]:
        L += 1

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

    _loop()  # pylint: disable=no-value-for-parameter

    # Degenerate bound: when the effective ``L`` equals ``2 ** len(x_wires)`` its
    # low ``n`` bits are all zero, so the bitwise core above computes the constant
    # ``x < 0`` (always False) rather than the intended "always True". Bit ``n`` of
    # ``L`` is set exactly in this case, so a single conditional ``X`` fixes the
    # target wire. Using ``cond`` keeps this qjit/JAX-capture compatible (no Python
    # boolean conversion of a possibly-traced ``L``).
    bit_n = _get_specific_bit(L, len(x_wires))
    cond(bit_n, X)(wires=[used_work_wires[len(x_wires) - 1]])

    cond(comparator.startswith(">"), _negate_output)()


add_decomps(LeftClassicalComparator, _left_classical_comparator)
