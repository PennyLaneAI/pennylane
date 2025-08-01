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
Contains the Select template.
"""

import copy
import itertools
from collections import Counter, defaultdict

from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import CNOT, X, adjoint, ctrl
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

from .temporary_and import TemporaryAND


class Select(Operation):
    r"""The ``Select`` operator, also known as multiplexer or multiplexed operation,
    applies different operations depending on the state of designated control wires.

    .. math:: Select|i\rangle \otimes |\psi\rangle = |i\rangle \otimes U_i |\psi\rangle

    .. figure:: ../../../doc/_static/templates/subroutines/select.png
                    :align: center
                    :width: 70%
                    :target: javascript:void(0);

    If the applied operations :math:`\{U_i\}` are all single-qubit Pauli rotations about the
    same axis, with the angle determined by the control wires, this is also called a
    **uniformly controlled rotation** gate.

    .. seealso:: :class:`~.SelectPauliRot`

    Args:
        ops (list[Operator]): operations to apply
        control (Sequence[int]): the wires controlling which operation is applied.
            At least :math:`\lceil \log_2 K\rceil` wires are required for :math:`K` operations.
        work_wires (Union[Wires, Sequence[int], or int]): auxiliary wire(s) that may be
            utilized during the decomposition of the operator into native operations.
            For details, see the section on the unary iterator decomposition below.
        id (str or None): String representing the operation (optional)

    .. note::
        The position of the operation in the list determines which qubit state implements that
        operation. For example, when the qubit register is in the state :math:`|00\rangle`,
        we will apply ``ops[0]``. When the qubit register is in the state :math:`|10\rangle`,
        we will apply ``ops[2]``. To obtain the list position ``index`` for a given binary
        bitstring representing the control state we can use the following relationship:
        ``index = int(state_string, 2)``. For example, ``2 = int('10', 2)``.

    **Example**

    >>> dev = qml.device('default.qubit', wires=4)
    >>> ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2,3])]
    >>> @qml.qnode(dev)
    >>> def circuit():
    >>>     qml.Select(ops, control=[0,1])
    >>>     return qml.state()
    ...
    >>> print(qml.draw(circuit, level='device')())
    0: ─╭○─╭○─╭●─╭●────┤  State
    1: ─├○─├●─├○─├●────┤  State
    2: ─╰X─│──╰Y─├SWAP─┤  State
    3: ────╰X────╰SWAP─┤  State

    .. details::
        :title: Unary iterator decomposition

        Generically, ``Select`` is decomposed into one multi-controlled operator for each target
        operator. However, if auxiliary wires are available, a decomposition using a
        "unary iterator" can be applied. It was introduced by
        `Babbush et al. (2018) <https://arxiv.org/abs/1805.03662>`__.

        **Principle**

        Unary iteration leverages auxiliary wires to store intermediate values for reuse between
        the different multi-controlled operators, avoiding unnecessary recomputation.
        In addition to this caching functionality, unary iteration reduces the cost of the
        computation directly, because the involved reversible AND (or Toffoli) gates can be
        implemented at lower cost if the target is known to be in the :math:`|0\rangle` state
        (see :class:`~TemporaryAND`).

        For :math:`K` operators to be Select-applied, :math:`c=\lceil\log_2 K\rceil` control
        wires are required. Unary iteration demands an additional :math:`c-1` auxiliary wires.
        Below we first show an example for :math:`K` being a power of two, i.e., :math:`K=2^c`.
        Then we elaborate on implementation details for the case :math:`K<2^c`, which we call
        a *partial Select* operator.

        **Example**

        Assume that we want to Select-apply :math:`K=8=2^3` operators to two target wires,
        which requires :math:`c=\lceil \log_2 K\rceil=3` control wires. The generic
        decomposition for this takes the form

        .. code-block::

            0: ─╭○─────╭○─────╭○─────╭○─────╭●─────╭●─────╭●─────╭●─────┤
            1: ─├○─────├○─────├●─────├●─────├○─────├○─────├●─────├●─────┤
            2: ─├○─────├●─────├○─────├●─────├○─────├●─────├○─────├●─────┤
            3: ─├U(M0)─├U(M1)─├U(M2)─├U(M3)─├U(M4)─├U(M5)─├U(M6)─├U(M7)─┤
            4: ─╰U(M0)─╰U(M1)─╰U(M2)─╰U(M3)─╰U(M4)─╰U(M5)─╰U(M6)─╰U(M7)─┤.

        Unary iteration then uses :math:`c-1=2` auxiliary wires, denoted ``aux0`` and ``aux1``
        below, to first rewrite the control structure:

        .. code-block::

            0:    ─╭○───────○╮─╭○───────○╮─╭○───────○╮─╭○───────○╮─╭●───────●╮─╭●───────●╮─╭●───────●╮─╭●───────●╮─┤
            1:    ─├○───────○┤─├○───────○┤─├●───────●┤─├●───────●┤─├○───────○┤─├○───────○┤─├●───────●┤─├●───────●┤─┤
            aux0:  ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ │
            2:    ───├○───○┤─────├●───●┤─────├○───○┤─────├●───●┤─────├○───○┤─────├●───●┤─────├○───○┤─────├●───●┤───┤
            aux1:    ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯   │
            3:    ─────├U(M0)──────├U(M1)──────├U(M2)──────├U(M3)──────├U(M4)──────├U(M5)──────├U(M6)──────├U(M7)──┤
            4:    ─────╰U(M0)──────╰U(M1)──────╰U(M2)──────╰U(M3)──────╰U(M4)──────╰U(M5)──────╰U(M6)──────╰U(M7)──┤

        Here, we used the symbols

        .. code-block::

            0: ─╭●──       ─●─╮─
            1: ─├●──  and  ─●─┤─
            2:  ╰───       ───╯

        for :class:`~.TemporaryAND` and its adjoint, respectively, and skipped drawing the
        auxiliary wires in areas where they are guaranteed to be in the state :math:`|0\rangle`.
        We will need three simplification rules for pairs of ``TemporaryAND`` gates:

        .. code-block::

            ─○─╮─╭○──   ──     ─○─╮─╭○──   ─╭○─       ─○─╮─╭●──   ─╭●────
            ─○─┤─├○── = ──,    ─○─┤─├●── = ─│──, and  ─●─┤─├○── = ─│──╭●─.
            ───╯ ╰───   ──     ───╯ ╰───   ─╰X─       ───╯ ╰───   ─╰X─╰X─

        Applying these simplifications reduces the computational cost of the ``Select``
        template:

        .. code-block::

            0:    ─╭○────────────────╭○──────────────────╭●─────────────────────╭●─────────────────●╮─┤
            1:    ─├○────────────────│───────────────────│──╭●──────────────────│──────────────────●┤─┤
            aux0:  ╰─╭●─────╭●────●╮─╰X─╭●─────╭●─────●╮─╰X─╰X─╭●─────╭●─────●╮─╰X─╭●─────╭●─────●╮─╯ │
            2:    ───├○─────│─────●┤────├○─────│──────●┤───────├○─────│──────●┤────├○─────│──────●┤───┤
            aux1:    ╰─╭●───╰X─╭●──╯    ╰─╭●───╰X──╭●──╯       ╰─╭●───╰X──╭●──╯    ╰─╭●───╰X──╭●──╯   │
            3:    ─────├U(M0)──├U(M1)─────├U(M2)───├U(M3)────────├U(M4)───├U(M5)─────├U(M6)───├U(M7)──┤
            4:    ─────╰U(M0)──╰U(M1)─────╰U(M2)───╰U(M3)────────╰U(M4)───╰U(M5)─────╰U(M6)───╰U(M7)──┤

        An additional cost reduction then results from the fact that the ``TemporaryAND``
        gate and its adjoint require four and zero :class:`~T` gates, respectively,
        in contrast to the seven ``T`` gates required by a decomposition of :class:`~Toffoli`.

        For general :math:`c` and :math:`K=2^c`, the decomposition takes a similar form, with
        alternating control and auxiliary wires.

        An implementation of the unary iterator is achieved in the following steps:
        We first define a recursive sub-circuit ``R``;
        given :math:`L` operators and :math:`2 \lceil\log_2(L)\rceil + 1` control and
        auxiliary wires, there are three cases that ``R`` distinguishes. First, if ``L>1``,
        it applies the circuit

        .. code-block::

            aux_j:   ╭R   ─╭●────╭●────●─╮─
            j+1:     ├R = ─├○────│─────●─┤─
            aux_j+1: ╰R    ╰──R──╰X─R────╯ ,

        where each label ``R`` symbolizes a call to ``R`` itself, on the next recursion level.
        These next-level calls use
        :math:`L' = 2^{\lceil\log_2(L)\rceil-1}` (i.e. half of :math:`L`, rounded up to the next
        power of two) and :math:`L-L'` (i.e. the rest) operators, respectively.

        Second, if ``L=1``, the single operator is applied, controlled on the first control wire.
        Finally, if ``L=0``, ``R`` does not apply any operators.

        With ``R`` defined, we are ready to outline the main circuit structure:

        #. Apply the left-most ``TemporaryAND`` controlled on qubits ``0`` and ``1``.
        #. Split the target operators into four "quarters" (often with varying sizes)
           and apply the first quarter using ``R``.
        #. Apply ``[X(0), CNOT([0, "aux0"]), X(0)]``.
        #. Apply the second quarter using ``R``.
        #. Apply ``[CNOT([0, "aux0"]), CNOT([1, "aux0"])]``.
        #. Apply the third quarter using ``R``.
        #. Apply ``[CNOT([0, "aux0"])]``.
        #. Apply the last quarter using ``R``.
        #. Apply the right-most ``adjoint(TemporaryAND)`` controlled on qubits ``0`` and ``1``.

        **Partial Select decomposition**

        The unary iterator decomposition of the ``Select`` template can be
        simplified further if both of the following criteria are met:

        #. There are fewer target operators than would maximally be possible for the given
           number of control wires, i.e. :math:`K<2^c`.

        #. The state :math:`|\psi\rangle` of the control wires satisfies
           :math:`\langle j | \psi\rangle=0` for all computational basis states with :math:`j\geq K`.

        We do not derive this reduction here but discuss the modifications to the implementation
        above that result from it.

        Given :math:`K=2^c-b` operators, where :math:`c` is defined as above and we
        have :math:`0\leq b<2^{c-1}`, the nine steps above are modified into one of three variants.
        In each variant, the first :math:`2^{c-1}` operators are applied in two equal portions,
        containing :math:`2^{c-2}` operators each.
        After this, :math:`\ell=2^{c-1} -b` operators remain and the three circuit variants are
        distinguished, based on :math:`\ell`:

        - if :math:`\ell \geq 2^{c-2}`, the following, rather generic, circuit is applied:

          .. code-block::

              0:    ─╭○─────╭○─────╭●────────╭●─────●─╮─
              1:    ─├○─────│──────│──╭●─────│──────●─┤─
              aux0:  ╰──╭R──╰X─╭R──╰X─╰X─╭R──╰X─╭R────╯
              2:    ────├R─────├R────────├R─────├R──────
              aux1:     ╰R     ╰R        ╰R     ╰R      .

          Here, each operator with three ``R`` labels symbolizes a call to ``R``. The first
          call in the second half applies :math:`2^{\lceil\log_2(\ell)\rceil-1}` operators.
          Note that this case is triggered if :math:`K` is larger than or equal to
          :math:`\tfrac{3}{4}` of the maximal capacity for :math:`c` control wires.
          Also note how the two middle ``TemporaryAND`` gates were merged into two CNOTs,
          like for the non-partial Select operator.

        - if :math:`1<\ell < 2^{c-2}`, the following circuit is applied:

          .. code-block::

              0:    ─╭○─────╭○─────○─╮╭●─────╭●─────●─╮─
              1:    ─├○─────│──────●─┤│──────│────────│─
              aux0:  ╰──╭R──╰X─╭R────╯│      │        │
              2:    ────├R─────├R─────├○─────│──────●─┤─
              aux1:     ╰R     ╰R     ╰───R──╰X──R────╯

          where the second half may skip more than one control and auxiliary wire each.
          In this diagram, both the operators with three and one ``R`` labels represent calls to
          ``R``, with single-label instances applying fewer operators.
          The first call to ``R`` in the second half applies :math:`2^{\lceil\log_2(\ell)\rceil-1}`
          operators. The middle elbows act on distinct wire triples and can not be merged as
          above.

        - if :math:`\ell=1`, the following circuit is applied:

          .. code-block::

              0:    ─╭○─────╭○─────○─╮╭●──
              1:    ─├○─────│──────●─┤│───
              aux0:  ╰──╭R──╰X─╭R────╯│───
              2:    ────├R─────├R─────│───
              aux1:     ╰R     ╰R     ╰U  .

          Here, the three connected ``R`` labels symbolize a call to ``R`` and
          apply :math:`2^{c-2}` operators each.
          The controlled gate on the right applies the single remaining operator.

    """

    resource_keys = {"op_reps", "num_control_wires"}

    @property
    def resource_params(self):
        op_reps = tuple(resource_rep(type(op), **op.resource_params) for op in self.ops)
        return {"op_reps": op_reps, "num_control_wires": len(self.control)}

    def _flatten(self):
        return (self.ops), (self.control, self.hyperparameters["work_wires"])

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata) -> "Select":
        return cls(data, control=metadata[0], work_wires=metadata[1])

    def __repr__(self):
        return f"Select(ops={self.ops}, control={self.control})"

    def __init__(self, ops, control, work_wires=None, id=None):
        control = Wires(control)
        work_wires = Wires(() if work_wires is None else work_wires)
        self.hyperparameters["ops"] = tuple(ops)
        self.hyperparameters["control"] = control
        self.hyperparameters["work_wires"] = work_wires

        if 2 ** len(control) < len(ops):
            raise ValueError(
                f"Not enough control wires ({len(control)}) for the desired number of "
                + f"operations ({len(ops)}). At least {_ceil_log(len(ops))} control "
                + "wires are required."
            )

        if any(
            control_wire in Wires.all_wires([op.wires for op in ops]) for control_wire in control
        ):
            raise ValueError("Control wires should be different from operation wires.")

        for op in ops:
            QueuingManager.remove(op)

        target_wires = Wires.all_wires([op.wires for op in ops])
        self.hyperparameters["target_wires"] = target_wires

        all_wires = target_wires + control
        super().__init__(*self.data, wires=all_wires, id=id)

    def map_wires(self, wire_map: dict) -> "Select":
        new_ops = [o.map_wires(wire_map) for o in self.hyperparameters["ops"]]
        new_control = [wire_map.get(wire, wire) for wire in self.hyperparameters["control"]]
        return Select(new_ops, new_control)

    def __copy__(self):
        """Copy this op"""
        cls = self.__class__
        copied_op = cls.__new__(cls)

        new_data = copy.copy(self.data)

        for attr, value in vars(self).items():
            if attr != "data":
                setattr(copied_op, attr, value)

        copied_op.data = new_data

        return copied_op

    @property
    def data(self):
        """Create data property"""
        return tuple(d for op in self.ops for d in op.data)

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        for op in self.ops:
            op_num_params = op.num_params
            if op_num_params > 0:
                op.data = new_data[:op_num_params]
                new_data = new_data[op_num_params:]

    def decomposition(self):
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n

        A ``DecompositionUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_decomposition`.

        Returns:
            list[Operator]: decomposition of the operator

        **Example**

        >>> ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2,3])]
        >>> op = qml.Select(ops, control=[0,1])
        >>> op.decomposition()
        [MultiControlledX(wires=[0, 1, 2], control_values=[0, 0]),
         MultiControlledX(wires=[0, 1, 3], control_values=[0, 1]),
         Controlled(Y(2), control_wires=[0, 1], control_values=[True, False]),
         Controlled(SWAP(wires=[2, 3]), control_wires=[0, 1])]
        """
        return self.compute_decomposition(self.ops, control=self.control)

    @staticmethod
    def compute_decomposition(
        ops,
        control,
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            ops (list[Operator]): operations to apply
            control (Sequence[int]): the wires controlling which operation is applied

        Returns:
            list[Operator]: decomposition of the operator

        **Example**

        >>> ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2,3])]
        >>> qml.Select.compute_decomposition(ops, control=[0,1])
        [MultiControlledX(wires=[0, 1, 2], control_values=[0, 0]),
         MultiControlledX(wires=[0, 1, 3], control_values=[0, 1),
         Controlled(Y(2), control_wires=[0, 1], control_values=[True, False]),
         Controlled(SWAP(wires=[2, 3]), control_wires=[0, 1])]
        """
        return [
            ctrl(op, control, control_values=state)
            for state, op in zip(itertools.product([0, 1], repeat=len(control)), ops)
        ]

    @property
    def ops(self):
        """Operations to be applied."""
        return self.hyperparameters["ops"]

    @property
    def control(self):
        """The control wires."""
        return self.hyperparameters["control"]

    @property
    def target_wires(self):
        """The wires of the target operators."""
        return self.hyperparameters["target_wires"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.hyperparameters["control"] + self.hyperparameters["target_wires"]


# Decomposition of Select using full + multi-control strategy


def _multi_controlled_rep(target_rep, num_control_wires, state):
    return controlled_resource_rep(
        base_class=target_rep.op_type,
        base_params=target_rep.params,
        num_control_wires=num_control_wires,
        num_work_wires=0,
        num_zero_control_values=num_control_wires - sum(state),
    )


def _select_resources_full_mc(op_reps, num_control_wires):
    state_iterator = itertools.product([0, 1], repeat=num_control_wires)

    resources = defaultdict(int)
    for rep, state in zip(op_reps, state_iterator):
        resources[_multi_controlled_rep(rep, num_control_wires, state)] += 1
    return dict(resources)


# pylint: disable=unused-argument
@register_resources(_select_resources_full_mc)
def _select_decomp_full_mc(ops, control, work_wires, **_):
    state_iterator = itertools.product([0, 1], repeat=len(control))
    for state, op in zip(state_iterator, ops):
        ctrl(op, control, control_values=state)


add_decomps(Select, _select_decomp_full_mc)

# Decomposition of Select using partial + unary iterator strategy


def _ceil(a):
    return int(math.ceil(a))


def _ceil_log(a):
    return _ceil(math.log2(a))


def _add_first_k_units(ops, controls, work_wires, k):
    """Add all controlled-applied operators within the unary iterator scheme.

    This function is used for the outer-most recursion level, and then calls _add_k_units
    for the inner recursion levels.
    In the documentation, this is the outer level that is being described in detail, and
    ``_add_k_units`` corresponds to the subroutine ``R``.

    """
    assert k == len(ops) > 2

    needed_controls = 2 * _ceil_log(k) - 1
    assert len(controls) >= needed_controls, f"{len(controls)=}, {needed_controls=}"
    controls = controls[:needed_controls]

    and_wires = controls[:3]
    new_work_wires = work_wires + controls[:2]
    new_controls = controls[2:]

    a = _ceil_log(k)  # a >= 2 because k>2 by assertion above
    k01 = 2 ** (a - 1)  # First half of circuit will implement 2^(a-1)>=2 operators
    k0 = k1 = 2 ** (a - 2)  # First two quarters of circuit each implement 2^(a-2)>=1 operator(s).
    l = k - k01
    k2 = _ceil(2 ** (_ceil_log(l) - 1))
    k3 = k - k01 - k2

    # Open TemporaryAND (controlled on |00>) + first quarter + CX (controlled on |0>) + second quarter
    first_half = (
        [TemporaryAND(and_wires, control_values=(0, 0))]
        + _add_k_units(ops[:k0], new_controls, new_work_wires, k0)
        + [X(controls[0])]
        + [CNOT([controls[0], controls[2]])]
        + [X(controls[0])]
        + _add_k_units(ops[k0:k01], new_controls, new_work_wires, k1)
    )

    if l == 1:  # first variant

        # Single operation left to apply: Only the third quarter will be needed, and it will not need
        # TemporaryAND gates at all
        and_wires_sec_half = []
        new_controls_sec_half = controls
        new_work_wires_sec_half = work_wires
        # Closing TemporaryAND for first half
        middle_part = [adjoint(TemporaryAND)(and_wires, control_values=(0, 1))]

    else:
        c_bar = 2 * (_ceil_log(k) - _ceil_log(k - k01) - 1)
        and_wires_sec_half = [controls[0], controls[c_bar + 1], controls[c_bar + 2]]
        new_controls_sec_half = controls[c_bar + 2 :]
        new_work_wires_sec_half = work_wires + controls[: c_bar + 2]

        if c_bar > 0:  # second variant
            # Closing TemporaryAND for first half, opening TemporaryAND for second half
            middle_part = [
                adjoint(TemporaryAND)(and_wires, control_values=(0, 1)),
                TemporaryAND(and_wires_sec_half, control_values=(1, 0)),
            ]
        else:  # third variant
            middle_part = [CNOT(and_wires[::2]), CNOT(and_wires[1:])]

    second_half = _add_k_units(
        ops[k01 : k01 + k2], new_controls_sec_half, new_work_wires_sec_half, k2
    )
    if and_wires_sec_half:
        second_half += (
            [CNOT(and_wires_sec_half[::2])]
            + _add_k_units(ops[k0 + k1 + k2 :], new_controls_sec_half, new_work_wires_sec_half, k3)
            + [adjoint(TemporaryAND)(and_wires_sec_half)]
        )

    return first_half + middle_part + second_half


def _add_k_units(ops, controls, work_wires, k):
    """Add k controlled-applied operators within the unary iterator scheme, in a recursive
    manner.

    In the documentation, this subroutine is called ``R``.
    This is _not_ used for the outer-most recursion level, see _add_first_k_units instead.

    We are given ``K=len(ops)`` operators and ``2 * ⌈log_2(K)⌉ + 1`` control and auxiliary wires.
    If ``K=0``, nothing is applied.
    If ``K=1``, the single operator is applied, controlled on the first control wire.

    In all other cases, this function applies the circuit

    .. code-block::

        ─╭●────╭●────●─╮─
        ─├○────│─────●─┤─
         ╰──■──╰X─■────╯

    where each box symbolizes calls to itself on the next recursion level.
    The next-level calls to ``_add_k_units` use

    ``k_first = 2 ** (⌈log_2(k)⌉-1)`` (i.e. half of ``k``, rounded up to the next power of two)
    and
    ``k_second = k-k_first`` (i.e. the rest)

    operators, respectively. Accordingly, two fewer control wires are used.

    """
    assert k == len(ops) > 0
    num_bits = _ceil_log(k)
    needed_controls = 2 * num_bits + 1
    assert len(controls) >= needed_controls, f"{len(controls)=}, {needed_controls=}"

    if k == 1:
        assert num_bits == 0
        return [ctrl(ops[0], control=controls[0], control_values=[1], work_wires=work_wires)]

    assert num_bits != 0
    controls = controls[:1] + controls[-(needed_controls - 1) :]

    and_wires = controls[:3]
    new_work_wires = work_wires + controls[:2]
    new_controls = controls[2:]
    k_first = 2 ** (num_bits - 1)
    return (
        [TemporaryAND(and_wires, control_values=(1, 0))]
        + _add_k_units(ops[:k_first], new_controls, new_work_wires, k_first)
        + [CNOT(and_wires[::2])]
        + _add_k_units(ops[k_first:], new_controls, new_work_wires, k - k_first)
        + [adjoint(TemporaryAND)(and_wires)]
    )


# pylint: disable=unused-argument
def _select_resources_partial_unary(op_reps, num_control_wires):
    num_ops = len(op_reps)
    counts = Counter()

    if num_ops / 2 ** _ceil_log(num_ops) > 3 / 4:
        counts.update(
            {
                resource_rep(TemporaryAND): num_ops - 3,
                adjoint_resource_rep(TemporaryAND): num_ops - 3,
                CNOT: num_ops,
                X: 2,
            }
        )
    else:
        counts.update(
            {
                resource_rep(TemporaryAND): num_ops - 2,
                adjoint_resource_rep(TemporaryAND): num_ops - 2,
                CNOT: num_ops - 2,
                X: 2,
            }
        )

    for op in op_reps:
        counts[controlled_resource_rep(op.op_type, op.params, num_control_wires=1)] += 1

    return dict(counts)


@register_resources(_select_resources_partial_unary)
def _select_decomp_partial_unary(ops, control, work_wires, **_):
    r"""This function reproduces the unary iterator behaviour in https://arxiv.org/abs/1805.03662.
    For :math:`K` operators this decomposition requires at least :math:`c=\lceil\log_2 K\rceil`
    control wires (as usual for Select), and :math:`c-1` additional work wires.
    See the documentation of ``Select`` for details.

    .. note::

        This decomposition assumes that the state on the control wires does not have any overlap
        with :math:`|i\rangle` for :math:`i\geq K`.
    """
    # Note that there might be a slightly improved implementation if a modified embedding is used
    # during state preparation in a PrepSelPrep pattern.

    if len(ops) == 0:
        return []

    min_num_controls = max(_ceil_log(len(ops)), 1)
    assert len(control) >= min_num_controls
    control = control[-min_num_controls:]
    if len(work_wires) < len(control) - 1:
        raise ValueError(
            f"Can't use this decomposition with less than {len(control) - 1} work wires for {len(control)} controls."
        )
    if 1 <= len(ops) <= 2:
        # Don't need unary iterator, just control-apply the one/two operator(s) directly.
        return [
            ctrl(op, control=control[0], control_values=[i], work_wires=work_wires)
            for i, op in enumerate(ops)
        ]
    aux_control = [control[0]]
    for i in range(min_num_controls - 1):
        aux_control.append(control[i + 1])
        aux_control.append(work_wires[i])
    work_wires = work_wires[min_num_controls - 1 :]
    return _add_first_k_units(ops, aux_control, work_wires, len(ops))


add_decomps(Select, _select_decomp_partial_unary)
