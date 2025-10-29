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
from collections import Counter, defaultdict
from itertools import product

import numpy as np

from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    controlled_resource_rep,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import CNOT, X, adjoint, ctrl
from pennylane.queuing import QueuingManager, apply
from pennylane.wires import Wires

from .arithmetic import TemporaryAND


def _partial_select(K, control):
    r"""Compute the simplified control structure for a partial Select operator.

    Args:
        K (int): The number of operators in ``Select``.
        control (Sequence[hashable]): Control wires.

    Returns:
        list[list[tuple]]: a list of length ``K`` with each entry being a list of length two.
        The first entry in the ``j``\ th inner list is a tuple of control wires that control
        the application of the ``j``\ th operator in the partial ``Select``. The second entry
        in the ``j``\ th inner list is a tuple of control values for these control wires of
        the ``j``\ th operation.

    """
    c = len(control)
    # Here is the logic behind the function below:
    # For the j-th operation, the control values are given by the bit string of j.
    # The r-th control node will only be skipped if
    # a) its control value ``j_r`` is 0, and
    # b) flipping the control node from ``j_r=0`` to ``j_r=1`` yields a bitstring that corresponds
    #    to an integer ``k`` with ``k>=K``.
    # In the loop below, the ``if`` clause decides which nodes to _keep_, not skip, so that we
    # invert the condition "a) and b)" to "not a) or not b)", i.e. "``j_r=1`` or ``k<K``"
    # Finally, note that ``j_r = (j>>(c-1-r)) & 1`` and ``k=j+2**(c-1-r)``.
    controls = [
        [
            (control[r], j_c)
            for r in range(c)
            if (j_c := (j >> (c - 1 - r)) & 1) or j + 2 ** (c - 1 - r) < K
        ]
        for j in range(K)
    ]
    return (list(zip(*ctrl_)) for ctrl_ in controls)


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
        partial (bool): Whether the state on the wires provided in ``control`` are compatible with
            a `partial Select <https://pennylane.ai/compilation/partial-select>`__ decomposition.
            See the note below for details.
        id (str or None): String representing the operation (optional)

    .. note::
        The position of the operation in the list determines which qubit state implements that
        operation. For example, when the qubit register is in the state :math:`|00\rangle`,
        we will apply ``ops[0]``. When the qubit register is in the state :math:`|10\rangle`,
        we will apply ``ops[2]``. To obtain the list position ``index`` for a given binary
        bitstring representing the control state we can use the following relationship:
        ``index = int(state_string, 2)``. For example, ``2 = int('10', 2)``.

    .. note::
        Using ``partial=True`` assumes that the quantum state :math:`|\psi\rangle` on the
        ``control`` wires satisfies :math:`\langle j|\psi\rangle=0` for all :math:`j\in [K, 2^c)`,
        where :math:`K` is the number of operators (``len(ops)``) and :math:`c` is the number of
        control wires (``len(control)``).
        If you are unsure whether this condition is satisfied, set ``partial=False`` to guarantee
        a correct, even though more expensive, decomposition.
        For more details on the partial Select decomposition, see
        `its compilation page <https://pennylane.ai/compilation/partial-select>`__.

    **Example**

    >>> dev = qml.device('default.qubit', wires=4)
    >>> ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2, 3])]
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.Select(ops, control=[0,1])
    ...     return qml.state()
    ...
    >>> print(qml.draw(circuit, level='device')())
    0: ─╭○─╭○─╭●─╭●────┤ ╭State
    1: ─├○─├●─├○─├●────┤ ├State
    2: ─╰X─│──╰Y─├SWAP─┤ ├State
    3: ────╰X────╰SWAP─┤ ╰State

    If there are fewer operators to be applied than possible for the given number of control
    wires, we call the ``Select`` operator a `partial Select <https://pennylane.ai/compilation/partial-select>`__.
    In this case, the control structure can be simplified if the state on the control wires
    does not have overlap with the unused computational basis states (:math:`|j\rangle` with
    :math:`j>K-1`). Passing ``partial=True`` tells ``Select`` that this criterion is
    satisfied, and allows the decomposition to make use of the simplification:

    >>> ops = [qml.X(2), qml.X(3), qml.SWAP([2, 3])]
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.Select(ops, control=[0, 1], partial=True)
    ...     return qml.state()
    ...
    >>> print(qml.draw(circuit, level='device')())
    0: ─╭○────╭●────┤ ╭State
    1: ─├○─╭●─│─────┤ ├State
    2: ─╰X─│──├SWAP─┤ ├State
    3: ────╰X─╰SWAP─┤ ╰State

    Note how the first (second) control node of the second (third) operator was skipped.

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

    resource_keys = {"op_reps", "num_control_wires", "partial", "num_work_wires"}

    @property
    def resource_params(self):
        op_reps = tuple(resource_rep(type(op), **op.resource_params) for op in self.ops)
        return {
            "op_reps": op_reps,
            "num_control_wires": len(self.control),
            "partial": self.partial,
            "num_work_wires": len(self.work_wires),
        }

    def _flatten(self):
        return tuple(self.ops), (
            self.control,
            self.work_wires,
            self.partial,
        )

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, ops, control, **kwargs):
        return super()._primitive_bind_call(*ops, wires=control, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata) -> "Select":
        return cls(data, control=metadata[0], work_wires=metadata[1], partial=metadata[2])

    def __repr__(self):
        return f"Select(ops={self.ops}, control={self.control}, partial={self.partial})"

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, ops, control, work_wires=None, partial=False, id=None):
        control = Wires(control)
        work_wires = Wires(() if work_wires is None else work_wires)
        self.hyperparameters["ops"] = tuple(ops)
        self.hyperparameters["control"] = control
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["partial"] = partial

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
        new_control = [wire_map.get(wire, wire) for wire in self.control]
        new_work_wires = [wire_map.get(wire, wire) for wire in self.work_wires]
        return Select(new_ops, new_control, work_wires=new_work_wires, partial=self.partial)

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
        >>> from pprint import pprint
        >>> pprint(op.decomposition())
        [MultiControlledX(wires=[0, 1, 2], control_values=[False, False]),
        MultiControlledX(wires=[0, 1, 3], control_values=[False, True]),
        Controlled(Y(2), control_wires=[0, 1], control_values=[True, False]),
        Controlled(SWAP(wires=[2, 3]), control_wires=[0, 1])]

        """
        return self.compute_decomposition(
            self.ops, control=self.control, partial=self.partial, work_wires=self.work_wires
        )

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_decomposition(ops, control, partial: bool = False, work_wires=None):
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
        >>> decomp = qml.Select.compute_decomposition(ops, control=[0,1])
        >>> from pprint import pprint
        >>> pprint(decomp)
        [MultiControlledX(wires=[0, 1, 2], control_values=[False, False]),
        MultiControlledX(wires=[0, 1, 3], control_values=[False, True]),
        Controlled(Y(2), control_wires=[0, 1], control_values=[True, False]),
        Controlled(SWAP(wires=[2, 3]), control_wires=[0, 1])]

        """
        if partial:
            if len(ops) == 1:
                if QueuingManager.recording():
                    apply(ops[0])
                return list(ops)
            decomp_ops = [
                ctrl(op, ctrl_, control_values=values, work_wires=work_wires)
                for (ctrl_, values), op in zip(_partial_select(len(ops), control), ops)
            ]
            return decomp_ops

        ctrl_states = product([0, 1], repeat=len(control))
        return [
            ctrl(op, control, control_values=state, work_wires=work_wires)
            for state, op in zip(ctrl_states, ops)
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
    def work_wires(self):
        """The work wires of the Select template."""
        return self.hyperparameters["work_wires"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.hyperparameters["control"] + self.hyperparameters["target_wires"]

    @property
    def partial(self):
        """Operations to be applied."""
        return self.hyperparameters["partial"]


# Decomposition of Select using multi-control strategy


def _multi_controlled_rep(target_rep, num_control_wires, ctrl_state, num_work_wires):
    return controlled_resource_rep(
        base_class=target_rep.op_type,
        base_params=target_rep.params,
        num_control_wires=num_control_wires,
        num_work_wires=num_work_wires,
        num_zero_control_values=num_control_wires - sum(ctrl_state),
    )


def _select_resources_multi_control(op_reps, num_control_wires, partial, num_work_wires):
    resources = defaultdict(int)
    if partial:
        if len(op_reps) == 1:
            resources[op_reps[0]] += 1
        else:
            # Use dummy control values, we will only care about the length of the outputs
            ctrls_and_ctrl_states = _partial_select(len(op_reps), list(range(num_control_wires)))
            for (ctrl_, ctrl_state), rep in zip(ctrls_and_ctrl_states, op_reps):
                resources[_multi_controlled_rep(rep, len(ctrl_), ctrl_state, num_work_wires)] += 1
    else:
        state_iterator = product([0, 1], repeat=num_control_wires)

        for state, rep in zip(state_iterator, op_reps):
            resources[_multi_controlled_rep(rep, num_control_wires, state, num_work_wires)] += 1
    return dict(resources)


@register_resources(_select_resources_multi_control)
def _select_decomp_multi_control(*_, ops, control, work_wires, partial, **__):

    if partial:
        if len(ops) == 1:
            apply(ops[0])
        else:
            for (ctrl_, ctrl_state), op in zip(_partial_select(len(ops), control), ops):
                ctrl(op, ctrl_, control_values=ctrl_state, work_wires=work_wires)
    else:
        for ctrl_state, op in zip(product([0, 1], repeat=len(control)), ops):
            ctrl(op, control, control_values=ctrl_state, work_wires=work_wires)


add_decomps(Select, _select_decomp_multi_control)

# Decomposition of Select using unary iterator


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
        + [ctrl(X(controls[2]), control=controls[0], control_values=[0])]
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


def _select_resources_unary_not_partial(op_reps, num_control_wires, num_work_wires):
    resources = defaultdict(int)
    c = num_control_wires
    K = len(op_reps)
    num_work_wires = num_work_wires - (c - 1)

    if c == 1:
        for i, target_rep in enumerate(op_reps):
            resources[
                controlled_resource_rep(
                    base_class=target_rep.op_type,
                    base_params=target_rep.params,
                    num_control_wires=1,
                    num_zero_control_values=(1 - i),
                    num_work_wires=num_work_wires,
                )
            ] += 1
        return dict(resources)

    def _make_first_flipped_bits(c, i=0):
        """Compute the pattern [c-1, c-2, c-1, c-3, c-1, c-2, c-1, c-4...] recursively.

        For example, for ``c=4``, we get a first call (with ``i=0``) that produces
        ``output =_make_first_flipped_bit(4, 0) = sub_0 + [0] + sub_0``, where
        ``sub_0 = _make_first_flipped_bit(3, 1) = sub_1 + [1] + sub_1``, where
        ``sub_1 = _make_first_flipped_bit(2, 2) = sub_2 + [2] + sub_2``, where
        ``sub_2 = _make_first_flipped_bit(1, 3) = [3]``.

        Overall this gives
        ``sub_1 = [3, 2, 3]``
        ``sub_0 = [3, 2, 3, 1, 3, 2, 3]``
        ``output = [3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3]``.
        """
        if c == 1:
            return [i]
        sub = _make_first_flipped_bits(c - 1, i=i + 1)
        return sub + [i] + sub

    # c-1 left elbows at the beginning and c-1-max(a,1) left elbows for each of the target
    # operators, except the last one, where a is the first flipped bit. Same for right elbows.
    first_flipped_bits = np.array(_make_first_flipped_bits(c)[: K - 1], dtype=int)
    num_elbows = c - 1 + np.sum(c - 1 - np.clip(first_flipped_bits, a_min=1, a_max=None))

    resources[resource_rep(TemporaryAND)] += num_elbows
    resources[adjoint_resource_rep(TemporaryAND)] += num_elbows
    more_than_a_quarter = int(K > 2 ** (c - 2))
    more_than_a_half = int(K > 2 ** (c - 1))
    resources[resource_rep(CNOT)] += K - 1 + more_than_a_half - more_than_a_quarter
    resources[
        controlled_resource_rep(
            base_class=X, base_params={}, num_control_wires=1, num_zero_control_values=1
        )
    ] += more_than_a_quarter
    for op_rep in op_reps:
        resources[
            controlled_resource_rep(
                op_rep.op_type, op_rep.params, num_control_wires=1, num_work_wires=num_work_wires
            )
        ] += 1

    return dict(resources)


# pylint: disable=unused-argument
def _select_resources_unary(op_reps, num_control_wires, partial, num_work_wires):
    num_ops = len(op_reps)
    if num_ops == 0:
        return {}
    if not partial:
        return _select_resources_unary_not_partial(op_reps, num_control_wires, num_work_wires)

    if num_ops == 1:
        return {op_reps[0]: 1}
    counts = Counter()

    if num_ops == 2:
        return {
            controlled_resource_rep(
                op_rep.op_type,
                op_rep.params,
                num_control_wires=1,
                num_work_wires=num_work_wires,
                num_zero_control_values=1 - i,
            ): 1
            for i, op_rep in enumerate(op_reps)
        }
    if num_ops / 2 ** _ceil_log(num_ops) > 3 / 4:
        counts.update(
            {
                resource_rep(TemporaryAND): num_ops - 3,
                adjoint_resource_rep(TemporaryAND): num_ops - 3,
                CNOT: num_ops - 1,
                controlled_resource_rep(X, {}, num_control_wires=1, num_zero_control_values=1): 1,
            }
        )
    else:
        counts.update(
            {
                resource_rep(TemporaryAND): num_ops - 2,
                adjoint_resource_rep(TemporaryAND): num_ops - 2,
                CNOT: num_ops - 3,
                controlled_resource_rep(X, {}, num_control_wires=1, num_zero_control_values=1): 1,
            }
        )

    num_work_wires = num_work_wires - (num_control_wires - 1)
    for op in op_reps:
        counts[
            controlled_resource_rep(
                op.op_type, op.params, num_control_wires=1, num_work_wires=num_work_wires
            )
        ] += 1

    return dict(counts)


def _select_decomp_unary_not_partial(ops, control, work_wires):
    """Decompose Select operator into unary iterator, without applying the partial Select
    reduction. The control structure is simpler without the Select reduction, so that we do
    not use the same recursive structure as for ``_select_decomp_unary`` but a simple ``for`` loop
    instead.

    Args:
        ops (Sequence[Operator]): Operators applied by the Select unary iterator.
        control (Sequence[hashable]): Control wires. Should be at least ``ceil(log2(len(ops)))`` many.
        work_wires (Sequence[hashable]): Work wires. Should be at least ``len(control)-1`` many.

    Returns:
        Sequence[Operator]: Decomposition of a non-partial ``Select`` using unary iteration.

    Denote the number of control qubits as ``c`` and the number of operators as ``K``.
    Arrange the control wires and ``c-1`` work wires as ``["c0", "c1", "w0", "c2", "w1", ...]``.

    We begin with a ladder of ``TemporaryAND`` operators:

    ```
    c0: ─╭○──────────
    c1: ─├○──────────
    w0: ─╰──╭●───────
    c2: ────├○───────
    w1: ────╰──╭●────
     :         :
     :            :
    wp: ──────────╰──
    t1: ─────────────
     :
    tn: ─────────────
    ```

    Here, the ``p`` in ``wp`` is the number of work wires, and ``n`` is the number of target
    wires.
    Then we iterate over the target operators and perform the following steps for each, except
    for the last operator.

    1. Apply the operator, controlled on the last wire ``"wp"`` of the control structure, to the
       target wires ``["t1", ... "tn"]``.
    2. For the ``k``-th operator (in 0-based indexing), find the position ``a`` of the most
       significant bit that flips when incrementing ``k`` to ``k+1``. It will be the position of
       the last bit that is ``0`` in the bit string of ``k``.
    3. Apply right elbows, starting from the lower end of the control structure up to ``a`` (exclusive).
    4. Apply gates that result from merging a right with the next left elbow. This is always a
       ``CNOT`` on the first and last gate of the elbows, but might be complemented with
       ``X`` gates or a second ``CNOT`` gate if ``a=1`` or ``a=0``, respectively.
    5. Apply left elbows corresponding to the right elbows applied in step 3, but starting at
       position ``a`` (exclusive).

    As an example, let ``c=4`` and the operator ``SWAP(["t1", "t2"])`` be the ``k=3``-rd operator
    that is applied, the following subcircuit is appended (using that ``k=0011_2`` and
    ``k+1=0100_2``, so that ``a=1`` is the position of the most significant flipped bit)

    ```
    c0: ─────────────╭●──────────
    c1: ─────────────│───────────
    w0: ──────────●╮─╰X──╭●──────
    c2: ──────────●┤─────├○──────
    w1: ───────●╮──╯─────╰──╭●───
    c3: ───────●┤───────────├○───
    w2: ─╭●─────╯───────────╰────
    t1: ─├SWAP───────────────────
    t2: ─╰SWAP───────────────────
         1.      3.  4.   5.
    ```

    To conclude, we control-apply the last target operator, which was excluded from the loop above,
    and apply a ladder of right elbows across the full control structure, starting at the
    low end:

    ```
    c0: ──────────●╮── *
    c1: ──────────○┤── *
    w0: ───────●╮──╯──
    c2: ───────●┤───── *
    w1: ────●╮──╯─────
     :       :
     :    :
    wp: ──╯───────────
    t1: ──────────────
     :
    tn: ──────────────
    ```

    The value of the control node (filled or open) on the control wires (marked with an
    asterisk (*), i.e. without the work wires) depends on the total number ``K`` of operators
    and can be computed as

    ```
    control_values = np.binary_repr(K-1, width=c)
    ```
    """

    c = len(control)
    K = len(ops)
    if c == 1:
        # Don't need unary iterator, just control-apply the one/two operator(s) directly.
        new_ops = [
            ctrl(op, control=control, control_values=[i], work_wires=work_wires)
            for i, op in enumerate(ops)
        ]
        return new_ops

    # Validate work wires
    p = len(work_wires)
    if p < c - 1:
        raise ValueError(
            f"Can't use this decomposition with less than {c - 1} work wires for {c} controls. "
            f"Got {p} work wires: {work_wires}."
        )

    unary_work_wires = work_wires[: c - 1]
    new_work_wires = work_wires[c - 1 :] + control
    aux_control = [control[0]]
    for ctrl_wire, work_wire in zip(control[1:], unary_work_wires, strict=False):
        aux_control.append(ctrl_wire)
        aux_control.append(work_wire)
    # Create triples of wires to which elbows are applied
    unary_triples = [aux_control[2 * i : 2 * i + 3] for i in range(c - 1)]

    # Apply initial ladder of left elbows
    ops_decomp = [
        TemporaryAND(triple, control_values=((0, 0) if i == 0 else (1, 0)))
        for i, triple in enumerate(unary_triples)
    ]

    first_bit_has_flipped = False
    for k, op in enumerate(ops[:-1]):
        # For all but the last target operator, do the following:
        # 1. apply target operator, always controlled on last unary iteration wire
        ops_decomp.append(ctrl(op, control=aux_control[-1], work_wires=new_work_wires))

        # 2. find the most significant bit ``a`` that flips when incrementing from k to k+1
        first_flip_bit = c - 1 - list(np.binary_repr(k, width=c)[::-1]).index("0")

        # 3. apply the ladder of right elbows up to the most significant flipped bit (exclusive)
        sub_triples = unary_triples[max(first_flip_bit, 1) : c - 1]
        ops_decomp.extend([adjoint(TemporaryAND(triple)) for triple in reversed(sub_triples)])

        # 4. apply gates that result from merging a right and left elbow (``inter_ops``)
        if first_flip_bit == 1:
            c0, c1, c2 = unary_triples[0]
            if first_bit_has_flipped:
                inter_ops = [CNOT([c0, c2])]
            else:
                inter_ops = [ctrl(X(c2), control=c0, control_values=[0])]
        elif first_flip_bit == 0:
            c0, c1, c2 = unary_triples[0]
            inter_ops = [CNOT([c0, c2]), CNOT([c1, c2])]
            first_bit_has_flipped = True
        else:
            inter_ops = [CNOT(unary_triples[first_flip_bit - 1][::2])]
        ops_decomp.extend(inter_ops)

        # 5. apply the ladder of elbows starting at the most significant flipped bit (exclusive)
        ops_decomp.extend([TemporaryAND(triple, control_values=(1, 0)) for triple in sub_triples])

    # For the last target operator, apply controlled target op and then the "closing"
    # ladder of right elbows
    closing_ctrl_bits = list(map(int, np.binary_repr(K - 1, width=c)))
    ops_decomp.append(ctrl(ops[-1], control=aux_control[-1], work_wires=new_work_wires))
    ops_decomp.extend(
        [
            adjoint(TemporaryAND(triple, control_values=(1, val)))
            for val, triple in zip(closing_ctrl_bits[2:], reversed(unary_triples[: c - 1]))
        ]
    )
    ops_decomp.append(adjoint(TemporaryAND(unary_triples[0], control_values=closing_ctrl_bits[:2])))
    return ops_decomp


def _unary_condition(op_reps, num_control_wires, partial, num_work_wires):
    return num_work_wires >= num_control_wires - 1


@register_condition(_unary_condition)
@register_resources(_select_resources_unary)
def _select_decomp_unary(*_, ops, control, work_wires, partial, **__):
    r"""This function reproduces the unary iterator behaviour in https://arxiv.org/abs/1805.03662.
    For :math:`K` operators this decomposition requires at least :math:`c=\lceil\log_2 K\rceil`
    control wires (as usual for Select), and :math:`c-1` additional work wires.
    See the documentation of ``Select`` for details.

    The ``partial`` argument controls whether the reduction to partial Select is performed,
    see the documentation of ``Select`` and https://pennylane.ai/compilation/partial-select for
    details.
    """

    K = len(ops)
    if K == 0:
        return []

    # Validate number of control wires
    c = len(control)
    min_num_controls = max(_ceil_log(K), 1)
    if c < min_num_controls:
        raise ValueError(
            f"At least {min_num_controls} control wires are required to implement Select of "
            f"{K} operators, but only {c} control wires were provided: {control}."
        )

    if not partial:
        return _select_decomp_unary_not_partial(ops, control, work_wires)

    # Due to partial=True, we are allowed to restrict to a subset of the control wires
    control = control[-min_num_controls:]
    if 1 <= K <= 2:
        if K == 1 and partial:
            # Can skip control for partial Select and a single op
            if QueuingManager.recording():
                apply(ops[0])
            return list(ops)
        # Don't need unary iterator, just control-apply the one/two operator(s) directly.
        new_ops = [
            ctrl(op, control=control[-1], control_values=[i], work_wires=work_wires)
            for i, op in enumerate(ops)
        ]
        return new_ops

    # Validate work wires
    p = len(work_wires)
    if p < c - 1:
        raise ValueError(
            f"Can't use this decomposition with less than {c - 1} work wires for {c} controls. "
            f"Got {p} work wires: {work_wires}."
        )

    # Arrange control and work wires into common register
    aux_control = [control[0]]
    for i in range(min_num_controls - 1):
        aux_control.append(control[i + 1])
        aux_control.append(work_wires[i])
    work_wires = work_wires[min_num_controls - 1 :]
    return _add_first_k_units(ops, aux_control, work_wires, len(ops))


add_decomps(Select, _select_decomp_unary)


# Decomposition of Select using one work wire to control the target operations


def _select_multi_control_work_wire_resources(op_reps, num_control_wires, num_work_wires, partial):
    resources = defaultdict(int)

    if partial:
        if len(op_reps) == 1:
            resources[_multi_controlled_rep(op_reps[0], 1, [1], num_work_wires - 1)] += 1
            resources[
                _multi_controlled_rep(
                    resource_rep(X), num_control_wires, [0] * num_control_wires, num_work_wires - 1
                )
            ] += 2
        else:
            # Use dummy control values, we will only care about the length of the outputs
            ctrls_and_ctrl_states = _partial_select(len(op_reps), list(range(num_control_wires)))
            for (ctrl_, ctrl_state), rep in zip(ctrls_and_ctrl_states, op_reps):
                resources[_multi_controlled_rep(rep, 1, [1], num_work_wires - 1)] += 1
                resources[
                    _multi_controlled_rep(
                        resource_rep(X), len(ctrl_), ctrl_state, num_work_wires - 1
                    )
                ] += 2
    else:
        state_iterator = product([0, 1], repeat=num_control_wires)

        for state, rep in zip(state_iterator, op_reps):

            resources[_multi_controlled_rep(rep, 1, [1], num_work_wires - 1)] += 1
            resources[
                _multi_controlled_rep(resource_rep(X), num_control_wires, state, num_work_wires - 1)
            ] += 2
    return dict(resources)


# pylint: disable=unused-argument
def _work_wire_condition(op_reps, num_control_wires, partial, num_work_wires):
    return num_work_wires >= 1


@register_condition(_work_wire_condition)
@register_resources(_select_multi_control_work_wire_resources)
def _select_decomp_multi_control_work_wire(*_, ops, control, work_wires, partial, **__):
    """
    Multi-controlled gate decomposition, in which, instead of directly controlling the target operator with all control
    wires, an auxiliary qubit is employed to encode whether the control condition is satisfied. The target operator
    is then applied as a single-qubit controlled gate from this auxiliary qubit.
    An example of this decomposition can be found in Figure 1(a):  https://arxiv.org/abs/1812.00954
    """

    if len(ops) == 0:
        return []

    if partial:
        if len(ops) == 1:
            ctrl(
                X(work_wires[:1]),
                control,
                control_values=[0] * len(control),
                work_wires=work_wires[1:],
            )
            ctrl(ops[0], control=work_wires[:1], work_wires=work_wires[1:])
            ctrl(
                X(work_wires[:1]),
                control,
                control_values=[0] * len(control),
                work_wires=work_wires[1:],
            )
            return []

        ctrls_and_ctrl_states = _partial_select(len(ops), control)
        for (ctrl_, ctrl_state), op in zip(ctrls_and_ctrl_states, ops, strict=True):
            ctrl(X(work_wires[:1]), ctrl_, control_values=ctrl_state, work_wires=work_wires[1:])
            ctrl(op, control=work_wires[:1], work_wires=work_wires[1:])
            ctrl(X(work_wires[:1]), ctrl_, control_values=ctrl_state, work_wires=work_wires[1:])
        return []

    for ctrl_state, op in zip(product([0, 1], repeat=len(control)), ops, strict=False):
        ctrl(X(work_wires[:1]), control, control_values=ctrl_state, work_wires=work_wires[1:])
        ctrl(op, control=work_wires[:1], work_wires=work_wires[1:])
        ctrl(X(work_wires[:1]), control, control_values=ctrl_state, work_wires=work_wires[1:])
    return []


add_decomps(Select, _select_decomp_multi_control_work_wire)

# pylint: disable=protected-access
if Select._primitive is not None:

    @Select._primitive.def_impl
    def _(*args, n_wires, **kwargs):
        ops, control = args[:-n_wires], args[-n_wires:]
        return type.__call__(Select, ops, control=control, **kwargs)
