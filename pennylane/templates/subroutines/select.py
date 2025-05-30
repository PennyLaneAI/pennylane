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

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.decomposition import add_decomps, adjoint_resource_rep, register_resources
from pennylane.operation import Operation
from pennylane.ops import CNOT, Hadamard, S, T, X, adjoint, ctrl
from pennylane.templates.subroutines.temporary_and import TemporaryAnd
from pennylane.wires import Wires


class Select(Operation):
    r"""Applies different operations depending on the state of
    designated control qubits.

    .. math:: Select|i\rangle \otimes |\psi\rangle = |i\rangle \otimes U_i |\psi\rangle

    .. figure:: ../../../doc/_static/templates/subroutines/select.png
                    :align: center
                    :width: 70%
                    :target: javascript:void(0);

    This operator is also known as **multiplexer**, or multiplexed operation.
    If the applied operations :math:`\{U_i\}` are all single-qubit Pauli rotations about the
    same axis, with the angle determined by the control qubits, this is also called a
    **uniformly controlled rotation** gate.

    .. seealso:: :class:`~.SelectPauliRot`

    Args:
        ops (list[Operator]): operations to apply
        control (Sequence[int]): the wires controlling which operation is applied
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

    """

    def _flatten(self):
        return (self.ops), (self.control)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata) -> "Select":
        return cls(data, metadata)

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
                + "wires required."
            )

        if any(
            control_wire in Wires.all_wires([op.wires for op in ops]) for control_wire in control
        ):
            raise ValueError("Control wires should be different from operation wires.")

        for op in ops:
            qml.QueuingManager.remove(op)

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
        states = list(itertools.product([0, 1], repeat=len(control)))
        decomp_ops = [
            ctrl(op, control, control_values=states[index]) for index, op in enumerate(ops)
        ]
        return decomp_ops

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
        """The wires of the input operators."""
        return self.hyperparameters["target_wires"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.hyperparameters["control"] + self.hyperparameters["target_wires"]


def _ceil(a):
    return int(math.ceil(a))


def _ceil_log(a):
    return _ceil(math.log2(a))


def _add_first_k_units(ops, controls, work_wires, k):
    """Add all controlled-applied operators within the unary iterator scheme, in a recursive
    manner.

    This function is used for the outer-most recursion level, and then calls _add_k_units
    for the inner recursion levels.

    Given ``k=len(ops)`` operators and ``2 * ⌈log_2(k)⌉ - 1`` control wires,
    this function applies one of two circuits.
    If ``k>=3/4 * 2**⌈log_2(k)⌉`` (i.e. ``k`` is larger or equal to the maximal capacity for
    the given number of controls), the following circuit is applied:
    ```
    ─╭○─────╭○─────╭●────────╭●─────●─╮─
    ─├○─────│──────│──╭●─────│──────●─┤─
     ╰──╭■──╰X─╭■──╰X─╰X─╭■──╰X─╭■────╯
    ────├■─────├■────────├■─────├■──────
        ╰■     ╰■        ╰■     ╰■
    ```
    where each box symbolizes a call to _add_k_units on the next recursion level.
    Here, two temporary ``AND`` gates were merged into the two single CNOTs and into the pair of
    CNOTs each.

    If ``k<3/4 * 2**⌈log_2(k)⌉`` (i.e. ``k`` is smaller than the maximal capacity for
    the given number of controls), the following circuit is applied instead:
    ```
    ─╭○─────╭○─────○─╮╭●─────╭●─────●─╮─
    ─├○─────│──────●─┤│──────│────────│─
     ╰──╭■──╰X─╭■────╯│      │        │
    ────├■─────├■─────├○─────│──────●─┤─
        ╰■     ╰■     ╰───■──╰X──■────╯
    ```

    That is, the identity merging the central two temporary ``AND`` gates can not be applied,
    and instead the second half, which takes a reduced form, is applied somewhat independently
    of the first half.
    """
    assert k == len(ops) > 2

    needed_controls = 2 * _ceil_log(k) - 1
    assert len(controls) >= needed_controls, f"{len(controls)=}, {needed_controls=}"
    controls = controls[:needed_controls]

    and_wires = controls[:3]
    new_work_wires = work_wires + controls[:2]
    new_controls = controls[2:]

    _k = 2 ** (_ceil_log(k) - 1)
    k0 = _k // 2
    k1 = _k - k0
    k2 = _ceil(2 ** (_ceil_log(k - _k) - 1))
    k3 = k - _k - k2

    first_half = (
        [X(and_wires[0]), X(and_wires[1])]
        + [TemporaryAnd(and_wires)]
        + [X(and_wires[0]), X(and_wires[1])]
        + _add_k_units(ops[:k0], new_controls, new_work_wires, k0)
        + [ctrl(X(controls[2]), control=controls[0], control_values=[0])]
        + _add_k_units(ops[k0 : k0 + k1], new_controls, new_work_wires, k1)
    )

    gap_bitstring = np.binary_repr(k - _k, width=(len(controls) + 1) // 2)
    assert gap_bitstring[0] == "0"
    i = list(gap_bitstring).index("1")
    and_wires_sec_half = [controls[0], controls[2 * i - 1], controls[2 * i]]
    new_controls_sec_half = controls[2 * i :]
    new_work_wires_sec_half = work_wires + controls[: 2 * i]
    if and_wires == and_wires_sec_half:
        middle_part = [CNOT(and_wires[::2]), CNOT(and_wires[1:])]
    else:
        middle_part = [
            X(and_wires[0]),
            adjoint(TemporaryAnd)(and_wires),
            X(and_wires[0]),
            X(and_wires_sec_half[1]),
            TemporaryAnd(and_wires_sec_half),
            X(and_wires_sec_half[1]),
        ]

    second_half = (
        _add_k_units(
            ops[k0 + k1 : k0 + k1 + k2], new_controls_sec_half, new_work_wires_sec_half, k2
        )
        + [CNOT(and_wires_sec_half[::2])]
        + _add_k_units(ops[k0 + k1 + k2 :], new_controls_sec_half, new_work_wires_sec_half, k3)
        + [adjoint(TemporaryAnd)(and_wires_sec_half)]
    )

    return first_half + middle_part + second_half


def _add_k_units(ops, controls, work_wires, k):
    """Add k controlled-applied operators within the unary iterator scheme, in a recursive
    manner.

    This is _not_ used for the outer-most recursion level, see _add_first_k_units instead.

    Given ``k=len(ops)`` operators and ``2 * ⌈log_2(k)⌉ + 1`` control wires,
    this function applies the circuit
    ```
    ─╭●────╭●────●─╮─
    ─├○────│─────●─┤─
     ╰──■──╰X─■────╯
    ```
    where each box symbolizes calls to _add_k_units on the next recursion level.
    The next-level calls to _add_k_units use

    ``k_first = 2 ** (⌈log_2(k)⌉-1)`` (i.e. half of ``k``, rounded up to the next power of two)
    and
    ``k_second = k-k_first`` (i.e. the rest)

    operators, respectively. Accordingly, two control wires less are used.
    """
    assert k == len(ops) > 0
    num_bits = _ceil_log(k)
    needed_controls = 2 * num_bits + 1
    assert len(controls) == needed_controls, f"{len(controls)=}, {needed_controls=}"

    if k == 1:
        return [ctrl(ops[0], control=controls[-1], control_values=[1], work_wires=work_wires)]

    and_wires = controls[:3]
    new_work_wires = work_wires + controls[:2]
    new_controls = controls[2:]
    k_first = 2 ** (num_bits - 1)
    return (
        [X(and_wires[1]), TemporaryAnd(and_wires), X(and_wires[1])]
        + _add_k_units(ops[:k_first], new_controls, new_work_wires, k_first)
        + [CNOT(and_wires[::2])]
        + _add_k_units(ops[k_first:], new_controls, new_work_wires, k - k_first)
        + [adjoint(TemporaryAnd)(and_wires)]
    )


# todo:
def _unary_select_resources():
    return {
        Hadamard: 2,
        CNOT: 3,
        T: 2,
        adjoint_resource_rep(T, {}): 2,
        adjoint_resource_rep(S, {}): 1,
    }


@register_resources(_unary_select_resources)
def _unary_select(ops, control, work_wires):
    r"""This function reproduces the unary iterator behaviour in https://arxiv.org/abs/1805.03662.
    For :math:`L` operators this decomposition requires at least :math:`c=\lceil\log_2 L\rceil`
    control wires (as usual for Select), and :math:`c-1` additional work wires.

    Note that there might be a slightly improved implementation if a modified embedding is allowed.
    """
    if len(ops) == 0:
        return []

    min_num_controls = max(_ceil_log(len(ops)), 1)
    assert len(control) >= min_num_controls
    control = control[:min_num_controls]
    if len(work_wires) < len(control) - 1:
        raise ValueError(
            f"Can't use this decomposition with less than {len(control) - 1} work wires for {len(control)} controls."
        )
    if 1 <= len(ops) <= 2:
        # Don't need unary iterator!
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


add_decomps(Select, _unary_select)
