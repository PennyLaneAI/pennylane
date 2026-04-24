# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains the TemporaryAND template, which also is known as Elbow.
"""

from typing import Literal

from pennylane import math, ops
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    change_op_basis_resource_rep,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.ops.op_math import adjoint
from pennylane.ops.op_math.controlled import ControlledOp
from pennylane.ops.op_math.controlled_ops import _check_and_convert_control_values
from pennylane.ops.qubit import X
from pennylane.wires import Wires, WiresLike


class TemporaryAND(ControlledOp):
    r"""TemporaryAND(wires, control_values, work_wires, work_wire_type)

    The ``TemporaryAND`` operation (also known as ``Elbow``) is a controlled ``X`` gate
    that assumes the target qubit is initialized in :math:`|0\rangle`. This assumption
    enables cheaper decompositions. The ``Adjoint(TemporaryAND)`` uncomputes the operation
    and assumes the target output to be :math:`|0\rangle`.

    For the standard three-qubit case (2 controls), the decomposition uses a phase-baked
    circuit (see Fig. 4 of `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_), and the
    uncompute is performed with a mid-circuit measurement.

    For more than two controls, the decomposition is a ladder of three-qubit
    ``TemporaryAND`` gates acting on the controls and a sequence of work wires.
    At least ``num_controls - 2`` work wires are required.

    .. note::

        For correct usage of this operation, the user must ensure
        that before computation the input state of the target wire is :math:`|0\rangle`,
        and that after uncomputation the output state of the target wire is :math:`|0\rangle`,
        when using ``TemporaryAND`` or ``Adjoint(TemporaryAND)``, respectively.
        Otherwise, behaviour may differ from the expected ``AND``.

    **Details:**

    * Number of wires: Any (at least 3). The last wire is the target; all others are controls.
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the subsystem the gate acts on. All but the last wire
            are the control wires, the last one is the target wire.
        control_values (tuple[bool or int]): The values on the control wires for which
            the target operator is applied. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
            Default is ``(1, 1, ...)``, corresponding to a traditional ``AND`` gate.
        work_wires (Union[Wires, Sequence[int], or int]): optional work wires used to decompose
            multi-controlled (``num_controls > 2``) ``TemporaryAND`` into a ladder of
            3-qubit ``TemporaryAND`` gates.
        work_wire_type (str): whether the work wires are ``"zeroed"`` or ``"borrowed"``.
            Defaults to ``"borrowed"``.

    .. seealso:: The alias :class:`~Elbow`.

    **Example**

    Three-wire case (2 controls):

    .. code-block:: python

        @qp.set_shots(1)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            # |0000>
            qp.X(0) # |1000>
            qp.X(1) # |1100>
            # The target wire is in state |0>, so we can apply TemporaryAND
            qp.TemporaryAND([0,1,2]) # |1110>
            qp.CNOT([2,3]) # |1111>
            # The target wire will be in state |0> after adjoint(TemporaryAND) gate is applied,
            # so we can apply adjoint(TemporaryAND)
            qp.adjoint(qp.TemporaryAND([0,1,2])) # |1101>
            return qp.sample(wires=[0,1,2,3])

    >>> print(qp.draw(circuit)())
    0: ──X─╭●─────●╮─┤ ╭Sample
    1: ──X─├●─────●┤─┤ ├Sample
    2: ────╰⊕─╭●──⊕╯─┤ ├Sample
    3: ───────╰X─────┤ ╰Sample

    Multi-wire case (more than 2 controls, with ``work_wires``):

    .. code-block:: python

        n = 4
        wires = list(range(n))
        work = list(range(n, 2 * n - 2))  # num_controls - 2 work wires

        @qp.transforms.decompose(
            gate_set={"TemporaryAND", "Adjoint(TemporaryAND)"}
        )
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.TemporaryAND(wires=wires, work_wires=work)
            return qp.state()
    """

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    name = "TemporaryAND"
    """str: The operator name. Kept as ``"TemporaryAND"`` so that the gate-set
    matching and registered decomposition rules work transparently despite
    ``TemporaryAND`` now being a :class:`~.ControlledOp`."""

    resource_keys = {
        "num_control_wires",
        "num_zero_control_values",
        "num_work_wires",
        "work_wire_type",
    }

    def _flatten(self):
        return (), (self.wires, tuple(self.control_values), self.work_wires, self.work_wire_type)

    @classmethod
    def _unflatten(cls, _, metadata):
        return cls(
            wires=metadata[0],
            control_values=metadata[1],
            work_wires=metadata[2],
            work_wire_type=metadata[3],
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # @classmethod
    # def _primitive_bind_call(
    #     cls, wires, control_values=None, work_wires=None, work_wire_type="borrowed", id=None
    # ):
    #     return cls._primitive.bind(
    #         *wires,
    #         n_wires=len(wires),
    #         control_values=control_values,
    #         work_wires=work_wires,
    #         work_wire_type=work_wire_type,
    #     )

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        wires: WiresLike,
        control_values: None | bool | list[bool] | int | tuple | list[int] = None,
        work_wires: WiresLike = (),
        work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
        id=None,
    ):
        wires = Wires(() if wires is None else wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        if len(wires) < 3:
            raise ValueError(
                f"TemporaryAND: wrong number of wires. {len(wires)} wire(s) given. "
                f"Need at least 3 (2 controls and 1 target)."
            )

        control_wires = wires[:-1]
        target_wires = wires[-1:]

        control_values = _check_and_convert_control_values(control_values, control_wires)

        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(X, wires=target_wires)
        super().__init__(
            base,
            control_wires=control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
            id=id,
        )

    def __repr__(self):
        cvals = tuple(int(v) for v in self.control_values)
        if all(cvals):
            return f"TemporaryAND(wires={self.wires})"
        return f"TemporaryAND(wires={self.wires}, control_values={cvals})"

    @property
    def wires(self):
        return self.control_wires + self.target_wires

    @property
    def resource_params(self) -> dict:
        return {
            "num_control_wires": len(self.control_wires),
            "num_zero_control_values": len([val for val in self.control_values if not val]),
            "num_work_wires": len(self.work_wires),
            "work_wire_type": self.work_wire_type,
        }

    def adjoint(self):
        return adjoint(TemporaryAND)(
            wires=self.wires,
            control_values=tuple(int(v) for v in self.control_values),
            work_wires=self.work_wires,
            work_wire_type=self.work_wire_type,
        )

    def map_wires(self, wire_map: dict):
        # We override ``Controlled.map_wires`` so that a ``TemporaryAND`` remains a
        # ``TemporaryAND`` after wire relabeling, rather than being simplified to a
        # ``Toffoli``/``MultiControlledX`` through ``ctrl``.
        new_control_wires = [wire_map.get(w, w) for w in self.control_wires]
        new_target_wires = [wire_map.get(w, w) for w in self.target_wires]
        new_work_wires = [wire_map.get(w, w) for w in self.work_wires]
        return TemporaryAND(
            wires=new_control_wires + new_target_wires,
            control_values=tuple(int(v) for v in self.control_values),
            work_wires=new_work_wires,
            work_wire_type=self.work_wire_type,
        )


# ---------------------------------------------------------------------------
# 3-wire (2 control) decomposition.
# ---------------------------------------------------------------------------


def _temporary_and_three_wire_resources(num_control_wires, **__):  # pylint: disable=unused-argument
    number_xs = 4  # worst case scenario
    prod_rep = resource_rep(
        ops.Prod,
        resources={
            resource_rep(ops.Hadamard): 1,
            resource_rep(ops.T): 1,
            resource_rep(ops.CNOT): 1,
            adjoint_resource_rep(ops.T, {}): 1,
        },
    )
    return {
        resource_rep(ops.X): number_xs,
        change_op_basis_resource_rep(prod_rep, ops.CNOT, prod_rep): 1,
        adjoint_resource_rep(ops.S, {}): 1,
    }


@register_condition(lambda num_control_wires, **_: num_control_wires == 2)
@register_resources(_temporary_and_three_wire_resources, exact=False)
def _temporary_and(wires: WiresLike, control_values, **__):
    ops.cond(math.logical_not(control_values[0]), ops.X)(wires[0])
    ops.cond(math.logical_not(control_values[1]), ops.X)(wires[1])

    ops.change_op_basis(
        ops.prod(
            ops.adjoint(ops.T(wires=wires[2])),
            ops.CNOT(wires=[wires[1], wires[2]]),
            ops.T(wires=wires[2]),
            ops.H(wires[2]),
        ),
        ops.CNOT(wires=[wires[0], wires[2]]),
        ops.prod(
            ops.H(wires[2]),
            ops.adjoint(ops.T(wires=wires[2])),
            ops.CNOT(wires=[wires[1], wires[2]]),
            ops.T(wires=wires[2]),
        ),
    )

    ops.adjoint(ops.S(wires=wires[2]))

    ops.cond(math.logical_not(control_values[0]), ops.X)(wires[0])
    ops.cond(math.logical_not(control_values[1]), ops.X)(wires[1])


add_decomps(TemporaryAND, _temporary_and)


# pylint: disable=unused-argument
def _adjoint_temporary_and_resources(base_class=None, base_params=None):
    return {ops.Hadamard: 1, ops.MidMeasure: 1, ops.CZ: 1}


@register_condition(
    lambda base_params=None, **_: base_params is not None
    and base_params.get("num_control_wires", 0) == 2
)
@register_resources(_adjoint_temporary_and_resources)
def _adjoint_TemporaryAND(wires: WiresLike, **kwargs):  # pylint: disable=unused-argument
    r"""The implementation of adjoint TemporaryAND by mid-circuit measurements as found in https://arxiv.org/abs/1805.03662."""
    ops.Hadamard(wires=wires[2])
    m_0 = ops.measure(wires[2], reset=True)
    ops.cond(m_0, ops.CZ)(wires=[wires[0], wires[1]])


add_decomps("Adjoint(TemporaryAND)", _adjoint_TemporaryAND)


# ---------------------------------------------------------------------------
# Multi-wire (num_controls > 2) decomposition using a ladder of 3-wire TemporaryANDs
# ---------------------------------------------------------------------------


def _multi_temporary_and_resources(num_control_wires, num_zero_control_values, **__):
    """Resources of the multi-control decomposition.

    It produces ``num_control_wires - 1`` 3-wire ``TemporaryAND`` gates, plus two ``X`` gates
    per zero-control value (for flipping the control bit before and after the operation).
    """
    # We build num_controls - 1 three-wire TemporaryAND gates
    # The count already accounts for the 3-wire version because each 3-wire
    # TemporaryAND has num_control_wires=2 (and we request its resource_rep here).
    return {
        resource_rep(
            TemporaryAND,
            num_control_wires=2,
            num_zero_control_values=0,
            num_work_wires=0,
            work_wire_type="borrowed",
        ): num_control_wires
        - 1,
        resource_rep(ops.X): 2 * num_zero_control_values,
    }


@register_condition(
    lambda num_control_wires, num_work_wires, **__: num_control_wires > 2
    and num_work_wires >= num_control_wires - 2
)
@register_resources(_multi_temporary_and_resources)
def _multi_temporary_and_decomp_with_work_wires(
    wires,
    control_values,
    work_wires,
    **__,
):
    """Multi-controlled TemporaryAND via a ladder of 3-wire TemporaryANDs.

    Requires at least ``num_controls - 2`` work wires.
    """
    control_wires = wires[:-1]
    c = len(control_wires)

    # Flip zero-control wires
    zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
    for w in zero_control_wires:
        ops.X(w)

    num_needed = c - 1
    # First elbow ties the first two controls together on work_wires[0]
    TemporaryAND(
        wires=[control_wires[0], control_wires[1], work_wires[0]],
    )

    for i in range(1, num_needed):
        _wires = (
            [work_wires[i - 1], control_wires[i + 1], work_wires[i]]
            if i < num_needed - 1
            else [work_wires[i - 1], control_wires[i + 1], wires[-1]]
        )
        TemporaryAND(wires=_wires)

    # Flip zero-control wires back
    for w in zero_control_wires:
        ops.X(w)


add_decomps(TemporaryAND, _multi_temporary_and_decomp_with_work_wires)


# ---------------------------------------------------------------------------
# Helpers for writing resource reps of the default 3-wire TemporaryAND.
# ---------------------------------------------------------------------------


def _default_temporary_and_resource_params(num_zero_control_values: int = 0) -> dict:
    """Return the resource parameters of a 3-wire (2-control) ``TemporaryAND``.

    Used internally by callers that previously invoked ``resource_rep(TemporaryAND)``
    without any keyword arguments (implicitly referring to the 3-wire case).
    """
    return {
        "num_control_wires": 2,
        "num_zero_control_values": num_zero_control_values,
        "num_work_wires": 0,
        "work_wire_type": "borrowed",
    }


Elbow = TemporaryAND
r"""Elbow(wires, control_values, work_wires, work_wire_type)

The Elbow, or :class:`~TemporaryAND` operator.

.. seealso:: The alias :class:`~TemporaryAND` for more details.

**Details:**

* Number of wires: Any (at least 3).

Args:
    wires (Sequence[int] or int): the subsystem the gate acts on.
        All but the last wire are control wires and the last one is the target wire.
    control_values (tuple[bool or int]): The values on the control wires for which
        the target operator is applied. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        Default is ``(1, 1, ...)``, corresponding to a traditional ``AND`` gate.
    work_wires (Union[Wires, Sequence[int], or int]): optional work wires used for the
        multi-controlled decomposition.
    work_wire_type (str): whether the work wires are ``"zeroed"`` or ``"borrowed"``.
        Defaults to ``"borrowed"``.
"""
