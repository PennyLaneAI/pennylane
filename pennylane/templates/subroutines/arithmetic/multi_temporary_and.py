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
r"""
Contains the ``MultiTemporaryAND`` template: a ``MultiControlledX``-like gate whose
decomposition is implemented through a ladder of :class:`~.TemporaryAND` operations
when enough work wires are available.
"""

from typing import Literal

from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import X, ctrl
from pennylane.ops.op_math.controlled_ops import (
    _check_and_convert_control_values,
)
from pennylane.wires import Wires, WiresLike

from .temporary_and import TemporaryAND


class MultiTemporaryAND(Operation):
    r"""A multi-controlled ``X`` gate whose decomposition is realised through a ladder of
    :class:`~.TemporaryAND` operations when enough work wires are available.

    The circuit with four control and two work wires has the following repeating structure:

    .. code-block::

        control0: ─╭●───────┤
        control1: ─├●───────┤
        control2: ─│──╭●────┤
        control3: ─│──│──╭●─┤
           work0: ─╰⊕─├●─│──┤
           work1: ────╰⊕─├●─┤
         target0: ───────╰⊕─┤

    .. note::

        Unlike :class:`~.MultiControlledX`, the control and target wires are passed as two
        **separate** arguments (rather than one concatenated ``wires`` list whose last entry
        is implicitly the target). The decomposition is registered against this class and
        emits ``len(control_wires) - 2`` :class:`~.TemporaryAND` gates (plus at most one
        final :class:`~.CNOT`) when ``len(work_wires) >= len(control_wires) - 2``.

    **Details:**

    * Number of wires: ``len(control_wires) + 1 + len(work_wires)`` (at least 2 qubits
      are involved in the operation itself, plus any optional work wires).
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        control_wires (Union[Wires, Sequence[int], int]): the wires the operation is
            controlled on. At least one control wire is required.
        target_wire (Union[Wires, Sequence[int], int]): the single wire the ``X``
            gate is applied to when every control wire matches its control value.
        control_values (Union[bool, list[bool], int, list[int], str, None]): the value(s)
            the control wire(s) should take. Integers other than 0 or 1 will be treated
            as ``int(bool(x))``. Strings of the form ``"101"`` are also accepted.
            Defaults to ``None``, which corresponds to controlling on the
            :math:`|1\rangle` state of every control wire.
        work_wires (Union[Wires, Sequence[int], int]): optional work wires that enable
            the efficient :class:`~.TemporaryAND` ladder decomposition. At least
            ``len(control_wires) - 1`` work wires are required to trigger that rule.
        work_wire_type (str): whether the work wires are ``"zeroed"`` (in the
            :math:`|0\rangle` state) or ``"borrowed"`` (in an arbitrary state).
            Defaults to ``"borrowed"``.

    .. seealso:: :class:`~.MultiControlledX`, :class:`~.TemporaryAND`.

    **Example**

    We set up an example with ``n=4`` ``control wires`` and ``n - 2`` work wires.

    .. code-block:: python

        import pennylane as qp
        from pennylane import MultiTemporaryAND

        qp.decomposition.enable_graph()

        control_wires = [f"control{i}" for i in range(4)]
        target_wire = [f"target{i}" for i in range(1)]
        work_wires = [f"work{i}" for i in range(3)]

        @qp.transforms.decompose(
            gate_set={"TemporaryAND", "CNOT"}
        )
        @qp.qnode(qp.device("default.qubit"))
        def qnode():
            MultiTemporaryAND(
                control_wires=control_wires,
                target_wire=target_wire,
                work_wires=work_wires,
                work_wire_type="zeroed"
            )
            return qp.state()

    We can now see that the decomposition uses a
    :class:`~.TemporaryAND` ladder instead of a :class:`~.MultiControlledX`:

    >>> print(qp.draw(qnode, wire_order=control_wires+work_wires+target_wire)())
    control0: ─╭●───────┤  State
    control1: ─├●───────┤  State
    control2: ─│──╭●────┤  State
    control3: ─│──│──╭●─┤  State
       work0: ─╰⊕─├●─│──┤  State
       work1: ────╰⊕─├●─┤  State
     target0: ───────╰⊕─┤  State
    """

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = {
        "num_control_wires",
        "num_zero_control_values",
        "num_work_wires",
        "work_wire_type",
    }

    name = "MultiTemporaryAND"

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        control_wires: WiresLike,
        target_wire: WiresLike,
        control_values: None | bool | list[bool] | int | list[int] | str = None,
        work_wires: WiresLike = (),
        work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
        id=None,
    ):
        control_wires = Wires(control_wires)
        target_wire = Wires(target_wire)
        work_wires = Wires(work_wires)

        if len(control_wires) == 0:
            raise ValueError("MultiTemporaryAND requires at least one control wire; got 0.")
        if len(target_wire) != 1:
            raise ValueError(
                f"MultiTemporaryAND requires exactly one target wire; got {len(target_wire)}."
            )
        if work_wire_type not in {"zeroed", "borrowed"}:
            raise ValueError(
                f"work_wire_type must be either 'zeroed' or 'borrowed'. Got '{work_wire_type}'."
            )

        control_values = tuple(_check_and_convert_control_values(control_values, control_wires))

        if len(Wires.shared_wires([control_wires, target_wire])) != 0:
            raise ValueError("Target wire must be different from the control wires.")
        if len(Wires.shared_wires([work_wires, control_wires + target_wire])) != 0:
            raise ValueError(
                "Work wires must be different from the control wires and the target wire."
            )

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["target_wire"] = target_wire
        self.hyperparameters["control_values"] = control_values
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["work_wire_type"] = work_wire_type

        all_wires = control_wires + target_wire + work_wires
        super().__init__(wires=all_wires, id=id)

    def __repr__(self):
        cv = self.hyperparameters["control_values"]
        ww = self.hyperparameters["work_wires"]
        base = (
            f"MultiTemporaryAND(control_wires={self.control_wires.tolist()}, "
            f"target_wire={self.target_wire.tolist()}"
        )
        if not all(cv):
            base += f", control_values={list(cv)}"
        if len(ww) > 0:
            base += f", work_wires={ww.tolist()}, work_wire_type={self.work_wire_type!r}"
        return base + ")"

    def _flatten(self):
        hp = self.hyperparameters
        metadata = (
            hp["control_wires"],
            hp["target_wire"],
            hp["control_values"],
            hp["work_wires"],
            hp["work_wire_type"],
        )
        return (), metadata

    @classmethod
    def _unflatten(cls, _, metadata):
        control_wires, target_wire, control_values, work_wires, work_wire_type = metadata
        return cls(
            control_wires=control_wires,
            target_wire=target_wire,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def control_wires(self) -> Wires:
        """The wires that the operation is controlled on."""
        return self.hyperparameters["control_wires"]

    @property
    def target_wire(self) -> Wires:
        """The single target wire the ``X`` gate is applied to."""
        return self.hyperparameters["target_wire"]

    @property
    def target_wires(self) -> Wires:
        """Alias for :attr:`target_wire`, matching the :class:`~.ControlledOp` interface."""
        return self.hyperparameters["target_wire"]

    @property
    def control_values(self) -> tuple[bool, ...]:
        """The control values for each control wire."""
        return self.hyperparameters["control_values"]

    @property
    def work_wires(self) -> Wires:
        """The work wires that may be used in the decomposition."""
        return self.hyperparameters["work_wires"]

    @property
    def work_wire_type(self) -> str:
        """Whether the work wires are ``"zeroed"`` or ``"borrowed"``."""
        return self.hyperparameters["work_wire_type"]

    @property
    def resource_params(self) -> dict:
        return {
            "num_control_wires": len(self.control_wires),
            "num_zero_control_values": sum(1 for v in self.control_values if not v),
            "num_work_wires": len(self.work_wires),
            "work_wire_type": self.work_wire_type,
        }

    def map_wires(self, wire_map: dict) -> "MultiTemporaryAND":
        new_control_wires = [wire_map.get(w, w) for w in self.control_wires]
        new_target_wire = [wire_map.get(w, w) for w in self.target_wire]
        new_work_wires = [wire_map.get(w, w) for w in self.work_wires]
        return MultiTemporaryAND(
            control_wires=new_control_wires,
            target_wire=new_target_wire,
            control_values=self.control_values,
            work_wires=new_work_wires,
            work_wire_type=self.work_wire_type,
        )


# Decomposition rule:


def _multi_temporary_and_resources(num_control_wires, **__):
    if num_control_wires == 1:
        return {resource_rep(X): 1}
    return {resource_rep(TemporaryAND): num_control_wires - 1}


@register_condition(
    lambda num_control_wires, num_work_wires, **_: (
        num_control_wires == 1 or num_work_wires >= num_control_wires - 2
    )
)
@register_resources(_multi_temporary_and_resources)
def _multi_temporary_and_decomp_with_work_wires(  # pylint: disable=unused-argument,too-many-arguments
    wires,
    control_wires,
    target_wire,
    control_values,
    work_wires,
    work_wire_type,
    **__,
):
    """Decomposition using a :class:`~.TemporaryAND` ladder.

    Needs ``len(control_wires) - 1`` work wires (or zero work wires when there is a
    single control wire, in which case the operation reduces to a single (X or CNOT)).
    """
    c = len(control_wires)
    if c == 1:
        # Reduces to (X or CNOT) depending on the control value.
        ctrl(
            X(target_wire),
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )
        return

    num_needed = c - 1
    TemporaryAND(
        wires=[control_wires[0], control_wires[1], work_wires[0]],
        control_values=(control_values[0], control_values[1]),
    )

    for i in range(1, num_needed):
        _wires = (
            [work_wires[i - 1], control_wires[i + 1], work_wires[i]]
            if i < num_needed - 1
            else [work_wires[i - 1], control_wires[i + 1], target_wire[0]]
        )
        TemporaryAND(
            wires=_wires,
            control_values=(True, control_values[i + 1]),
        )


add_decomps(
    MultiTemporaryAND,
    _multi_temporary_and_decomp_with_work_wires,
)
