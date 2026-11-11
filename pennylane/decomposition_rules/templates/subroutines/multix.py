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

"""Built-in decomposition rules for templates.subroutines.multix."""

from pennylane import capture, compiler, math
from pennylane.allocation import allocate
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition_rules.op_math.controlled2 import (
    flip_zero_control as flip_zero_control2,
)
from pennylane.decomposition_rules.op_math.pow2 import pow_involutory
from pennylane.decomposition_rules.op_math.prod2 import _multi_temporary_and_all_ones
from pennylane.decomposition_rules.op_math.symbolicop2 import self_adjoint
from pennylane.ops import CNOT, PauliX, adjoint, cond
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.templates.subroutines.multix import MultiX
from pennylane.typing import TensorLike, Wire
from pennylane.wires import DynamicWire, WiresLike, is_abstract_qubit


# Resources function for MultiX
def _multix_resources(bitstring: TensorLike, wires: WiresLike):  # pylint: disable=unused-argument
    # The total number of PauliX gates used depends on the bitstring.
    # Specifically, sum(bitstring) gates are used by MultiX, not len(bitstring).
    # However, if bitstring is an AbstractArray, only the shape of bitstring is known.
    # Therefore, the resource count can only supply the *worst-case* scenario instead.
    # Hence why exact=False is used when registering the resource.
    return {PauliX: len(wires)}


# Decomposition function for MultiX
@register_resources(_multix_resources, exact=False)
def _multix_decomposition(bitstring: TensorLike, wires: WiresLike) -> None:
    # Dynamically-allocated wires (``AbstractQubit`` handles, e.g. from ``qp.allocate``) cannot
    # be stacked into a numeric array for ``for_loop``-based dynamic indexing, so unroll instead
    # (``bitstring`` may still be traced data, indexed with a static index). Remove this fallback
    # once dynamic-wire indexing is supported (shortcut.com/story/129521).
    if any(is_abstract_qubit(w) or isinstance(w, DynamicWire) for w in wires):
        if compiler.active() or capture.enabled():
            bitstring = math.array(bitstring, like="jax")
        for i, wire in enumerate(wires):
            cond(bitstring[i], PauliX)(wires=wire)
        return

    if compiler.active() or capture.enabled():
        bitstring = math.array(bitstring, like="jax")
        wires = math.array(wires, like="jax")

    @for_loop(0, len(wires), 1)
    def locally_apply_paulix(i):
        cond(bitstring[i], PauliX)(wires=wires[i])

    locally_apply_paulix()  # pylint: disable=no-value-for-parameter


add_decomps(MultiX, _multix_decomposition)

add_decomps("Adjoint(MultiX)", self_adjoint)

add_decomps("Pow(MultiX)", pow_involutory)


def _controlled_multix_ladder_resources(base, control_wires, **_):
    # num_control_wires - 1 TemporaryAND gates to compute the AND ladder, as many to uncompute it,
    # and up to one CNOT per target wire (worst case, hence exact=False).
    num_control_wires = len(control_wires)
    return {
        TemporaryAND: num_control_wires - 1,
        adjoint(TemporaryAND(Wire[3])): num_control_wires - 1,
        CNOT: len(base.wires),
    }


def _multix_ladder_fanout(base, control_wires, work_wires):
    """Computes ``AND(control_wires)`` into a single work wire with the ``TemporaryAND`` ladder in
    :func:`~._multi_temporary_and_all_ones`, fans that wire out with a ``CNOT`` to every set bit of
    ``base``, then uncomputes the ladder. Requires ``len(control_wires) - 1`` wires in
    ``work_wires``, all in the zero state."""
    bitstring = base.bitstring
    wires = base.wires
    if compiler.active() or capture.enabled():
        bitstring = math.array(bitstring, like="jax")

    num_needed = len(control_wires) - 1
    target_wire = _multi_temporary_and_all_ones(control_wires, work_wires)

    def _apply_cnot(wire):
        CNOT(wires=[target_wire, wire])

    if any(
        is_abstract_qubit(w) or isinstance(w, DynamicWire)
        for w in (*wires, *work_wires, *control_wires)
    ):
        for i, wire in enumerate(wires):
            cond(bitstring[i], _apply_cnot)(wire)
    else:
        wires = math.array(wires, like="jax")

        @for_loop(0, len(wires), 1)
        def _locally_apply_cnot(i):
            cond(bitstring[i], _apply_cnot)(wires[i])

        _locally_apply_cnot()  # pylint: disable=no-value-for-parameter

    # Uncompute the ladder. Rebuilt inline rather than via adjoint(_multi_temporary_and_all_ones)(),
    # since adjoint-of-a-qfunc doesn't support DynamicRegister arguments under capture.
    for i in range(num_needed - 1, 0, -1):
        adjoint(TemporaryAND)(wires=[work_wires[i - 1], control_wires[i + 1], work_wires[i]])
    adjoint(TemporaryAND)(wires=[control_wires[0], control_wires[1], work_wires[0]])


def _controlled_multix_ladder_extra_work_wires(control_wires, work_wires, work_wire_type, **_):
    # Allocate only the shortfall beyond any zeroed work wires already supplied.
    num_needed = len(control_wires) - 1
    num_available = len(work_wires) if work_wire_type == "zeroed" else 0
    return {"zeroed": max(0, num_needed - num_available)}


@register_condition(lambda control_wires, **_: len(control_wires) > 1 and not compiler.active())
@register_resources(
    _controlled_multix_ladder_resources,
    work_wires=_controlled_multix_ladder_extra_work_wires,
    exact=False,
)
def _controlled_multix_ladder(base, control_wires, work_wires, work_wire_type, **_):
    """Controlled-``MultiX`` decomposition that loads the fanout through a single work wire
    holding ``AND(control_wires)``, rather than repeating the multi-control structure once per
    target wire. Uses as many of the caller-supplied zeroed work wires as needed, dynamically
    allocating the rest."""
    num_needed = len(control_wires) - 1
    available = list(work_wires[:num_needed]) if work_wire_type == "zeroed" else []
    num_to_allocate = num_needed - len(available)
    with allocate(num_to_allocate, state="zero", restored=True) as allocated:
        _multix_ladder_fanout(base, control_wires, work_wires=available + list(allocated))


add_decomps("C(MultiX)", flip_zero_control2(_controlled_multix_ladder))
