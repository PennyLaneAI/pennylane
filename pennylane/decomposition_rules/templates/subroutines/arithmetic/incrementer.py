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

"""Built-in decomposition rules for templates.subroutines.arithmetic.incrementer."""

from pennylane import capture, compiler, math
from pennylane.capture import hint
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition_rules.op_math.controlled2 import (
    flip_zero_control as flip_zero_control2,
)
from pennylane.decomposition_rules.templates.subroutines.qrom import _interleave_controls
from pennylane.ops import CNOT, MultiControlledX, PauliX, X, adjoint, change_op_basis, cond
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.templates.subroutines.arithmetic.incrementer import Incrementer
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.templates.subroutines.multix import MultiX
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires, is_abstract_qubit


def _core_incrementer_resources(num_wires):
    """Resources of the incrementer's core MCX ladder, as a function of the number of wires in
    the ladder (excluding work wires)."""
    resources = {X: 1}
    if num_wires > 1:
        # Forward ladder
        resources[TemporaryAND] = num_wires - 2
        # Backward ladder and trailing CNOT
        resources[CNOT] = num_wires - 2 + 1
        resources[_adjoint_abstract(TemporaryAND)] = num_wires - 2
    return resources


def _incrementer_resources(wires, work_wires=None):  # pylint: disable=unused-argument
    return _core_incrementer_resources(len(wires))


def _has_dynamic_qubits(wires):
    """Whether any wire is a dynamically-allocated ``AbstractQubit`` handle (e.g. from
    ``qp.allocate``), which cannot be stacked into a numeric array for the dynamic-slicing
    ladder construction below. This forces the fallback decomposition; remove once dynamic-wire
    indexing is supported (shortcut.com/story/129521)."""
    return any(is_abstract_qubit(w) for w in wires)


def _work_wire_condition(wires, work_wires):
    return (len(work_wires) + 1) >= len(wires) and not _has_dynamic_qubits(wires)


def _ctrl_work_wire_condition(base, control_wires, work_wires, **_):
    num_wires = len(base.increment_wires) + len(control_wires)
    num_work_wires = len(base.work_wires) + len(work_wires)
    return (num_work_wires + 1) >= num_wires and not _has_dynamic_qubits(base.increment_wires)


def _work_wire_inverse_condition(wires, work_wires):
    return not _work_wire_condition(wires, work_wires)


def _decompose_mcxs(wires, work_wires, control_wires=None):
    if control_wires is None:
        wires = wires[::-1]
        num_controls = 0
    else:
        if compiler.active() or capture.enabled() and control_wires.shape[0] > 0:
            wires = math.concatenate([wires, math.atleast_1d(control_wires)])
        else:
            wires = wires + control_wires
        wires = wires[::-1]
        num_controls = len(control_wires)

    def _increment():
        # Construct the wires on which the ladder will act.
        interleaved = _interleave_controls(wires, work_wires)
        triples = [interleaved[2 * i : 2 * i + 3] for i in range(len(wires) - 2)]
        if compiler.active() or capture.enabled():
            interleaved = math.array(interleaved, like="jax")
            triples = math.array(triples, like="jax")

        # Forward ladder
        @for_loop(len(wires) - 2)
        def forward_ladder(k):
            TemporaryAND(triples[k])

        forward_ladder()  # pylint: disable=no-value-for-parameter

        # The backward ladder applies a CNOT in all iterations except for k < num_controls - 2
        num_ladder_iters = len(wires) - 2
        num_ladder_cnots = num_ladder_iters - max(num_controls - 2, 0)
        cnot_prob = num_ladder_cnots / num_ladder_iters if num_ladder_iters > 0 else 0.0

        # Backward ladder
        @for_loop(len(wires) - 3, -1, -1)
        def backward_adder(k):
            cond(k >= num_controls - 2, hint(CNOT, {"branch-prob": cnot_prob}))(
                [interleaved[2 * k + 2], interleaved[2 * k + 3]]
            )
            adjoint(TemporaryAND)(triples[k])

        backward_adder()  # pylint: disable=no-value-for-parameter

        if num_controls <= 1:
            # Trailing CNOT
            CNOT(wires[:2])

    cond(len(wires) > 1, _increment)()


def _incrementer_fallback_resources(wires, work_wires):
    num_wires = len(wires)
    num_work_wires = len(work_wires)
    resources = {}

    for i in range(num_wires, 1, -1):
        resources[MultiControlledX(Wire[i], work_wires=Wire[num_work_wires])] = 1

    resources[PauliX] = 1

    return resources


@register_condition(_work_wire_inverse_condition)
@register_resources(_incrementer_fallback_resources, name="incrementer_multicontrolledx")
def _incrementer_multicontrolledx(wires, work_wires, **_):
    # This loop is intentionally a plain Python loop rather than a ``qp.for_loop``: each
    # iteration emits a ``MultiControlledX`` with a different number of controls (``i - 1``),
    # so the loop body has no fixed structure and cannot be traced as a dynamic loop. Keeping
    # it unrolled lets this fallback run under program capture without any extra gating.
    num_wires = len(wires)

    for i in range(num_wires, 1, -1):
        MultiControlledX(
            wires[num_wires - i :][::-1],
            [1] * (i - 1),
            work_wires=work_wires,
        )

    X(wires[-1])


@register_condition(_work_wire_condition)
@register_resources(_incrementer_resources, name="incrementer_elbow_ladder")
def _incrementer_elbow_ladder(wires, work_wires, **_):
    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")

    _decompose_mcxs(wires, work_wires)
    X(wires[-1])


def _controlled_incrementer_resources(base, control_wires, **_):
    num_control_wires = len(control_wires)
    resources = _core_incrementer_resources(len(base.increment_wires) + num_control_wires)
    resources[X] = 0
    if num_control_wires > 2:
        resources[CNOT] -= num_control_wires - 2
    resources[CNOT] -= num_control_wires > 1
    return resources


@register_condition(_ctrl_work_wire_condition)
@register_resources(_controlled_incrementer_resources, name="ctrl_incrementer_elbow_ladder")
def _ctrl_incrementer_elbow_ladder(base, control_wires, work_wires, **_):
    wires = base.increment_wires
    base_work_wires = base.work_wires

    if compiler.active() or capture.enabled():
        wires, work_wires, control_wires = (
            math.array(wires, like="jax"),
            math.array(work_wires, like="jax"),
            math.array(control_wires, like="jax"),
        )
        base_work_wires = math.array(base_work_wires, like="jax")
        if base_work_wires.shape[0] > 0 and work_wires.shape[0] > 0:
            # Both arrays are already 1D at this point, guaranteed by the ``shape[0]`` check
            work_wires = math.concatenate([base_work_wires, work_wires])
        elif base_work_wires.shape[0] > 0 and work_wires.shape[0] == 0:
            work_wires = base_work_wires
    else:
        work_wires = base_work_wires + Wires(work_wires)

    _decompose_mcxs(wires, work_wires, control_wires)


def _adjoint_incrementer_resources(base):
    num_wires = len(base.increment_wires)
    multix = MultiX(Bool[num_wires], Wire[num_wires])
    incrementer = Incrementer(Wire[num_wires], work_wires=Wire[len(base.work_wires)])
    return {_change_op_basis_abstract(multix, incrementer, multix): 1}


@register_resources(_adjoint_incrementer_resources, name="adjoint_incrementer_bit_flips")
def _adjoint_incrementer_bit_flips(base):
    """Decompose a decrement into an increment conjugated by bit flips, ``x - 1 = ~(~x + 1)``."""
    wires = base.increment_wires
    ones = [1] * len(wires)
    change_op_basis(
        MultiX(ones, wires),
        Incrementer(wires, work_wires=base.work_wires),
        MultiX(ones, wires),
    )


add_decomps(Incrementer, _incrementer_elbow_ladder, _incrementer_multicontrolledx)

add_decomps("C(Incrementer)", flip_zero_control2(_ctrl_incrementer_elbow_ladder))

add_decomps("Adjoint(Incrementer)", _adjoint_incrementer_bit_flips)
