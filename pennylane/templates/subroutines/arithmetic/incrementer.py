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
"""Contains the implementation of the incrementer template."""

from pennylane.capture import enabled
from pennylane.control_flow import for_loop
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.math import array
from pennylane.operation import Operator
from pennylane.ops import CNOT, MultiControlledX, PauliX, X, adjoint, cond
from pennylane.wires import Wires, WiresLike

from .temporary_and import TemporaryAND

has_jax = True
try:
    from jax import lax
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_jax = False  # pragma: no cover


class Incrementer(Operator):
    """
    Increment the input `wires` by one, using zeroed `work_wires`.

    Args:
        wires (Wires): The wires that the incrementer acts on.
        work_wires (Wires): The auxiliary wires that the incrementer may use in its decomposition.

    We use a left elbow ladder together with a CNOT+right elbow uncompute ladder.
    This is a manually reduced decomposition of the standard incrementer via MCX gates if
    work wires are available.

    Generic decomposition:
    0: ─╭X────────────────┤
    1: ─├●─╭X─────────────┤
    2: ─├●─├●─╭X──────────┤
    3: ─├●─├●─├●─╭X───────┤
    4: ─├●─├●─├●─├●─╭X────┤
    5: ─╰●─╰●─╰●─╰●─╰●──X─┤

    Decompose all MCX gates into elbows and CNOTs:
    0   : ─────────────╭X──────────────────────────────────────────────────────────────────────────┤
    1   : ──────────╭●─│───●╮──────────────────────╭X──────────────────────────────────────────────┤
    2   : ───────╭●─│──│────│──●╮───────────────╭●─│───●╮───────────────╭X─────────────────────────┤
    3   : ────╭●─│──│──│────│───│──●╮────────╭●─│──│────│──●╮────────╭●─│───●╮────────╭X───────────┤
    4   : ─╭●─│──│──│──│────│───│───│──●╮─╭●─│──│──│────│───│──●╮─╭●─│──│────│──●╮─╭●─│───●╮─╭X────┤
    5   : ─├●─│──│──│──│────│───│───│──●┤─├●─│──│──│────│───│──●┤─├●─│──│────│──●┤─├●─│───●┤─╰●──X─┤
    aux0: ─│──│──├⊕─├●─│───●┤──⊕┤───│───│─│──│──│──│────│───│───│─│──│──│────│───│─│──│────│───────┤
    aux1: ─│──├⊕─╰●─│──│────│──●╯──⊕┤───│─│──├⊕─├●─│───●┤──⊕┤───│─│──│──│────│───│─│──│────│───────┤
    aux2: ─╰⊕─╰●────│──│────│──────●╯──⊕╯─╰⊕─╰●─│──│────│──●╯──⊕╯─╰⊕─├●─│───●┤──⊕╯─│──│────│───────┤
    aux3: ──────────╰⊕─╰●──⊕╯───────────────────╰⊕─╰●──⊕╯────────────╰⊕─╰●──⊕╯─────╰⊕─╰●──⊕╯───────┤

    Cancel neighbouring right and left elbows (moving some work wire usage around in the process)
    0   : ─────────────╭X───────────────────────────────┤
    1   : ──────────╭●─│───●╮─╭X────────────────────────┤
    2   : ───────╭●─│──│────│─│──●╮──╭X─────────────────┤
    3   : ────╭●─│──│──│────│─│───│──│──●╮─╭X───────────┤
    4   : ─╭●─│──│──│──│────│─│───│──│───│─│───●╮─╭X────┤
    5   : ─├●─│──│──│──│────│─│───│──│───│─│───●┤─╰●──X─┤
    aux0: ─│──│──├⊕─├●─│───●┤─╰●─⊕┤──│───│─│────│───────┤
    aux1: ─│──├⊕─╰●─│──│────│────●╯──╰●─⊕┤─│────│───────┤
    aux2: ─╰⊕─╰●────│──│────│───────────●╯─╰●──⊕╯───────┤
    aux3: ──────────╰⊕─╰●──⊕╯───────────────────────────┤

    We see a leading ladder of left elbows and a backwards ladder of CNOT+right elbow pairs.
    This circuit is derived, e.g., in
    `Gidney's blog <https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html>`__,
    see "Incrementer from n-2 Zeroed bits".

    The Controlled(Incrementer) decomposition provided is a similar decomposition to the default,
    except that there are is no X gate at the end of the circuit, and the MCXs have one
    additional control. It is therefore 'cut-off', and we can follow the same logic as the default
    decomposition, excluding only the trivial X which is not decomposed into elbows and CNOTs
    or cancelled in any case.

    Generic decomposition:
    0: ─╭X────────────────┤
    1: ─├●─╭X─────────────┤
    2: ─├●─├●─╭X──────────┤
    3: ─├●─├●─├●─╭X───────┤
    4: ─├●─├●─├●─├●─╭X────┤
    5: ─├●─├●─├●─├●─├●─╭X─┤
    6: ─╰●─╰●─╰●─╰●─╰●─╰●─┤

    Optimized controlled decomposition:
    0 : ─╭|Ψ⟩─────────────╭X────────────────────────────────┤
    1 : ─├|Ψ⟩──────────╭●─│──╭●╮─╭X─────────────────────────┤
    2 : ─├|Ψ⟩───────╭●─│──│──│─│─│──╭●╮─╭X──────────────────┤
    3 : ─├|Ψ⟩────╭●─│──│──│──│─│─│──│─│─│──╭●╮─╭X───────────┤
    4 : ─├|Ψ⟩─╭●─│──│──│──│──│─│─│──│─│─│──│─│─│──╭●╮─╭X────┤
    5 : ─╰|Ψ⟩─├●─│──│──│──│──│─│─│──│─│─│──│─│─│──├●┤─├●─╭X─┤
    6 : ──────├⊕─├●─│──│──│──│─│─│──│─│─│──├●┤─├●─├⊕╯─│──│──┤
    7 : ──────│──├⊕─├●─│──│──│─│─│──├●┤─├●─├⊕╯─│──│───│──│──┤
    8 : ──────│──│──├⊕─├●─│──├●┤─├●─├⊕╯─│──│───│──│───│──│──┤
    9 : ──────│──│──│──├⊕─├●─├⊕╯─│──│───│──│───│──│───│──│──┤
    11: ──────╰●─╰●─╰●─╰●─╰●─╰●──╰●─╰●──╰●─╰●──╰●─╰●──╰●─╰●─┤

    """

    resource_keys = {"num_wires", "num_work_wires"}

    def __init__(self, wires: WiresLike, work_wires: WiresLike = ()):
        wires = Wires(wires)
        work_wires = Wires(() if work_wires is None else work_wires)
        self.hyperparameters["work_wires"] = work_wires

        super().__init__(wires=wires + work_wires)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def resource_params(self):
        return {
            "num_wires": len(self.wires),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
        }


def _incrementer_resources(num_wires, num_work_wires, **_):
    num_wires = num_wires - num_work_wires
    resources = {}
    if num_wires > 1:
        # Forward ladder
        resources[resource_rep(TemporaryAND)] = num_wires - 2
        # Backward ladder plus trailing CNOT
        resources[resource_rep(CNOT)] = num_wires - 2 + 1
        resources[adjoint_resource_rep(TemporaryAND, {})] = num_wires - 2
    resources[resource_rep(X)] = 1
    return resources


def _work_wire_condition(num_wires, num_work_wires, **_):
    return (num_work_wires + 1) >= (num_wires - num_work_wires)


def _base_work_wire_condition(base_params, num_control_wires, num_work_wires, **_):
    return _work_wire_condition(
        base_params["num_wires"] + num_control_wires,
        base_params["num_work_wires"] + num_work_wires,
        **_,
    )


def _work_wire_inverse_condition(num_wires, num_work_wires, **_):
    return not _work_wire_condition(num_wires, num_work_wires)


def _decompose_mcxs(wires, work_wires, control_wires=None):
    if control_wires is None:
        wires = wires[::-1][len(work_wires) :]
    else:
        wires = control_wires + wires[: -len(work_wires)]
        wires = wires[::-1]

    def _increment():
        # Construct the wires on which the ladder will act.
        zipped = sum(zip(wires[1:], work_wires), start=tuple())
        if enabled():
            zipped = array(zipped, like="jax")
        all_wires = wires[:1] + zipped

        # Forward ladder
        @for_loop(len(wires) - 2)
        def forward_ladder(k):
            if enabled():
                TemporaryAND(lax.dynamic_slice(all_wires, (2 * k,), (3,)))
            else:
                TemporaryAND(all_wires[2 * k : 2 * k + 3])

        forward_ladder()  # pylint: disable=no-value-for-parameter

        # Backward ladder
        @for_loop(len(wires) - 3, -1, -1)
        def backward_adder(k):
            CNOT([all_wires[2 * k + 2], all_wires[2 * k + 3]])
            if enabled():
                adjoint(TemporaryAND)(lax.dynamic_slice(all_wires, (2 * k,), (3,)))
            else:
                adjoint(TemporaryAND)(all_wires[2 * k : 2 * k + 3])

        backward_adder()  # pylint: disable=no-value-for-parameter

        # Trailing CNOT
        CNOT(wires[:2])

    cond(len(wires) > 1, _increment)()


def _incrementer_fallback_resources(num_wires, num_work_wires, **_):
    resources = {}

    for i in range(num_wires - num_work_wires - 1, 1, -1):
        resources[
            resource_rep(
                MultiControlledX,
                num_control_wires=i - 1,
                num_zero_control_values=0,
                num_work_wires=num_work_wires,
                work_wire_type="borrowed",
            )
        ] = 1

    resources[resource_rep(PauliX)] = 1

    return resources


@register_condition(_work_wire_inverse_condition)
@register_resources(_incrementer_fallback_resources)
def _incrementer_fallback_decomposition(wires, work_wires, **_):

    if enabled():
        wires = array(wires, like="jax")

    if len(work_wires) > 0:
        wires = wires[: -len(work_wires)]

    @for_loop(len(wires) - 1, 1, -1)
    def flip_wires(i, wires, num_wires):
        MultiControlledX(
            [wires[wire + (num_wires - i)] for wire in range(i)][::-1],
            [1 for _ in range(i - 1)],
            work_wires=work_wires,
        )
        return wires, num_wires

    flip_wires(wires, len(wires))  # pylint: disable=no-value-for-parameter

    X(wires[-1])


@register_condition(_work_wire_condition)
@register_resources(_incrementer_resources)
def _incrementer_decomposition(wires, work_wires, **_):

    if enabled():
        wires = array(wires, like="jax")

    _decompose_mcxs(wires, work_wires)
    X(wires[-len(work_wires) - 1])


def _controlled_incrementer_resources(base_params, **_):
    resources = _incrementer_resources(base_params["num_wires"], base_params["num_work_wires"])
    resources[resource_rep(X)] = 0
    return resources


def _control_values_condition(num_zero_control_values, **_):
    return num_zero_control_values == 0


@register_condition(_base_work_wire_condition)
@register_condition(_control_values_condition)
@register_resources(_controlled_incrementer_resources)
def _controlled_incrementer_decomposition(
    *_,
    control_wires,
    work_wires,
    base,
    **__,
):
    wires = base.wires
    work_wires = base.hyperparameters["work_wires"] + work_wires

    if enabled():
        wires, work_wires, control_wires = (
            array(wires, like="jax"),
            array(work_wires, like="jax"),
            array(control_wires, like="jax"),
        )

    _decompose_mcxs(wires, work_wires, control_wires)


add_decomps(Incrementer, _incrementer_decomposition)
add_decomps(Incrementer, _incrementer_fallback_decomposition)
add_decomps("C(Incrementer)", _controlled_incrementer_decomposition)
