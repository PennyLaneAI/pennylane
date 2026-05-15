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
       0: ─────────────╭X──────────────────────────────────────────────────────────────────────────┤
       1: ──────────╭●─│───●╮──────────────────────╭X──────────────────────────────────────────────┤
       2: ───────╭●─│──│────│──●╮───────────────╭●─│───●╮───────────────╭X─────────────────────────┤
       3: ────╭●─│──│──│────│───│──●╮────────╭●─│──│────│──●╮────────╭●─│───●╮────────╭X───────────┤
       4: ─╭●─│──│──│──│────│───│───│──●╮─╭●─│──│──│────│───│──●╮─╭●─│──│────│──●╮─╭●─│───●╮─╭X────┤
       5: ─├●─│──│──│──│────│───│───│──●┤─├●─│──│──│────│───│──●┤─├●─│──│────│──●┤─├●─│───●┤─╰●──X─┤
    aux0: ─│──│──├⊕─├●─│───●┤──⊕┤───│───│─│──│──│──│────│───│───│─│──│──│────│───│─│──│────│───────┤
    aux1: ─│──├⊕─╰●─│──│────│──●╯──⊕┤───│─│──├⊕─├●─│───●┤──⊕┤───│─│──│──│────│───│─│──│────│───────┤
    aux2: ─╰⊕─╰●────│──│────│──────●╯──⊕╯─╰⊕─╰●─│──│────│──●╯──⊕╯─╰⊕─├●─│───●┤──⊕╯─│──│────│───────┤
    aux3: ──────────╰⊕─╰●──⊕╯───────────────────╰⊕─╰●──⊕╯────────────╰⊕─╰●──⊕╯─────╰⊕─╰●──⊕╯───────┤

    Cancel neighbouring right and left elbows (moving some work wire usage around in the process)
       0: ─────────────╭X───────────────────────────────┤
       1: ──────────╭●─│───●╮─╭X────────────────────────┤
       2: ───────╭●─│──│────│─│──●╮──╭X─────────────────┤
       3: ────╭●─│──│──│────│─│───│──│──●╮─╭X───────────┤
       4: ─╭●─│──│──│──│────│─│───│──│───│─│───●╮─╭X────┤
       5: ─├●─│──│──│──│────│─│───│──│───│─│───●┤─╰●──X─┤
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
    """

    resource_keys = {"num_wires"}

    def __init__(self, wires: WiresLike, work_wires: WiresLike = ()):
        wires = Wires(wires)
        work_wires = Wires(() if work_wires is None else work_wires)
        self.hyperparameters["work_wires"] = work_wires

        super().__init__(wires=wires + work_wires)

    @property
    def resource_params(self):
        return {
            "num_wires": len(self.wires),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
        }


def _incrementer_resources(num_wires):
    resources = {}
    if num_wires > 1:
        # Forward ladder
        resources[resource_rep(TemporaryAND)] = num_wires - 2
        # Backward ladder plus trailing CNOT
        resources[resource_rep(CNOT)] = num_wires - 3 + 1
        resources[adjoint_resource_rep(TemporaryAND, {})] = num_wires - 3
    resources[resource_rep(X)] = 1
    return resources


def _work_wire_condition(num_wires, num_work_wires, **_):
    return num_work_wires >= num_wires - num_work_wires - 1


def _decompose_mcxs(wires, work_wires, control_wires=None):
    if control_wires is None:
        wires = wires[::-1][len(work_wires) :]
    else:
        wires = control_wires + wires[: -len(work_wires)]
        wires = wires[::-1]

    def _increment():
        # Construct the wires on which the ladder will act.
        all_wires = wires[:1] + list(sum(zip(wires[1:], work_wires), start=tuple()))

        if enabled():
            all_wires = array(all_wires, like="jax")

        # Forward ladder
        @for_loop(len(wires) - 2)
        def forward_ladder(k):
            TemporaryAND(all_wires[2 * k : 2 * k + 3])

        forward_ladder()  # pylint: disable=no-value-for-parameter

        # Backward ladder
        @for_loop(len(wires) - 3, -1, -1)
        def backward_adder(k):
            CNOT([all_wires[2 * k + 2], all_wires[2 * k + 3]])
            adjoint(TemporaryAND)(all_wires[2 * k : 2 * k + 3])

        backward_adder()  # pylint: disable=no-value-for-parameter

        # Trailing CNOT
        CNOT(wires[:2])

    cond(len(wires) > 1, _increment)()


def _incrementer_fallback_resources(num_wires, num_work_wires, **_):
    resources = {}

    for i in range(1, num_wires):
        resources[
            resource_rep(
                MultiControlledX,
                num_control_wires=i,
                num_zero_control_values=0,
                num_work_wires=num_work_wires,
            )
        ] = 1

    resources[resource_rep(PauliX)] = 1

    return resources


@register_resources(_incrementer_fallback_resources)
def _incrementer_fallback_decomposition(wires, work_wires, **_):

    if enabled():
        wires = array(wires, like="jax")

    if len(work_wires) > 0:
        wires = wires[: -len(work_wires)]

    @for_loop(len(wires) - 1, 1, -1)
    def flip_wires(i, num_wires):
        MultiControlledX(
            [wire + (num_wires - i) for wire in range(i)][::-1], [1 for _ in range(i - 1)]
        )
        return num_wires

    flip_wires(len(wires))  # pylint: disable=no-value-for-parameter

    X(wires[-1])


@register_condition(_work_wire_condition)
@register_resources(_incrementer_resources)
def _incrementer_decomposition(wires, work_wires, **_):

    if enabled():
        wires = array(wires, like="jax")

    _decompose_mcxs(wires, work_wires)
    X(wires[-len(work_wires) - 1])


def _controlled_incrementer_resources(num_wires):
    resources = _incrementer_resources(num_wires)
    resources[resource_rep(X)] = 0
    return resources


def _control_values_condition(control_values, **_):
    return not sum(map(lambda val: not val, control_values))


@register_condition(_work_wire_condition)
@register_condition(_control_values_condition)
@register_resources(_controlled_incrementer_resources)
def _controlled_incrementer_decomposition(
    *_,
    wires,
    control_wires,
    work_wires,
    base,
    **__,
):
    _decompose_mcxs(base.wires, work_wires, control_wires)


add_decomps(Incrementer, _incrementer_decomposition)
add_decomps(Incrementer, _incrementer_fallback_decomposition)
add_decomps("C(Incrementer)", _controlled_incrementer_decomposition)
