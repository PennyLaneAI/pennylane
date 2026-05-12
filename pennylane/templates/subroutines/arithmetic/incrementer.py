from pennylane.wires import WiresLike, Wires

from pennylane.decomposition import add_decomps, register_resources, resource_rep, adjoint_resource_rep

from pennylane.templates import TemporaryAND

from pennylane.ops import adjoint, CNOT, X

from pennylane.operation import Operator


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
    """

    resource_keys = {"num_wires"}

    def __init__(self, wires: WiresLike, work_wires: WiresLike = ()):
        wires = Wires(wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        super().__init__(wires=wires + work_wires)

    @property
    def resource_params(self):
        return {
            "num_wires": len(self.wires),
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


@register_resources(_incrementer_resources)
def _incrementer_decomposition(wires, work_wires):
    wires = wires[::-1]
    if len(wires) > 1:
        # Construct the wires on which the ladder will act.
        all_wires = wires[:1] + list(sum(zip(wires[1:], work_wires), start=tuple()))
        # Forward ladder
        for k in range(len(wires) - 2):
            TemporaryAND(all_wires[2 * k : 2 * k + 3])
        # Backward ladder
        for k in range(len(wires) - 3, -1, -1):
            CNOT([all_wires[2 * k + 2], all_wires[2 * k + 3]])
            adjoint(TemporaryAND)(all_wires[2 * k : 2 * k + 3])
        # Trailing CNOT
        CNOT(wires[:2])
    X(wires[0])


add_decomps(Incrementer, _incrementer_decomposition)