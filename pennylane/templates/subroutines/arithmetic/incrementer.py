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
"""Contains the implementation of the Incrementer template."""

from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import CNOT, MultiControlledX, PauliX, X, adjoint
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.controlled2 import flip_zero_control as flip_zero_control2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike

from .temporary_and import TemporaryAND


class Incrementer(Operator2):
    """
    Increment the input ``wires`` by one, using zeroed ``work_wires``.

    Args:
        wires (Wires): The wires that the incrementer acts on.
        work_wires (Wires): The auxiliary wires that the incrementer may use in its decomposition.

    **Example**

    Here we add :math:`2 + 1` to get :math:`3`, using the `Incrementer`.

    .. code-block:: python

        from pennylane import qnode, device, sample, BasisEmbedding, Incrementer
        import numpy as np

        wires = [0, 1, 2]
        work_wires = [3, 4]
        init_state = [0, 1, 0]  # binary representation of 2

        dev = device("default.qubit", wires=wires + work_wires)

        @qnode(dev, shots=1)
        def increment(wires, init_state, work_wires=None):
            BasisEmbedding(init_state, wires)
            Incrementer(wires, work_wires)
            return sample()

        result = increment(wires, init_state, work_wires)[0]

    >>> result[:len(wires)]
    array([0, 1, 1])

    The result incremented the binary value in the non-work wires by 1: :math:`(010)_2 + (001)_2 = (011)_2`.

    .. details::
        :title: Decomposition
        :href: decomposition

        We use a left elbow ladder together with a :class:`~.CNOT` + right :class:`~.TemporaryAND` uncompute ladder.
        This is a manually reduced decomposition of the standard incrementer via :class:`~.MultiControlledX` gates if
        work wires are available.

        Generic decomposition:

        .. code-block::

            0: ─╭X────────────────┤
            1: ─├●─╭X─────────────┤
            2: ─├●─├●─╭X──────────┤
            3: ─├●─├●─├●─╭X───────┤
            4: ─├●─├●─├●─├●─╭X────┤
            5: ─╰●─╰●─╰●─╰●─╰●──X─┤

        Decompose all MCX gates into elbows and CNOTs:

        .. code-block::

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

        .. code-block::

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

        The ``Controlled(Incrementer)`` decomposition provided is a similar decomposition to the default,
        except that there is no ``X`` gate at the end of the circuit, and the ``MultiControlledX`` gates have one
        additional control. It is therefore 'cut-off', and we can follow the same logic as the default
        decomposition, excluding only the trivial X which is not decomposed into elbows and CNOTs
        or cancelled in any case.

        Generic decomposition:

        .. code-block::

            0: ─╭X────────────────┤
            1: ─├●─╭X─────────────┤
            2: ─├●─├●─╭X──────────┤
            3: ─├●─├●─├●─╭X───────┤
            4: ─├●─├●─├●─├●─╭X────┤
            5: ─├●─├●─├●─├●─├●─╭X─┤
            6: ─╰●─╰●─╰●─╰●─╰●─╰●─┤

        Optimized controlled decomposition (controlled on wire 12):

        .. code-block::

            0   : ────────────────╭X────────────────────────────────────┤
            1   : ─────────────╭●─│───●╮─╭X─────────────────────────────┤
            2   : ──────────╭●─│──│────│─│───●╮─╭X──────────────────────┤
            3   : ───────╭●─│──│──│────│─│────│─│───●╮─╭X───────────────┤
            4   : ────╭●─│──│──│──│────│─│────│─│────│─│───●╮─╭X────────┤
            5   : ─╭●─│──│──│──│──│────│─│────│─│────│─│────│─│───●╮─╭X─┤
            aux0: ─├⊕─├●─│──│──│──│────│─│────│─│────│─│───●┤─╰●──⊕┤─│──┤
            aux1: ─│──╰⊕─├●─│──│──│────│─│────│─│───●┤─╰●──⊕╯──────│─│──┤
            aux2: ─│─────╰⊕─├●─│──│────│─│───●┤─╰●──⊕╯─────────────│─│──┤
            aux3: ─│────────╰⊕─├●─│───●┤─╰●──⊕╯────────────────────│─│──┤
            aux4: ─│───────────╰⊕─╰●──⊕╯───────────────────────────│─│──┤
            12  : ─╰●─────────────────────────────────────────────●╯─╰●─┤

    """

    wire_argnames = ("wires", "work_wires")
    arg_specs = {"wires": Wire[-1], "work_wires": Wire[-1]}

    def __init__(self, wires: WiresLike, work_wires: WiresLike = ()):
        wires = Wires(wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        super().__init__(wires, work_wires=work_wires)

    # pylint: disable=arguments-differ
    def __abstract_init__(self, wires, work_wires=()):
        super().__abstract_init__(wires=Wire[len(wires)], work_wires=Wire[len(work_wires)])

    @property
    def increment_wires(self):
        """The wires whose encoded integer is incremented, excluding ``work_wires``."""
        return self.arguments["wires"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.increment_wires + self.work_wires


def _core_incrementer_resources(num_wires):
    """Resources of the work-wire decomposition, as a function of the (bare) number of wires
    that are incremented (i.e. excluding any control or work wires)."""
    resources = {X: 1}
    if num_wires > 1:
        # Forward ladder
        resources[TemporaryAND] = num_wires - 2
        # Backward ladder and trailing CNOT
        resources[CNOT] = num_wires - 2 + 1
        resources[_adjoint_abstract(TemporaryAND)] = num_wires - 2
    return resources


def _incrementer_resources(wires, **_):
    return _core_incrementer_resources(len(wires))


def _work_wire_condition(wires, work_wires, **_):
    return (len(work_wires) + 1) >= len(wires)


def _base_work_wire_condition(base, control_wires, work_wires, **_):
    num_wires = len(base.increment_wires) + len(control_wires)
    num_work_wires = len(base.work_wires) + len(work_wires)
    return (num_work_wires + 1) >= num_wires


def _work_wire_inverse_condition(wires, work_wires, **_):
    return not _work_wire_condition(wires, work_wires)


def _decompose_mcxs(wires, work_wires, control_wires=None):
    if control_wires is None:
        wires = wires[::-1]
        num_controls = 0
    else:
        wires = (wires + control_wires)[::-1]
        num_controls = len(control_wires)

    if len(wires) <= 1:
        return

    # Construct the wires on which the ladder will act.
    zipped = sum(zip(wires[1:], work_wires), start=tuple())
    all_wires = wires[:1] + zipped

    # Forward ladder
    for k in range(len(wires) - 2):
        TemporaryAND(all_wires[2 * k : 2 * k + 3])

    # Backward ladder
    for k in range(len(wires) - 3, -1, -1):
        if k >= num_controls - 2:
            CNOT([all_wires[2 * k + 2], all_wires[2 * k + 3]])
        adjoint(TemporaryAND)(all_wires[2 * k : 2 * k + 3])

    if num_controls <= 1:
        # Trailing CNOT
        CNOT(wires[:2])


def _incrementer_fallback_resources(wires, work_wires, **_):
    num_wires = len(wires)
    num_work_wires = len(work_wires)
    resources = {}

    for i in range(num_wires, 1, -1):
        resources[MultiControlledX(Wire[i], work_wires=Wire[num_work_wires])] = 1

    resources[PauliX] = 1

    return resources


@register_condition(_work_wire_inverse_condition)
@register_resources(_incrementer_fallback_resources)
def _incrementer_fallback_decomposition(wires, work_wires, **_):
    num_wires = len(wires)

    for i in range(num_wires, 1, -1):
        MultiControlledX(
            wires[num_wires - i :][::-1],
            [1 for _ in range(i - 1)],
            work_wires=work_wires,
        )

    X(wires[-1])


@register_condition(_work_wire_condition)
@register_resources(_incrementer_resources)
def _incrementer_decomposition(wires, work_wires, **_):
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


@register_condition(_base_work_wire_condition)
@register_resources(_controlled_incrementer_resources)
def _controlled_incrementer_decomposition(
    *_,
    control_wires,
    work_wires,
    base,
    **__,
):
    wires = base.increment_wires
    work_wires = base.work_wires + Wires(work_wires)

    _decompose_mcxs(wires, work_wires, control_wires)


add_decomps(Incrementer, _incrementer_decomposition)
add_decomps(Incrementer, _incrementer_fallback_decomposition)
add_decomps("C(Incrementer)", flip_zero_control2(_controlled_incrementer_decomposition))
