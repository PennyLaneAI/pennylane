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
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike


class Incrementer(Operator2):
    r"""
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
        :title: Available decompositions

        .. decomposition-table:: qp.Incrementer
            :symbols: :math:`n \geq 2` ``wires`` and :math:`w` ``work_wires``
            :example: :math:`n=4`

            .. decomposition-rule:: incrementer_elbow_ladder
                :image: /_static/decomposition_thumbnails/incrementer_elbow_ladder.svg
                :requires: :math:`w \geq n - 1` work wires

                :class:`~.TemporaryAND` | :math:`n - 2` | 2
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`n - 2` | 2
                :class:`~.CNOT` | :math:`n - 1` | 3
                :class:`~.X` | :math:`1` | 1

            .. decomposition-rule:: incrementer_multicontrolledx
                :image: /_static/decomposition_thumbnails/incrementer_multi_controlled_x.svg
                :requires: :math:`w < n - 1` work wires

                :class:`~.MultiControlledX` with :math:`1, \dots, n-1` controls | :math:`n-1` | 3
                :class:`~.X` | :math:`1` | 1

            .. decomposition-rule:: adjoint_incrementer_bit_flips
                :image: /_static/decomposition_thumbnails/incrementer_adjoint.svg
                :region: adjoint

                :class:`~.MultiX` on :math:`n` wires | :math:`2`
                :class:`~.Incrementer` | :math:`1`

            .. decomposition-rule:: flip_zero_ctrl_values(ctrl_incrementer_elbow_ladder)
                :image: /_static/decomposition_thumbnails/incrementer_controlled.svg
                :region: controlled
                :requires: :math:`n + c - 1` work wires in total, from ``work_wires`` and the
                    work wires of the controlled operator
                :where: :math:`c` is the number of control wires, each control wire with
                    control value :math:`0` adds two :class:`~.PauliX` gates, not included in the
                    counts, and the circuit shows :math:`c = 2`

                :class:`~.TemporaryAND` | :math:`n + c - 2`
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`n + c - 2`
                :class:`~.CNOT` | :math:`n`

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

    def __init__(self, wires: WiresLike, work_wires: WiresLike | None = None):
        work_wires = Wires([] if work_wires is None else work_wires)
        super().__init__(wires, work_wires=work_wires)

    @property
    def increment_wires(self):
        """The wires whose encoded integer is incremented, excluding ``work_wires``."""
        return self.arguments["wires"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.increment_wires + self.work_wires
