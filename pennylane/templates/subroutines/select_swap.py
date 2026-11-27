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
"""
This submodule contains the template SelectSwap.
"""

from collections.abc import Sequence

from pennylane import capture, math
from pennylane.core.operator import Operator
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import SWAP, ctrl
from pennylane.typing import Bool, TensorLike, Wire
from pennylane.wires import Wires, WiresLike

from .qrom import preprocess_qrom_inputs
from .qrom_load import (
    QROMLoad,
    _measure_and_reset,
    _measure_and_reset_resources,
    _phase_fixup,
    _phase_fixup_resources,
    _select_ops,
    _select_swap_sizes,
    _select_swap_table,
    _swap_ops,
)


class SelectSwap(Operator):
    r"""Load bitstrings into a zeroed target register with Select-SWAP, leaving the work wires in a
    garbage state.

    For :math:`L` classical bitstrings :math:`b_0,\dots,b_{L-1}`, ``SelectSwap`` reads the index
    :math:`i` from the control register and writes :math:`b_i` into the target register. Both the
    target register and the work wires have to start out in :math:`|0\rangle`:

    .. math::

        \text{SelectSwap}|i\rangle|0\rangle|0\rangle_{\text{work}} = \begin{cases}
            |i\rangle|b_i\rangle|g_i\rangle_{\text{work}} & \text{if } i < L,\\
            |i\rangle|0\rangle|g_i\rangle_{\text{work}} & \text{otherwise,}
        \end{cases}

    where :math:`|g_i\rangle` is a garbage state of the work wires that depends on :math:`i`.

    So this is :class:`~.QROMLoad` without the promise to clean up after itself. Like
    ``QROMLoad``, it is decomposed with `Select-SWAP <https://arxiv.org/abs/1812.00954>`__: it
    loads several bitstrings in parallel, into the target register *and* into work wires, and then
    swaps the right one into the target register. ``QROMLoad`` needs a second round of this to
    return the work wires to :math:`|0\rangle`, whereas ``SelectSwap`` simply leaves the other
    bitstrings behind on the work wires, which cuts the cost by more than half.

    To unload the data again, apply ``qp.adjoint`` of ``SelectSwap`` with the same wires. This
    requires the garbage state of the work wires to be left untouched in the meantime.
    Unloading with ``Adjoint(SelectSwap)`` can be decomposed with the measurement-based
    uncomputation from Appendix C of [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__],
    which measures the garbage together with the target register instead of undoing the SWAPs.
    For long bitstrings, this is cheaper than undoing the SWAPs before uncomputing ``QROMLoad``.

    .. warning::

        If the target register or the work wires do not start out in :math:`|0\rangle`, the result
        is generally *not* the XOR you would get from :class:`~.QROM`. Use :class:`~.QROMLoad` if
        you need the work wires back, and :class:`~.QROM` if you cannot assume a zeroed target.

    .. seealso:: :class:`~.QROM`, :class:`~.QROMLoad`

    Args:
        bitstrings (TensorLike | Sequence[str]): the :math:`L` bitstrings, either as a 2D array of
            bits or as a sequence of strings like ``"0110"``. ``bitstrings[i]`` is loaded if the
            control register is in the state :math:`|i\rangle`.
        control_wires (WiresLike): the control register holding the index :math:`i`, most
            significant bit first. Requires at least :math:`\lceil\log_2 L\rceil` wires.
        target_wires (WiresLike): the target register, with one wire per bit of the bitstrings.
            Needs to start in :math:`|0\rangle`.
        work_wires (WiresLike): auxiliary wires, which need to start in :math:`|0\rangle` and may
            be left in a garbage state. The more work wires, the more bitstrings are loaded in
            parallel. Only as many as pay off are used, though, so not all work wires are
            necessarily used.

    **Example**

    Consider eight 2-bit strings, store the index :math:`i=5` in the control register, and
    apply ``SelectSwap`` with three work wires:

    .. code-block:: python

        bitstrings = ["01", "11", "10", "00", "11", "01", "00", "10"]
        wires = {"control_wires": [0, 1, 2], "target_wires": [3, 4], "work_wires": [5, 6, 7]}

        @qp.qnode(qp.device("default.qubit"), shots=1)
        def circuit(unload=False):
            qp.BasisState([1, 0, 1], wires=[0, 1, 2])
            qp.SelectSwap(bitstrings, **wires)
            if unload:
                qp.adjoint(qp.SelectSwap(bitstrings, **wires))
            return qp.sample(wires=[3, 4]), qp.sample(wires=[5, 6, 7])

    >>> target, work = circuit()
    >>> print(target, work)
    [[0 1]] [[1 1 0]]

    The target register holds ``bitstrings[5]``, as expected. The work wires 5 and 6 hold
    ``bitstrings[4]``, which was loaded in parallel and then swapped out of the target register.
    Wire 7 was only borrowed temporarily and is back in :math:`|0\rangle`. Applying the adjoint
    of ``SelectSwap`` unloads the data and cleans up the garbage again:

    >>> target, work = circuit(unload=True)
    >>> print(target, work)
    [[0 0]] [[0 0 0]]

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.SelectSwap
            :symbols: :math:`L` bitstrings of length :math:`n`, :math:`c` control wires and
                :math:`w` work wires
            :example: :math:`L=32`, :math:`n=1`, :math:`c=5` and :math:`w=5`

            .. decomposition-rule:: select_swap
                :image: /_static/decomposition_thumbnails/select_swap.svg
                :where: :math:`k = 2^j` bitstrings are loaded in parallel, with :math:`j` chosen to
                    minimize the cost among all :math:`j` with
                    :math:`n(k - 1) + \max(c - j - 1, 0) \leq w`, or :math:`k = 1` if
                    :math:`w < c - 1`

                :class:`~.QROMLoad` with :math:`\lceil L/k \rceil` bitstrings of length :math:`kn` on :math:`c - j` control wires | :math:`1` | 1
                :class:`~.SWAP` with 1 control | :math:`n(k - 1)` | 3

            .. decomposition-rule:: measurement
                :image: /_static/decomposition_thumbnails/select_swap_adjoint_measurement.svg
                :lightbox:
                :region: adjoint
                :requires: program capture, as with :func:`qjit(capture=True) <.qjit>`,
                    and :math:`c \geq 1` control wires
                :where: the :math:`nk` target and work wires holding the :math:`k` bitstrings
                    loaded in parallel, with :math:`k` as for ``select_swap``, are measured and
                    reset. Then, the :math:`s` least-significant control wires are encoded into a
                    one-hot register of :math:`m = 2^s` auxiliary wires, taken from the target
                    and work wires, with :math:`s \leq \min(c, \lfloor \log_2 (n + w) \rfloor)`
                    chosen to minimize the estimated T-count, and
                    :math:`L' = k \lceil L/k \rceil`

                :class:`~.QROM` with :math:`\lceil L'/m \rceil` bitstrings of length :math:`m` on :math:`c - s` control wires and :math:`n + w - m` work wires | :math:`1` | 1
                :func:`PauliMeasure("X") <.pauli_measure>` | :math:`nk` | 4
                :func:`PauliMeasure("Z") <.pauli_measure>` | :math:`nk` | 4
                :class:`~.PauliX` (conditional) | :math:`nk` | 4
                :class:`~.PauliX` | :math:`2` | 2
                :class:`~.TemporaryAND` | :math:`m - 1` | 3
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`m - 1` | 3
                :class:`~.CNOT` | :math:`2(m - 1)` | 6
                :class:`~.Hadamard` | :math:`2m` | 8

    """

    dynamic_argnames = ("bitstrings",)
    wire_argnames = ("control_wires", "target_wires", "work_wires")

    arg_specs = {
        "bitstrings": Bool[-1, -1],
        "control_wires": Wire[-1],
        "target_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        bitstrings: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
    ):
        args = (bitstrings, control_wires, target_wires, work_wires)
        super().__init__(*preprocess_qrom_inputs(*args))

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.control_wires + self.target_wires + self.work_wires


def _select_swap_resources(bitstrings, control_wires, target_wires, work_wires):
    num_control_wires_select, num_work_wires_select, _, num_work_wires_cswap, depth = (
        _select_swap_sizes(bitstrings, control_wires, target_wires, work_wires)
    )
    num_targets = len(target_wires)
    num_columns = -(-len(bitstrings) // depth)
    select_rep = QROMLoad(
        Bool[num_columns, depth * num_targets],
        Wire[num_control_wires_select],
        Wire[depth * num_targets],
        Wire[min(num_work_wires_select, max(0, num_control_wires_select - 1))],
    )
    resources = {select_rep: 1}
    if depth > 1:
        cswap_rep = ctrl(
            SWAP(Wire[2]),
            control=Wire[1],
            work_wires=Wire[num_work_wires_cswap],
            work_wire_type="zeroed",
        )
        resources[cswap_rep] = num_targets * (depth - 1)
    return resources


@register_resources(_select_swap_resources, name="select_swap")
def _select_swap(bitstrings, control_wires, target_wires, work_wires):
    num_control_wires_select, _, num_work_wires_swap, num_work_wires_cswap, depth = (
        _select_swap_sizes(bitstrings, control_wires, target_wires, work_wires)
    )

    swap_work_wires = work_wires[:num_work_wires_swap]
    num_work_wires_select = max(0, num_control_wires_select - 1)
    select_work_wires = work_wires[
        num_work_wires_swap : num_work_wires_swap + num_work_wires_select
    ]
    # The Select block returns its work wires to |0>, so the CSWAPs can borrow one of them.
    cswap_work_wires = work_wires[num_work_wires_swap : num_work_wires_swap + num_work_wires_cswap]
    swap_wires = Wires(target_wires) + Wires(swap_work_wires)

    _select_ops(
        bitstrings,
        depth,
        target_wires,
        swap_wires,
        control_wires[:num_control_wires_select],
        select_work_wires,
        zeroed_targets=True,
    )
    _swap_ops(control_wires[num_control_wires_select:], swap_wires, target_wires, cswap_work_wires)


def _adjoint_select_swap_measurement_resources(base):
    *_, depth = _select_swap_sizes(
        base.bitstrings, base.control_wires, base.target_wires, base.work_wires
    )
    num_targets = len(base.target_wires)
    num_entries = depth * -(-len(base.bitstrings) // depth)
    resources = _phase_fixup_resources(
        num_entries, len(base.control_wires), num_targets + len(base.work_wires)
    )
    for op, count in _measure_and_reset_resources(depth * num_targets).items():
        resources[op] += count
    return resources


@register_condition(lambda base: capture.enabled() and len(base.control_wires) > 0)
@register_resources(_adjoint_select_swap_measurement_resources, exact=False, name="measurement")
def _adjoint_select_swap_measurement(base):
    r"""Uncomputation of SelectSwap with measurements and a phase fixup, following Appendix C of
    `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__.

    After ``SelectSwap``, the :math:`k` columns of :math:`n` wires made up of the target wires and
    the first :math:`n(k-1)` work wires hold the :math:`k` bitstrings that were loaded in
    parallel, permuted by the SWAP network depending on the least-significant address bits. All
    other work wires are in the :math:`|0\rangle` state.

    These :math:`nk` wires are measured in the :math:`X` basis and reset to :math:`|0\rangle`.
    With outcomes :math:`m_p` on column :math:`p`, this leaves the phase
    :math:`(-1)^{\sum_p m_p\cdot b_{\pi_i(p)}}` on each address :math:`i`, where
    :math:`b_{\pi_i(p)}` is the bitstring that ``SelectSwap`` moved into column :math:`p`.
    Unlike when inverting the SWAP network first, this phase depends on all address bits, and it
    is undone with the same phase fixup as in the measurement-based uncomputation of
    :class:`~.QROMLoad`, with all target and work wires as auxiliary wires.
    """
    bitstrings = math.cast(base.bitstrings, int)
    target_wires, work_wires = base.target_wires, base.work_wires
    _, _, num_work_wires_swap, _, depth = _select_swap_sizes(
        bitstrings, base.control_wires, target_wires, work_wires
    )

    outcomes = _measure_and_reset(Wires(target_wires) + Wires(work_wires[:num_work_wires_swap]))
    table = _select_swap_table(bitstrings, depth)
    fixups = math.dot(table, math.cast(math.stack(outcomes), int)) % 2
    _phase_fixup(fixups, base.control_wires, Wires(target_wires) + Wires(work_wires))


add_decomps(SelectSwap, _select_swap)
add_decomps("Adjoint(SelectSwap)", _adjoint_select_swap_measurement)
