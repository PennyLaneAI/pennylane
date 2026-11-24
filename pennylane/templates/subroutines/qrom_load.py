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
This submodule contains the template QROMLoad.
"""

from collections.abc import Sequence

from pennylane.core.operator import Operator2
from pennylane.typing import Bool, TensorLike, Wire
from pennylane.wires import WiresLike

from .qrom import (
    preprocess_qrom_inputs,
)


class QROMLoad(Operator2):
    r"""Load bitstrings into a zeroed target register, depending on the index stored in a control
    register.

    For :math:`L` classical bitstrings :math:`b_0,\dots,b_{L-1}`, ``QROMLoad`` reads the index
    :math:`i` from the control register and writes :math:`b_i` into the target register, which
    has to start out in :math:`|0\rangle`:

    .. math::

        \text{QROMLoad}|i\rangle|0\rangle = \begin{cases}
            |i\rangle|b_i\rangle & \text{if } i < L,\\
            |i\rangle|0\rangle & \text{otherwise.}
        \end{cases}

    So this is :class:`~.QROM` with the promise of a zeroed target, and that promise unlocks
    reduced quantum resources: given enough work wires to hold additional copies of the target
    register, ``QROMLoad`` is decomposed with `Select-SWAP <https://arxiv.org/abs/1812.00954>`__.
    It loads several bitstrings in parallel and then swaps the right one into the target register,
    trading qubits for fewer T gates. With fewer work wires, or if loading in parallel does not pay
    off, it simply falls back to ``QROM``.

    To unload the data again, apply ``qp.adjoint`` of ``QROMLoad``. If you can leave the work
    wires in a garbage state until unloading, use :class:`~.SelectSwap` instead, which skips
    cleaning them up and thus costs less than half. Unloading with ``Adjoint(QROMLoad)`` can be
    decomposed with the measurement-based uncomputation from Appendix C of
    [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__].

    .. warning::

        If the target register does not start out in :math:`|0\rangle`, the result depends on the
        decomposition and generally is *not* the XOR you would get from :class:`~.QROM`. Use
        :class:`~.QROM` if you cannot assume a zeroed target.

    .. seealso:: :class:`~.QROM`, :class:`~.SelectSwap`, :class:`~.BBQRAM`,
        :class:`~.QROMStatePreparation`

    Args:
        bitstrings (TensorLike | Sequence[str]): the :math:`L` bitstrings, either as a 2D array of
            bits or as a sequence of strings like ``"0110"``. ``bitstrings[i]`` is loaded if the
            control register is in the state :math:`|i\rangle`.
        control_wires (WiresLike): the control register holding the index :math:`i`, most
            significant bit first. Requires at least :math:`\lceil\log_2 L\rceil` wires.
        target_wires (WiresLike): the target register, with one wire per bit of the bitstrings.
            Needs to start in :math:`|0\rangle`.
        work_wires (WiresLike): auxiliary wires, which need to start in :math:`|0\rangle`. The more
            work wires, the more bitstrings Select-SWAP can load in parallel. It only uses as many
            as pay off, though, so not all work wires are necessarily used.

    **Example**

    Consider four 3-bit strings, store the index :math:`i=2` in the control register, and
    apply ``QROMLoad``:

    .. code-block:: python

        bitstrings = ["010", "111", "110", "000"]

        @qp.qnode(qp.device("default.qubit"), shots=1)
        def circuit():
            qp.BasisState([1, 0], wires=[0, 1])
            qp.QROMLoad(bitstrings, control_wires=[0, 1], target_wires=[2, 3, 4], work_wires=[5])
            return qp.sample(wires=[2, 3, 4])

    >>> print(circuit())
    [[1 1 0]]

    This is exactly ``bitstrings[2]``, loaded into the target register.

    .. details::
        :title: Cheaper decomposition with Select-SWAP

        For the example above, we could just have used :class:`~.QROM` instead, and without work
        wires ``QROMLoad`` even falls back to ``QROM`` itself. To demonstrate the savings we can
        get from additional work space if the target wires are guaranteed to start in the state
        :math:`|0\rangle`, let us compare the two operators on a larger example: 256 bitstrings
        of length 2, with 11 work wires. We let :func:`~.estimate` count the T gates in their
        decompositions into Clifford+T gates:

        .. code-block:: python

            bitstrings = np.random.default_rng(42).integers(0, 2, size=(256, 2))
            reg = qp.registers({"control": 8, "target": 2, "work": 11})
            args = (bitstrings, reg["control"], reg["target"], reg["work"])

            def t_count(op_type):
                @qp.qjit(capture=True)
                @qp.qnode(qp.device("null.qubit", wires=21))
                def circuit():
                    op_type(*args)
                    return qp.probs()

                counts = qp.estimate(circuit, target=qp.gate_sets.CLIFFORD_T)().resources
                return counts.quantum_operations["T"] + counts.quantum_operations["Adjoint(T)"]

        >>> t_count(qp.QROM), t_count(qp.QROMLoad)
        (1012, 656)

        So ``QROMLoad`` saves more than a third of the T gates. To see where the savings come from,
        let us draw the Select-SWAP decomposition:

        >>> op = qp.QROMLoad(*args)
        >>> rule = qp.list_decomps(qp.QROMLoad)["select_swap"]
        >>> print(qp.draw(rule, show_matrices=False)(**op.arguments))
         0: ────────────────────────────────────────╭◑────────────────────────────────────────────── ···
         1: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         2: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         3: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         4: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         5: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         6: ────────────────╭●────╭●────╭●────╭●────│─────────╭●────╭●────╭●────╭●────────────────── ···
         7: ────╭●────╭●────│─────│─────│─────│─────│─────────│─────│─────│─────│─────╭●────╭●────── ···
         8: ──H─│─────├SWAP─│─────├SWAP─│─────│─────├QROM(M0)─│─────│─────├SWAP─│─────├SWAP─│──────H ···
         9: ──H─├SWAP─│─────├SWAP─│─────│─────│─────├QROM(M0)─│─────│─────│─────├SWAP─│─────├SWAP──H ···
        10: ────│─────╰SWAP─│─────│─────│─────├SWAP─├QROM(M0)─├SWAP─│─────│─────│─────╰SWAP─│─────── ···
        11: ────╰SWAP───────│─────│─────├SWAP─│─────├QROM(M0)─│─────├SWAP─│─────│───────────╰SWAP─── ···
        12: ────────────────│─────╰SWAP─│─────│─────├QROM(M0)─│─────│─────╰SWAP─│─────────────────── ···
        13: ────────────────╰SWAP───────│─────│─────├QROM(M0)─│─────│───────────╰SWAP─────────────── ···
        14: ────────────────────────────│─────╰SWAP─├QROM(M0)─╰SWAP─│─────────────────────────────── ···
        15: ────────────────────────────╰SWAP───────├QROM(M0)───────╰SWAP─────────────────────────── ···
        16: ────────────────────────────────────────├work─────────────────────────────────────────── ···
        17: ────────────────────────────────────────├work─────────────────────────────────────────── ···
        18: ────────────────────────────────────────├work─────────────────────────────────────────── ···
        19: ────────────────────────────────────────├work─────────────────────────────────────────── ···
        20: ────────────────────────────────────────╰work─────────────────────────────────────────── ···
        <BLANKLINE>
         0: ··· ─────────────────────────────────────╭◑────────────────────────────────────────────┤
         1: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         2: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         3: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         4: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         5: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         6: ··· ─────────────╭●────╭●────╭●────╭●────│─────────╭●────╭●────╭●────╭●────────────────┤
         7: ··· ─╭●────╭●────│─────│─────│─────│─────│─────────│─────│─────│─────│─────╭●────╭●────┤
         8: ··· ─│─────├SWAP─│─────├SWAP─│─────│─────├QROM(M0)─│─────│─────├SWAP─│─────├SWAP─│─────┤
         9: ··· ─├SWAP─│─────├SWAP─│─────│─────│─────├QROM(M0)─│─────│─────│─────├SWAP─│─────├SWAP─┤
        10: ··· ─│─────╰SWAP─│─────│─────│─────├SWAP─├QROM(M0)─├SWAP─│─────│─────│─────╰SWAP─│─────┤
        11: ··· ─╰SWAP───────│─────│─────├SWAP─│─────├QROM(M0)─│─────├SWAP─│─────│───────────╰SWAP─┤
        12: ··· ─────────────│─────╰SWAP─│─────│─────├QROM(M0)─│─────│─────╰SWAP─│─────────────────┤
        13: ··· ─────────────╰SWAP───────│─────│─────├QROM(M0)─│─────│───────────╰SWAP─────────────┤
        14: ··· ─────────────────────────│─────╰SWAP─├QROM(M0)─╰SWAP─│─────────────────────────────┤
        15: ··· ─────────────────────────╰SWAP───────├QROM(M0)───────╰SWAP─────────────────────────┤
        16: ··· ─────────────────────────────────────├work─────────────────────────────────────────┤
        17: ··· ─────────────────────────────────────├work─────────────────────────────────────────┤
        18: ··· ─────────────────────────────────────├work─────────────────────────────────────────┤
        19: ··· ─────────────────────────────────────├work─────────────────────────────────────────┤
        20: ··· ─────────────────────────────────────╰work─────────────────────────────────────────┤

        The inner ``QROM``\ s only read the first six control wires, so they have 64 instead of 256
        entries. Each entry stacks four bitstrings, which are loaded onto the target wires 8 and 9
        *and* the work wires 10 to 15 at once. The two remaining control wires (wires 6 and 7) then
        pick the requested bitstring with controlled ``SWAP`` gates around each ``QROM``, moving it
        into the target register. ``QROMLoad`` weighs the reduced cost of two smaller ``QROM`` calls
        and the four controlled ``SWAP`` blocks against the cost of a single full-sized ``QROM``,
        and picks the number of bitstrings to load in parallel that minimizes the combined cost,
        among the powers of two that fit into the work wires.
        Here, eight bitstrings in parallel would not fit, and two would be more expensive than four.

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.QROMLoad
            :symbols: :math:`L` bitstrings of length :math:`n`, :math:`c` control wires and
                :math:`w` work wires
            :example: :math:`L=32`, :math:`n=1`, :math:`c=5` and :math:`w=5`

            .. decomposition-rule:: select_swap
                :image: /_static/decomposition_thumbnails/qrom_load_select_swap.svg
                :lightbox:
                :requires: :math:`w \geq c - 1` work wires, and :math:`k > 1` to be the cheapest
                    choice
                :where: :math:`k = 2^j` bitstrings are loaded in parallel, with :math:`j` chosen to
                    minimize the cost among all :math:`j` with
                    :math:`n(k - 1) + \max(c - j - 1, 0) \leq w`

                :class:`~.QROM` with :math:`\lceil L/k \rceil` bitstrings of length :math:`kn` on :math:`c - j` control wires | :math:`2` | 2
                :class:`~.SWAP` with 1 control | :math:`4n(k - 1)` | 12
                :class:`~.Hadamard` | :math:`2n` | 2

            .. decomposition-rule:: to_qrom
                :image: /_static/decomposition_thumbnails/qrom_load_to_qrom.svg

                :class:`~.QROM` | :math:`1` | 1

            .. decomposition-rule:: measurement
                :image: /_static/decomposition_thumbnails/qrom_load_adjoint_measurement.svg
                :lightbox:
                :region: adjoint
                :requires: program capture, as with :func:`qjit(capture=True) <.qjit>`,
                    and :math:`c \geq 1` control wires
                :where: the :math:`j` least-significant control wires are encoded into a one-hot
                    register of :math:`k = 2^j` auxiliary wires, taken from the target wires
                    after their reset and the work wires, with
                    :math:`j \leq \min(c, \lfloor \log_2 (n + w) \rfloor)` chosen to minimize the
                    estimated T-count. With enough auxiliary wires, this approximately minimizes
                    the Toffoli count :math:`\lceil L/k \rceil + k`

                :class:`~.QROM` with :math:`\lceil L/k \rceil` bitstrings of length :math:`k` on :math:`c - j` control wires and :math:`n + w - k` work wires | :math:`1` | 1
                :func:`PauliMeasure("X") <.pauli_measure>` | :math:`n` | 1
                :func:`PauliMeasure("Z") <.pauli_measure>` | :math:`n` | 1
                :class:`~.PauliX` (conditional) | :math:`n` | 1
                :class:`~.PauliX` | :math:`2` | 2
                :class:`~.TemporaryAND` | :math:`k - 1` | 3
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`k - 1` | 3
                :class:`~.CNOT` | :math:`2(k - 1)` | 6
                :class:`~.Hadamard` | :math:`2k` | 8

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
