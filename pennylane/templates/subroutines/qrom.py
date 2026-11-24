# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This submodule contains the template QROM.
"""

from collections.abc import Sequence

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.math import ceil_log2
from pennylane.typing import AbstractArray, Bool, TensorLike, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


def preprocess_qrom_inputs(bitstrings, control_wires, target_wires, work_wires):
    """Pre-process and validate the inputs to :class:`~.QROM` or :class:`~.QROMLoad` (excluding
    ``clean`` for the latter).
    """
    control_wires = Wires(control_wires)
    target_wires = Wires(target_wires)
    work_wires = Wires(() if work_wires is None else work_wires)

    if not isinstance(bitstrings, AbstractArray):
        if isinstance(bitstrings[0], str):
            bitstrings = [[int(bit) for bit in bitstring] for bitstring in bitstrings]

        if isinstance(bitstrings, (list, tuple)):
            bitstrings = math.array(bitstrings, dtype=bool)

        else:
            bitstrings = bitstrings.astype(bool)

    wire_args = {
        "control_wires": control_wires,
        "target_wires": target_wires,
        "work_wires": work_wires,
    }
    validate_no_wire_overlaps(wire_args)

    if 2 ** len(control_wires) < bitstrings.shape[0]:
        raise ValueError(
            f"Not enough control wires ({len(control_wires)}) for the desired number of "
            f"bitstrings ({bitstrings.shape[0]}). At least {ceil_log2(bitstrings.shape[0])} "
            "control wires are required."
        )

    if bitstrings.shape[1] != len(target_wires):
        raise ValueError("Bitstring length must match the number of target wires.")

    return bitstrings, control_wires, target_wires, work_wires


class QROM(Operator2):
    r"""Flip bits of a target register, depending on the index stored in a control register.

    For :math:`L` classical bitstrings :math:`b_0,\dots,b_{L-1}`, ``QROM`` reads the index
    :math:`i` from the control register and XORs :math:`b_i` onto the target register:

    .. math::

        \text{QROM}|i\rangle|t\rangle = \begin{cases}
            |i\rangle|t\oplus b_i\rangle & \text{if } i < L,\\
            |i\rangle|t\rangle & \text{otherwise.}
        \end{cases}

    The target register can be in any state, including superpositions. ``QROM`` is
    its own inverse, so applying it a second time uncomputes the data again.

    If you know that the target register starts out in :math:`|0\rangle`, use :class:`~.QROMLoad`
    instead. Given enough work wires, it switches to a cheaper decomposition, which ``QROM``
    cannot do. The same goes for uncomputation: if the target is known to end up in
    :math:`|0\rangle`, use the adjoint of :class:`~.QROMLoad`. If, in addition, you can afford to
    leave garbage in the work wires until unloading, use :class:`~.SelectSwap`.

    .. seealso:: :class:`~.QROMLoad`, :class:`~.SelectSwap`, :class:`~.BBQRAM`,
        :class:`~.QROMStatePreparation`

    Args:
        bitstrings (TensorLike | Sequence[str]): the :math:`L` bitstrings, either as a 2D array of
            bits or as a sequence of strings like ``"0110"``. ``bitstrings[i]`` is XORed onto the
            target register if the control register is in the state :math:`|i\rangle`.
        control_wires (WiresLike): the control register holding the index :math:`i`, most
            significant bit first. Requires at least :math:`\lceil\log_2 L\rceil` wires.
        target_wires (WiresLike): the target register, with one wire per bit of the bitstrings.
        work_wires (WiresLike): auxiliary wires, which need to start in :math:`|0\rangle` and are
            returned to it. Up to ``len(control_wires) - 1`` work wires make the decomposition
            cheaper; more are not used.

    .. note::

        QROM and QRAM sound alike, but QRAM is about reading *and* writing data stored in a
        quantum memory, whereas QROM only reads classical data that is hardcoded into the circuit.

    **Example**

    Let us take four 3-bit strings, store the index :math:`i=2` in the control register, and
    prepare the target register in some basis state :math:`|t\rangle=|101\rangle`:

    .. code-block:: python

        bitstrings = ["010", "111", "110", "000"]

        @qp.qnode(qp.device("default.qubit"), shots=1)
        def circuit():
            qp.BasisState([1, 0], wires=[0, 1])
            qp.BasisState([1, 0, 1], wires=[2, 3, 4])
            qp.QROM(bitstrings, control_wires=[0, 1], target_wires=[2, 3, 4], work_wires=[5])
            return qp.sample(wires=[2, 3, 4])

    >>> print(circuit())
    [[0 1 1]]

    This is the prepared target register state XORed with the :math:`i=2` bitstring,
    :math:`101\oplus 110=011`, as expected.

    Since indices :math:`i\geq L` leave the target untouched, a ``QROM`` controlled on
    :math:`|0\rangle` is just a ``QROM`` with an additional control wire in front:

    >>> bitstrings = ["01", "11", "10"]
    >>> op = qp.QROM(bitstrings, control_wires=[0, 1, 2], target_wires=[3, 4], work_wires=[])
    >>> base = qp.QROM(bitstrings, control_wires=[1, 2], target_wires=[3, 4], work_wires=[])
    >>> ctrl_op = qp.ctrl(base, control=0, control_values=0)
    >>> np.allclose(qp.matrix(op), qp.matrix(ctrl_op, wire_order=op.wires))
    True

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.QROM
            :stacked:
            :symbols: :math:`L > 2` bitstrings of length :math:`n`, :math:`c` control wires and
                :math:`w` work wires
            :example: :math:`L=8`, :math:`n=2`, :math:`c=3`, :math:`w=2`

            .. decomposition-rule:: unary_iteration
                :image: /_static/decomposition_thumbnails/qrom_unary_iteration.svg
                :lightbox:
                :where: the iteration tree spans :math:`c' = \min(c, w + 1)` control wires and the
                    other :math:`e = c - c'` control wires control each load, so that the tree
                    iterates over :math:`B = \lceil L / 2^e \rceil` blocks of bitstrings,
                    :math:`N = c' + B - 2 - \mathrm{wt}(B - 1) - [B > 2^{c' - 1}]`,
                    :math:`\mathrm{wt}(k)` is the number of ones in the binary representation of
                    :math:`k`, and :math:`[\cdot]` is 1 if the condition holds and 0 otherwise

                :class:`~.TemporaryAND` | :math:`N` if :math:`w \geq 1` | 5
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`N` if :math:`w \geq 1` | 5
                :class:`~.CNOT` | :math:`B - 1 + [B > 2^{c' - 1}]` if :math:`w \geq 1` | 8
                :class:`~.PauliX` | :math:`2[B > 2^{c' - 2}]` if :math:`w \geq 1` | 2
                :class:`~.MultiX` with :math:`e + 1` controls | :math:`L` | 8

            .. decomposition-rule:: measurement
                :image: /_static/decomposition_thumbnails/qrom_measurement_based.svg
                :lightbox:
                :requires: :math:`w \geq c - 1` work wires
                :where: :math:`K = 2^{\lceil \log_2 L \rceil}`, :math:`e = c - \lceil \log_2 L \rceil`
                    extra control wires, :math:`A = K - 3 + 2[e > 0]` temporary ANDs that are
                    uncomputed with measurements, and :math:`[\cdot]` is 1 if the condition holds
                    and 0 otherwise. Each X-type measurement acts on a work wire and on the target
                    wires flipped by the XOR of two bitstrings, so its width depends on the data

                :class:`~.TemporaryAND` | :math:`A + \max(e - 1, 0)` | 5
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`\max(e - 1, 0)` | 0
                :func:`PauliMeasure("X...X") <.pauli_measure>` | :math:`A` | 5
                :func:`PauliMeasure("Z") <.pauli_measure>` | :math:`A` | 5
                :class:`~.CZ` (conditional) | :math:`A` | 5
                :class:`~.PauliZ` (conditional) | :math:`A` | 5
                :class:`~.PauliX` (conditional) | :math:`A` | 5
                :class:`~.MultiX` (conditional) | :math:`A` | 5
                :class:`~.MultiX` | :math:`[e = 0]` | 1
                :class:`~.MultiX` with 1 control | :math:`1 + [e = 0]` | 2
                :class:`~.MultiControlledX` with 1 control | :math:`[e = 0]` | 1
                :class:`~.CNOT` | :math:`3 + \max(K/2 - 4, 0)` if :math:`e = 0`, :math:`K/2 - 1` if :math:`e > 0` | 3
                :class:`~.PauliX` | :math:`[e = 1]` | 0
                :class:`Adjoint(PauliX) <.PauliX>` | :math:`[e = 1]` | 0

            .. decomposition-rule:: measurement
                :image: /_static/decomposition_thumbnails/qrom_measurement_based.svg
                :lightbox:
                :region: adjoint
                :requires: :math:`w \geq c - 1` work wires
                :where: :class:`~.QROM` is self-adjoint, so this is the same rule as above

                Gates of ``measurement`` above | same counts

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
