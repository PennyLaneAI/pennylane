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
This submodule contains the template for QROM.
"""

from collections.abc import Sequence

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.math import ceil_log2
from pennylane.typing import AbstractArray, Bool, TensorLike, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


class QROM(Operator2):
    r"""Applies the QROM operator.

    This operator encodes bitstrings associated with indexes:

    .. math::
        \text{QROM}|i\rangle|0\rangle = |i\rangle |b_i\rangle,

    where :math:`b_i` is the bitstring associated with index :math:`i`.

    Args:
        bitstrings (TensorLike): the data to be encoded
        control_wires (WiresLike):
            The register that stores the index for the entry of the classical data we want to
            read.
        target_wires (Sequence[int]): the wires where the bitstring is loaded
        work_wires (Sequence[int]): the auxiliary wires used for the computation
        clean (bool): if True, the work wires are not altered by operator, default is ``True``

    .. seealso:: :class:`~.BBQRAM`, :class:`~.QROMStatePreparation`

    .. note::
        QRAM and QROM, though similar, have different applications and purposes. QRAM is intended
        for read-and-write capabilities, where the stored data can be loaded and changed. QROM is
        designed to only load stored data into a quantum register.

    **Example**

    In this example, the QROM operator is applied to encode the third bitstring, associated with index 2, in the target wires.

    .. code-block:: python

        # a list of bitstrings is defined
        bitstrings = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]

        dev = qp.device("default.qubit")

        @qp.qnode(dev, shots=1)
        def circuit():

            # the third index is encoded in the control wires [0, 1]
            qp.BasisState([1, 0], wires = [0,1])

            qp.QROM(bitstrings = bitstrings,
                    control_wires = [0,1],
                    target_wires = [2,3,4],
                    work_wires = [5,6,7])

            return qp.sample(wires = [2,3,4])

    >>> print(circuit())
    [[1 1 0]]

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.QROM
            :stacked:
            :symbols: :math:`K` bitstrings of length :math:`b`, :math:`c \geq 2` control wires,
                :math:`w` work wires and ``clean=True``
            :example: :math:`K=4`, :math:`b=2`, :math:`c=2`, :math:`w=2`, bitstrings
                ``[[0, 0], [0, 1], [1, 0], [1, 1]]``

            .. decomposition-rule:: select_swap
                :image: /_static/decomposition_thumbnails/qrom_select_swap.svg
                :requires: :math:`w \geq c - 1` work wires and :math:`\lambda \geq 2`
                :where: :math:`\lambda` is the number of bitstrings loaded in parallel: the largest
                    power of 2 with :math:`\lambda \leq K`, :math:`b(\lambda - 1) \leq w` and
                    :math:`w - b(\lambda - 1) \geq c - \log_2 \lambda - 1`. The nested
                    :class:`~.QROM` loads :math:`\lceil K/\lambda \rceil` bitstrings of length
                    :math:`\lambda b`

                :class:`~.QROM` | :math:`2` | 2
                :class:`~.Hadamard` | :math:`2b` | 4
                :class:`~.CSWAP` | :math:`4b(\lambda - 1)` | 8

            .. decomposition-rule:: unary_iteration
                :image: /_static/decomposition_thumbnails/qrom_unary_iteration.svg
                :where: :math:`t = \min(c, w + 1)` control wires span the unary iteration tree and
                    the other :math:`e = c - t` are added as controls to each load,
                    :math:`B = \lceil K / 2^e \rceil`,
                    :math:`N = t + B - 2 - \mathrm{wt}(B-1) - [B > 2^{t-1}]`, with
                    :math:`\mathrm{wt}` the number of ones in the binary representation, and
                    :math:`[\cdot]` equal to 1 if the condition holds and 0 otherwise. With no work
                    wires (:math:`w = 0`), each bitstring is instead loaded by a :class:`~.MultiX`
                    with :math:`c` controls

                :class:`~.TemporaryAND` | :math:`N` | 1
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`N` | 1
                :class:`~.CNOT` | :math:`B - 1 + [B > 2^{t-1}]` | 4
                :class:`~.X` | :math:`2[B > 2^{t-2}]` | 2
                :class:`~.MultiX` with :math:`e + 1` controls | :math:`K` | 4

            .. decomposition-rule:: measurement
                :image: /_static/decomposition_thumbnails/qrom_measurement.svg
                :requires: :math:`w \geq c - 1` work wires and compilation with :func:`~.qjit`
                :where: :math:`L = 2^c` (the data is padded to :math:`L` entries, assuming
                    :math:`c = \lceil \log_2 K \rceil`) and :math:`A = L - 3`. Gates marked as
                    conditional are applied depending on mid-circuit measurement outcomes,
                    and the number of :class:`~.CNOT` gates depends on the bitstrings

                :class:`~.MultiX` | :math:`1 + A` (:math:`A` conditional) | 2
                :class:`~.TemporaryAND` | :math:`A` | 1
                :class:`~.X` with 1 control on :math:`|0\rangle` | :math:`1` | 1
                :class:`~.CNOT` | at most :math:`\max(3, L/2 - 1) + 2b` | 5
                :func:`PauliMeasure <.pauli_measure>` (:math:`X`-type, :math:`\leq b + 1` wires) | :math:`A` | 1
                :func:`PauliMeasure <.pauli_measure>` (:math:`Z`, 1 wire) | :math:`A` | 1
                :class:`~.CZ` (conditional) | :math:`A` | 1
                :class:`~.Z` (conditional) | :math:`A` | 1
                :class:`~.X` (conditional) | :math:`A` | 1

    .. details::
        :title: Usage Details

        This template takes as input three different sets of wires. The first one is ``control_wires`` which is used
        to encode the desired index. Therefore, if we have :math:`m` bitstrings, we need
        at least :math:`\lceil \log_2(m)\rceil` control wires.

        The second set of wires is ``target_wires`` which stores the bitstrings.
        For instance, if the bitstring is ``[0, 1, 1, 0]``, we will need four target wires. Internally,
        the bitstrings are encoded using the :class:`~.MultiX` template.


        The ``work_wires`` are auxiliary qubits used to reduce the gate complexity of the
        operator. These wires are dynamically partitioned into two sets: one for the
        :class:`~.Select` block and another to facilitate parallel data loading via a
        `SWAP network <https://pennylane.ai/compilation/swap-network>`__.

        The template determines the depth, :math:`\lambda` (a power of 2),
        based on the available ``work_wires``. Let :math:`b` be the length of the bitstrings.
        The number of wires allocated to the SWAP network is :math:`k_{swap} = b \cdot (\lambda - 1)`.
        The remaining wires, :math:`k_{select}`, are assigned to the :class:`~.Select` block.

        To ensure the decomposition is valid, the template guarantees that
        :math:`k_{select} \geq c - \log_2(\lambda) - 1`, where :math:`c` is the number of
        control wires, updating the depth if needed.

        The QROM template has two variants. The first one (``clean = False``) is based on [`arXiv:1812.00954 <https://arxiv.org/abs/1812.00954>`__] that alternates the state in the ``work_wires``.
        The second one (``clean = True``), based on [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__], solves that issue by
        returning ``work_wires`` to their initial state. This technique can be applied when the ``work_wires`` are not
        initialized to zero.

        .. note::

            More ``control_wires`` than the minimum :math:`\lceil \log_2(m) \rceil` may be
            provided. The extra wires are treated as the most-significant address bits: the data
            is loaded only when they are all in :math:`|0\rangle`, and the operation acts as the
            identity otherwise. This turns ``QROM`` into a *controlled* load gated by those extra
            wires.

    """

    dynamic_argnames = ("bitstrings",)
    wire_argnames = ("control_wires", "target_wires", "work_wires")
    compilable_argnames = ("clean",)

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
        clean=True,
    ):

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

        super().__init__(bitstrings, control_wires, target_wires, work_wires, clean)

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.control_wires + self.target_wires + self.work_wires
