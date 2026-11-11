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
from pennylane.ops import CNOT, CZ, X, Z, cond, pauli_measure
from pennylane.typing import AbstractArray, Bool, TensorLike, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .arithmetic import TemporaryAND
from .multix import MultiX


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
    ):  # pylint: disable=too-many-arguments,disable=too-many-positional-arguments

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


def _measurement_uncompute(work_wire, ctrl_wires, targets, product):
    """Measurement-based uncomputation from Fig 18a) https://arxiv.org/abs/2211.15465

    Args:
        work_wire: the AND output wire to uncompute. Third wire on the figure.
        ctrl_wires: [ctrl0, ctrl1] -- the two AND control wires (for CZ correction). First and second qubit on the figure.
        targets: target register wires.
        product: bitstring indicating the X positions in the target register.
    """
    x_wires = [targets[i] for i, bit in enumerate(product) if bit == 1]

    m1 = pauli_measure("X" + "X" * len(x_wires), [work_wire, *x_wires])

    cond(m1 == 1, CZ)(wires=ctrl_wires)
    # Without this, the reset below would leave a global phase of -1 if both outcomes are 1
    cond(m1 == 1, Z)(wires=work_wire)

    m2 = pauli_measure("Z", [work_wire])
    cond(m2 == 1, X)(wires=work_wire)
    cond(m2 == 1, MultiX)(product, wires=targets)


def _measurement_qrom_inner(controls, targets, bitstrings):
    """Inner binary recursion with measurement-based uncomputation.

    Each level opens a TemporaryAND, recurses into left/right halves,
    then uncomputes via measurement. The XOR product between subtree
    bases is absorbed into the measurement.

    Args:
        controls: interleaved [flag, sel, work, sel2, work2, ...]
        targets: target register wires
        bitstrings: The set of k strings to be loaded in the decomposition. They do not necessarily match the QROM input values.

    """

    k = len(bitstrings)
    if k <= 1:
        return

    num_bits = ceil_log2(k)
    needed = 2 * num_bits + 1
    controls = list(controls[:1]) + list(controls[-(needed - 1) :])

    flag, sel, work = controls[0], controls[1], controls[2]
    child_controls = controls[2:]

    k_left = 2 ** (num_bits - 1)

    if k > 2:
        TemporaryAND([flag, sel, work], control_values=[1, 0])
        _measurement_qrom_inner(child_controls, targets, bitstrings[:k_left])
        CNOT(wires=[flag, work])
        _measurement_qrom_inner(child_controls, targets, bitstrings[k_left:])
    else:
        TemporaryAND([flag, sel, work], control_values=[1, 1])

    product = math.bitwise_xor(bitstrings[0], bitstrings[k_left])
    _measurement_uncompute(work, [flag, sel], targets, product)
