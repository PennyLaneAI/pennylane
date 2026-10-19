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
r"""
Contains the PhaseGradientStatePrep template.
"""

from collections import Counter

import numpy as np

import pennylane as qp
from pennylane import math
from pennylane.core.operator import StatePrepBase2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.exceptions import WireError
from pennylane.typing import AbstractWires, Wire
from pennylane.wires import Wires, WiresLike

from .phase_gradient_decomp_data import GLOBAL_PHASES, INITIAL_PPRS, PI_OVER_8_PPRS


class PhaseGradientStatePrep(StatePrepBase2):
    r"""Prepares a `phase gradient state <https://pennylane.ai/compilation/phase-gradient>`__.

    For :math:`b` wires and :math:`B=2^b`, the prepared state is

    .. math::

        |\nabla_b\rangle = \frac{1}{\sqrt{B}} \sum_{k=0}^{B-1} e^{-2\pi i \frac{k}{B}} |k\rangle,

    where the first wire holds the most significant bit of :math:`k`.
    Adding an integer :math:`M` to this state with a (semi-in-place) adder like
    :class:`~.SemiAdder` imprints the phase :math:`e^{2\pi i \frac{M}{B}}`, which makes it a
    catalytic resource state for implementing rotation gates.
    See the `compilation hub <https://pennylane.ai/compilation/phase-gradient>`__ for more details.

    .. seealso:: Compiling :class:`~.RZ` gates to phase gradient operations with
        :func:`~.transforms.rz_phase_gradient`.

    .. note::

        This template prepares the phase gradient state only if the input state is
        :math:`|0\rangle^{\otimes b}`.

    Args:
        wires (WiresLike): the wires on which to prepare the phase gradient state

    The state is a product state, with the :math:`j`-th wire (counting from :math:`0`) in the state
    :math:`\tfrac{1}{\sqrt{2}}(|0\rangle + e^{-i\pi/2^j}|1\rangle)`. It is prepared by applying a
    :class:`~.Hadamard` gate to each wire, followed by phase gates with angles :math:`-\pi/2^j`.
    The first three of these are the discrete gates :class:`~.Z`, :math:`S^\dagger` and
    :math:`T^\dagger`, the remaining ones are realized with :class:`~.PhaseShift`.

    **Example**

    >>> dev = qp.device("default.qubit")
    >>> @qp.qnode(dev)
    ... def circuit():
    ...     qp.PhaseGradientStatePrep(wires=range(3))
    ...     return qp.state()
    >>> B = 2**3
    >>> np.allclose(circuit(), np.exp(-2j * np.pi * np.arange(B) / B) / np.sqrt(B))
    True

    The decomposition is supported on up to 30 wires and consists of :class:`~.PPR_8` gates,
    at most one :class:`~.PPR_2` or :class:`~.PPR_4` gate per wire, and a :class:`~.GlobalPhase`.

    """

    arg_specs = {"wires": Wire[-1]}

    def __init__(self, wires: WiresLike):
        super().__init__(wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "|∇z⟩"

    def state_vector(self, wire_order: WiresLike | None = None):
        num_op_wires = len(self.wires)
        dim = 2**num_op_wires
        op_vector = np.exp((-2j * np.pi / dim) * np.arange(dim)) / np.sqrt(dim)
        op_vector = math.reshape(op_vector, (2,) * num_op_wires)

        if wire_order is None or Wires(wire_order) == self.wires:
            return op_vector

        wire_order = Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise WireError(f"Custom wire_order must contain all {self.name} wires")

        num_total_wires = len(wire_order)
        indices = tuple(
            [Ellipsis] + [slice(None)] * num_op_wires + [0] * (num_total_wires - num_op_wires)
        )
        ket = np.zeros([2] * num_total_wires, dtype=np.complex128)
        ket[indices] = op_vector

        if self.wires != wire_order[:num_op_wires]:
            current_order = self.wires + list(Wires.unique_wires([wire_order, self.wires]))
            desired_order = [current_order.index(w) for w in wire_order]
            ket = ket.transpose(desired_order)

        return ket


def _ppr_counts(num_wires):
    counts = Counter()
    for init, pprs in zip(INITIAL_PPRS[:num_wires], PI_OVER_8_PPRS[:num_wires], strict=True):
        if init is not None:
            counts[init] += 1
        counts.update((8 if p.isupper() else -8, p.upper()) for p in pprs)
    return counts


def _ppr(denominator, pauli_word, wires):
    op_class = {2: qp.PPR_2, 4: qp.PPR_4, 8: qp.PPR_8}[abs(denominator)]
    return op_class(1 if denominator > 0 else -1, pauli_word, wires=wires)


def _phase_gradient_state_prep_ppr_resources(wires: AbstractWires):
    resources = {
        _ppr(denominator, pauli, wires=Wire[1]): count
        for (denominator, pauli), count in _ppr_counts(len(wires)).items()
    }
    resources[qp.GlobalPhase] = 1
    return resources


@register_condition(lambda wires: len(wires) <= len(PI_OVER_8_PPRS))
@register_resources(_phase_gradient_state_prep_ppr_resources)
def _phase_gradient_state_prep_ppr_decomposition(wires: WiresLike):
    # Wire j is prepared in the state (|0> + exp(-i pi / 2**j)|1>) / sqrt(2) by the PPR
    # ``INITIAL_PPRS[j]`` (angle denominator, Pauli word; no gate if ``None``), followed by the π/8
    # PPRs in ``PI_OVER_8_PPRS[j]``, where upper (lower) case letters denote PPRs with angle π/8
    # (-π/8). Together with the global phase given by summing ``GLOBAL_PHASES`` over the wires,
    # the state prepared on 30 wires deviates from the phase gradient state by
    # less than 1e-12 in the 2-norm.

    num_wires = len(wires)
    for wire, init, pprs in zip(
        wires, INITIAL_PPRS[:num_wires], PI_OVER_8_PPRS[:num_wires], strict=True
    ):
        if init is not None:
            _ppr(*init, wires=wire)
        for p in pprs:
            qp.PPR_8(1 if p.isupper() else -1, p.upper(), wires=wire)
    qp.GlobalPhase(sum(GLOBAL_PHASES[: len(wires)]))


add_decomps(
    PhaseGradientStatePrep,
    _phase_gradient_state_prep_ppr_decomposition,
)
