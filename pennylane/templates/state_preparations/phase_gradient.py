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

# Portions of this file (the rotation sequences used in
# _phase_gradient_state_prep_ppr_decomposition below) are derived from pygridsynth:
# https://github.com/quantum-programming/pygridsynth
#
# MIT License
#
# Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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

    The decomposition is supported on up to 30 wires and consists of :class:`~.PPR` gates (and
    a :class:`~.GlobalPhase`).

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


# Wires 0 to 2 are based on the exact gates Z, S^dagger and T^dagger. For j >= 3, the PPRs are
# based on Clifford+T approximations of RZ(-pi / 2**j) obtained with pygridsynth v2.0.0
# The gate sequence for wire j is obtained by
#     pygridsynth.gridsynth_gates(-mpmath.pi / 2**j, mpmath.mpf(epsilon), up_to_phase=True)
# with the following tolerances epsilon (wire: epsilon):
#     3: 2.5e-12, 4: 4e-13, 5: 5e-13, 6: 3e-13, 7: 6e-12, 8: 3e-13, 9: 1e-12, 10: 2e-13,
#     11: 6e-13, 12: 2e-12, 13: 4e-13, 14: 2.1e-12, 15: 3e-13, 16: 8e-14, 17: 1.3e-12,
#     18: 7e-13, 19: 3e-13, 20: 3e-12, 21: 2e-12, 22: 2e-13, 23: 6e-13, 24: 2e-13, 25: 3e-13,
#     26: 5e-13, 27: 8e-14, 28: 2.66e-13, 29: 5.3e-13
# The sequence, and the preceding Hadamard, is processed into a pure π/8 PPR sequence, up to a
# leading Clifford PPR.

_INITIAL_PPRS = (
    (-4, "Y"),
    (4, "X"),
    (4, "X"),
    (2, "X"),
    (4, "X"),
    (4, "Y"),
    (4, "X"),
    (-4, "X"),
    (2, "X"),
    (-4, "X"),
    (-4, "X"),
    (4, "Y"),
    (4, "X"),
    (-4, "Y"),
    (-4, "X"),
    None,
    (2, "X"),
    (-4, "X"),
    (2, "X"),
    (4, "Y"),
    (-4, "Y"),
    (4, "X"),
    (4, "Y"),
    (4, "X"),
    (2, "X"),
    (4, "X"),
    None,
    (4, "Y"),
    (4, "X"),
    (2, "X"),
)

# fmt: off
_PI_OVER_8_PPRS = (
    "",
    "",
    "Z",
    "yZyXyXZyZyXyzyzXyzXyXZXZyXZXZXYZYxZxZxyZxyZyZxyxzyzyzyzyxyZyZyZyZyXyXyXZyXZyXyXyzyxyxzyzyxzyzyxyZxyxyxzyxzxYzxYxYxZYZXY",
    "xyxzyzXzXyzXyXZyXyXZyXyzXyzXzYzxzxzyxzxzxYxZxZYxZxyxzxYxYxZYZYxYzxYzxYzYzxYxYxZYxYxZYxYzxYzxYxZxyxzyzXyzXyzXzXzYXYXYZXZXYXY",
    "ZyZyXyzyxyZxyxzxYxYxYzYXYXzXzXzYzYXYXYXzXyXyzXzXyXZyZxyZxZxZYxZxZYxZYZYZYZYZYxZxZYxZxyxyxzxzxzyxzxYzYXzXyzyzXzXzXyXyXZXZXYZX",
    "xZYxYzYXYXYXYZXZXZyXyzXyXyzXzYzxYxZxZYZYxZxZYZYxZYxZYxYzYzYzYXzYXzYzYzxYxZxyxzyxyZyZyZxZYxYzYzYXzXzXzYXYZXZXYXYXYZXZXZXYZXZXYZX",
    "zXzXzYzxYzxzyzyxzyzXyzyzXyzXzYzxYxZYxZYZXYZYZYxZxZxZxyZyZxZYZYZXZXZyZyZxZxyxzyzyxzxYzxzxzxYxZYZYxYxZxyZyXZXZyZyZyX",
    "YxZYZYZXYXYZXZXZyXZXZXYZXZyXyXyzXzYzYzYzxYzxYzYzxzxzxYxYzxzxYzYzYXYXzYzYXzXyXZXYXzXzXzXzXyXyXZXYXzXzYzYXYZYxZYZYZXYZYZXZXYXY",
    "ZYxZxyZyXyXyXZXYXzYzxzxzyzyzXyXyXZXYZXZyXyXyzyzXyzXyzXzXzYzxYxZYxYzxzxYzYXzYzYXzXyXyzyzyxzyzyzyzXzXzYXYXYZYZYxZxZxZYxZYZY",
    "XZyXyzyzyxzyxyZxZxyxzxYxZxyZxZxZxyZxZxZYxZxZYxZYZXZXZyXZyZxyxzyzyzyxyZxZxyxzyzXzXyXZyZyZyZyXyzXyzyzXzYXYZXYXYXYZXYXYZYxZxyxzxYxZYZY",
    "YXYZYZYZXYXYXzYzYzxYxZxZYxYxYxYzxzyxzyxyxzyxyZxZYxYxYzxzyzyzXzYzxzxYzYXYZXYZYxYzYXzYXzYXzXyzXzYXzYXYZXYZYZYZXYXzXyzXzXyXZyZyX",
    "ZXZyZxZYxZYxZxZYZXZyZyXZyXZXZXYZXZyZyXZyXZXZXYZXZyZyZxZxZxyxyZyZyXZyZyZxyZxZYxYxZYZYZXYZXZyXyXyXZyXyXyzXzXyzyxyZyXyXyX",
    "zXzYXzYXzXzXyXZyZxyxzxzxzxYzYzYXzYzxYxZxZxyZyZxZxZxyxzxYxZxyZxyZxZxZYxYxZYxZxZYZXZyXyXyXyzXyXyzXyXyzXyXyzXyXZyZxyZyXZXZXYXYZY",
    "zyxyZxyxzyxzxzxYxYzxzxYxZxZYZXZyXyzyxzxYxYxZYZXZXZXZyXyzXzXzYzxzxzxYxZxZYxYzYXzXzYzYzYXzYXzYzYzxYxZYZXYXYXzXyzXyXZXZyZyX",
    "YzxYxYxZxyZxyxzxzyzyxzyxyZxyZxyxyZxyZxZYZYxZYZXYXYZXZyZyZyZyZyZxZYxYzxYzYXYZXZXZXYZXYZXZyZxZxZYxZYZXYXzXzXyzyxzxYxZYxZYxYzYXY",
    "xyxyxzyxzyzyxzyzXyzyxyxzxzyxzyxyZyZyXZXZXYZXYZXYXYXzYXzXyXyzXzYzYXzXzYzYXYZYxYzYzxYzxYxZYxZxZYZXYXYZYxYxZxyxzyzXyXyzyxzyzXyXZXZX",
    "XYZYxZYxZxyxyxyZxZYxYxZYxZxZxyZxZxZxyxyxyxyZxyZyZyXyzyzXzXyzyxyZyXyzyxyxzxYzYzxYxYxZxZxZYxYzxYzxYxZxyxzxYzYXzXzYXYXzXyXZ",
    "XyzXzYXzXzYXzXzYXYZYZXZXYZXYXYZXYXYXYXzXyzXzYXzYzYzYXzXyXyzXzXzYzYXYXYZYxZYZYZXYZXZyXZXYZYxYxZxZxZxyZxyZyZyXZXYXzYzxYzYzYXY",
    "yxzyzXyXyzyxzxzyzyxzxYxYxYzxzxYxYzYzxYzxzyzXzXyzyzyzyzyxzyxzxYxZYxZxZxZYZXZyXZyXZyXZXYXYZXYXYXYZYZYZXZyZyZyZyZxZxZYxZYZXZXZXY",
    "YZXYXYXYZYZXZXZyZyZyXZyZyZxyxzyxyZyZxZYZXZyXZXZyZxZYZYxYzxYzxYzxYxYzYXzXzYXzYXzXzXzYXzYXzXzXyzXzXzYzxzxzyxzxzyxzxYxZxZYZY",
    "xyxyZyXyzyzyzXzXyXyzXzXyzXyzyzyxzyxzxzxYzYXzYXzXyXZyZxZxZYxZYZYZXYZYZYZYxZYZYZYZXZyZyXZXZXZXZyZxyxyZxyZxyxzxYzxYzYXYXY",
    "yxzxzyzXyXZyXyzXyXZXYXzYXYZXZyXyzyzyzyzyxzxzxzyxyZyZxZYxZxyxyxzxYzxzxzyxzxzxYzYzxzyxzxzxYzYXYZYxZYxYxYzYXYXYZYxYxYxYxZYZXZyX",
    "zxzyzXyzXyzXzYXzXzXyXZXYXzYzxYxYzYzxYxYxZxZYxYxZYZYxYxYzxYxZYZXZXZXYZXZyXyXyzXyXZyXZyZxyxzyxzxzyzXyXyXZXZyXZXYZYZXYXYXYXYXY",
    "YzYXzYzxzxYxZYZXYZXYZXZyZyZyZyZyZxyZyZyZxyxzxYzYXzXzXyzXyXyzyzXzYXYXYXYZXYZYxYxYzYzxzyzyzXzYzYXYZXYXYXzXzXyXyXZyZyXyXZXYXYZXZyX",
    "ZxyxyxzxzxYzxzyxyZyXZyXyzXyXZXZXYZXZyXZXYXYZYxYzYXzXzYXzYzYzxzyzXzYzxzxYzxYxYzYzxzyxzyxzyzXyXyzyxzyxyZxyZyXZXZyZxyxyZxZYxZYZY",
    "xYzYzYXYXzYXzYzYzxzyzXyXyXyzyzyxzyzyzyxyxzyzyzyzyxyxyZyXyXZXZyZxyxyxyxyxzyzXzXzYzxYzxzyzXzXyXyXyzXyzXyzXzYXzYXYXzXzXzXzYXYXY",
    "YxYzYzxYxZxyZyZyXZyZyXZyXZyZxyZxyZyZyXZyZxZYxYxYzYXYZYZYZYZXYZYxZxZYxYxYxYxZxyxyxzyxyxyZxyxyZxZxyxzyzyxzyxyZyZxZYZXZXYXzYXYXzYXY",
    "xYzxYzxYzYXYXYXYZYZYZYZXYXzYXzYXYZXYXYZYZYZYxZYZXZyXyXyXyzXzXyzyzyxzxzxzyzXyXyzyzyzyzXyXZXYZYZXYXzXyzXyXZyZyXZyZyZxyZxyZxZxZYZYZY",
    "xYxZxZxyxzyzXyXyzXyzyzyxzyzXyXyzXyzXzXzXyzXyzyxyxyxyxyxyZxZYZYZXZXZXYXYXzXyzXzYzxzyzXzYzxYzYzxzyzyzyzXzYzxzyzXyzyxzxzyzXzYzYXzXyX",
)

_GLOBAL_PHASES = (
    0.0, 0.0, -0.39269908169872414, 0.19634954085053827, 2.4543692606172054, -1.9144080232811331,
    1.5953400194010237, 0.7976700097031784, -0.7792622402458426, -3.138524692014499,
    -0.7838641826095573, -3.1408256631960407, 1.5711798219926387, -0.785206415798946,
    -1.1780013712959445, -2.7488456349914205, -1.5707723583450672, 2.356206474417838,
    0.7854041555095971, 0.3927020777549259, 1.5707978248243835, 0.7853989124122822,
    -1.178096870589138, -2.356194302938653, -2.748893478264377, -1.9634953616802364,
    -2.748893548484341, -2.7488935601876974, 2.7488935777426624, -1.5707963238689968,
)
# fmt: on


def _ppr_counts(num_wires):
    counts = Counter()
    for init, pprs in zip(_INITIAL_PPRS[:num_wires], _PI_OVER_8_PPRS[:num_wires], strict=True):
        if init is not None:
            counts[init] += 1
        counts.update((8 if p.isupper() else -8, p.upper()) for p in pprs)
    return counts


def _phase_gradient_state_prep_ppr_resources(wires: AbstractWires):
    resources = {
        qp.PPR(denominator, pauli, wires=Wire[1]): count
        for (denominator, pauli), count in _ppr_counts(len(wires)).items()
    }
    resources[qp.GlobalPhase] = 1
    return resources


@register_condition(lambda wires: len(wires) <= len(_PI_OVER_8_PPRS))
@register_resources(_phase_gradient_state_prep_ppr_resources)
def _phase_gradient_state_prep_ppr_decomposition(wires: WiresLike):
    # Wire j is prepared in the state (|0> + exp(-i pi / 2**j)|1>) / sqrt(2) by the PPR
    # ``_INITIAL_PPRS[j]`` (angle denominator, Pauli word; no gate if ``None``), followed by the π/8
    # PPRs in ``_PI_OVER_8_PPRS[j]``, where upper (lower) case letters denote PPRs with angle π/8
    # (-π/8). Together with the global phase given by summing ``_GLOBAL_PHASES`` over the wires,
    # the state prepared on 30 wires deviates from the phase gradient state by
    # less than 1e-12 in the 2-norm.

    num_wires = len(wires)
    for wire, init, pprs in zip(
        wires, _INITIAL_PPRS[:num_wires], _PI_OVER_8_PPRS[:num_wires], strict=True
    ):
        if init is not None:
            qp.PPR(*init, wires=wire)
        for p in pprs:
            qp.PPR(8 if p.isupper() else -8, p.upper(), wires=wire)
    qp.GlobalPhase(sum(_GLOBAL_PHASES[: len(wires)]))


add_decomps(
    PhaseGradientStatePrep,
    _phase_gradient_state_prep_ppr_decomposition,
)
