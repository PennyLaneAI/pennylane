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

import numpy as np

import pennylane as qp
from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import StatePrepBase2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.exceptions import WireError
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.typing import AbstractWires, Wire
from pennylane.wires import Wires, WiresLike


class PhaseGradientStatePrep(StatePrepBase2):
    r"""Prepares a `phase gradient state <https://pennylane.ai/compilation/phase-gradient>`__.

    For :math:`b` wires and :math:`B=2^b`, the prepared state is

    .. math::

        |\nabla_b\rangle = \frac{1}{\sqrt{B}} \sum_{k=0}^{B-1} e^{-2\pi i \frac{k}{B}} |k\rangle,

    where the first wire holds the most significant bit of :math:`k`.
    Adding an integer :math:`M` to this state with a (semi-in-place) adder like
    :class:`~.SemiAdder` imprints the global phase :math:`e^{2\pi i \frac{M}{B}}`, which makes it a
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

    The decomposition consists of :class:`~.Hadamard` gates and phase gates:

    >>> print(qp.draw(qp.PhaseGradientStatePrep(wires=range(5)).decomposition)())
    0: ──H──Z─────────┤
    1: ──H──S†────────┤
    2: ──H──T†────────┤
    3: ──H──Rϕ(-0.39)─┤
    4: ──H──Rϕ(-0.20)─┤
    """

    arg_specs = {"wires": Wire[-1]}

    def __init__(self, wires: WiresLike):
        super().__init__(wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "|∇⟩"

    def state_vector(self, wire_order: WiresLike | None = None):
        num_op_wires = len(self.wires)
        dim = 2**num_op_wires
        op_vector = np.exp(-2j * np.pi * np.arange(dim) / dim) / np.sqrt(dim)
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


def _phase_gradient_state_prep_resources(wires: AbstractWires):
    num_wires = len(wires)
    resources = {qp.Hadamard: num_wires}
    if num_wires > 0:
        resources[qp.Z] = 1
    if num_wires > 1:
        resources[_adjoint_abstract(qp.S)] = 1
    if num_wires > 2:
        resources[_adjoint_abstract(qp.T)] = 1
    if num_wires > 3:
        resources[qp.PhaseShift] = num_wires - 3
    return resources


@register_resources(_phase_gradient_state_prep_resources)
def _phase_gradient_state_prep_decomposition(wires: WiresLike):
    num_wires = len(wires)
    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")

    @for_loop(num_wires)
    def hadamard_loop(i):
        qp.Hadamard(wires[i])

    hadamard_loop()  # pylint: disable=no-value-for-parameter

    if num_wires > 0:
        qp.Z(wires[0])
    if num_wires > 1:
        qp.adjoint(qp.S(wires[1]))
    if num_wires > 2:
        qp.adjoint(qp.T(wires[2]))

    @for_loop(3, num_wires)
    def phase_shift_loop(i):
        qp.PhaseShift(-np.pi * 2.0**-i, wires[i])

    phase_shift_loop()  # pylint: disable=no-value-for-parameter


add_decomps(PhaseGradientStatePrep, _phase_gradient_state_prep_decomposition)
