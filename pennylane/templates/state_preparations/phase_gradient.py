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

from pennylane import math
from pennylane.core.operator import StatePrepBase2
from pennylane.exceptions import WireError
from pennylane.typing import Wire
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

    .. seealso::

        Compiling :class:`~.RZ` gates to phase-gradient operations with
        :func:`~.transforms.rz_phase_gradient`.

        Preparation of this state with Clifford gates, adders, and mid-circuit measurements via
        `Jones distillation <https://arxiv.org/abs/1303.3066>`__ with
        :func:`~.transforms.decompositions.make_phase_gradient_distillation_decomp`.

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
    A second decomposition, also supported on up to 30 wires, consists of :class:`~.Hadamard`,
    :class:`~.S`, :class:`~.T`, Pauli gates, and a :class:`~.GlobalPhase`, with as many
    :class:`~.T` gates as there are :class:`~.PPR_8` gates in the first decomposition.

    Alternatively, the Jones distillation rule uses Clifford gates, :class:`~.SemiAdder`, and
    mid-circuit measurements and supports any number of wires. Construct it and pass it to
    ``fixed_decomps``:

    >>> qp.decomposition.enable_graph()
    >>> registers = qp.registers({"grad": 5, "aux": 5, "work": 4})
    >>> rule = qp.transforms.decompositions.make_phase_gradient_distillation_decomp(
    ...     registers["aux"], registers["work"], mode="postselect"
    ... )
    >>> @qp.transforms.decompose(
    ...     gate_set={"Hadamard", "PauliZ", "Adjoint(S)", "SemiAdder", "MidMeasure"},
    ...     fixed_decomps={qp.PhaseGradientStatePrep: rule},
    ... )
    ... @qp.qnode(qp.device("null.qubit", wires=14))
    ... def distilled_circuit():
    ...     qp.PhaseGradientStatePrep(registers["grad"])
    ...     return qp.state()
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
