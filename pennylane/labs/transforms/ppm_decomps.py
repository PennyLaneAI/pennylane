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
Decomposition rules for standard operators like ``Hadamard`` or ``CNOT`` into PPMs.
"""

import pennylane as qp
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure, pauli_measure
from pennylane.wires import WiresLike


def make_hadamard_ppm_decomp(work_wire: WiresLike):
    r"""Produce a decomposition rule of :class:`~.Hadamard` into one PauliZ, 2 PPMs and
    one Pauli correction. Uses one work wire, which needs to be in the measurement resource
    state :math:`|+\rangle`.

    Args:
        work_wire (WireLike): Work wire to use for the decomposition rule. It is expected that
            we can index into this object via ``work_wire[0]``.

    Returns:
        DecompositionRule: A decomposition rule to be used with :func:`~.decompose`.

    **Example**

    show example usage
    """

    def _hadamard_ppm_resources():
        return {qp.resource_rep(PauliMeasure): 2, qp.Z: 1, qp.Y: 1}

    @qp.register_resources(_hadamard_ppm_resources)
    def _hadamard_ppm(wires: WiresLike, **__):
        # Need a |+> state on the work wire!
        qp.Z(wires)
        both_wires = [wires[0], work_wire[0]]
        m0 = pauli_measure("YY", both_wires)
        m1 = pauli_measure("X", work_wire[0])
        qp.cond(m0 == m1, qp.Y(wires))

    return _hadamard_ppm


def make_cnot_ppm_decomp(work_wire: WiresLike):
    r"""Produce a decomposition rule of :class:`~.CNOT` into 3 PPMs and
    two Pauli corrections. Uses one work wire, which needs to be in the state :math:`|0\rangle`.

    Args:
        work_wire (WireLike): Work wire to use for the decomposition rule. It is expected that
            we can index into this object via ``work_wire[0]``.

    Returns:
        DecompositionRule: A decomposition rule to be used with :func:`~.decompose`.

    **Example**

    show example usage
    """

    def _cnot_ppm_resources():
        return {qp.resource_rep(PauliMeasure): 3, qp.Z: 1, qp.X: 1}

    @qp.register_resources(_cnot_ppm_resources)
    def _cnot_ppm(wires: WiresLike, **__):
        m0 = pauli_measure("ZX", [wires[0], work_wire[0]])
        m1 = pauli_measure("ZX", [work_wire[0], wires[1]])
        m2 = pauli_measure("X", work_wire[0])
        qp.cond(m1, qp.Z(wires[0]))
        qp.cond(m0 != m2, qp.X(wires[1]))

    return _cnot_ppm


def make_cz_ppm_decomp(work_wire: WiresLike):
    r"""Produce a decomposition rule of :class:`~.CZ` into 3 PPMs and
    two Pauli corrections. Uses one work wire, which needs to be in the state :math:`|0\rangle`.

    Args:
        work_wire (WireLike): Work wire to use for the decomposition rule. It is expected that
            we can index into this object via ``work_wire[0]``.

    Returns:
        DecompositionRule: A decomposition rule to be used with :func:`~.decompose`.

    **Example**

    show example usage
    """

    def _cz_ppm_resources():
        return {qp.resource_rep(PauliMeasure): 3, qp.Z: 2}

    @qp.register_resources(_cz_ppm_resources)
    def _cz_ppm(wires: WiresLike, **__):
        m0 = pauli_measure("ZX", [wires[0], work_wire[0]])
        m1 = pauli_measure("ZZ", [work_wire[0], wires[1]])
        m2 = pauli_measure("X", work_wire[0])
        qp.cond(m1, qp.Z(wires[0]))
        qp.cond(m0 != m2, qp.Z(wires[1]))

    return _cz_ppm
