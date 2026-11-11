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

"""Built-in decomposition rules for templates.subroutines.qubitization_thc."""

import numpy as np

from pennylane import capture
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import GlobalPhase, Hadamard, Z, change_op_basis
from pennylane.templates.subroutines.alias_sampling_thc import (
    AliasSamplingTHC,
    _num_index_wires,
    alias_sampling_thc_wires,
)
from pennylane.templates.subroutines.flip_sign import FlipSign
from pennylane.templates.subroutines.qubitization_thc import QubitizationTHC, _alias_wire_split
from pennylane.templates.subroutines.select_thc import SelectTHC
from pennylane.templates.subroutines.superposition_thc import SuperpositionTHC
from pennylane.typing import Wire


def _prepare_registers(
    M, N, aleph, index_wires, prep_garbage_wires, work_wires
):  # pylint: disable=too-many-arguments
    r"""Split the input registers into the sub-registers of the three THC oracles.

    The layout is the one documented by :func:`qubitization_thc_wires`. The returned
    ``alias_work`` is split rather than interleaved: :class:`~.AliasSamplingTHC` keeps every
    wire it leaves entangled in the lower part of its work register, so that part is fed
    from the reflected register and the restored tail from the shared clean pool, leaving
    no wire in ``prep_garbage_wires`` that ends in :math:`\lvert 0 \rangle`.

    Returns:
        dict: the registers each sub-template receives, plus the three flags that
        ``PREPARE`` hands to ``SELECT`` and the register the reflection acts on
    """
    n = _num_index_wires(M)
    index = list(index_wires)
    garbage = list(prep_garbage_wires)
    clean = list(work_wires)

    n_sup = alias_sampling_thc_wires(M, N, aleph)["superposition_work_wires"]
    swap_index, n_garbage = _alias_wire_split(M, N, aleph)

    superposition_work = garbage[:n_sup]
    spin_wires = garbage[-2:]

    # SuperpositionTHC returns every work wire to |0> except its three flags (indices 0, 3
    # and 6), so the rest are reflected wires that are free to carry alias garbage; the
    # alias adjoint clears them again before SuperpositionTHC's adjoint runs.
    garbage_pool = (
        [w for i, w in enumerate(superposition_work) if i not in (0, 3, 6)] + garbage[n_sup:-2]
    )[:n_garbage]
    # AliasSamplingTHC keeps all of its garbage in the lower part of its work register:
    # contiguous address, QROM output, symmetrization flag and sampling register. The
    # restored tail (inequality-test flag, comparator and QROM scratch) comes from the
    # shared clean pool.
    alias_work = garbage_pool + clean

    return {
        "mu_wires": index[:n],
        "nu_wires": index[n:],
        # SuperpositionTHC runs first, while the shared pool is still untouched, and it
        # forwards anything past its fixed 3n+5 layout to its comparators and
        # multi-controlled gates. Given only the minimum it has *zero* scratch for those.
        "superposition_work": superposition_work + clean,
        "alias_work": alias_work,
        # SuperpositionTHC's flags: index 3 is true when nu = M (the one-body sentinel
        # column) and index 6 when the superposition was prepared correctly.
        "edge_flag": superposition_work[3],
        "success_flag": superposition_work[6],
        # AliasSamplingTHC holds its mu <-> nu symmetrization flag one wire above the
        # alternate edge flag, below the sampling register.
        "swap_flag": alias_work[swap_index],
        "spin_wires": spin_wires,
        "reflected": index + garbage,
    }


def _prepare_select(
    M,
    N,
    zeta,
    t_ell,
    chi,
    t_eigenvectors,
    aleph,
    beth,
    registers,
    system_wires,
    index_wires,
    gradient_wires,
    work_wires,
    num_batches,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    r"""Return ``PREPARE``:math:`^\dagger \cdot` ``SELECT`` :math:`\cdot` ``PREPARE`` as a
    single :func:`~.change_op_basis`, given the registers of :func:`_prepare_registers`."""
    mu_wires, nu_wires = registers["mu_wires"], registers["nu_wires"]
    spin_wires = registers["spin_wires"]
    sign_wire = registers["alias_work"][alias_sampling_thc_wires(M, N, aleph)["sign_wire"]]

    def prepare():
        SuperpositionTHC(M, N, mu_wires, nu_wires, registers["superposition_work"])
        AliasSamplingTHC(
            M,
            N,
            zeta,
            t_ell,
            mu_wires,
            nu_wires,
            registers["edge_flag"],
            registers["alias_work"],
            aleph,
        )
        # The two spin flags are the |+> controls that route each V onto a spin sector.
        for wire in spin_wires:
            Hadamard(wire)

    def select():
        # The sign of each LCU coefficient must be applied an *odd* number of times between
        # PREPARE and PREPARE^dagger, so it cannot sit inside ``prepare``: the adjoint would
        # square the sign away and the walk would block encode the coefficient *magnitudes*
        # instead.
        Z(sign_wire)
        SelectTHC(
            chi,
            t_eigenvectors,
            beth,
            system_wires,
            list(index_wires),
            [
                registers["success_flag"],
                registers["edge_flag"],
                registers["swap_flag"],
                spin_wires[0],
                spin_wires[1],
            ],
            gradient_wires,
            work_wires,
            num_batches=num_batches,
        )

    return change_op_basis(prepare, select)


def _qubitization_thc_resources(
    zeta,
    t_ell,
    chi,
    t_eigenvectors,
    aleph,
    beth,
    system_wires,
    index_wires,
    prep_garbage_wires,
    gradient_wires,
    work_wires,
    num_batches=1,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Return the top-level resources of the QubitizationTHC decomposition."""
    M = len(zeta)
    N = 2 * len(chi[0])

    # The sub-register sizes only depend on how the input registers are split, so the
    # split is replayed on placeholder labels to keep it in sync with the decomposition.
    sizes = [len(reg) for reg in (system_wires, index_wires, prep_garbage_wires)]
    sizes += [len(gradient_wires), len(work_wires)]
    offsets = np.cumsum([0] + sizes)
    system, index, garbage, gradient, work = (
        list(range(int(start), int(stop))) for start, stop in zip(offsets[:-1], offsets[1:])
    )
    registers = _prepare_registers(M, N, aleph, index, garbage, work)

    # `_prepare_select` returns change_op_basis. Under capture this is unrolled eagerly
    # (binds inner gates into the jaxpr and returns None), so we pause here while we build the
    # resource key. compute_resources already stops queuing but does not pause capture.
    # This can be removed once https://github.com/PennyLaneAI/pennylane/issues/10162 is resolved.
    with capture.pause():
        prepselprep = _prepare_select(
            M,
            N,
            zeta,
            t_ell,
            chi,
            t_eigenvectors,
            aleph,
            beth,
            registers,
            system,
            index,
            gradient,
            work,
            num_batches,
        )

    num_reflected = len(registers["reflected"])
    reflection = FlipSign([0] * num_reflected, Wire[num_reflected], work_wires=Wire[len(work)])

    return {
        abstractify(prepselprep): 1,
        reflection: 1,
        GlobalPhase: 1,
    }


@register_resources(_qubitization_thc_resources)
def _qubitization_thc_decomp(
    zeta,
    t_ell,
    chi,
    t_eigenvectors,
    aleph,
    beth,
    system_wires,
    index_wires,
    prep_garbage_wires,
    gradient_wires,
    work_wires,
    num_batches=1,
    **_,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    M = len(zeta)
    N = 2 * len(chi[0])

    registers = _prepare_registers(M, N, aleph, index_wires, prep_garbage_wires, work_wires)
    _prepare_select(
        M,
        N,
        zeta,
        t_ell,
        chi,
        t_eigenvectors,
        aleph,
        beth,
        registers,
        system_wires,
        index_wires,
        gradient_wires,
        work_wires,
        num_batches,
    )

    # R = 2|0><0| - I on the full PREPARE register. The global sign is fixed so that the
    # |0> block is + H / lambda: the sign flip of |0> that a bare I - 2|0><0| would give
    # is exactly what SELECT's rewriting of n = (1 - V) / 2 already supplies.
    reflected = registers["reflected"]

    # SELECT and PREPARE both restore work_wires, so they are zeroed auxiliary wires here
    # and make the multi-controlled Z much cheaper.
    FlipSign([0] * len(reflected), reflected, work_wires=work_wires)
    GlobalPhase(np.pi)


add_decomps(QubitizationTHC, _qubitization_thc_decomp)
