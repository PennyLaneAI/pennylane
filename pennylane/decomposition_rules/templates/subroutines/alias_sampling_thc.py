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

"""Built-in decomposition rules for templates.subroutines.alias_sampling_thc."""

from collections import defaultdict
from functools import lru_cache

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.templates.subroutines.alias_sampling import (
    _apply_hadamards,
    _build_alias_tables,
)
from pennylane.ops import SWAP, Hadamard, adjoint, ctrl
from pennylane.templates.subroutines.alias_sampling_thc import (
    AliasSamplingTHC,
    _build_thc_pairs,
    _num_address_wires,
)
from pennylane.templates.subroutines.arithmetic.left_quantum_comparator import LeftQuantumComparator
from pennylane.templates.subroutines.arithmetic.out_square import OutSquare
from pennylane.templates.subroutines.arithmetic.semi_adder import SemiAdder
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.templates.subroutines.qrom import QROM
from pennylane.typing import Wire
from pennylane.wires import Wires


def _lcu_signs(M, entries, weights):
    r"""Signs of the LCU coefficients that multiply the ``SELECT`` unitaries.

    The stored sign is *not* the sign of the raw weight. ``SELECT`` applies products of
    the reflections :math:`V_\mu = I - 2 c^\dagger_\mu c_\mu`, so the Hamiltonian is
    rewritten with :math:`c^\dagger_\mu c_\mu = (I - V_\mu) / 2`: the two-body block
    carries a *product* of two number operators and the one-body block a single one,

    .. math::

        \zeta_{\mu\nu} n_\mu n_\nu \to +\tfrac{1}{4} \zeta_{\mu\nu} V_\mu V_\nu ,
        \qquad t_\ell n_\ell \to -\tfrac{1}{2} t_\ell V_\ell ,

    so the one-body column (:math:`\nu = M`) enters with an extra minus sign while the
    two-body column does not. Everything else (the identity terms dropped above) only
    shifts the block encoded operator by a multiple of the identity.

    Args:
        M (int): the THC rank
        entries (Sequence[tuple[int, int]]): the THC index pairs
        weights (Sequence[float]): the aligned (signed) weights

    Returns:
        list[int]: one sign bit per entry, ``0`` for a positive LCU coefficient and ``1``
        for a negative one
    """
    # Flip only the one-body column. Negate both branches to block encode ``-H`` instead.
    coeffs = [-w if nu == M else w for w, (_, nu) in zip(weights, entries, strict=True)]
    return [int(c < 0) for c in coeffs]


@lru_cache(maxsize=16)
def _build_qrom_data(
    M, N, zeta, t_ell, num_index_wires, aleph
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    r"""Pack the alias tables into the bitstrings consumed by :class:`~.QROM`.

    The QROM is addressed by the contiguous two-body index
    ``s = mu + nu (nu + 1) / 2`` (matching :func:`_compute_contiguous_register`). Each
    row concatenates, in order: ``sign`` and ``alt_sign``, ``mu_alt``
    (``num_index_wires`` bits), ``nu_alt`` (``num_index_wires`` bits), the ``aleph``-bit
    keep threshold, and the ``alt_edge`` flag.

    The keep threshold and alternate index are produced by the classical
    ``_build_alias_tables`` of :class:`~.AliasSampling`
    (Walker/Vose, ``mu = aleph`` bits); the signs (see :func:`_lcu_signs`) and the
    ``alt_edge`` sentinel are derived from the THC pair enumeration.

    Args:
        M (int): the THC rank
        N (int): the number of spin orbitals
        zeta (tuple[tuple[float]]): the THC central tensor, shape ``(M, M)``
        t_ell (tuple[float]): the one-body eigenvalues, shape ``(N // 2,)``
        num_index_wires (int): number of wires per index register (``len(mu_wires)``)
        aleph (int): number of bits used for the keep-probability comparison

    Returns:
        list[list[int]]: the QROM data, one bitstring (list of ints) per address
    """
    entries, weights = _build_thc_pairs(M, N, zeta, t_ell)
    probs = [abs(w) for w in weights]
    signs = _lcu_signs(M, entries, weights)

    alt, keep = _build_alias_tables(probs, aleph)

    n_bits = 2 + 2 * num_index_wires + aleph + 1
    data = [[0] * n_bits for _ in range(len(entries))]
    for i, (mu, nu) in enumerate(entries):
        s = mu + (nu**2 + nu) // 2
        alt_i = alt[i]
        mu_alt, nu_alt = entries[alt_i]
        data[s] = (
            [signs[i], signs[alt_i]]
            + math.int_to_binary(mu_alt, num_index_wires).tolist()
            + math.int_to_binary(nu_alt, num_index_wires).tolist()
            + math.int_to_binary(keep[i], aleph).tolist()
            + [1 if nu_alt == M else 0]
        )
    return data


def _compute_contiguous_register(M, N, mu_wires, nu_wires, work_wires):
    r"""Compute the contiguous address ``s = mu + nu (nu + 1) / 2`` into ``work_wires``.

    Uses ``nu (nu + 1) / 2 = (nu^2 + nu) / 2`` via ``OutSquare`` (``nu^2``) followed by
    ``SemiAdder`` (``+ nu``), a right shift by one bit (division by two, implemented
    with SWAPs), and a final ``SemiAdder`` (``+ mu``). The result lands on the first
    ``n_d`` work wires.
    """
    n_d = _num_address_wires(M, N)
    OutSquare(nu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d], output_wires_zeroed=True)
    SemiAdder(nu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d - 1])
    SemiAdder(mu_wires, work_wires[: n_d - 1], work_wires[n_d : 2 * n_d - 2])


def _cswap_pair(flag, left, right, work_wires):
    """CSWAP each pair of wires in ``left`` / ``right`` controlled on ``flag``."""
    n = min(len(left), len(right))
    if n == 0:
        return

    if compiler.active() or capture.enabled():
        left = math.array(left, like="jax")
        right = math.array(right, like="jax")

    @for_loop(n)
    def _loop(i):
        ctrl(
            SWAP(wires=[left[i], right[i]]),
            control=[flag],
            work_wires=work_wires,
            work_wire_type="zeroed",
        )

    _loop()  # pylint: disable=no-value-for-parameter


def _symmetrize(mu_wires, nu_wires, swap_flag, edge_flag, work_wires):
    """Swap ``mu`` and ``nu`` when ``swap_flag`` is 1 and ``edge_flag`` is 0."""
    n = len(mu_wires)
    if n == 0:
        return

    joint_flag = work_wires[0]
    _cswap_work = work_wires[1:]
    TemporaryAND([swap_flag, edge_flag, joint_flag], control_values=(1, 0))
    _cswap_pair(joint_flag, mu_wires, nu_wires, _cswap_work)
    adjoint(TemporaryAND([swap_flag, edge_flag, joint_flag], control_values=(1, 0)))


def _alias_sampling_thc_resources(
    M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph
):  # pylint: disable=too-many-arguments,unused-argument
    n = len(mu_wires)
    n_d = _num_address_wires(M, N)
    n_work = len(work_wires)
    # ``f`` and the QROM work pool are spelled exactly as in the decomposition below, where the
    # pool is ``work_wires[f + 2 * aleph + 3:]``. Declaring a different size hands the graph a
    # resource rep that no emitted QROM ever matches, and an op with no node in the graph
    # silently bypasses fixed_decomps and falls back to QROM.decomposition().
    f = n_d + 1 + 2 * n
    n_qrom_target = 2 + 2 * n + aleph + 1
    n_qrom_work = max(n_work - (f + 2 * aleph + 3), 0)
    data = _build_qrom_data(M, N, zeta, t_ell, n, aleph)
    qrom = QROM(
        data,
        control_wires=Wire[n_d - 1],
        target_wires=Wire[n_qrom_target],
        work_wires=Wire[n_qrom_work],
        clean=True,
    )
    out_sq = OutSquare(Wire[n], Wire[n_d], Wire[n_d], output_wires_zeroed=True)
    adder_0 = SemiAdder(Wire[n], Wire[n_d], Wire[n_d - 1])
    adder_1 = SemiAdder(Wire[n], Wire[n_d - 1], Wire[n_d - 2])

    # The comparator does not restore its work wires, so the keep-value swaps only get the
    # tail of the pool; the symmetrization swaps run after the adjoint comparator and get
    # all of it but the joint flag. Both slices are spelled as in the decomposition below.
    n_cmp_work = min(aleph - 1, n_qrom_work)
    lqc = LeftQuantumComparator(
        Wire[aleph],
        Wire[aleph],
        Wire[1],
        Wire[n_cmp_work],
        comparator="<=",
    )
    keep_cswap = ctrl(
        SWAP(wires=Wire[2]),
        control=Wire[1],
        work_wires=Wire[n_qrom_work - n_cmp_work],
        work_wire_type="zeroed",
    )
    sym_cswap = ctrl(
        SWAP(wires=Wire[2]),
        control=Wire[1],
        work_wires=Wire[max(n_qrom_work - 1, 0)],
        work_wire_type="zeroed",
    )
    resources = defaultdict(int)
    resources[out_sq] += 1
    resources[adder_0] += 1
    resources[adder_1] += 1
    resources[qrom] += 1
    resources[Hadamard] += aleph + 1
    resources[lqc] += 1
    resources[adjoint(lqc)] += 1
    resources[TemporaryAND] += 1
    resources[adjoint(TemporaryAND(Wire[3]))] += 1
    resources[keep_cswap] += 2 * n + 2
    resources[sym_cswap] += n
    return dict(resources)


@register_resources(_alias_sampling_thc_resources)
def _alias_sampling_thc_decomp(
    M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph, **_
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    n = len(mu_wires)
    n_d = _num_address_wires(M, N)
    work_wires = list(work_wires)
    mu_wires = list(mu_wires)
    nu_wires = list(nu_wires)
    edge_flag = Wires(edge_flag)[0]

    # Work wires are used as follows (order is changed, compared to Fig.4 in Lee et al)
    # The following entries store a value by the end of the template and are not reset
    # [:n_d-1]        : ν(ν+1)//2 + μ after _compute_contiguous_register
    # n_d-1           : QROM loads the sign θ_s
    # n_d             : QROM loads the alternate sign {θ_alt}_s
    # [n_d+1:n_d+1+n] : QROM loads the alternate {μ_alt}_s
    # [n_d+1+n:n_d+1+2n] : QROM loads the alternate {ν_alt}_s
    # Call f = n_d+1+2n
    # [f:f+ℵ]         : QROM loads the keep values
    # [f+ℵ]           : alternate qubit for the input edge flag (not in Fig.4)
    # [f+ℵ+1]         : flag for symmetrization SWAPs
    # [f+ℵ+2:f+2ℵ+2]  : Sampling register to compare keep values against.
    # The following register is reset to zero
    # [f+2ℵ+2]        : The comparator flag for sampling keep values
    # The following registers are reset to zero, and overlap partially
    # [f+2ℵ+3:]       : Work wires for QROM
    # [f+2ℵ+3:f+3ℵ+2] : Work wires for keep value comparator, dirty until its adjoint runs
    # [f+3ℵ+2:]       : Work wires for keep value CSWAPs, the part the comparator leaves zeroed
    # [f+2ℵ+3:]       : Work wires for symmetrization CSWAPs, once the comparator is undone

    contiguous_register = work_wires[: n_d - 1]
    sign_wire = work_wires[n_d - 1]
    alt_sign_wire = work_wires[n_d]
    alt_mu_wires = work_wires[n_d + 1 : n_d + 1 + n]
    alt_nu_wires = work_wires[n_d + 1 + n : (f := n_d + 1 + 2 * n)]
    keep_wires = work_wires[f : f + aleph]
    alt_edge_flag = work_wires[f + aleph]
    symmetrize_flag = work_wires[f + aleph + 1]

    # Reset to zero and disjoint
    sample_reg = work_wires[f + aleph + 2 : f + 2 * aleph + 2]
    sample_flag = work_wires[f + 2 * aleph + 2]

    # Reset to zero and overlapping
    qrom_work = work_wires[f + 2 * aleph + 3 :]
    cmp_work = qrom_work[: aleph - 1]
    keep_cswap_work = qrom_work[aleph - 1 :]
    sym_cswap_work = qrom_work

    # work_wires includes the output contiguous_register and additional zeroed work wires that
    # are returned to zero, so we do not need to account for them explicitly.
    _compute_contiguous_register(M, N, mu_wires, nu_wires, work_wires)

    data = _build_qrom_data(M, N, zeta, t_ell, n, aleph)
    QROM(
        data,
        control_wires=contiguous_register,
        target_wires=work_wires[n_d - 1 : f + aleph] + [alt_edge_flag],
        work_wires=qrom_work,
    )

    _apply_hadamards(sample_reg)
    LeftQuantumComparator(keep_wires, sample_reg, sample_flag, work_wires=cmp_work, comparator="<=")

    _cswap_pair(sample_flag, mu_wires, alt_mu_wires, keep_cswap_work)
    _cswap_pair(sample_flag, nu_wires, alt_nu_wires, keep_cswap_work)
    _cswap_pair(
        sample_flag, [edge_flag, sign_wire], [alt_edge_flag, alt_sign_wire], keep_cswap_work
    )

    adjoint(
        LeftQuantumComparator(
            keep_wires, sample_reg, sample_flag, work_wires=cmp_work, comparator="<="
        )
    )

    Hadamard(symmetrize_flag)
    _symmetrize(mu_wires, nu_wires, symmetrize_flag, edge_flag, sym_cswap_work)


add_decomps(AliasSamplingTHC, _alias_sampling_thc_decomp)
