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

"""Built-in decomposition rules for templates.subroutines.select_thc."""

from collections import Counter
from functools import partial
from math import pi

import numpy as np

from pennylane import capture, math
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, SWAP, Hadamard, S, X, Z, adjoint, change_op_basis, ctrl
from pennylane.templates.subroutines.arithmetic.semi_adder import SemiAdder
from pennylane.templates.subroutines.multix import MultiX
from pennylane.templates.subroutines.qrom import QROM
from pennylane.templates.subroutines.select_thc import SelectTHC, _angle_batches, select_thc_wires


def _cascade_angles(leaf):
    r"""Givens angles :math:`\theta_p` mapping ``leaf`` onto :math:`\lvert e_0 \rangle`.

    Zeroes the vector from its last coordinate inwards, so row ``0`` of the resulting
    :math:`U` is the normalized ``leaf`` and :math:`U^\dagger Z_1 U` is the reflection about
    it. Only :math:`N/2 - 1` angles are needed rather than a full :math:`\mathcal{O}(N^2)`
    Givens network, because that sandwich depends on :math:`U` only through row ``0``.

    Args:
        leaf (tuple[float]): the vector to rotate, shape ``(N/2,)``. Its norm is irrelevant;
            the angles are scale invariant.

    Returns:
        the ``N/2 - 1`` angles, where entry ``p`` belongs to the pair
        ``(p, p + 1)``. Empty for a single orbital.

    Raises:
        ValueError: if ``leaf`` is the zero vector.
    """
    vec = math.reshape(math.asarray(leaf, dtype=float), -1)
    if math.shape(vec)[0] <= 1:
        return math.zeros(0)
    if math.norm(vec) < 1e-15:
        raise ValueError("Cannot build a rotation from a zero vector.")

    # Each step of the cascade replaces vec[p] by the norm of the suffix it has just
    # zeroed, so the angles have a closed form. Only the last coordinate keeps its sign,
    # which is what puts the final angle in (-pi, 0] when leaf[-1] is negative.
    suffix = math.sqrt(math.cumsum(vec[::-1] ** 2)[::-1])
    suffix = math.concatenate([suffix[:-1], vec[-1:]])
    return math.arctan2(suffix[1:], vec[:-1])


def _build_qrom_givens_data(chi, t_eigenvectors, beth, one_body_table, batches):
    r"""Build the incremental ``QROM`` tables of quantized Givens angles.

    Each row holds the ``beth``-bit angles that rotate one leaf onto
    :math:`\lvert e_0 \rangle`, one field per pair in the batch; addresses not backed by a
    leaf are all-zero and load the identity. Since ``QROM(clean=True)`` writes with an
    XOR, table :math:`b` is stored as its difference with table :math:`b - 1`, which carries
    the angle register from one batch to the next instead of erasing and reloading it. The
    first table is absolute, so re-loading it clears the register.

    Args:
        chi (tuple[tuple[float]]): the THC leaf matrix, shape ``(M, N/2)``, addressed by the index
            register.
        t_eigenvectors (tuple[tuple[float]]): the one-body eigenvectors as columns, shape
            ``(N/2, N/2)``. Ignored when ``one_body_table=False``.
        beth (int): bits of precision per Givens angle.
        one_body_table (bool): if ``True``, the address is extended by the one-body flag and
            the eigenvectors are loaded above the ``chi`` rows.
        batches (Sequence[Sequence[int]]): the Givens pairs of each batch, in application
            order, as returned by ``_angle_batches``.

    Returns:
        list[list[list[int]]]: one ``QROM`` table per batch, empty when there are no pairs
        to rotate.
    """
    chi = math.asarray(chi, dtype=float)
    M, n_half = chi.shape
    if not batches:
        return []

    block = 1 << math.ceil_log2(max(M, n_half))
    addresses = list(range(M))
    leaves = list(chi)
    if one_body_table:
        addresses += [block + ell for ell in range(n_half)]
        t_eig = math.asarray(t_eigenvectors, dtype=float)
        leaves += list(math.transpose(t_eig))

    # The quantized angles are the last point where this is array math: ``QROM`` takes
    # classical bitstrings, so the grid indices are pulled out as Python ints here and the
    # tables below are built with plain integer arithmetic.
    levels = 1 << beth
    thetas = math.stack([_cascade_angles(leaf) for leaf in leaves])
    grid = math.floor(-math.mod(2.0 * thetas, 4.0 * pi) / (4.0 * pi) * levels)
    grid = [[int(value) % levels for value in row] for row in grid]

    # Only the last batch can be short, so the first sets the width of the angle register
    # and the rest fill a prefix. The table stops at the last address backed by a leaf:
    # `QROM`` pads the rest with the identity, so cost falls with the number of rows but
    # it also scales with ``2 ** len(control_wires)``, so both are kept as tight as possible.
    n_rows = block + n_half if one_body_table else M
    width = len(batches[0]) * beth
    leaf_of = dict(zip(addresses, range(len(leaves))))

    def _row(address, batch):
        if address not in leaf_of:
            return [0] * width
        angles = grid[leaf_of[address]]
        bits = [(angles[p] >> (beth - 1 - j)) & 1 for p in batch for j in range(beth)]
        return bits + [0] * (width - len(bits))

    tables = [[_row(address, batch) for address in range(n_rows)] for batch in batches]
    return [tables[0]] + [
        [[cur ^ prev for cur, prev in zip(row, previous)] for row, previous in zip(table, before)]
        for table, before in zip(tables[1:], tables[:-1])
    ]


def _apply_loaded_rotation(
    psi_down, angle_wires, beth, pairs, gradient_wires, work_wires
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    r"""Apply the rotations of one batch, whose angles are held in ``angle_wires``.

    This is the phase-gradient compilation of Section III.C of `Lee et al. (2021)
    <https://arxiv.org/abs/2011.03494>`_. Applying this to the Givens rotation, we write it as

    .. math::

        \mathrm{SE}(\theta) = \left(H_0\,\mathrm{CNOT}_{01}\right)^\dagger
        \left[R_y(-\theta/2)_0 \otimes R_y(-\theta/2)_1\right]
        \left(H_0\,\mathrm{CNOT}_{01}\right) ,
        \qquad R_y(\alpha) = S H R_z(\alpha) H S^\dagger ,

    both :math:`R_y` carry the same angle, so each becomes one addition of the
    same loaded value under a fixed Clifford: two additions per Givens pair, and no
    arbitrary-angle rotation anywhere. The control is moved onto the loaded bits by a ``CNOT`` layer,
    so no controlled adder is needed.

    Args:
        psi_down (Sequence[int]): the ``N/2`` spatial orbitals :math:`U` acts on
        angle_wires (Sequence[int]): the loaded angle register, ``beth`` bits per pair
        beth (int): bits of precision per Givens angle
        pairs (Sequence[int]): the Givens pairs of this batch, in application order
        gradient_wires (Sequence[int]): the ``beth + 1`` wires of the phase gradient register
        work_wires (Sequence[int]): at least ``beth`` zeroed wires. The first ``beth`` are the
            scratch of :class:`~.SemiAdder`.
    """
    adder_work = work_wires[:beth]

    for slot, p in enumerate(pairs):
        bits = angle_wires[slot * beth : (slot + 1) * beth]

        add = adjoint(SemiAdder)
        lower, upper = psi_down[p], psi_down[p + 1]
        Hadamard(lower)
        CNOT(wires=[lower, upper])
        for wire in (lower, upper):
            adjoint(S)(wire)
            Hadamard(wire)
            ctrl(MultiX([1] * len(bits), wires=bits), control=wire)
            add(bits, gradient_wires, adder_work)
            ctrl(MultiX([1] * len(bits), wires=bits), control=wire)
            Z(wire)
            Hadamard(wire)
            S(wire)
        CNOT(wires=[lower, upper])
        Hadamard(lower)


def _select_half(
    chi,
    t_eigenvectors,
    beth,
    system_wires,
    index_wires,
    flag_wires,
    gradient_wires,
    work_wires,
    num_batches=1,
    one_body_table=False,
    skip_one_body=False,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    r"""Apply one :math:`V = U^\dagger Z_1 U` sandwich of the THC ``SELECT`` oracle.

    This is the kernel of Fig. 5 of `Lee et al. (2021)
    <https://arxiv.org/abs/2011.03494>`_. It is emitted as a single
    :func:`~.change_op_basis` so that a control placed on this sandwich falls only on the
    :math:`Z_1` reflection.

    Args:
        chi (tuple[tuple[float]]): the THC leaf matrix, shape ``(M, N/2)``
        t_eigenvectors (tuple[tuple[float]]): the one-body eigenvectors as columns, shape
            ``(N/2, N/2)``. Required when ``one_body_table=True``, ignored otherwise.
        beth (int): bits of precision per Givens angle
        system_wires (Sequence[int]): the ``N`` spin orbitals
        index_wires (Sequence[int]): the ``ceil(log2(M + 1))`` wires of the THC index
            addressing the rotation table
        flag_wires (Sequence[int]): three wires, the success flag, the one-body
            flag and the spin flag
        gradient_wires (Sequence[int]): the ``beth + 1`` wires holding the phase gradient state.
            This is assumed to be prepared on entry and left unchanged, as it is reused between
            ``PREPARE`` and ``SELECT`` oracles.
        work_wires (Sequence[int]): clean scratch, returned to :math:`\lvert 0 \rangle`
        num_batches (int): the number of batches the Givens angles are loaded in. The
            default of ``1`` loads all of them at once; larger values shrink
            ``work_wires`` at the cost of more ``QROM`` loads.
        one_body_table (bool): flag to check if the ``QROM`` address is extended by the
            one-body flag and the one-body rotations are loaded
        skip_one_body (bool): if ``False``, the :math:`Z_1` reflection is controlled on the
            success flag alone, as in the first sandwich of Fig. 5. If ``True``, it carries
            a second control, the open circle on the :math:`\lvert \nu = M \rangle`
            wire of the second sandwich in Fig. 5, so the reflection is switched off on the
            one-body block and this half acts as the identity there.

    Raises:
        ValueError: if a register has the wrong size, or if ``t_eigenvectors`` has the
            wrong shape when ``one_body_table=True``.
    """
    chi = math.asarray(chi, dtype=float)
    M, n_half = chi.shape
    n = math.ceil_log2(M + 1)
    req = select_thc_wires(M, 2 * n_half, beth, num_batches)
    batches, width = _angle_batches(n_half, num_batches)
    n_angle = width * beth if batches else 0

    if len(system_wires) != 2 * n_half:
        raise ValueError(f"system_wires must have {2 * n_half} entries; got {len(system_wires)}.")
    if len(index_wires) != n:
        raise ValueError(f"index_wires must have {n} entries for M={M}; got {len(index_wires)}.")
    if len(flag_wires) != 3:
        raise ValueError(f"flag_wires must have 3 entries; got {len(flag_wires)}.")
    if len(gradient_wires) != beth + 1:
        raise ValueError(
            f"gradient_wires must have beth + 1 = {beth + 1} entries; got {len(gradient_wires)}."
        )
    if len(work_wires) < req["work_wires"]:
        raise ValueError(
            f"work_wires must have at least {req['work_wires']} entries for M={M}, "
            f"N/2={n_half}, beth={beth}; got {len(work_wires)}."
        )
    if one_body_table and math.shape(t_eigenvectors) != (n_half, n_half):
        raise ValueError(
            f"t_eigenvectors must have shape ({n_half}, {n_half}); "
            f"got {math.shape(t_eigenvectors)}."
        )

    psi_down, psi_up = list(system_wires[:n_half]), list(system_wires[n_half:])
    succ, edge, spin = flag_wires
    angle_wires = list(work_wires[:n_angle])
    # The QROM restores its work wires before the adder runs, so the two share the pool.
    # ``angle_wires`` holds the loaded angles while the sandwich is open, so ``qrom_work`` is
    # the only part of the pool that is zeroed for the ops between the two QROM calls.
    qrom_work = list(work_wires[n_angle:])
    swap_work = list(work_wires)
    gradient_wires = list(gradient_wires)

    tables = _build_qrom_givens_data(chi, t_eigenvectors, beth, one_body_table, batches)
    # The index register is sized by PREPARE, which needs the ``nu = M`` sentinel, but this
    # QROM only addresses ``mu < M`` or ``ell < N/2``, so its top wire is always in |0>.
    n_mu = math.ceil_log2(max(M, n_half))
    qrom = {
        "control_wires": ([edge] if one_body_table else []) + list(index_wires)[n - n_mu :],
        "target_wires": angle_wires,
        "work_wires": qrom_work,
        "clean": True,
    }

    z_control, z_values = ([succ, edge], [1, 0]) if skip_one_body else ([succ], [1])

    def _basis():
        # Route V onto the spin-up block when the spin flag is set, then apply U,
        # one batch of angles at a time.
        for down, up in zip(psi_down, psi_up):
            ctrl(
                SWAP(wires=[down, up]),
                control=spin,
                work_wires=swap_work,
                work_wire_type="zeroed",
            )
        for b, batch in enumerate(batches):
            QROM(tables[b], **qrom)
            _apply_loaded_rotation(psi_down, angle_wires, beth, batch, gradient_wires, qrom_work)

    def _reflect():
        # Reflect on the first orbital, switched off on the one-body block.
        ctrl(
            Z(psi_down[0]),
            control=z_control,
            control_values=z_values,
            work_wires=qrom_work,
            work_wire_type="zeroed",
        )

    def _unbasis():
        adjoint(_basis)()

    return change_op_basis(_basis, _reflect, _unbasis)


def _index_swaps(
    mu_wires, nu_wires, edge, spin1, spin2, work_wires
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    r"""Exchange the two index registers and the two spin flags off the one-body block.

    The swaps are conditioned on :math:`\nu \neq M`, so they are emitted as a
    :func:`~.change_op_basis` around an ``X`` on ``edge``: a control placed on this block then
    falls only on the swaps, not on the two bit flips.

    Args:
        mu_wires (Sequence[int]): the wires of the first THC index
        nu_wires (Sequence[int]): the wires of the second THC index
        edge (int): the one-body sentinel flag
        spin1 (int): the spin flag of the first sandwich
        spin2 (int): the spin flag of the second sandwich
        work_wires (Sequence[int]): zeroed scratch for the controlled swaps
    """

    def _swaps():
        for a, b in zip(mu_wires, nu_wires, strict=True):
            ctrl(
                SWAP(wires=[a, b]),
                control=edge,
                work_wires=work_wires,
                work_wire_type="zeroed",
            )
        ctrl(
            SWAP(wires=[spin1, spin2]),
            control=edge,
            work_wires=work_wires,
            work_wire_type="zeroed",
        )

    return change_op_basis(partial(X, wires=edge), _swaps)


def _select_thc_resources(
    chi,
    t_eigenvectors,
    beth,
    system_wires,
    index_wires,
    flag_wires,
    gradient_wires,
    work_wires,
    num_batches=1,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Return the top-level resources of the SelectTHC decomposition."""
    sizes = [
        len(system_wires),
        len(index_wires),
        len(flag_wires),
        len(gradient_wires),
        len(work_wires),
    ]
    offsets = np.cumsum([0] + sizes)
    concrete_wires = [
        range(int(start), int(stop)) for start, stop in zip(offsets[:-1], offsets[1:])
    ]
    system, index, flags, gradient, work = concrete_wires
    n = math.ceil_log2(len(chi) + 1)
    mu_wires = list(index[:n])
    nu_wires = list(index[n:])
    succ, edge, _, spin1, spin2 = flags

    # `_select_half` returns change_op_basis. Under capture that unrolls eagerly
    # (binds inner gates into the jaxpr and returns None), so pause while we
    # build the resource keys. compute_resources already stops queuing but does
    # not pause capture. See #10162.
    with capture.pause():
        first_half = _select_half(
            chi,
            t_eigenvectors,
            beth,
            system,
            mu_wires,
            [succ, edge, spin1],
            gradient,
            work,
            num_batches=num_batches,
            one_body_table=True,
        )
        second_half = _select_half(
            chi,
            t_eigenvectors,
            beth,
            system,
            nu_wires,
            [succ, edge, spin2],
            gradient,
            work,
            num_batches=num_batches,
            skip_one_body=True,
        )
        swaps = _index_swaps(mu_wires, nu_wires, edge, spin1, spin2, work)

    resources = Counter(abstractify(op) for op in (first_half, second_half, swaps))
    resources[X] += 1
    return resources


@register_resources(_select_thc_resources)
def _select_thc_decomp(
    chi,
    t_eigenvectors,
    beth,
    system_wires,
    index_wires,
    flag_wires,
    gradient_wires,
    work_wires,
    num_batches=1,
    **_,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    M = len(chi)
    n = math.ceil_log2(M + 1)
    mu_wires = index_wires[:n]
    nu_wires = index_wires[n : 2 * n]
    succ, edge, swap, spin1, spin2 = flag_wires

    # 1. V on mu in the first spin sector, the only sandwich acting on the one-body block.
    _select_half(
        chi,
        t_eigenvectors,
        beth,
        system_wires,
        mu_wires,
        [succ, edge, spin1],
        gradient_wires,
        work_wires,
        num_batches=num_batches,
        one_body_table=True,
    )

    # 2. V on nu in the other spin sector, switched off on the one-body block.
    _select_half(
        chi,
        t_eigenvectors,
        beth,
        system_wires,
        nu_wires,
        [succ, edge, spin2],
        gradient_wires,
        work_wires,
        num_batches=num_batches,
        skip_one_body=True,
    )

    # 3. Exchange the two indices and the two spin flags, and flip the qubit that
    #    controls the mu <-> nu swap. This is the "X on the ancilla qubit and swapping the mu and nu
    #    registers" step between Eqs. (38) and (39) of arXiv:2011.03494, and it is what makes
    #    SELECT self-inverse.
    _index_swaps(mu_wires, nu_wires, edge, spin1, spin2, work_wires)
    X(swap)


add_decomps(SelectTHC, _select_thc_decomp)
