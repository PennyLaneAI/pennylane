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
"""Contains the templates for the tensor hypercontraction ``SELECT`` oracle."""

from itertools import islice

import numpy as np

import pennylane as qp


def _cascade_angles(leaf):
    r"""Givens angles :math:`\theta_p` mapping ``leaf`` onto :math:`\lvert e_0 \rangle`.

    Zeroes the vector from its last coordinate inwards, so row ``0`` of the resulting
    :math:`U` is the normalized ``leaf`` and :math:`U^\dagger Z_1 U` is the reflection about
    it. Only :math:`N/2 - 1` angles are needed rather than a full :math:`\mathcal{O}(N^2)`
    Givens network, because that sandwich depends on :math:`U` only through row ``0``.

    Args:
        leaf (tensor_like): the vector to rotate, shape ``(N/2,)``. Its norm is irrelevant;
            the angles are scale invariant.

    Returns:
        numpy.ndarray: the ``N/2 - 1`` angles, where entry ``p`` belongs to the pair
        ``(p, p + 1)``. Empty for a single orbital.

    Raises:
        ValueError: if ``leaf`` is the zero vector.
    """
    vec = np.asarray(leaf, dtype=float).reshape(-1)
    if vec.size <= 1:
        return np.zeros(0)
    if np.linalg.norm(vec) < 1e-15:
        raise ValueError("Cannot build a rotation from a zero vector.")

    # Each step of the cascade replaces vec[p] by the norm of the suffix it has just
    # zeroed, so the angles have a closed form. Only the last coordinate keeps its sign,
    # which is what puts the final angle in (-pi, 0] when leaf[-1] is negative.
    suffix = np.sqrt(np.cumsum(vec[::-1] ** 2)[::-1])
    suffix[-1] = vec[-1]
    return np.arctan2(suffix[1:], vec[:-1])


def _angle_batches(n_half, num_batches):
    r"""Split the Givens pairs into contiguous batches, in application order.

    Args:
        n_half (int): the number of spatial orbitals, ``N/2``
        num_batches (int): the number of batches to split the ``N/2 - 1`` pairs into

    Returns:
        tuple[list[list[int]], int]: the pairs of each batch, followed by the width of the
        widest batch. Pair ``p`` acts on the spatial orbitals ``(p, p + 1)`` and the ladder
        runs from ``N/2 - 2`` down to ``0``. Only the last batch may be short, so the width
        sizes the angle register and the number of batches returned can be smaller than
        ``num_batches``.
    """
    pairs = list(reversed(range(max(n_half - 1, 0))))
    width = max(int(np.ceil(len(pairs) / num_batches)), 1)
    return [pairs[i : i + width] for i in range(0, len(pairs), width)], width


def _build_qrom_givens_data(chi, t_eigenvectors, beth, one_body_table, batches):
    r"""Build the incremental ``QROM`` tables of quantized Givens angles.

    Each row holds the ``beth``-bit angles that rotate one leaf onto
    :math:`\lvert e_0 \rangle`, one field per pair in the batch; addresses not backed by a
    leaf are all-zero and load the identity. Since ``qp.QROM(clean=True)`` writes with an
    XOR, table :math:`b` is stored as its difference with table :math:`b - 1`, which carries
    the angle register from one batch to the next instead of erasing and reloading it. The
    first table is absolute, so re-loading it clears the register.

    Args:
        chi (tensor_like): the THC leaf matrix, shape ``(M, N/2)``, addressed by the index
            register.
        t_eigenvectors (tensor_like): the one-body eigenvectors as columns, shape
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
    chi = np.asarray(chi, dtype=float)
    M, n_half = chi.shape
    if not batches:
        return []

    block = 1 << qp.math.ceil_log2(M + 1)
    addresses = list(range(M))
    leaves = list(chi)
    if one_body_table:
        addresses += [block + ell for ell in range(n_half)]
        leaves += list(np.asarray(t_eigenvectors, dtype=float).T)

    levels = 1 << beth
    thetas = np.array([_cascade_angles(leaf) for leaf in leaves])
    k = np.round(np.mod(2.0 * thetas, 4.0 * np.pi) / (4.0 * np.pi) * levels).astype(int) % levels
    bits = (k[:, :, None] >> np.arange(beth - 1, -1, -1)) & 1

    # Only the last batch can be short, so the first sets the width of the angle register
    # and the rest fill a prefix.
    n_rows = block * (2 if one_body_table else 1)
    tables = np.zeros((len(batches), n_rows, len(batches[0]) * beth), dtype=int)
    for b, batch in enumerate(batches):
        tables[b, addresses, : len(batch) * beth] = bits[:, batch].reshape(len(leaves), -1)

    return [table.tolist() for table in np.concatenate([tables[:1], tables[:-1] ^ tables[1:]])]


def _apply_loaded_rotation(
    psi_down, angle_wires, beth, pairs, gradient_wires, adder_work, adjoint=False
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    r"""Apply the rotations of one batch, whose angles are held in ``angle_wires``.

    This is the phase-gradient compilation of Section III.C of `Lee et al. (2021)
    <https://arxiv.org/abs/2011.03494>`_. Applying this to the Givens rotation, we write it as

    .. math::

        \mathrm{SE}(\theta) = \left(H_0\,\mathrm{CNOT}_{01}\right)^\dagger
        \left[R_y(-\theta/2)_0 \otimes R_y(-\theta/2)_1\right]
        \left(H_0\,\mathrm{CNOT}_{01}\right) ,
        \qquad R_y(\alpha) = S H R_z(\alpha) H S^\dagger ,

    both :math:`R_y` carry the same angle, so each becomes one controlled addition of the
    same loaded value under a fixed Clifford: two additions per Givens pair, and no
    arbitrary-angle rotation anywhere.

    Args:
        psi_down (Sequence[int]): the ``N/2`` spatial orbitals :math:`U` acts on
        angle_wires (Sequence[int]): the loaded angle register, ``beth`` bits per pair
        beth (int): bits of precision per Givens angle
        pairs (Sequence[int]): the Givens pairs of this batch, in application order
        gradient_wires (Sequence[int]): the ``beth`` wires of the phase gradient register
        adder_work (Sequence[int]): ``beth - 1`` clean wires for :class:`~pennylane.SemiAdder`
        adjoint (bool): if ``True``, apply the inverse rotation
    """
    for slot, p in reversed(list(enumerate(pairs))) if adjoint else enumerate(pairs):
        bits = angle_wires[slot * beth : (slot + 1) * beth]

        # Subtracting rather than adding gives the forward rotation; the adjoint rotation
        # is the same Clifford frame with the addition running the other way.
        add = qp.SemiAdder if adjoint else qp.adjoint(qp.SemiAdder)
        lower, upper = psi_down[p], psi_down[p + 1]
        qp.Hadamard(lower)
        qp.CNOT(wires=[lower, upper])
        for wire in (lower, upper):
            qp.adjoint(qp.S)(wire)
            qp.Hadamard(wire)
            qp.ctrl(add, control=wire)(bits, gradient_wires, adder_work)
            qp.Hadamard(wire)
            qp.S(wire)
        qp.CNOT(wires=[lower, upper])
        qp.Hadamard(lower)


def select_thc_wires(M, N, beth, num_batches=1):
    r"""Return the wire counts required by :func:`select_thc`.

    Args:
        M (int): the THC rank.
        N (int): the number of spin orbitals.
        beth (int): bits of precision per Givens angle. The angle grid has spacing
            :math:`2\pi / 2^{\mathrm{beth}}`, so the worst-case error of any one angle
            is :math:`\pi / 2^{\mathrm{beth}}`.
        num_batches (int): the number of batches the Givens angles are loaded in. The
            default of ``1`` loads all of them at once. See the note below.

    Returns:
        dict: ``{"system_wires": n_system, "index_wires": n_index,
        "flag_wires": n_flag, "gradient_wires": n_gradient, "work_wires": n_work}``.

        * ``system_wires`` (``N``): the spin orbitals, the ``N/2`` spin-down spatial
          orbitals followed by the ``N/2`` spin-up ones.
        * ``index_wires`` (``2 * ceil(log2(M + 1))``): :math:`\mu` followed by
          :math:`\nu`, as produced by ``PREPARE``. The ``+ 1`` inside the logarithm
          leaves room for the one-body sentinel value :math:`\nu = M`.
        * ``flag_wires`` (``5``): the success flag, the one-body sentinel flag,
          ``PREPARE``'s :math:`\mu \leftrightarrow \nu` symmetrization flag,
          and the two spin flags.
        * ``gradient_wires`` (``beth``): the phase gradient register. See the note below.
        * ``work_wires`` (``ceil((N/2 - 1) / num_batches) * beth +
          max(ceil(log2(M + 1)), beth - 1)``): the minimum clean scratch, returned to
          :math:`\lvert 0 \rangle`. Zero when ``N/2 == 1``, where the sandwich is a lone
          :math:`Z_1` and no angles are loaded.

    Raises:
        ValueError: if ``M``, ``N``, ``beth`` or ``num_batches`` is not a positive
            integer, or if the one-body block does not fit the index register.

    .. note::

        Only ``work_wires`` is a minimum; the other four are exact and must be matched.
        Extra work wires are forwarded to the internal ``qp.QROM``, which uses them for a
        ``SelectSwap`` decomposition that lowers the T-gate count at the cost of those
        qubits.

    .. note::

        ``gradient_wires`` must be prepared by the caller in the phase gradient state

        .. math::

            \lvert \phi \rangle = \frac{1}{\sqrt{2^{\mathrm{beth}}}}
            \sum_{k=0}^{2^{\mathrm{beth}} - 1}
            e^{-2 \pi i k / 2^{\mathrm{beth}}} \lvert k \rangle ,

        a product state that ``beth`` ``Hadamard`` and ``beth`` ``PhaseShift`` gates prepare.
        The ``SELECT`` oracle leaves it unchanged, so it is
        deliberately not allocated internally: one register is prepared once and shared by
        ``PREPARE`` and ``SELECT``.

    **Example**

    >>> from pennylane.labs.templates import select_thc_wires
    >>> select_thc_wires(M=3, N=4, beth=4)
    {'system_wires': 4, 'index_wires': 4, 'flag_wires': 5, 'gradient_wires': 4, 'work_wires': 7}

    """
    for name, value in (("M", M), ("N", N), ("beth", beth), ("num_batches", num_batches)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    n = qp.math.ceil_log2(M + 1)
    n_half = N // 2
    if n_half > 2**n:
        raise ValueError(
            f"The one-body rotations do not fit the index register: N // 2 = {n_half} "
            f"exceeds 2 ** ceil(log2(M + 1)) = {2**n} for M = {M}."
        )

    batches, width = _angle_batches(n_half, num_batches)
    n_angle = width * beth if batches else 0
    return {
        "system_wires": N,
        "index_wires": 2 * n,
        "flag_wires": 5,
        "gradient_wires": beth,
        "work_wires": n_angle + max(n, beth - 1) if n_angle else 0,
    }


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

    This is the kernel of Figs. 5 and 7 of `Lee et al. (2021)
    <https://arxiv.org/abs/2011.03494>`_.

    Args:
        chi (tensor_like): the THC leaf matrix, shape ``(M, N/2)``
        t_eigenvectors (tensor_like): the one-body eigenvectors as columns, shape
            ``(N/2, N/2)``. Required when ``one_body_table=True``, ignored otherwise.
        beth (int): bits of precision per Givens angle
        system_wires (Sequence[int]): the ``N`` spin orbitals
        index_wires (Sequence[int]): the ``ceil(log2(M + 1))`` wires of the THC index
            addressing the rotation table
        flag_wires (Sequence[int]): three wires, the success flag, the one-body
            flag and the spin flag
        gradient_wires (Sequence[int]): the ``beth`` wires holding the phase gradient state.
            This is assumed to be prepared on entry and left unchanged, as it is reused between
            ``PREPARE`` and ``SELECT`` oracles.
        work_wires (Sequence[int]): clean scratch, returned to :math:`\lvert 0 \rangle`
        num_batches (int): the number of batches the Givens angles are loaded in. The
            default of ``1`` loads all of them at once; larger values shrink
            ``work_wires`` at the cost of more ``QROM`` loads.
        one_body_table (bool): flag to check if the the ``QROM`` address is extended by the
            one-body flag and the one-body rotations are loaded
        skip_one_body (bool): flag to check if reflection is switched off when the one-body
            flag is set, so this sandwich acts as the identity on the one-body block

    Raises:
        ValueError: if a register has the wrong size, or if ``t_eigenvectors`` has the
            wrong shape when ``one_body_table=True``.
    """
    chi = np.asarray(chi, dtype=float)
    M, n_half = chi.shape
    n = qp.math.ceil_log2(M + 1)
    req = select_thc_wires(M, 2 * n_half, beth, num_batches)
    batches, width = _angle_batches(n_half, num_batches)
    n_angle = width * beth if batches else 0

    if len(system_wires) != 2 * n_half:
        raise ValueError(f"system_wires must have {2 * n_half} entries; got {len(system_wires)}.")
    if len(index_wires) != n:
        raise ValueError(f"index_wires must have {n} entries for M={M}; got {len(index_wires)}.")
    if len(flag_wires) != 3:
        raise ValueError(f"flag_wires must have 3 entries; got {len(flag_wires)}.")
    if len(gradient_wires) != beth:
        raise ValueError(
            f"gradient_wires must have beth = {beth} entries; got {len(gradient_wires)}."
        )
    if len(work_wires) < req["work_wires"]:
        raise ValueError(
            f"work_wires must have at least {req['work_wires']} entries for M={M}, "
            f"N/2={n_half}, beth={beth}; got {len(work_wires)}."
        )
    if one_body_table and np.shape(t_eigenvectors) != (n_half, n_half):
        raise ValueError(
            f"t_eigenvectors must have shape ({n_half}, {n_half}); "
            f"got {np.shape(t_eigenvectors)}."
        )

    psi_down, psi_up = list(system_wires[:n_half]), list(system_wires[n_half:])
    succ, edge, spin = flag_wires
    angle_wires = list(work_wires[:n_angle])
    # The QROM restores its work wires before the adder runs, so the two share the pool.
    qrom_work = list(work_wires[n_angle:])
    adder_work = qrom_work[: beth - 1]
    gradient_wires = list(gradient_wires)

    tables = _build_qrom_givens_data(chi, t_eigenvectors, beth, one_body_table, batches)
    qrom = {
        "control_wires": ([edge] if one_body_table else []) + list(index_wires),
        "target_wires": angle_wires,
        "work_wires": qrom_work,
        "clean": True,
    }

    # 1. Route V onto the spin-up block when the spin flag is set.
    for down, up in zip(psi_down, psi_up):
        qp.CSWAP(wires=[spin, down, up])

    # 2. Apply U, one batch of angles at a time
    for b, batch in enumerate(batches):
        qp.QROM(tables[b], **qrom)
        _apply_loaded_rotation(psi_down, angle_wires, beth, batch, gradient_wires, adder_work)

    # 3. Reflect on the first orbital, switched off on the one-body block
    z_control, z_values = ([succ, edge], [1, 0]) if skip_one_body else ([succ], [1])
    qp.ctrl(qp.Z(psi_down[0]), control=z_control, control_values=z_values)

    # 4. Apply Adjoint U, one batch of angles at a time
    for b in reversed(range(len(batches))):
        _apply_loaded_rotation(
            psi_down, angle_wires, beth, batches[b], gradient_wires, adder_work, adjoint=True
        )
        qp.adjoint(qp.QROM(tables[b], **qrom))

    # 5. Undo the spin routing.
    for down, up in zip(psi_down, psi_up):
        qp.CSWAP(wires=[spin, down, up])


def select_thc(
    chi,
    t_eigenvectors,
    beth,
    system_wires,
    index_wires,
    flag_wires,
    gradient_wires,
    work_wires,
    num_batches=1,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    r"""Self-inverse Hamiltonian selection oracle for tensor hypercontraction (THC).

    Implements the ``SELECT`` of Figs. 5 and 7 of `Lee et al. (2021)
    <https://arxiv.org/abs/2011.03494>`_. On the subspace flagged by the success wire this
    applies

    .. math::

        \lvert \mu \rangle \lvert \nu \rangle \lvert \psi \rangle \;\longmapsto\;
        \lvert \nu \rangle \lvert \mu \rangle \,
        V_\nu^{\beta} V_\mu^{\alpha} \lvert \psi \rangle ,
        \qquad V_\mu = U_\mu^\dagger Z_1 U_\mu = I - 2 c_\mu^\dagger c_\mu ,

    reducing to the one-body :math:`V_{T, \ell}^{\sigma}` when the one-body sentinel flag
    is set.

    The Hermitian operator that the walk operator block-encodes is given by:

    .. math::

        \langle P \rvert \mathrm{SELECT} \lvert P \rangle = \tfrac{1}{2}
        \left( V_\mu^{\alpha} V_\nu^{\beta} + V_\nu^{\alpha} V_\mu^{\beta} \right) ,

    up to the sign :math:`(-1)^{s_{\mu\nu}}` of the sampled coefficient, which ``PREPARE``
    applies as a controlled phase on its sign qubit.

    Use :func:`select_thc_wires` for the register sizes.

    Args:
        chi (tensor_like): the THC leaf matrix, shape ``(M, N/2)``
        t_eigenvectors (tensor_like): eigenvectors of the modified one-body matrix
            :math:`T'` as columns, shape ``(N/2, N/2)``
        beth (int): bits of precision per Givens angle
        system_wires (Sequence[int]): the ``N`` spin orbitals
        index_wires (Sequence[int]): ``2 * ceil(log2(M + 1))`` wires, :math:`\mu`
            followed by :math:`\nu`, as left by ``PREPARE``.
        flag_wires (Sequence[int]): this includes five wires, in order the success flag, the
            one-body flag, ``PREPARE``'s :math:`\mu \leftrightarrow \nu`
            symmetrization flag, and the two spin flags.
        gradient_wires (Sequence[int]): the ``beth`` wires holding the phase gradient state.
            This is assumed to be prepared on entry and left unchanged, as it is reused between
            ``PREPARE`` and ``SELECT`` oracles.
        work_wires (Sequence[int]): clean scratch, returned to :math:`\lvert 0 \rangle`
        num_batches (int): the number of batches the Givens angles are loaded in. The
            default of ``1`` loads all of them at once; larger values shrink
            ``work_wires`` at the cost of more ``QROM`` loads.

    Raises:
        ValueError: if a register has the wrong size.


    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp
        from pennylane.labs.templates import select_thc, select_thc_wires

        M, N, beth = 3, 4, 3
        rng = np.random.default_rng(0)
        chi = rng.standard_normal((M, N // 2))
        t_eigenvectors = np.linalg.qr(rng.standard_normal((N // 2, N // 2)))[0]

        sizes = select_thc_wires(M, N, beth)
        wires = qp.registers({name: size for name, size in sizes.items()})
        n_total = sum(sizes.values())

        @qp.qnode(qp.device("default.qubit", wires=n_total))
        def circuit():
            qp.PauliX(wires["flag_wires"][0])              # success flag
            for w in wires["flag_wires"][3:]:              # the two spin flags
                qp.Hadamard(w)
            for j, w in enumerate(wires["gradient_wires"]):     # phase gradient state
                qp.Hadamard(w)
                qp.PhaseShift(-2 * np.pi * 2 ** (beth - 1 - j) / 2**beth, wires=w)
            select_thc(
                chi, t_eigenvectors, beth, wires["system_wires"], wires["index_wires"],
                wires["flag_wires"], wires["gradient_wires"], wires["work_wires"],
            )

    """
    M = np.asarray(chi, dtype=float).shape[0]
    n = qp.math.ceil_log2(M + 1)

    if len(index_wires) != 2 * n:
        raise ValueError(
            f"index_wires must have 2 * ceil(log2(M + 1)) = {2 * n} entries for M={M}; "
            f"got {len(index_wires)}."
        )
    if len(flag_wires) != 5:
        raise ValueError(f"flag_wires must have 5 entries; got {len(flag_wires)}.")

    iw_iter = iter(index_wires)
    mu_wires = list(islice(iw_iter, n))
    nu_wires = list(islice(iw_iter, n))
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

    # 3. Exchange the two indices and the two spin flags, and flip PREPARE's
    #    symmetrization flag
    for a, b in zip(mu_wires, nu_wires):
        qp.ctrl(qp.SWAP(wires=[a, b]), control=edge, control_values=0)
    qp.ctrl(qp.SWAP(wires=[spin1, spin2]), control=edge, control_values=0)
    qp.X(swap)
