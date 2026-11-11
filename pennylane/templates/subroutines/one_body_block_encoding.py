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
"""Block-encoding of a one-body operator."""

import numpy as np

from pennylane.core.operator import Operator2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .alias_sampling import alias_sampling_wires


def one_body_block_encoding_wires(norbs, alias_sampling_nbits):
    r"""Returns the sizes of the three wire registers required by :class:`~.OneBodyBlockEncoding`.

    :class:`~.OneBodyBlockEncoding` acts on three disjoint registers whose sizes are fixed by
    ``norbs`` and ``alias_sampling_nbits``. Use this function to size them before allocating wires.

    The registers are:
        * ``prep_wires``: the PREP register whose :math:`|\vec 0\rangle` block holds the encoded
          operator, :math:`2 \lceil \log_2 n_\text{orbs} \rceil + 3 \mu + 1` wires
        * ``system_wires``: the state register :math:`|\psi\rangle` the operator acts on,
          :math:`2 n_\text{orbs}` wires
        * ``work_wires``: clean work wires that start and end in :math:`|0\rangle`, at least
          :math:`\lceil \log_2 n_\text{orbs} \rceil` of them

    Here :math:`\mu` is ``alias_sampling_nbits``.

    Args:
        norbs (int): number of spatial orbitals
        alias_sampling_nbits (int): number of bits of precision used for the alias-sampling
            coefficients

    Returns:
        dict[str, int]: the required number of wires for ``prep_wires`` and ``system_wires``, which
        are exact, and the minimum number for ``work_wires``

    **Example**

    >>> qp.one_body_block_encoding_wires(4, alias_sampling_nbits=2)
    {'prep_wires': 11, 'system_wires': 8, 'work_wires': 2}

    """
    req = alias_sampling_wires(norbs, alias_sampling_nbits)
    return {
        "prep_wires": req["target_wires"] + 1 + req["temp_wires"],  # |p> + |sigma> + garbage
        "system_wires": 2 * norbs,
        "work_wires": max(req["work_wires"], req["target_wires"]),  # clean work wires
    }


def _validate_op_matrix(op_matrix):
    """Check that ``op_matrix`` is a nested tuple defining a real symmetric matrix.

    Raises:
        ValueError: if ``op_matrix`` is not a nested ``tuple``, or does not define a square, real,
            finite, symmetric matrix of at least two spatial orbitals
    """
    if not isinstance(op_matrix, tuple) or not all(isinstance(row, tuple) for row in op_matrix):
        raise ValueError(
            "op_matrix must be a tuple of tuples of floats, because it is compile-time static "
            f"data and has to be hashable; got {type(op_matrix).__name__}. Convert an array with "
            "tuple(tuple(float(entry) for entry in row) for row in matrix)."
        )

    norbs = len(op_matrix)
    if norbs == 0 or any(len(row) != norbs for row in op_matrix):
        raise ValueError(
            f"op_matrix must be square; got {norbs} row(s) of lengths "
            f"{tuple(len(row) for row in op_matrix)}."
        )
    if norbs < 2:
        raise ValueError(
            "op_matrix must have at least two spatial orbitals, because a single one leaves the "
            f"PREP index register empty; got norbs={norbs}."
        )
    for row in op_matrix:
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise ValueError(f"op_matrix entries must be real numbers; got {entry!r}.")

    arr = np.asarray(op_matrix, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("op_matrix must be finite.")
    if not np.allclose(arr, arr.T):
        raise ValueError("op_matrix must be symmetric (o_pq = o_qp).")


class OneBodyBlockEncoding(Operator2):
    r"""Block-encoding of a one-body operator.

    Implements :math:`\hat{B} = -\,\text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP}`,
    following `arXiv:2602.20270 <https://arxiv.org/abs/2602.20270>`_ (Fig. 12). The
    :math:`|\vec 0\rangle` block of :math:`\hat{B}` on ``prep_wires`` is :math:`\hat O / \lambda`,
    where :math:`\hat O` is the non-identity part of a one-body operator

    .. math::

        \begin{align}
            \hat{O} &= \sum_{pq,\sigma} o_{pq}\, \hat c^\dagger_{p\sigma} \hat c_{q\sigma}
            - \Big( \sum_p \mu_p \Big) \hat 1 \\
            &= \sum_{p,\sigma} \mu_p\, \hat V^\dagger \hat n_{p\sigma} \hat V
            - \Big( \sum_p \mu_p \Big) \hat 1 \\
            &= -\sum_{p,\sigma} \frac{\mu_p}{2}\, \hat V^\dagger \hat Z_{p\sigma} \hat V ,
        \end{align}

    The eigenvalues :math:`\mu_p` and the eigenvector matrix :math:`V` come from the
    diagonalization :math:`o_{pq} = V \operatorname{diag}(\mu) V^T` of ``op_matrix``. The operator
    :math:`\hat V` is the orbital rotation that this classical matrix generates, applied by
    :class:`~.BasisRotation`; the unhatted :math:`V` in that identity is the matrix itself, not an
    operator. The occupation operator is :math:`\hat n_{p\sigma} = (\hat 1 - \hat Z_{p\sigma})/2`,
    the convention shared with ``qp.fermi.jordan_wigner`` and ``qp.qchem.hf_state``, in which
    :math:`|1\rangle` is occupied. The paper's version of the last line carries :math:`+\mu_p/2`
    instead, because it uses :math:`\hat n = (\hat 1 + \hat z)/2`.

    The normalization of the block-encoding is :math:`\lambda = \sum_p |\mu_p|`. The leading minus
    sign of :math:`\hat{B}` is applied as a :class:`~.GlobalPhase`, costing no gates, and is what
    makes the encoded block :math:`+\hat O / \lambda` rather than :math:`-\hat O / \lambda`.

    This is the block-encoding only, not the qubitization walk operator: composing it with a
    reflection about :math:`|\vec 0\rangle` on ``prep_wires`` gives the walk operator of Sec. III A
    of the same reference, whose powers block-encode Chebyshev polynomials of
    :math:`\hat O / \lambda`.

    Use :func:`~.one_body_block_encoding_wires` for the required register sizes.

    .. seealso:: :class:`~.AliasSampling` for the alias sampling used within this operator.

    .. warning::

        ``system_wires`` uses the spin-blocked ``sigma * norbs + p`` ordering, where ``p`` is the
        spatial-orbital index and ``sigma`` the spin index, matching the indices of the sums
        above. Passing a register laid out in the interleaved ``2 * p + sigma`` ordering produced
        by ``qp.qchem`` gives a silently incorrect block-encoding; no error is raised.

    Args:
        op_matrix (tuple[tuple[float]]): the real symmetric one-body matrix of shape
            ``(norbs, norbs)``, where ``norbs >= 2`` is the number of spatial orbitals.
        alias_sampling_nbits (int): number of bits of precision used for the alias-sampling
            coefficients
        prep_wires (WiresLike): the PREP register; the encoded operator is the
            :math:`|\vec 0\rangle` block on these wires
        system_wires (WiresLike): the ``2 * norbs`` system spin-orbitals, ordered
            spin-blocked: ``system_wires[sigma * norbs + p]`` holds spatial orbital ``p`` of spin
            sector ``sigma``, so the first ``norbs`` wires are one spin sector and the last
            ``norbs`` the other. This differs from the interleaved ``2 * p + sigma`` ordering
            produced by ``qp.qchem``; the occupation convention (:math:`|1\rangle` is
            occupied) is shared
        work_wires (WiresLike): work wires that start in :math:`|0\rangle` and are returned to
            :math:`|0\rangle`. At least :math:`\lceil \log_2 n_\text{orbs} \rceil` of them are
            required; extra wires are forwarded to the internal :class:`~.QROM` and
            :class:`~.Select` to lower the T-gate count

    Raises:
        ValueError: if ``op_matrix`` is not a nested ``tuple``, or is not square, not real, not
            symmetric, or smaller than ``2 x 2``
        ValueError: if ``prep_wires`` or ``system_wires`` do not have exactly the sizes reported
            by :func:`~.one_body_block_encoding_wires`, or if ``work_wires`` has fewer than the
            reported minimum
        ValueError: if ``op_matrix`` is zero, so that :math:`\lambda = \sum_p |\mu_p| = 0` and
            the block-encoding cannot be normalized

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp

        op_matrix = ((1.0, 2.0), (2.0, 1.0))
        req = qp.one_body_block_encoding_wires(len(op_matrix), alias_sampling_nbits=2)
        all_wires = qp.registers(req)

        @qp.qnode(qp.device("default.qubit", wires=sum(req.values())))
        def circuit():
            qp.OneBodyBlockEncoding(op_matrix, 2, **all_wires)
            return qp.probs(wires=all_wires["prep_wires"])

    The probability of finding the PREP register back in :math:`|\vec 0 \rangle` is the squared
    norm of the encoded block acting on the system state, here the vacuum:
    :math:`\| \hat O |\vec 0\rangle / \lambda \|^2 = \big(\sum_p \mu_p / \lambda\big)^2
    = (2/4)^2`.

    >>> print(np.round(circuit()[0], 3))
    0.25

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.OneBodyBlockEncoding
            :stacked:
            :symbols: :math:`K` = ``norbs`` spatial orbitals and :math:`\nu` negative
                eigenvalues of ``op_matrix``
            :example: the operator above with ``alias_sampling_nbits=1``, :math:`K=2`, :math:`\nu=1`

            .. decomposition-rule:: _one_body_block_encoding_decomp
                :image: /_static/decomposition_thumbnails/one_body_block_encoding_lcu.svg
                :where: :math:`\delta = 1` if :math:`\nu > 0` and :math:`\delta = 0` otherwise

                :class:`~.AliasSampling` | :math:`1` | 1
                :class:`Adjoint(AliasSampling) <.AliasSampling>` | :math:`1` | 1
                :class:`~.Hadamard` | :math:`4K + 2` | 10
                :class:`~.BasisRotation` on :math:`K` wires | :math:`4` | 4
                :class:`~.LeftClassicalComparator` | :math:`\delta` | 1
                :class:`Adjoint(LeftClassicalComparator) <.LeftClassicalComparator>` | :math:`\delta` | 1
                :class:`~.Z` | :math:`\delta` | 1
                :class:`~.QROM` | :math:`1` | 1
                :class:`~.GlobalPhase` | :math:`1` | 1

    """

    wire_argnames = ("prep_wires", "system_wires", "work_wires")
    compilable_argnames = ("op_matrix", "alias_sampling_nbits")
    arg_specs = {
        "prep_wires": Wire[-1],
        "system_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        op_matrix,
        alias_sampling_nbits: int,
        prep_wires: WiresLike,
        system_wires: WiresLike,
        work_wires: WiresLike,
    ):  # pylint: disable=too-many-arguments
        _validate_op_matrix(op_matrix)
        norbs = len(op_matrix)

        if (
            isinstance(alias_sampling_nbits, bool)
            or not isinstance(alias_sampling_nbits, int)
            or alias_sampling_nbits < 1
        ):
            raise ValueError(
                f"alias_sampling_nbits must be a positive integer, got {alias_sampling_nbits!r}."
            )

        # lambda = sum_p |mu_p| normalizes the block-encoding, so it has to be non-zero.
        if not np.any(np.asarray(op_matrix, dtype=float)):
            raise ValueError(
                "op_matrix must be non-zero: the block-encoding normalization "
                "lambda = sum_p |mu_p| must be a positive value."
            )

        prep_wires = Wires(prep_wires)
        system_wires = Wires(system_wires)
        work_wires = Wires([] if work_wires is None else work_wires)

        req = one_body_block_encoding_wires(norbs, alias_sampling_nbits)
        for name, register in (("prep_wires", prep_wires), ("system_wires", system_wires)):
            if len(register) != req[name]:
                raise ValueError(
                    f"{name} must have {req[name]} wires for norbs={norbs}, "
                    f"alias_sampling_nbits={alias_sampling_nbits}; got {len(register)}."
                )
        if len(work_wires) < req["work_wires"]:
            raise ValueError(
                f"work_wires must have at least {req['work_wires']} wire(s) for norbs={norbs}, "
                f"alias_sampling_nbits={alias_sampling_nbits}; got {len(work_wires)}."
            )

        validate_no_wire_overlaps(
            {
                "prep_wires": prep_wires,
                "system_wires": system_wires,
                "work_wires": work_wires,
            }
        )

        super().__init__(op_matrix, alias_sampling_nbits, prep_wires, system_wires, work_wires)
