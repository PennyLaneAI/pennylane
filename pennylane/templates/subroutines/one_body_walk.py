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
"""Qubitization walk operator for the block-encoding of a one-body operator."""

import numpy as np

from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import Hadamard, Z, adjoint, ctrl
from pennylane.typing import Bool, Complex, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .alias_sampling import AliasSampling, alias_sampling_wires
from .arithmetic.left_classical_comparator import LeftClassicalComparator
from .multix import MultiX
from .qchem.basis_rotation import BasisRotation
from .select import Select


def one_body_walk_wires(norbs, alias_sampling_nbits):
    r"""Returns the sizes of the three wire registers required by :class:`~.OneBodyWalk`.

    :class:`~.OneBodyWalk` acts on three disjoint registers whose sizes are fixed by ``norbs``
    and ``alias_sampling_nbits``. Use this function to size them before allocating wires.

    The registers are:
        * ``prep_wires``: the full PREP register that the reflection acts on,
          :math:`2 \lceil \log_2 n_\text{orbs} \rceil + 3 \mu + 1` wires
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

    >>> qp.one_body_walk_wires(4, alias_sampling_nbits=2)
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
            finite, symmetric matrix
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
    for row in op_matrix:
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise ValueError(f"op_matrix entries must be real numbers; got {entry!r}.")

    arr = np.asarray(op_matrix, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("op_matrix must be finite.")
    if not np.allclose(arr, arr.T):
        raise ValueError("op_matrix must be symmetric (o_pq = o_qp).")


def _walk_data(op_matrix):
    r"""Return the classical data the walk needs from ``op_matrix``.

    Diagonalizes ``op_matrix`` as :math:`o = V \operatorname{diag}(\mu) V^T` and returns the
    PREP weights :math:`|\mu_p|`, the number of negative eigenvalues, and the orbital-rotation
    matrix handed to :class:`~.BasisRotation`. The row/column sign fixing keeps
    :math:`\operatorname{det} V = +1` so that the Givens decomposition of ``BasisRotation``
    carries no extra phase flip.

    ``eigh`` returns the eigenvalues in ascending order, so the negative ones are exactly the
    indices :math:`0, \dots, n_\text{neg} - 1`. That is what lets the decomposition apply the sign
    of :math:`\mu_p` with a single ``p < n_neg`` comparison instead of a per-index phase.

    Args:
        op_matrix (tuple[tuple[float]]): the validated real symmetric one-body matrix

    Returns:
        tuple[tuple[float], int, numpy.ndarray]: the weights ``absmu`` as a hashable tuple, the
        number of negative eigenvalues ``n_neg``, and the ``unitary_matrix`` for
        ``BasisRotation``
    """
    mat = np.asarray(op_matrix, dtype=float)
    norbs = mat.shape[0]

    mu, vmat = np.linalg.eigh(mat)  # o = vmat diag(mu) vmat.T

    dvec = np.array([(-1.0) ** i for i in range(norbs)])
    col = dvec if np.linalg.det(vmat) > 0 else np.concatenate([-dvec[:1], dvec[1:]])
    unitary_matrix = vmat * dvec[:, None] * col[None, :]

    absmu = tuple(float(value) for value in np.abs(mu))
    n_neg = int(np.sum(mu < 0))
    return absmu, n_neg, unitary_matrix


class OneBodyWalk(Operator2):
    r"""Qubitization walk operator that block-encodes a one-body operator.

    Implements :math:`\hat{W} = \hat{R} \cdot \text{PREP}^\dagger \cdot
    \text{SEL} \cdot \text{PREP}`, with :math:`\hat{R} = \hat 1 - 2|0\rangle\langle 0|`
    the reflection on ``prep_wires``, following `arXiv:2602.20270
    <https://arxiv.org/abs/2602.20270>`_ (Fig. 12 for the block-encoding, Sec. III A for the
    walk operator). The :math:`|\vec 0\rangle` block of the walk is :math:`\hat O / \lambda`,
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

    The normalization of the block-encoding is :math:`\lambda = \sum_p |\mu_p|`.

    Use :func:`~.one_body_walk_wires` for the required register sizes.

    .. seealso:: :class:`~.AliasSampling` for the alias sampling used within this operator.

    .. warning::

        ``system_wires`` uses the spin-blocked ``sigma * norbs + p`` ordering, where ``p`` is the
        spatial-orbital index and ``sigma`` the spin index, matching the indices of the sums
        above. Passing a register laid out in the interleaved ``2 * p + sigma`` ordering produced
        by ``qp.qchem`` gives a silently incorrect block-encoding; no error is raised.

    Args:
        op_matrix (tuple[tuple[float]]): the real symmetric one-body matrix of shape
            ``(norbs, norbs)``, where ``norbs`` is the number of spatial orbitals.
        alias_sampling_nbits (int): number of bits of precision used for the alias-sampling
            coefficients
        prep_wires (WiresLike): the full PREP register, reflected by
            :math:`\hat{\mathcal{R}}`
        system_wires (WiresLike): the ``2 * norbs`` system spin-orbitals, ordered
            spin-blocked: ``system_wires[sigma * norbs + p]`` holds spatial orbital ``p`` of spin
            sector ``sigma``, so the first ``norbs`` wires are one spin sector and the last
            ``norbs`` the other. This differs from the interleaved ``2 * p + sigma`` ordering
            produced by ``qp.qchem``; the occupation convention (:math:`|1\rangle` is
            occupied) is shared
        work_wires (WiresLike): work wires that start in :math:`|0\rangle` and are returned to
            :math:`|0\rangle`. At least :math:`\lceil \log_2 n_\text{orbs} \rceil` of them are
            required; extra wires are forwarded to the internal :class:`~.QROM` and
            multi-controlled :math:`Z` to lower the T-gate count

    Raises:
        ValueError: if ``op_matrix`` is not a nested ``tuple``, or is not square, not real, or not
            symmetric
        ValueError: if ``prep_wires`` or ``system_wires`` do not have exactly the sizes reported
            by :func:`~.one_body_walk_wires`, or if ``work_wires`` has fewer than the reported
            minimum
        ValueError: if ``op_matrix`` is zero, so that :math:`\lambda = \sum_p |\mu_p| = 0` and
            the block-encoding cannot be normalized

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp

        op_matrix = ((1.0, 2.0), (2.0, 1.0))
        req = qp.one_body_walk_wires(len(op_matrix), alias_sampling_nbits=2)
        all_wires = qp.registers(req)

        @qp.qnode(qp.device("default.qubit", wires=sum(req.values())))
        def circuit():
            qp.OneBodyWalk(op_matrix, 2, **all_wires)
            return qp.probs(wires=all_wires["prep_wires"])

    The probability of finding the PREP register back in :math:`|\vec 0 \rangle` is the squared
    norm of the encoded block acting on the system state, here the vacuum:
    :math:`\| \hat O |\vec 0\rangle / \lambda \|^2 = \big(\sum_p \mu_p / \lambda\big)^2
    = (2/4)^2`.

    >>> print(np.round(circuit()[0], 3))
    0.25

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

        req = one_body_walk_wires(norbs, alias_sampling_nbits)
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


def _split_prep_wires(prep_wires, norbs, alias_sampling_nbits):
    """Split ``prep_wires`` into the index, spin and garbage sub-registers."""
    n_index = alias_sampling_wires(norbs, alias_sampling_nbits)["target_wires"]
    index_wires = prep_wires[:n_index]
    spin_wire = prep_wires[n_index]
    garbage_wires = prep_wires[n_index + 1 :]
    return index_wires, spin_wire, garbage_wires


def _one_body_walk_resources(
    op_matrix, alias_sampling_nbits, prep_wires, system_wires, work_wires
):  # pylint: disable=too-many-arguments,unused-argument
    norbs = len(op_matrix)
    absmu, n_neg, _ = _walk_data(op_matrix)
    n_prep, n_work = len(prep_wires), len(work_wires)
    n_index = alias_sampling_wires(norbs, alias_sampling_nbits)["target_wires"]
    n_garbage = n_prep - n_index - 1

    prep = AliasSampling(
        absmu,
        alias_sampling_nbits,
        target_wires=Wire[n_index],
        temp_wires=Wire[n_garbage],
        work_wires=Wire[n_work],
    )
    rotation = BasisRotation(Complex[norbs, norbs], wires=Wire[norbs])
    select = Select(
        [Z(Wire[1])] * (2 * norbs),
        control=Wire[n_index + 1],
        work_wires=Wire[n_work],
        partial=True,
    )
    reflection = ctrl(
        Z(Wire[1]),
        control=Wire[n_prep - 1],
        control_values=[1] * (n_prep - 1),
        work_wires=Wire[n_work],
        work_wire_type="zeroed",
    )

    resources = {
        prep: 1,
        adjoint(prep): 1,
        Hadamard: 2,
        rotation: 2,
        adjoint(rotation): 2,
        select: 1,
        MultiX(Bool[n_prep], Wire[n_prep]): 2,
        reflection: 1,
    }

    if n_neg > 0:
        compare = LeftClassicalComparator(
            x_wires=Wire[n_index],
            L=n_neg,
            target_wire=Wire[1],
            work_wires=Wire[max(n_index - 1, 0)],
            comparator="<",
        )
        resources[compare] = 1
        resources[adjoint(compare)] = 1
        resources[Z] = 1

    return resources


@register_resources(_one_body_walk_resources)
def _one_body_walk_decomp(
    op_matrix, alias_sampling_nbits, prep_wires, system_wires, work_wires, **_
):  # pylint: disable=too-many-arguments
    norbs = len(op_matrix)
    absmu, n_neg, unitary_matrix = _walk_data(op_matrix)

    index_wires, spin_wire, garbage_wires = _split_prep_wires(
        prep_wires, norbs, alias_sampling_nbits
    )

    # PREP
    AliasSampling(
        absmu,
        alias_sampling_nbits,
        target_wires=index_wires,
        temp_wires=garbage_wires,
        work_wires=work_wires,
    )
    Hadamard(spin_wire)

    # SEL = V^dagger . (multiplexed Z_{p,sigma}) . V as a matrix product, so V is applied first.
    # The signs of mu_p are phased in before the multiplexed Z.
    for s in (0, 1):
        BasisRotation(
            unitary_matrix=unitary_matrix,
            wires=system_wires[s * norbs : (s + 1) * norbs],
        )

    # Carry the sign of mu_p as a -1 phase on |p> rather than scaling the multiplexed Z: Select
    # controls a bare Pauli far more cheaply than a scaled SProd. The negative eigenvalues occupy
    # the first n_neg indices because eigh sorts them ascending, so one `p < n_neg` test flags them.
    if n_neg > 0:
        n_index = len(index_wires)
        compare_kwargs = {
            "x_wires": index_wires,
            "L": n_neg,
            "target_wire": work_wires[0],
            "work_wires": work_wires[1:n_index],
            "comparator": "<",
        }
        LeftClassicalComparator(**compare_kwargs)
        Z(work_wires[0])
        adjoint(LeftClassicalComparator)(**compare_kwargs)

    ops = [Z(system_wires[s * norbs + p]) for p in range(norbs) for s in (0, 1)]
    # We can use `partial=True` because PREP puts amplitude only on |p> with p < norbs,
    # so the control register has no support on basis states with no matching op.
    Select(ops, control=list(index_wires) + [spin_wire], work_wires=work_wires, partial=True)

    for s in (0, 1):
        adjoint(
            BasisRotation(
                unitary_matrix=unitary_matrix,
                wires=system_wires[s * norbs : (s + 1) * norbs],
            )
        )
    # PREP^dagger
    adjoint(
        AliasSampling(
            absmu,
            alias_sampling_nbits,
            target_wires=index_wires,
            temp_wires=garbage_wires,
            work_wires=work_wires,
        )
    )
    Hadamard(spin_wire)

    # R = I - 2|0><0| on the PREP register (index + spin + garbage). R|0> = -|0> is what makes
    # the |0> block of the walk +O/lambda rather than -O/lambda.
    n_prep = len(prep_wires)
    MultiX([True] * n_prep, wires=prep_wires)

    # AliasSampling returns work_wires to |0>, so they are available here as clean work wires.
    ctrl(
        Z(prep_wires[-1]),
        control=prep_wires[:-1],
        control_values=[1] * (n_prep - 1),
        work_wires=work_wires,
        work_wire_type="zeroed",
    )
    MultiX([True] * n_prep, wires=prep_wires)


add_decomps(OneBodyWalk, _one_body_walk_decomp)
