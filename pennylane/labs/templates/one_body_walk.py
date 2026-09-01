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
"""Qubitization walk operator for block-encoding of a one-particle operator."""

import pennylane as qp
from pennylane.labs.templates import alias_sampling, alias_sampling_wires


def one_body_walk_wires(norbs, alias_sampling_precision):
    r"""Sizes of the three wire registers required by :func:`one_body_walk`.

    The registers are:
        * ``prep_wires``: the full PREP register that the reflection acts on
        * ``system_wires``: the state register :math:`|\psi\rangle` the operator acts on
        * ``work_wires``: clean scratch that returns to ``|0>``

    Args:
        norbs (int): number of spatial orbitals
        alias_sampling_precision (int): number of bits needed for alias-sampling coefficient precision.

    Returns:
        dict[str, int]: number of wires for ``prep_wires``, ``system_wires``, ``work_wires``.
    """
    req = alias_sampling_wires(norbs, alias_sampling_precision)
    return {
        "prep_wires": req["target_wires"] + 1 + req["temp_wires"],  # |p> + |sigma> + garbage
        "system_wires": 2 * norbs,
        "work_wires": req["work_wires"],  # clean scratch
    }


def one_body_walk(op_matrix, alias_sampling_precision, prep_wires, system_wires, work_wires):
    r"""Walk operator for the block-encoding of a one-particle operator.

    Implements :math:`\hat{\mathcal{W}} = \hat{\mathcal{R}} \cdot \text{PREP}^\dagger \cdot
    \text{SEL} \cdot \text{PREP}`, with :math:`\hat{\mathcal{R}} = \hat 1 - 2|0\rangle\langle 0|`
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

    where :math:`o_{pq} = V \operatorname{diag}(\mu) V^T`, :math:`\hat V` is the orbital rotation
    diagonalizing ``op_matrix``, and :math:`\hat n_{p\sigma} = (\hat 1 - \hat Z_{p\sigma})/2` is
    the occupation convention shared with ``qp.fermi.jordan_wigner`` and ``qp.qchem.hf_state``,
    in which :math:`|1\rangle` is occupied. The paper's version of the last line carries
    :math:`+\mu_p/2` because it uses :math:`\hat n = (\hat 1 + \hat z)/2` instead.

    The normalization of the block-encoding is :math:`\lambda = \sum_p |\mu_p|`.

    Args:
        op_matrix (array): The real symmetric one-body matrix, shape ``(N, N)``, where N is the number
            of spatial orbitals.
        alias_sampling_precision (int): number of bits needed for alias-sampling coefficient precision
        prep_wires (Sequence[int]): the full PREP register, reflected by ``R``
        system_wires (Sequence[int]): wires for representing the ``2 N`` system spin-orbitals
        work_wires (Sequence[int]): clean scratch returned to ``|0>``
    """
    norbs = qp.math.shape(op_matrix)[0]

    if qp.math.shape(op_matrix) != (norbs, norbs):
        raise ValueError(f"op_matrix must be square; got {qp.math.shape(op_matrix)}.")
    if qp.math.iscomplexobj(op_matrix):
        raise ValueError("op_matrix must be real.")
    if not qp.math.allclose(op_matrix, qp.math.transpose(op_matrix)):
        raise ValueError("op_matrix must be symmetric (o_pq = o_qp).")

    req = one_body_walk_wires(norbs, alias_sampling_precision)
    for name, seq in (("prep_wires", prep_wires), ("system_wires", system_wires)):
        if len(seq) != req[name]:
            raise ValueError(
                f"{name} must have {req[name]} wires for norbs={norbs}, "
                f"alias_sampling_precision={alias_sampling_precision}; got {len(seq)}."
            )
    if len(work_wires) < req["work_wires"]:
        raise ValueError(
            f"work_wires must have at least {req['work_wires']} wires for norbs={norbs}, "
            f"alias_sampling_precision={alias_sampling_precision}; got {len(work_wires)}."
        )

    prep_wires = qp.wires.Wires(prep_wires)
    system_wires = qp.wires.Wires(system_wires)
    work_wires = qp.wires.Wires(work_wires)

    # Split prep_wires: index |p> (na) + spin |sigma> (1) + temp register.
    na = alias_sampling_wires(norbs, alias_sampling_precision)["target_wires"]
    index_wires, (spin_wire, *garbage_wires) = prep_wires[:na], prep_wires[na:]

    mu, vmat = qp.math.linalg.eigh(op_matrix)  # o = vmat diag(mu) vmat.T

    dvec = qp.math.stack([(-1.0) ** i for i in range(norbs)])
    col = dvec if qp.math.linalg.det(vmat) > 0 else qp.math.concatenate([-dvec[:1], dvec[1:]])
    unitary_matrix = vmat * dvec[:, None] * col[None, :]
    absmu = qp.math.abs(mu)
    signs = qp.math.where(absmu > 0, qp.math.sign(mu), 1.0)

    # PREP
    alias_sampling(
        absmu,
        alias_sampling_precision,
        target_wires=index_wires,
        temp_wires=garbage_wires,
        work_wires=work_wires,
    )
    qp.Hadamard(spin_wire)
    # SEL = V . (multiplexed Z_{p,sigma}) . V^dagger, with the signs of mu_p phased in first
    for s in (0, 1):
        qp.BasisRotation(
            wires=[system_wires[s * norbs + p] for p in range(norbs)], unitary_matrix=unitary_matrix
        )

    # Carry the sign of mu_p as a -1 phase on |p> rather than scaling the multiplexed Z: Select
    # controls a bare Pauli far more cheaply than a scaled SProd.
    n_index = len(index_wires)
    for p, sign in enumerate(signs):
        if sign >= 0:
            continue
        zeros = [w for w, b in zip(index_wires, format(p, f"0{n_index}b")) if b == "0"]
        for w in zeros:
            qp.X(w)
        if n_index == 1:
            qp.Z(index_wires[0])
        else:
            qp.ctrl(
                qp.Z(index_wires[-1]),
                control=index_wires[:-1],
                work_wires=work_wires,
                work_wire_type="zeroed",
            )
        for w in zeros:
            qp.X(w)

    ops = [qp.Z(system_wires[s * norbs + p]) for p in range(norbs) for s in (0, 1)]
    # PREP puts amplitude only on |p> with p < norbs, so the control register has no support on
    # basis states with no matching op and the cheaper partial-Select decomposition is valid.
    qp.Select(ops, control=index_wires + [spin_wire], work_wires=work_wires, partial=True)

    for s in (0, 1):
        qp.adjoint(qp.BasisRotation)(
            wires=[system_wires[s * norbs + p] for p in range(norbs)], unitary_matrix=unitary_matrix
        )
    # PREP^dagger
    qp.adjoint(alias_sampling)(
        absmu,
        alias_sampling_precision,
        target_wires=index_wires,
        temp_wires=garbage_wires,
        work_wires=work_wires,
    )
    qp.Hadamard(spin_wire)

    # R = I - 2|0><0| on the PREP register (index + spin + garbage). R|0> = -|0> is what makes
    # the |0> block of the walk +O/lambda rather than -O/lambda.
    for w in prep_wires:
        qp.X(w)

    # alias_sampling returns work_wires to |0>, so they are available here as clean ancillas for
    # a much cheaper multi-controlled Z.
    qp.ctrl(
        qp.Z(prep_wires[-1]),
        control=prep_wires[:-1],
        control_values=[1] * (len(prep_wires) - 1),
        work_wires=work_wires,
        work_wire_type="zeroed",
    )
    for w in prep_wires:
        qp.X(w)
