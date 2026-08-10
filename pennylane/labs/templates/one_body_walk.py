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

import numpy as np

import pennylane as qp
from pennylane.labs.templates import alias_sampling, alias_sampling_wires


def one_body_walk_wires(norbs, alias_sampling_precision):
    r"""Sizes of the three wire registers required by :func:`one_body_walk`.

    The registers are:
        * ``prep_wires``: the full PREP register that the reflection acts on.
        * ``system_wires``: the state register ``|psi>`` the operator acts on.
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

    Implements :math:`\hat{\mathcal{W}} = \text{PREP} \cdot
    \text{SEL} \cdot \text{PREP}^\dagger \cdot \hat{\mathcal{R}}` (Fig. 12 of
    `arXiv:2602.20270 <https://arxiv.org/abs/2602.20270>`_), block-encoding the
    non-identity part of a one-body operator

    .. math::

        \hat{O} = \sum_{p,\sigma} \frac{\mu_p}{2}\, \hat V^\dagger \hat z_{p\sigma} \hat V ,

    where :math:`o_{pq} = V \operatorname{diag}(\mu) V^T` and :math:`\lambda = \sum_p |\mu_p|`.

    Args:
        op_matrix (array): The real symmetric one-body matrix, shape ``(N, N)``, where N is the number
            of spatial orbitals.
        alias_sampling_precision (int): alias-sampling coefficient precision
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

    prep_wires, system_wires, work_wires = list(prep_wires), list(system_wires), list(work_wires)

    # Split prep_wires: index |p> (na) + spin |sigma> (1) + temp register.
    na = alias_sampling_wires(norbs, alias_sampling_precision)["target_wires"]
    index_wires = prep_wires[:na]
    spin_wire = prep_wires[na]
    garbage_wires = prep_wires[na + 1 :]

    mu, vmat = np.linalg.eigh(op_matrix)  # o = vmat diag(mu) vmat.T
    if np.linalg.det(vmat) < 0:
        vmat[:, 0] *= -1

    dmat = qp.math.diag([(-1.0) ** i for i in range(norbs)])
    unitary_matrix = dmat @ vmat @ dmat
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
    # SEL = V . (signed multiplexed Z_{p,sigma}) . V^dagger
    for s in (0, 1):
        qp.BasisRotation(
            wires=[system_wires[s * norbs + p] for p in range(norbs)], unitary_matrix=unitary_matrix
        )
    ops = [
        qp.s_prod(signs[p], qp.Z(system_wires[s * norbs + p])) for p in range(norbs) for s in (0, 1)
    ]
    qp.Select(ops, control=index_wires + [spin_wire], work_wires=work_wires)
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

    # R = I - 2|0><0| on the PREP register (index + spin + garbage).
    for w in prep_wires:
        qp.X(w)
    qp.ctrl(
        qp.Z(prep_wires[-1]), control=prep_wires[:-1], control_values=[1] * (len(prep_wires) - 1)
    )
    for w in prep_wires:
        qp.X(w)
