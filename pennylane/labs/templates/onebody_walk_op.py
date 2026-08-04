# Copyright 2025 Xanadu Quantum Technologies Inc.

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


def one_body_walk_wires(n_orbitals, mu_bits):
    r"""Return the wire registers required by :func:`one_body_walk`.

    Args:
        n_orbitals (int): number of spatial orbitals :math:`N_a` (PREP has ``L = N_a`` states).
        mu_bits (int): alias-sampling coefficient precision.

    Returns:
        dict: ``sys_wires`` (``2 N_a``), ``p_reg`` (``ceil(log2 N_a)``), ``sigma`` (1),
        ``temp_wires``, ``work_wires`` (from :func:`alias_sampling_wires`), and ``anc``
        (all PREP ancillas: ``p_reg + [sigma] + temp_wires + work_wires``).
    """
    nso = 2 * n_orbitals
    req = alias_sampling_wires(n_orbitals, mu_bits)
    na = req["target_wires"]
    sys_wires = list(range(nso))
    p_reg = list(range(nso, nso + na))
    sigma = nso + na
    t0 = sigma + 1
    temp_wires = list(range(t0, t0 + req["temp_wires"]))
    w0 = temp_wires[-1] + 1
    work_wires = list(range(w0, w0 + req["work_wires"]))
    anc = p_reg + [sigma] + temp_wires + work_wires
    return {
        "sys_wires": sys_wires, "p_reg": p_reg, "sigma": sigma,
        "temp_wires": temp_wires, "work_wires": work_wires, "anc": anc,
    }


def one_body_walk(o_matrix, mu_bits, wires=None):
    r"""Walk operator for the block-encoding of a one-particle operator.

    Implements :math:`\hat{\mathcal{W}} = \hat{\mathcal{R}} \cdot \texttt{PREP} \cdot
    \texttt{SEL} \cdot \texttt{PREP}^\dagger` (Fig. 12 / Eq. (44) of
    `arXiv:2602.20270 <https://arxiv.org/abs/2602.20270>`_), which block-encodes the
    non-identity part of a one-body (spin-conserving) operator

    .. math::

        \hat{O} = \sum_{p,\sigma} \frac{\mu_p}{2}\, \hat V^\dagger \hat z_{p\sigma} \hat V ,

    where :math:`o_{pq} = V \operatorname{diag}(\mu) V^T` and :math:`\lambda = \sum_p |\mu_p|`.
    The walk acts as :math:`e^{\pm i \arccos(\hat O / \lambda)}` on the block-encoded subspace.

    * ``PREP`` uses coherent alias sampling (:func:`~.alias_sampling`) to load
      :math:`\sqrt{|\mu_p| / \lambda}` on the index register and a Hadamard on the
      spin qubit; the sign of :math:`\mu_p` is carried into ``SEL``.
    * ``SEL`` applies the orbital rotation :math:`V`, a signed multiplexed Pauli-:math:`Z` selected
      by the index and spin
      registers (:class:`~.Select`), and :math:`V^\dagger`.
    * ``R`` is the reflection :math:`2|0\rangle\langle 0| - I` on the PREP ancillas.

    Args:
        one_body (tensor_like): real symmetric one-body matrix :math:`o_{pq}`, shape
            ``(N_a, N_a)``.
        alias_sampling_precision (int): alias-sampling coefficient precision.
        wires (dict or None): registers from :func:`one_body_walk_wires`; built from
            contiguous integers if ``None``.
    """
    n_orbitals = qp.math.shape(one_body)[0]
    if wires is None:
        wires = one_body_walk_wires(n_orbitals, alias_sampling_precision)

    mu, vmat = np.linalg.eigh(one_body)      # o = vmat diag(mu) vmat.T
    signs, absmu = np.sign(mu), np.abs(mu)
    sys_wires, p_reg, sigma = wires["sys_wires"], wires["p_reg"], wires["sigma"]
    anc = wires["anc"]

    def so(p, s):
        return sys_wires[s * n_orbitals + p]

    def prep():
        alias_sampling(absmu, alias_sampling_precision, target_wires=p_reg,
                       temp_wires=wires["temp_wires"], work_wires=wires["work_wires"])
        qp.Hadamard(sigma)

    # PREP
    prep()
    # SEL = V . (signed multiplexed Z_{p,sigma}) . V^dagger
    for s in (0, 1):
        qp.BasisRotation(wires=[so(p, s) for p in range(n_orbitals)], unitary_matrix=vmat.T)
    ops = [qp.s_prod(signs[p], qp.Z(so(p, s))) for p in range(n_orbitals) for s in (0, 1)]
    qp.Select(ops, control=p_reg + [sigma])          # control: p_reg (MSB) + sigma (LSB)
    for s in (0, 1):
        qp.adjoint(qp.BasisRotation)(wires=[so(p, s) for p in range(n_orbitals)], unitary_matrix=vmat.T)
    # PREP^dagger
    qp.adjoint(prep)()
    # R = 2|0><0| - I on PREP ancillas (X-wrapped multi-controlled Z)
    for w in anc:
        qp.X(w)
    qp.ctrl(qp.Z(anc[-1]), control=anc[:-1], control_values=[1] * (len(anc) - 1))
    for w in anc:
        qp.X(w)
