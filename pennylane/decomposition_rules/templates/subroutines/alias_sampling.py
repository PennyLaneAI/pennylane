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

"""Built-in decomposition rules for templates.subroutines.alias_sampling."""

from itertools import islice

import numpy as np

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.math import ceil_log2
from pennylane.ops import CSWAP, RZ, GlobalPhase, Hadamard, adjoint, ctrl
from pennylane.templates.subroutines.alias_sampling import AliasSampling, UniformPrep
from pennylane.templates.subroutines.arithmetic.left_classical_comparator import (
    LeftClassicalComparator,
)
from pennylane.templates.subroutines.arithmetic.left_quantum_comparator import LeftQuantumComparator
from pennylane.templates.subroutines.qrom import QROM
from pennylane.typing import Wire


def _apply_hadamards(wires):
    """Apply a Hadamard to each wire, using ``for_loop`` (unrolled when not tracing)."""
    n = len(wires)
    if n == 0:
        return
    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")

    @for_loop(n)
    def _loop(i):
        Hadamard(wires[i])

    _loop()  # pylint: disable=no-value-for-parameter


def _uniform_prep_resources(n_states, target_wires, work_wires):
    # pylint: disable=unused-argument
    k = (n_states & -n_states).bit_length() - 1
    L = n_states >> k
    logL = ceil_log2(L)
    resources = {Hadamard: len(target_wires)}
    if L == 1:
        return resources

    lcc = LeftClassicalComparator(
        x_wires=Wire[logL],
        value=L,
        target_wire=Wire[1],
        work_wires=Wire[max(logL - 1, 0)],
        comparator="<",
    )
    resources[lcc] = 1
    resources[RZ] = 1
    resources[adjoint(lcc)] = 1
    resources[Hadamard] += 2 * logL
    resources[ctrl(GlobalPhase(0.0), control=Wire[logL], control_values=[0] * logL)] = 1
    resources[GlobalPhase] = 1
    return resources


@register_resources(_uniform_prep_resources)
def _uniform_prep_decomp(n_states, target_wires, work_wires, **_):
    k = (n_states & -n_states).bit_length() - 1
    L = n_states >> k
    logL = ceil_log2(L)

    _apply_hadamards(target_wires)

    if L == 1:
        return

    flr = L.bit_length() - 1
    theta = np.arccos(1.0 - (2**flr) / L)
    data_L = target_wires[:logL]
    w_used = work_wires[1:logL]

    LeftClassicalComparator(
        x_wires=data_L, value=L, target_wire=work_wires[0], work_wires=w_used, comparator="<"
    )
    RZ(theta, wires=work_wires[0])
    adjoint(LeftClassicalComparator)(
        x_wires=data_L, value=L, target_wire=work_wires[0], work_wires=w_used, comparator="<"
    )

    _apply_hadamards(data_L)
    ctrl(GlobalPhase(-theta), control=data_L, control_values=[0] * logL)
    _apply_hadamards(data_L)
    GlobalPhase(np.pi / 2)


add_decomps(UniformPrep, _uniform_prep_decomp)


def _build_alias_tables(probs, mu):
    r"""Compute the classical alias-sampling tables ``alt`` and ``keep``.

    O(L) iterative matching (Walker/Vose) for the coherent alias sampling of
    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_. Returns integers
    :math:`\mathrm{alt}_\ell \in [0, L)` and :math:`\mathrm{keep}_\ell \in [0, 2^\mu)`
    satisfying the normalization constraint (Eq. 39):

    .. math::

        \frac{\mathrm{keep}_\ell + \sum_{k \,:\, \mathrm{alt}_k = \ell}
        (2^\mu - \mathrm{keep}_k)}{2^\mu L} = \widetilde{\rho}_\ell .

    Args:
        probs (Sequence[float]): non-negative weights (normalized internally).
        mu (int): number of bits for ``keep`` and the ``sigma`` register.

    Returns:
        tuple[list[int], list[int]]: ``(alt, keep)``, each of length ``L``.

    .. note::

        ``keep_l`` holds :math:`\mu` bits (range :math:`[0, 2^\mu - 1]`). Columns
        not touched by the matching loop keep their defaults ``alt_l = l`` and a
        full ``keep``; these are self-aliased, so the ``keep`` value cancels in the
        constraint above and capping at :math:`2^\mu - 1` is exact.
    """
    probs = np.asarray(probs, dtype=float)
    if np.any(probs < 0) or not np.all(np.isfinite(probs)):
        raise ValueError("probs must be non-negative and finite")

    L = len(probs)

    total = probs.sum()
    if total <= 0:
        raise ValueError("probs must sum to a positive value")

    n = 2**mu
    scaled = (L * probs / total).astype(float)
    alt = list(range(L))
    keep = [n] * L  # default: self-aliased, full keep (covers leftover columns)

    # Use this threshold instead of 1.0 to avoid floating-point issues when L is large and the
    # scaled values are very close to 1.0. The threshold is set to 1.0 - 1.0/(2*n) to ensure that the scaled values are
    # correct with respect to the \mu bits of precision.
    threshold = 1.0 - 1.0 / (2 * n)
    small_mask = scaled < threshold
    small = np.where(small_mask)[0].tolist()
    large = np.where(~small_mask)[0].tolist()

    while small and large:
        s = small.pop()
        g = large.pop()
        keep[s] = int(round(scaled[s] * n))
        alt[s] = g
        scaled[g] += scaled[s] - 1.0
        if scaled[g] < threshold:
            small.append(g)
        else:
            large.append(g)

    keep = np.clip(keep, 0, n - 1).tolist()
    return alt, keep


def _split_temp_wires(temp_wires, mu, logL):
    tw_iter = iter(temp_wires)
    sigma_wires = list(islice(tw_iter, mu))
    alt_wires = list(islice(tw_iter, logL))
    keep_wires = list(islice(tw_iter, mu))
    flag = next(tw_iter)
    cmp_work = list(islice(tw_iter, max(mu - 1, 0)))
    return sigma_wires, alt_wires, keep_wires, flag, cmp_work


def _qrom_data(probs, mu, logL):
    L = len(probs)
    alt, keep = _build_alias_tables(probs, mu)
    data = [[0] * (logL + mu) for _ in range(2**logL)]
    for l in range(L):
        alt_bits = [int(b) for b in format(alt[l], f"0{logL}b")] if logL else []
        data[l] = alt_bits + [int(b) for b in format(keep[l], f"0{mu}b")]
    return data


def _alias_sampling_resources(probs, mu, target_wires, temp_wires, work_wires):
    # pylint: disable=unused-argument
    L = len(probs)
    logL = ceil_log2(L)
    n_target = len(target_wires)
    n_work = len(work_wires)
    data = _qrom_data(probs, mu, logL)
    qrom = QROM(
        data,
        control_wires=Wire[n_target],
        target_wires=Wire[logL + mu],
        work_wires=Wire[max(n_work - 1, 0)],
        clean=True,
    )
    lqc = LeftQuantumComparator(Wire[mu], Wire[mu], Wire[1], Wire[max(mu - 1, 0)], comparator="<=")
    return {
        UniformPrep(L, Wire[n_target], Wire[n_work]): 1,
        Hadamard: mu,
        qrom: 1,
        lqc: 1,
        CSWAP: logL,
    }


@register_resources(_alias_sampling_resources)
def _alias_sampling_decomp(probs, mu, target_wires, temp_wires, work_wires, **_):
    L = len(probs)
    logL = ceil_log2(L)
    sigma_wires, alt_wires, keep_wires, flag, cmp_work = _split_temp_wires(temp_wires, mu, logL)
    data = _qrom_data(probs, mu, logL)

    UniformPrep(L, target_wires, work_wires)
    _apply_hadamards(sigma_wires)
    QROM(
        data,
        control_wires=list(target_wires),
        target_wires=list(alt_wires) + list(keep_wires),
        work_wires=work_wires[1:],
        clean=True,
    )
    LeftQuantumComparator(keep_wires, sigma_wires, flag, cmp_work, comparator="<=")

    n_swap = min(len(target_wires), len(alt_wires))
    if n_swap == 0:
        return
    if compiler.active() or capture.enabled():
        target_wires = math.array(target_wires, like="jax")
        alt_wires = math.array(alt_wires, like="jax")

    @for_loop(n_swap)
    def _swap(i):
        CSWAP(wires=[flag, target_wires[i], alt_wires[i]])

    _swap()  # pylint: disable=no-value-for-parameter


add_decomps(AliasSampling, _alias_sampling_decomp)
