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
"""Contains the templates for Alias Sampling."""

from itertools import islice

import numpy as np

import pennylane as qp
from pennylane.labs.templates import LeftClassicalComparator, LeftQuantumComparator


def uniform_prep_ops(n_states, target_wires, work_wires):
    r"""Prepare a uniform superposition over ``n_states`` basis states.

    .. math::

        \frac{1}{\sqrt{n_\text{states}}} \sum_{i=0}^{n_\text{states}-1} |i\rangle

    Uses ``Hadamard`` gates when ``n_states`` is a power of two, otherwise the
    amplitude-amplification circuit of `arXiv:1805.03662
    <https://arxiv.org/abs/1805.03662>`_ (Figure 12).

    Args:
        n_states (int): the number of states to prepare.
        target_wires (Sequence[int]): wires on which to prepare the superposition.
        work_wires (Sequence[int]): auxiliary qubits, returned to zero.
    """
    if n_states < 1:
        raise ValueError("n_states must be at least 1")

    k = (n_states & -n_states).bit_length() - 1
    L = n_states >> k
    logL = qp.math.ceil_log2(L)

    expected_target = k + logL
    if len(target_wires) != expected_target:
        raise ValueError(
            f"target_wires must have {expected_target} wires for n_states={n_states} "
            f"(k={k}, logL={logL}); got {len(target_wires)}."
        )

    data_L = target_wires[:logL]
    for w in target_wires:
        qp.Hadamard(w)

    if L == 1:
        return

    expected_work = 1 + max(logL - 1, 1)
    if len(work_wires) < expected_work:
        raise ValueError(
            f"work_wires must have at least {expected_work} wires for n_states={n_states} "
            f"(k={k}, logL={logL}); got {len(work_wires)}."
        )
    flr = L.bit_length() - 1  # floor(log2 L)
    theta = np.arccos(1.0 - (2**flr) / L)
    w_used = work_wires[1:expected_work]

    LeftClassicalComparator(
        x_wires=data_L, L=L, target_wire=work_wires[0], work_wires=w_used, comparator="<"
    )
    qp.RZ(theta, wires=work_wires[0])
    qp.adjoint(LeftClassicalComparator)(
        x_wires=data_L, L=L, target_wire=work_wires[0], work_wires=w_used, comparator="<"
    )

    for w in data_L:
        qp.Hadamard(w)
    qp.ctrl(qp.GlobalPhase(-theta), control=data_L, control_values=[0] * logL)
    for w in data_L:
        qp.Hadamard(w)

    qp.GlobalPhase(np.pi / 2)


def _build_alias_tables(probs, mu):
    r"""Compute the classical alias-sampling tables ``alt`` and ``keep``.

    O(L) iterative matching (Walker/Vose) for the coherent alias sampling of
    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_. Returns integers
    :math:`\mathrm{alt}_\ell \in [0, L)` and :math:`\mathrm{keep}_\ell \in [0, 2^\mu)`
    satisfying the normalization constraint (Eq. requirekl):

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
    if np.any(probs < 0):
        raise ValueError("probs must be non-negative")
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


def alias_sampling_wires(n_states, mu):
    r"""Return the wire counts required by :func:`alias_sampling`.

    Args:
        n_states (int): the number of amplitudes ``L``.
        mu (int): number of ``keep`` / ``sigma`` bits.

    Returns:
        dict: ``{"target_wires": n_target, "temp_wires": n_temp, "work_wires": n_work}``.

        * ``target_wires`` (``ceil(log2 L)``): the output index register ``|l>``.
        * ``temp_wires`` (``3*mu + ceil(log2 L)``): sigma + alt + keep + flag +
          comparator scratch (``mu - 1`` wires that the comparator leaves dirty);
          left entangled with ``|l>`` and uncomputed by :math:`prepare^{\dagger}`.
        * ``work_wires`` (``1+max(log(L) - 1, 1)``): minimum clean
          scratch (UNIFORM_L flag + work, reused by QROM), returned to
          :math:`|0\rangle`.

    .. note::

        The reported ``work_wires`` is the minimum required by :func:`alias_sampling`.
        More work_wires can be added to be forwarded to the internal
        ``qml.QROM``, which uses them for a ``SelectSwap`` decomposition that lowers
        the T-gate count at the cost of the additional qubits. At exactly the
        minimum, ``QROM`` uses its unary decomposition (more T-gates, fewer qubits).
        ``target_wires`` and ``temp_wires`` are exact and must be matched exactly.
    """
    if isinstance(mu, bool) or not isinstance(mu, int) or mu < 1:
        raise ValueError(f"mu must be a positive integer, got {mu!r}.")
    if n_states < 1:
        raise ValueError("n_states must be at least 1.")

    logL = qp.math.ceil_log2(n_states)
    n_target = logL
    # sigma(mu) + alt(logL) + keep(mu) + flag(1) + comparator scratch(mu-1)
    n_temp = mu + logL + mu + 1 + max(mu - 1, 0)
    n_work = 1 + max(logL - 1, 1)
    return {"target_wires": n_target, "temp_wires": n_temp, "work_wires": n_work}


def alias_sampling(probs, mu, target_wires, temp_wires, work_wires):
    r"""Prepare a state with real and positive amplitudes via coherent alias sampling (Figure 11 of
    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_).

    Starting from all-zeros, prepares

    .. math::

        \sum_{\ell=0}^{L-1} \sqrt{\widetilde{\rho}_\ell}\; |\ell\rangle |\mathrm{temp}_\ell\rangle ,

    the :math:`\mu`-bit approximation of the target :math:`\sqrt{w_\ell / \lambda}`
    on the index register ``target_wires``.

    The gate sequence is: ``UNIFORM_L`` on ``wires``, ``H^mu`` on the sigma part of
    ``temp_wires``, a ``QROM`` load of ``alt_l``/``keep_l``, the inequality test
    ``keep_l <= sigma``, and a flag-controlled SWAP of ``wires`` with ``alt_l``.

    Use :func:`alias_sampling_wires` for the required register sizes.

    .. warning::

        ``temp_wires`` come out entangled with ``|l>`` (the "temp" register of the
        paper) and are not returned to :math:`|0\rangle`. In a prepare/select/prepare
        pattern, ``qml.adjoint(alias_sampling)`` (``prepare``-dagger) uncomputes
        them. ``work_wires`` are returned to :math:`|0\rangle` and may be reused.

    Args:
        probs (Sequence[float]): non-negative weights :math:`w_\ell` (length ``L``).
        mu (int): number of bits for ``keep`` and ``sigma``, representing the precision of the alias-sampling coefficients.
        target_wires (Sequence[int]): the output index register ``|l>``, size ``ceil(log2 L)``.
        temp_wires (Sequence[int]): the garbage register (sigma + alt + keep + flag +
            comparator scratch), left entangled; size ``3*mu + ceil(log2 L)``.
        work_wires (Sequence[int]): clean scratch, returned to :math:`|0\rangle`.

    """
    probs = np.asarray(probs, dtype=float)
    L = len(probs)
    if L < 1:
        raise ValueError("probs must have at least one entry.")

    logL = qp.math.ceil_log2(L)

    if isinstance(mu, bool) or not isinstance(mu, int) or mu < 1:
        raise ValueError(f"mu must be a positive integer, got {mu!r}.")

    req = alias_sampling_wires(L, mu)

    if len(target_wires) != req["target_wires"]:
        raise ValueError(
            f"target_wires must have {req['target_wires']} entries for L={L}; got {len(target_wires)}."
        )
    if len(temp_wires) != req["temp_wires"]:
        raise ValueError(
            f"temp_wires must have {req['temp_wires']} entries for L={L}, mu={mu}; "
            f"got {len(temp_wires)}."
        )
    if len(work_wires) < req["work_wires"]:
        raise ValueError(
            f"work_wires must have at least {req['work_wires']} entries for L={L}, mu={mu}; "
            f"got {len(work_wires)}."
        )

    # Split temp_wires: sigma (mu), alt (logL), keep (mu), flag (1),
    # comparator scratch (mu-1).
    tw_iter = iter(temp_wires)

    sigma_wires = list(islice(tw_iter, mu))
    alt_wires = list(islice(tw_iter, logL))
    keep_wires = list(islice(tw_iter, mu))
    flag = next(tw_iter)
    cmp_work = list(islice(tw_iter, max(mu - 1, 0)))

    alt, keep = _build_alias_tables(probs, mu)

    data = [[0] * (logL + mu) for _ in range(2**logL)]
    for l in range(L):
        alt_bits = [int(b) for b in format(alt[l], f"0{logL}b")] if logL else []
        data[l] = alt_bits + [int(b) for b in format(keep[l], f"0{mu}b")]

    # 1. UNIFORM_L over |l>.
    uniform_prep_ops(L, target_wires, work_wires)

    # 2. H^mu over sigma.
    for w in sigma_wires:
        qp.Hadamard(w)

    # 3. QROM: load alt_l, keep_l addressed by |l>.
    qp.QROM(
        data,
        control_wires=list(target_wires),
        target_wires=list(alt_wires) + list(keep_wires),
        work_wires=work_wires[1:],
        clean=True,
    )

    # 4. Inequality test keep_l <= sigma -> flag. The comparator leaves its
    #    scratch (cmp_work) dirty; it lives in temp_wires and is uncomputed by
    #    prepare-dagger, so it must not come from the clean work pool.
    LeftQuantumComparator(
        list(keep_wires), list(sigma_wires), flag, work_wires=cmp_work, comparator="<="
    )

    # 5. flag-controlled SWAP of |l> with |alt_l>.
    for wl, wa in zip(target_wires, alt_wires):
        qp.CSWAP(wires=[flag, wl, wa])
