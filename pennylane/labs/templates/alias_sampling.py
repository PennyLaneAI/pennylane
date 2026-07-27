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

import numpy as np

import pennylane as qp
from pennylane.labs.templates import LeftClassicalComparator


def uniform_prep_ops(n_states, target_wires, flag, work_wires):
    """
    This operation prepares a uniform superposition over a given number of
    basis states. The uniform superposition is defined as:

    .. math::

        \frac{1}{\sqrt{l}} \sum_{i=0}^{l} |i\rangle

    where :math:`l` is the number of states.

    This operation uses ``Hadamard`` gates to create the uniform superposition when
    the number of states is a power of two. If the number of states is not a power of two,
    the amplitude amplification technique defined in
    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_ is used.

    Args:
        n_states (int): the number of states to prepare
        target_wires (Sequence[int]): the wires on which to prepare the uniform superposition
        flag (int): a single wire to use as a flag qubit
        work_wires (Sequence[int]): the wires to use as auxiliary qubits

    """

    if n_states < 1:
        raise ValueError("n_states must be at least 1")

    k = (n_states & -n_states).bit_length() - 1
    L = n_states >> k
    logL = (L-1).bit_length()

    expected = k + logL
    if len(target_wires) != expected:
        raise ValueError(
            f"target_wires must have {expected} wires for n_states={n_states} "
            f"(k={k}, logL={logL}); got {len(target_wires)}."
        )


    data_L = target_wires[:logL]
    for w in target_wires:
        qp.Hadamard(w)

    if L == 1:
        # power of two: Hadamards already give the uniform state.
        return

    flr = (L).bit_length() - 1 #floor(log2(L))

    # reference draws arccos((2^flr - L)/L); we use the pi-complement
    # arccos(1 - 2^flr/L) to match PennyLane's RZ sign convention.
    theta = np.arccos(1.0 - (2 ** flr) / L)
    w_used = list(work_wires)[: max(logL - 1, 1)]

    # --- amplitude-amplification on the logL register ---
    LeftClassicalComparator(x_wires=data_L, L=L, target_wire=flag,
                            work_wires=w_used, comparator="<")
    qp.RZ(theta, wires=flag)
    qp.adjoint(LeftClassicalComparator)(x_wires=data_L, L=L, target_wire=flag,
                                        work_wires=w_used, comparator="<")

    for w in data_L:
        qp.Hadamard(w)

    # reflect about |0...0> on the logL register
    qp.ctrl(qp.GlobalPhase(-theta), control=data_L, control_values=[0] * logL)

    for w in data_L:
        qp.Hadamard(w)

def alias_tables(probs, mu):
    r"""Compute the classical alias-sampling tables ``alt`` and ``keep``.

    Implements the ``O(L)`` iterative-matching preprocessing of the coherent alias
    sampling scheme of `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_
    (equivalent to Walker's alias method [Walker (1974), Vose (1991)], as used in
    OpenFermion's ``utils/_lcu_util.py``).

    For a target distribution :math:`\rho_\ell = w_\ell / \lambda` over
    :math:`\ell \in [0, L)`, it returns integers :math:`\mathrm{alt}_\ell \in [0, L)`
    and :math:`\mathrm{keep}_\ell \in [0, 2^\mu)` satisfying the normalization
    constraint (Eq. requirekl of the paper):

    .. math::

        \frac{\mathrm{keep}_\ell + \sum_{k \,:\, \mathrm{alt}_k = \ell}
        (2^\mu - \mathrm{keep}_k)}{2^\mu L} = \widetilde{\rho}_\ell ,

    where :math:`\widetilde{\rho}_\ell` is the :math:`\mu`-bit approximation of
    :math:`\rho_\ell`.

    Args:
        probs (Sequence[float]): non-negative weights :math:`w_\ell`. They need not
            be normalized; they are normalized internally to :math:`\rho_\ell`.
        mu (int): number of bits used for ``keep`` and the ``sigma`` register.

    Returns:
        tuple[list[int], list[int]]: ``(alt, keep)``, each of length ``L``.

    .. note::

        ``keep_l`` holds :math:`\mu` bits, so its representable range is
        :math:`[0, 2^\mu - 1]`. A "full keep" column is its own alias
        (``alt_l == l``), where the ``keep`` value cancels in the constraint above,
        so capping it at :math:`2^\mu - 1` is exact; any other capping is a
        :math:`\le 1/2^\mu` discretization effect controlled by ``mu``.
    """
    probs = np.asarray(probs, dtype=float)
    if np.any(probs < 0):
        raise ValueError("probs must be non-negative")
    L = len(probs)
    total = probs.sum()
    if total <= 0:
        raise ValueError("probs must sum to a positive value")

    n = 2**mu
    scaled = (L * probs / total).astype(float)  # mean 1

    alt = list(range(L))
    keep = [n] * L

    small = [l for l in range(L) if scaled[l] < 1.0]
    large = [l for l in range(L) if scaled[l] >= 1.0]

    while small and large:
        s = small.pop()
        g = large.pop()
        keep[s] = int(round(scaled[s] * n))
        alt[s] = g
        scaled[g] = scaled[g] + scaled[s] - 1.0
        (small if scaled[g] < 1.0 else large).append(g)

    for l in small + large:
        keep[l] = n
        alt[l] = l

    keep = [min(max(k, 0), n - 1) for k in keep]
    return alt, keep


def alias_sampling_ops(
    probs, mu, l_wires, sigma_wires, alt_wires, keep_wires, flag, uniform_flag, uniform_work
):
    r"""Prepare an arbitrary state via coherent alias sampling (``subprepare``).

    Implements the generic ``subprepare`` circuit of Figure 11 of
    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_. Starting from the
    all-zeros state, it prepares

    .. math::

        \sum_{\ell=0}^{L-1} \sqrt{\widetilde{\rho}_\ell}\; |\ell\rangle |\mathrm{temp}_\ell\rangle ,

    a superposition over :math:`L` indices with amplitudes
    :math:`\sqrt{\widetilde{\rho}_\ell}` (the :math:`\mu`-bit approximation of the
    target :math:`\sqrt{w_\ell / \lambda}`), where :math:`|\mathrm{temp}_\ell\rangle`
    is an entangled junk register.

    The gate sequence is: ``UNIFORM_L`` on the index register, ``H^mu`` on the
    ``sigma`` register, a ``QROM`` load of ``alt_l`` and ``keep_l``, an inequality
    test ``keep_l <= sigma`` written to ``flag``, and a ``flag``-controlled SWAP of
    the index register with ``alt_l``.

    .. note::

        The ``sigma``, ``alt``, ``keep`` and ``flag`` registers are intentionally
        left entangled with the index register (the "temp" junk of the paper). They
        are *not* returned to :math:`|0\rangle` here; the surrounding algorithm
        uncomputes them with ``prepare``-dagger. Do not treat them as clean ancillas.

    Args:
        probs (Sequence[float]): non-negative weights :math:`w_\ell` (length ``L``).
        mu (int): number of bits for ``keep`` and ``sigma``. Higher ``mu`` gives a
            closer approximation of the target distribution.
        l_wires (Sequence[int]): the index register :math:`|\ell\rangle`, size
            ``ceil(log2 L)``.
        sigma_wires (Sequence[int]): the ``sigma`` register, size ``mu``.
        alt_wires (Sequence[int]): register holding ``alt_l``, size ``ceil(log2 L)``.
        keep_wires (Sequence[int]): register holding ``keep_l``, size ``mu``.
        flag (int): single wire for the inequality-test result.
        uniform_flag (int): flag wire for the ``UNIFORM_L`` sub-block.
        uniform_work (Sequence[int]): work wires for ``UNIFORM_L`` (also reused as
            ``QROM`` / comparator scratch, and returned to :math:`|0\rangle` there).
    """
    probs = np.asarray(probs, dtype=float)
    L = len(probs)
    logL = max((L - 1).bit_length(), 1)

    if len(l_wires) != logL:
        raise ValueError(f"l_wires must have {logL} wires for L={L}; got {len(l_wires)}.")
    if len(alt_wires) != logL:
        raise ValueError(f"alt_wires must have {logL} wires for L={L}; got {len(alt_wires)}.")
    if len(sigma_wires) != mu:
        raise ValueError(f"sigma_wires must have mu={mu} wires; got {len(sigma_wires)}.")
    if len(keep_wires) != mu:
        raise ValueError(f"keep_wires must have mu={mu} wires; got {len(keep_wires)}.")
    if mu < 1:
        raise ValueError(f"mu must be a positive integer, got {mu}.")

    alt, keep = alias_tables(probs, mu)

    # QROM data: address l -> alt_l (logL bits, big-endian) then keep_l (mu bits).
    data = [
        [int(b) for b in format(alt[l], f"0{logL}b")]
        + [int(b) for b in format(keep[l], f"0{mu}b")]
        for l in range(L)
    ]
    while len(data) < 2**logL:  # pad unused addresses
        data.append([0] * (logL + mu))

    # 1. UNIFORM_L: uniform superposition over l in [0, L).
    uniform_prep_ops(L, l_wires, uniform_flag, uniform_work)

    # 2. H^mu: uniform superposition over sigma in [0, 2^mu).
    for w in sigma_wires:
        qp.Hadamard(w)

    # 3. QROM: load alt_l and keep_l, addressed by |l>.
    qp.QROM(
        data,
        control_wires=list(l_wires),
        target_wires=list(alt_wires) + list(keep_wires),
        work_wires=list(uniform_work),
        clean=True,
    )

    # 4. Inequality test keep_l <= sigma -> flag. (Comparator scratch reuses
    #    uniform_work, which the comparator + its adjoint leave untouched here.)
    LeftQuantumComparator(
        list(keep_wires), list(sigma_wires), flag, work_wires=list(uniform_work), comparator="<="
    )

    # 5. flag-controlled SWAP of |l> with |alt_l>.
    for wl, wa in zip(l_wires, alt_wires):
        qp.ctrl(qp.SWAP(wires=[wl, wa]), control=flag)
