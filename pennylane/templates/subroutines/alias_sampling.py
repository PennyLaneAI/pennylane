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
"""Uniform state preparation and coherent alias sampling templates."""

from itertools import islice

import numpy as np

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.math import ceil_log2
from pennylane.ops import CSWAP, RZ, GlobalPhase, Hadamard, adjoint, ctrl
from pennylane.typing import AbstractWires, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .arithmetic.left_classical_comparator import LeftClassicalComparator
from .arithmetic.left_quantum_comparator import LeftQuantumComparator
from .qrom import QROM


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


class UniformPrep(Operator2):
    r"""Prepare a uniform superposition over the first ``n_states`` basis states.

    .. math::

        \frac{1}{\sqrt{n_\text{states}}} \sum_{i=0}^{n_\text{states}-1} |i\rangle

    Applies a layer of Hadamard gates when ``n_states`` is a power of two.
    Otherwise, uses the amplitude-amplification circuit described in
    Figure 12 of `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_.

    Args:
        n_states (int): the number of basis states to prepare
        target_wires (WiresLike): wires on which to prepare the superposition.
            Must have ``k + ceil(log2 L)`` wires, where ``n_states = 2**k * L`` with ``L`` odd.
        work_wires (WiresLike): auxiliary qubits, returned to zero. Unused when
            ``n_states`` is a power of two; otherwise at least ``ceil(log2 L)`` wires.

    Raises:
        ValueError: if ``n_states`` is less than 1, or if the wire registers have the wrong size

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp

        @qp.qnode(qp.device("default.qubit", wires=6))
        def circuit():
            qp.UniformPrep(n_states=5, target_wires=range(3), work_wires=range(3, 6))
            return qp.probs(wires=range(3))

    >>> print(np.round(circuit(), 3))
    [0.2 0.2 0.2 0.2 0.2 0.  0.  0. ]
    """

    wire_argnames = ("target_wires", "work_wires")
    compilable_argnames = ("n_states",)
    arg_specs = {"target_wires": Wire[-1], "work_wires": Wire[-1]}

    def __init__(self, n_states: int, target_wires: WiresLike, work_wires: WiresLike):
        if n_states < 1:
            raise ValueError("n_states must be at least 1")

        if isinstance(target_wires, AbstractWires):
            super().__init__(n_states, target_wires, work_wires)
            return

        target_wires = Wires(target_wires)
        work_wires = Wires([] if work_wires is None else work_wires)

        k = (n_states & -n_states).bit_length() - 1
        L = n_states >> k
        logL = ceil_log2(L)
        expected_target = k + logL
        if len(target_wires) != expected_target:
            raise ValueError(
                f"target_wires must have {expected_target} wires for n_states={n_states} "
                f"(k={k}, logL={logL}); got {len(target_wires)}."
            )

        validate_no_wire_overlaps({"target_wires": target_wires, "work_wires": work_wires})

        if L != 1:
            expected_work = logL  # flag + (logL - 1) comparator scratch
            if len(work_wires) < expected_work:
                raise ValueError(
                    f"work_wires must have at least {expected_work} wires for n_states={n_states} "
                    f"(k={k}, logL={logL}); got {len(work_wires)}."
                )

        super().__init__(n_states, target_wires, work_wires)


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
        L=L,
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
        x_wires=data_L, L=L, target_wire=work_wires[0], work_wires=w_used, comparator="<"
    )
    RZ(theta, wires=work_wires[0])
    adjoint(LeftClassicalComparator)(
        x_wires=data_L, L=L, target_wire=work_wires[0], work_wires=w_used, comparator="<"
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


def alias_sampling_wires(n_states, mu):
    r"""Compute the size of the three registers that :class:`~.AliasSampling` acts on.

    The three registers differ in what they hold and in whether they are restored, so their sizes
    are reported separately: ``target_wires`` carry the prepared state, ``temp_wires`` are left
    entangled with it, and ``work_wires`` are returned to :math:`|0\rangle` and can be reused.

    Args:
        n_states (int): the number of coefficients :math:`L` of the state to be prepared
        mu (int): number of bits of precision used for the ``keep`` and ``sigma`` registers

    Returns:
        dict: ``{"target_wires": n_target, "temp_wires": n_temp, "work_wires": n_work}``.

        * ``target_wires`` (``ceil(log2 L)``): the output index register :math:`|\ell \rangle`
        * ``temp_wires`` (``3*mu + ceil(log2 L)``): sigma + alt + keep + flag +
          comparator scratch (``mu - 1`` wires that the comparator leaves dirty);
          left entangled with :math:`|\ell \rangle` and uncomputed by :math:`prepare^{\dagger}`.
        * ``work_wires`` (``ceil(log2 L) - k``, where ``k`` is the number of trailing
          zero bits of ``L``): minimum clean scratch, returned to :math:`|0\rangle`. Only the odd part ``L / 2**k``
          needs amplitude amplification, so this is zero whenever ``L`` is a power
          of two.

    .. note::

        The reported ``work_wires`` is the minimum required by :class:`~.AliasSampling`.
        More work_wires can be added to be forwarded to the internal
        ``QROM``, which uses them for a ``SelectSwap`` decomposition that lowers
        the T-gate count at the cost of the additional qubits. At exactly the
        minimum, ``QROM`` uses its unary decomposition (more T-gates, fewer qubits).
        ``target_wires`` and ``temp_wires`` are exact and must be matched exactly.
    """
    if isinstance(mu, bool) or not isinstance(mu, int) or mu < 1:
        raise ValueError(f"mu must be a positive integer, got {mu!r}.")
    if n_states < 1:
        raise ValueError("n_states must be at least 1.")

    logL = ceil_log2(n_states)
    n_target = logL
    # sigma(mu) + alt(logL) + keep(mu) + flag(1) + comparator scratch(mu-1)
    n_temp = mu + logL + mu + 1 + max(mu - 1, 0)

    # UniformPrep only amplifies the odd part L of n_states = 2**k * L, which
    # costs ceil_log2(L) = logL - k wires (zero when n_states is a power of two).
    k = (n_states & -n_states).bit_length() - 1
    n_work = logL - k
    return {"target_wires": n_target, "temp_wires": n_temp, "work_wires": n_work}


def _canonicalize_probs(probs):
    """Turn ``probs`` into a hashable 1-D tuple of floats for compilable static data."""
    arr = np.asarray(probs, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"probs must be a 1-D sequence of weights, got shape {arr.shape}.")
    if arr.size < 1:
        raise ValueError("probs must have at least one entry.")
    if np.any(arr < 0) or not np.all(np.isfinite(arr)):
        raise ValueError("probs must be non-negative and finite")
    if arr.sum() <= 0:
        raise ValueError("probs must sum to a positive value")
    return tuple(float(p) for p in arr)


class AliasSampling(Operator2):
    r"""Prepare a state with real and positive amplitudes via coherent alias sampling.

    Starting from all-zeros, the circuit of Figure 11 in
    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_ prepares

    .. math::

        \sum_{\ell=0}^{L-1} \sqrt{\widetilde{\rho}_\ell}\; |\ell\rangle |\mathrm{temp}_\ell\rangle ,

    the :math:`\mu`-bit approximation of the target :math:`\sqrt{w_\ell / \lambda}`
    on the index register ``target_wires``, where :math:`\lambda = \sum_{\ell} w_\ell`
    and :math:`\widetilde{\rho}_\ell` is the :math:`\mu`-bit approximation of :math:`w_\ell / \lambda`,
    satisfying  :math:`|\widetilde{\rho}_\ell - w_\ell / \lambda| \le 2^{-\mu}`.

    The gate sequence is: ``UNIFORM_L`` on ``target_wires``, ``H^mu`` on the sigma part of
    ``temp_wires``, a ``QROM`` load of ``alt_l``/``keep_l``, the inequality test
    ``keep_l <= sigma``, and a flag-controlled SWAP of ``target_wires`` with ``alt_l``.

    Use :func:`~.alias_sampling_wires` for the required register sizes.

    .. warning::

        ``temp_wires`` come out entangled with :math:`|\ell\rangle` (the "temp" register of the
        paper) and are not returned to :math:`|0\rangle`. In a prepare/select/prepare
        pattern, ``qp.adjoint(AliasSampling(...))`` (``prepare``-dagger) uncomputes
        them. ``work_wires`` are returned to :math:`|0\rangle` and may be reused.

    Args:
        probs (Sequence[float]): non-negative weights :math:`w_\ell` (length ``L``)
        mu (int): number of bits for ``keep`` and ``sigma``, representing the precision of the alias-sampling coefficients
        target_wires (WiresLike): the output index register :math:`|\ell\rangle`, size ``ceil(log2 L)``
        temp_wires (WiresLike): the garbage register (sigma + alt + keep + flag +
            comparator scratch), left entangled; size ``3*mu + ceil(log2 L)``.
        work_wires (WiresLike): clean scratch, returned to :math:`|0\rangle`.

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp

        probs = np.array([0.1, 0.2, 0.3, 0.4])
        mu = 4

        req = qp.alias_sampling_wires(len(probs), mu)
        n_wires = sum(req.values())
        target_wires, temp_wires, work_wires = np.split(
            np.arange(n_wires), np.cumsum([req["target_wires"], req["temp_wires"]])
        )

        @qp.qnode(qp.device("default.qubit", wires=n_wires))
        def circuit():
            qp.AliasSampling(probs, mu, target_wires, temp_wires, work_wires)
            return qp.probs(wires=target_wires)

    >>> print(np.round(circuit(), 3))
    [0.094 0.203 0.297 0.406]
    """

    wire_argnames = ("target_wires", "temp_wires", "work_wires")
    compilable_argnames = ("probs", "mu")
    arg_specs = {
        "target_wires": Wire[-1],
        "temp_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        probs,
        mu: int,
        target_wires: WiresLike,
        temp_wires: WiresLike,
        work_wires: WiresLike,
    ):  # pylint: disable=too-many-arguments
        if isinstance(mu, bool) or not isinstance(mu, int) or mu < 1:
            raise ValueError(f"mu must be a positive integer, got {mu!r}.")

        probs = _canonicalize_probs(probs)
        if isinstance(target_wires, AbstractWires):
            super().__init__(probs, mu, target_wires, temp_wires, work_wires)
            return

        L = len(probs)

        target_wires = Wires(target_wires)
        temp_wires = Wires(temp_wires)
        work_wires = Wires([] if work_wires is None else work_wires)

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
        validate_no_wire_overlaps(
            {"target_wires": target_wires, "temp_wires": temp_wires, "work_wires": work_wires}
        )

        super().__init__(probs, mu, target_wires, temp_wires, work_wires)


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
