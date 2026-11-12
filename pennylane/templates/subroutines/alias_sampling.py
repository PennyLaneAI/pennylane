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

import numpy as np

from pennylane.core.operator import Operator2
from pennylane.math import ceil_log2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


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

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.UniformPrep
            :stacked:
            :symbols: :math:`n_\text{states} = 2^k L` with :math:`L` odd, and
                :math:`\ell = \lceil \log_2 L \rceil`
            :example: :math:`n_\text{states}=5` (:math:`k=0`, :math:`L=5`, :math:`\ell=3`)

            .. decomposition-rule:: _uniform_prep_decomp
                :image: /_static/decomposition_thumbnails/uniform_prep_amplification.svg
                :lightbox:
                :requires: :math:`\ell` work wires
                :where: :math:`\delta = 1` if :math:`L > 1` and :math:`\delta = 0` otherwise. The
                    controlled :class:`~.GlobalPhase` is controlled on :math:`|0\rangle`

                :class:`~.Hadamard` | :math:`k + 3\ell` | 9
                :class:`~.LeftClassicalComparator` | :math:`\delta` | 1
                :class:`Adjoint(LeftClassicalComparator) <.LeftClassicalComparator>` | :math:`\delta` | 1
                :class:`~.RZ` | :math:`\delta` | 1
                :class:`~.GlobalPhase` with :math:`\ell` controls | :math:`\delta` | 1
                :class:`~.GlobalPhase` | :math:`\delta` | 1
    """

    wire_argnames = ("target_wires", "work_wires")
    compilable_argnames = ("n_states",)
    arg_specs = {"target_wires": Wire[-1], "work_wires": Wire[-1]}

    def __init__(self, n_states: int, target_wires: WiresLike, work_wires: WiresLike):
        if n_states < 1:
            raise ValueError("n_states must be at least 1")

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


def _validate_probs(probs):
    """Require ``probs`` to already be a hashable 1-D tuple of floats."""
    if not isinstance(probs, tuple):
        raise ValueError(
            "probs must be a tuple of floats, because it is compile-time static "
            f"data and has to be hashable; got {type(probs).__name__}. Convert an array with "
            "tuple(arr)."
        )
    if any(isinstance(x, (tuple, list)) for x in probs):
        raise ValueError(
            "probs must be a tuple of floats, but it contains nested sequences; "
            "a flat tuple was expected. Convert an array with tuple(arr)."
        )
    arr = np.asarray(probs, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"probs must be a 1-D sequence of weights, got shape {arr.shape}.")
    if arr.size < 1:
        raise ValueError("probs must have at least one entry.")
    if np.any(arr < 0) or not np.all(np.isfinite(arr)):
        raise ValueError("probs must be non-negative and finite")
    if arr.sum() <= 0:
        raise ValueError("probs must sum to a positive value")


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
        probs (tuple[float]): non-negative weights :math:`w_\ell` in a tuple of length ``L``.
        mu (int): number of bits for ``keep`` and ``sigma``, representing the precision of the alias-sampling coefficients
        target_wires (WiresLike): the output index register :math:`|\ell\rangle`, size ``ceil(log2 L)``
        temp_wires (WiresLike): the garbage register (sigma + alt + keep + flag +
            comparator scratch), left entangled; size ``3*mu + ceil(log2 L)``.
        work_wires (WiresLike): clean scratch, returned to :math:`|0\rangle`.

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp

        probs = (0.1, 0.2, 0.3, 0.4)
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

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.AliasSampling
            :stacked:
            :symbols: :math:`L` weights in ``probs``, :math:`\ell = \lceil \log_2 L \rceil` and
                :math:`\mu` = ``mu`` bits
            :example: :math:`L=4`, :math:`\mu=2`

            .. decomposition-rule:: _alias_sampling_decomp
                :image: /_static/decomposition_thumbnails/alias_sampling_coherent.svg

                :class:`~.UniformPrep` | :math:`1` | 1
                :class:`~.Hadamard` | :math:`\mu` | 2
                :class:`~.QROM` | :math:`1` | 1
                :class:`~.LeftQuantumComparator` | :math:`1` | 1
                :class:`~.CSWAP` | :math:`\ell` | 2
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

        _validate_probs(probs)

        L = len(probs)

        target_wires = Wires(target_wires)
        temp_wires = Wires(temp_wires)
        work_wires = Wires([] if work_wires is None else work_wires)

        req = alias_sampling_wires(L, mu)
        if len(target_wires) != req["target_wires"]:
            raise ValueError(
                f"target_wires must have {req['target_wires']} entries for L={L}; "
                f"got {len(target_wires)}."
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
