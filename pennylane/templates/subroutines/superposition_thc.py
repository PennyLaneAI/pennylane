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
"""Contains the SuperpositionTHC template, used as a subroutine in tensor
hypercontraction (THC) qubitization."""

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


class SuperpositionTHC(Operator2):
    r"""Prepare the uniform superposition over the valid :math:`(\mu, \nu)` index
    pairs of the tensor hypercontraction (THC) representation.

    This template prepares the state used by the ``SELECT``/``PREPARE`` pair in THC
    qubitization given the THC rank :math:`M` and the number of spin orbitals :math:`N`.
    A single round of amplitude amplification is used to produce the state:

    .. math::

        \lvert 0 \rangle^{\otimes n} \lvert 0 \rangle^{\otimes n} \lvert 0 \rangle \;\mapsto\;
        \sqrt{p}\, \frac{1}{\sqrt{d}} \sum_{(\mu, \nu) \in \mathcal{S}}
        \lvert \mu \rangle \lvert \nu \rangle \lvert 1 \rangle
        \;+\; \sqrt{1 - p}\, \lvert \phi \rangle \lvert 0 \rangle ,

    where the last qubit is the success flag ``work_wires[6]``. The desired uniform
    superposition over the valid index set :math:`\mathcal{S}` is the component flagged by
    :math:`\lvert 1 \rangle`, while :math:`\lvert \phi \rangle` is a leftover (garbage) state
    flagged by :math:`\lvert 0 \rangle`. The success probability :math:`p` equals :math:`1`
    whenever :math:`d \geq 2^{2n} / 4` (the valid pairs make up at least a quarter of the
    :math:`2^{2n}` index combinations), in which case the single amplification round prepares
    the exact uniform superposition. When :math:`d < 2^{2n} / 4` the single round cannot reach
    unit probability, so a weight :math:`1 - p` is left in the :math:`\lvert 0 \rangle`-flagged
    garbage subspace. The valid index set :math:`\mathcal{S}` is

    .. math::

       \mathcal{S} = \{ (\mu, \nu) \mid 0 \leq \mu \le \nu < M \} \;\cup\;
       \{ (\mu, M) \mid 0 \leq \mu < N/2 \}

    and :math:`d = N/2 + M(M+1)/2` is its size.

    The construction follows the tensor hypercontraction state preparation of
    `Lee et al. (2021), Fig. 3 <https://arxiv.org/abs/2011.03494>`_.

    .. note::

        Every work wire is returned to the zero state, except:

        - ``work_wires[0]``: the amplitude amplification auxiliary wire. Its final cleaning
          rotation is omitted, so it is left in an arbitrary state and is not a flag. This state
          is intended to be uncomputed by the adjoint of ``SuperpositionTHC``.
        - ``work_wires[3]``: flag, true if the system is in the state :math:`\lvert \eta = M \rangle`.
        - ``work_wires[6]``: flag, true if the superposition has been prepared correctly.

    Args:
        M (int): The THC rank. Together with ``N``
            it determines the size :math:`d = N/2 + M(M+1)/2` of the prepared superposition.
        N (int): The number of spin orbitals. Used to count the one-body contribution
            :math:`N/2` to the valid index set.
        mu_wires (WiresLike): The :math:`n` wires that store the first THC index :math:`\mu`.
            At least :math:`n = \lceil \log_2(M + 1) \rceil` wires are required so the index
            registers can hold the one-body sentinel value :math:`M` (equivalently
            :math:`M \le 2^{n} - 1`).
        nu_wires (WiresLike): The :math:`n` wires that store the second THC index :math:`\nu`.
            Must contain the same number of wires as ``mu_wires``.
        work_wires (WiresLike): The auxiliary wires. The first seven wires are the ones shown in
            Fig. 3 of `Lee et al. (2021) <https://arxiv.org/abs/2011.03494>`_;
            the remaining wires are scratch space for the comparators and multi-controlled
            gates. At least :math:`3\,n + 5` zeroed work wires must be provided, where
            :math:`n` is the size of ``nu_wires``.

    **Example**

    The template prepares the THC index superposition on the ``mu_wires`` / ``nu_wires``
    registers. Here :math:`n = 3`, so the minimum number of work wires is :math:`3n + 5 = 14`.
    The prepared superposition is uniform over the valid index pairs
    :math:`\mathcal{S}` conditioned on the success flag ``work_wires[6] == 1``; we
    therefore read out the ``mu`` and ``nu`` registers together with that flag.

    .. code-block:: python

        import numpy as np
        import pennylane as qp

        n = 3
        M, N = 5, 2
        mu_wires = list(range(0, n))
        nu_wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n, 2 * n + 3 * n + 5))
        success_flag = work_wires[6]

        dev = qp.device("lightning.qubit", wires=2 * n + 3 * n + 5)

        @qp.qnode(dev)
        def circuit():
            qp.SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
            return qp.probs(mu_wires + nu_wires + [success_flag])

    The valid pairs are exactly those flagged in the success subspace, and each
    carries equal weight :math:`1 / d` with :math:`d = N/2 + M(M+1)/2`.

    >>> probs = circuit().reshape(2**n, 2**n, 2)
    >>> valid = np.where(probs > 1e-9)
    >>> valid_mu_nu = [tuple(map(int, arr)) for arr in zip(*valid[:2])]
    >>> valid_mu_nu
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]
    >>> d = N // 2 + M * (M + 1) // 2
    >>> len(valid_mu_nu) == d
    True
    >>> np.allclose(probs[valid], 1/d)
    True

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.SuperpositionTHC
            :stacked:
            :symbols: :math:`n` wires per index register
            :example: :math:`n=1` (:math:`M=1`, :math:`N=2`)

            .. decomposition-rule:: _superposition_thc
                :image: /_static/decomposition_thumbnails/superposition_thc_amplification.svg

                :class:`~.Hadamard` | :math:`6n` | 6
                :class:`~.RY` | :math:`2` | 2
                :class:`~.X` | :math:`4` | 4
                :class:`~.LeftClassicalComparator` | :math:`4` | 4
                :class:`Adjoint(LeftClassicalComparator) <.LeftClassicalComparator>` | :math:`4` | 4
                :class:`~.LeftQuantumComparator` | :math:`2` | 2
                :class:`Adjoint(LeftQuantumComparator) <.LeftQuantumComparator>` | :math:`2` | 2
                :class:`~.MultiControlledX` with :math:`n` controls | :math:`3` | 3
                :class:`~.X` with 3 controls | :math:`1` | 1
                :class:`~.TemporaryAND` | :math:`2` | 2
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`2` | 2
                :class:`~.CZ` with 3 controls | :math:`1` | 1
                :class:`~.FlipSign` on :math:`2n + 1` wires | :math:`1` | 1
                :class:`~.GlobalPhase` | :math:`1` | 1
    """

    wire_argnames = ("mu_wires", "nu_wires", "work_wires")
    compilable_argnames = ("M", "N")
    arg_specs = {
        "mu_wires": Wire[-1],
        "nu_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        M: int,
        N: int,
        mu_wires: WiresLike,
        nu_wires: WiresLike,
        work_wires: WiresLike,
    ):  # pylint: disable=too-many-arguments
        mu_wires = Wires(mu_wires)
        nu_wires = Wires(nu_wires)
        work_wires = Wires(work_wires)

        n = len(mu_wires)
        if len(nu_wires) != n:
            raise ValueError(
                f"mu_wires and nu_wires must contain the same number of wires, but got "
                f"{n} and {len(nu_wires)}."
            )

        min_work_wires = 3 * n + 5
        if len(work_wires) < min_work_wires:
            raise ValueError(
                f"At least {min_work_wires} work_wires (3 * len(mu_wires) + 5) should be "
                f"provided, but only {len(work_wires)} were given."
            )

        validate_no_wire_overlaps(
            {"mu_wires": mu_wires, "nu_wires": nu_wires, "work_wires": work_wires}
        )

        if M < N // 2 - 1:
            raise ValueError("M must be greater than or equal to N//2 - 1.")

        # The index registers must be able to hold the one-body sentinel value
        # ``M`` (the column flagged by ``nu = M``), so ``M <= 2 ** n - 1``.
        if M > 2**n - 1:
            raise ValueError(
                f"mu_wires and nu_wires each need at least ceil(log2(M + 1)) wires. "
                f"Got M={M} with {n} wires, which only allows M up to {2**n - 1}. "
                f"Provide at least {math.ceil_log2(M + 1)} wires per index register."
            )

        super().__init__(M, N, mu_wires, nu_wires, work_wires)

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.mu_wires + self.nu_wires + self.work_wires
