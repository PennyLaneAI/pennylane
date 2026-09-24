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

from collections import defaultdict

import numpy as np

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import (
    RY,
    GlobalPhase,
    Hadamard,
    MultiControlledX,
    X,
    Z,
    adjoint,
    ctrl,
)
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .arithmetic.left_classical_comparator import LeftClassicalComparator
from .arithmetic.left_quantum_comparator import LeftQuantumComparator
from .arithmetic.temporary_and import TemporaryAND
from .flip_sign import FlipSign
from .multix import MultiX


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


def _left_inequalities(
    M, N, mu_wires, nu_wires, work_wires, keep_eq=False
):  # pylint:disable=too-many-arguments
    r"""Apply the inequality tests that flag a valid THC index pair.

    Computes the comparisons that define the valid index set onto dedicated flag
    wires of the ancilla register (Fig. 3 of `Lee et al. (2021)
    <https://arxiv.org/abs/2011.03494>`_):

    * ``work_wires[1]``: :math:`\nu <= M` (classical comparison against the THC rank).
    * ``work_wires[2]``: :math:`\mu \leq \nu` (quantum comparison between the two registers).
    * ``work_wires[3]``: :math:`\nu = M` (classical equality against the THC rank,
      i.e. the one-body sentinel column). Only computed when ``keep_eq`` is ``False``.
    * ``work_wires[4]``: :math:`\mu \geq N/2` (classical comparison selecting two-body
      terms; equivalently, the one-body block keeps :math:`\mu < N/2`, the :math:`N/2`
      one-body terms).

    The auxiliary wires used on each comparator are drawn from disjoint slices of ``work_wires``
    starting at index ``7``, except for the one-body sentinel flag calculator, which resets its
    work wires directly.

    Args:
        M (int): The THC rank.
        N (int): The number of spin orbitals.
        mu_wires (WiresLike): The wires storing the first THC index :math:`\mu`.
        nu_wires (WiresLike): The wires storing the second THC index :math:`\nu`.
        work_wires (WiresLike): The auxiliary wires.
        keep_eq (bool): If ``False`` (the default, used in the forward passes and the
            first adjoint), ``work_wires[3]`` is computed via the zero-controlled
            ``MultiControlledX``. If ``True`` (used only in the final
            ``adjoint(_left_inequalities)``), that gate is skipped so the prepared
            ``work_wires[3]`` flag is left in place as an output.
    """
    mu_wires = Wires(mu_wires)
    nu_wires = Wires(nu_wires)
    work_wires = Wires(work_wires)

    n = len(mu_wires)

    if not keep_eq:
        # We check if the register is in state M.
        # To do so, we use the fact that a MultiControlledX with control_values = 0 detects if
        # the register is in state 0, and we shift that state with MultiX before and after.
        # TODO: Can we just move these bit flips into the control values?
        # TODO: Replace by TemporaryAND ladder if it does not cost the qubits for too long
        MultiX(math.int_to_binary(M, n), wires=nu_wires)
        MultiControlledX(
            wires=nu_wires + work_wires[3:4],
            control_values=[0] * n,
            work_wires=work_wires[7 : n + 6],
            work_wire_type="zeroed",
        )
        MultiX(math.int_to_binary(M, n), wires=nu_wires)

    LeftClassicalComparator(
        nu_wires,
        M,
        target_wire=work_wires[1],
        work_wires=work_wires[7 : 7 + n - 1],
        comparator="<=",
    )
    LeftQuantumComparator(
        mu_wires,
        nu_wires,
        target_wire=work_wires[2],
        work_wires=work_wires[7 + n - 1 : 7 + 2 * n - 1],
        comparator="<=",
    )

    LeftClassicalComparator(
        mu_wires,
        N // 2,
        target_wire=work_wires[4],
        work_wires=work_wires[7 + 2 * n - 1 : 7 + 3 * n - 2],
        comparator=">=",
    )


def _controlled_pauli(pauli, num_control_wires, num_work_wires, control_values=None):
    """Resources for a zeroed-work multi-controlled single-qubit Pauli."""
    return ctrl(
        pauli(Wire[1]),
        control=Wire[num_control_wires],
        control_values=control_values,
        work_wires=Wire[num_work_wires],
        work_wire_type="zeroed",
    )


def _superposition_thc_resources(M, N, mu_wires, nu_wires, work_wires):
    # pylint: disable=unused-argument
    r"""Returns the exact gate counts of the SuperpositionTHC decomposition."""

    n = len(mu_wires)
    num_work_wires = len(work_wires)

    # Number of borrowed work wires available to each gate: the Controlled gates use
    # extra_work = work_wires[4n+6:], and the MCX in _left_inequalities uses work_wires[3n+6:4n+6].
    extra_work = max(0, num_work_wires - (3 * n + 5))

    lcc_le = LeftClassicalComparator(Wire[n], M, Wire[1], Wire[n - 1], comparator="<=")
    lcc_gt = LeftClassicalComparator(Wire[n], N // 2, Wire[1], Wire[n - 1], comparator=">=")
    lqc = LeftQuantumComparator(Wire[n], Wire[n], Wire[1], Wire[n], comparator="<=")
    mcx = _controlled_pauli(X, n, n - 1, control_values=[0] * n)
    multix = MultiX(Bool[n], Wire[n])

    resources = defaultdict(int)

    resources[GlobalPhase] += 1
    resources[Hadamard] += 6 * n
    resources[X] += 4
    resources[RY] += 2
    resources[TemporaryAND] += 2
    resources[FlipSign([0] * (2 * n + 1), Wire[2 * n + 1], work_wires=Wire[extra_work])] += 1
    resources[adjoint(TemporaryAND(Wire[3]))] += 2
    resources[_controlled_pauli(X, 3, extra_work)] += 1
    resources[_controlled_pauli(Z, 3, extra_work)] += 1
    # _left_inequalities applied twice in the forward direction
    resources[lcc_le] += 2
    resources[lcc_gt] += 2
    resources[lqc] += 2
    resources[multix] += 6
    # _left_inequalities applied twice as an adjoint.
    resources[adjoint(lcc_le)] += 2
    resources[adjoint(lcc_gt)] += 2
    resources[adjoint(lqc)] += 2
    resources[mcx] += 3
    return dict(resources)


@register_resources(_superposition_thc_resources)
def _superposition_thc(M, N, mu_wires, nu_wires, work_wires, **_):
    # pylint: disable=too-many-arguments
    #
    # The first seven `work_wires` correspond to the flag/auxiliary register in Fig. 3 of
    # https://arxiv.org/pdf/2011.03494. After the routine, all work wires return to the zero
    # state except the flags work_wires[3] and work_wires[6], and work_wires[0], whose cleaning
    # rotation we omit. Note that the paper uses 1-based indexing, whereas we use 0-based indexing.

    mu_wires = Wires(mu_wires)
    nu_wires = Wires(nu_wires)
    work_wires = Wires(work_wires)

    n = len(mu_wires)
    extra_work = work_wires[7 + 3 * n - 2 :]

    # 1. Equal superposition over both index registers.
    for wire in mu_wires + nu_wires:
        Hadamard(wire)

    # 2. Rotation angle for the single round of amplitude amplification.
    n_total_vals = 2**n
    d = N // 2 + M * (M + 1) // 2
    frac_valid = d / n_total_vals**2
    limit = 0.5 / math.sqrt(frac_valid)
    cos_val = math.where(limit < 1.0, limit, 1.0)
    angle = 2 * math.arcsin(cos_val)

    RY(angle, wires=work_wires[0])

    # 3. Flag the valid index pairs, then mark the "success" subspace with a phase.
    _left_inequalities(M, N, mu_wires, nu_wires, work_wires)

    # Replace Toffolis by temporary ANDs. For this, we need to move the PauliX on work_wires[5]
    # around a little bit
    TemporaryAND(work_wires[3:6])
    X(wires=work_wires[5])
    ctrl(Z(work_wires[5]), control=work_wires[0:3], work_wires=extra_work, work_wire_type="zeroed")
    X(wires=work_wires[5])
    adjoint(TemporaryAND(work_wires[3:6]))

    # 4. Uncompute the flags and the amplitude-marking rotation. The closure keeps ``M``
    # and ``N`` concrete; passing them as traced arguments breaks the comparators, whose
    # classical operands must be compile-time constants.
    # ``lazy=False`` adjoints each op of the body in place rather than wrapping the body in a
    # region that is reversed later. The comparators uncompute their elbows through
    # ``Adjoint(TemporaryAND)``, whose measurement-based rule is not reversible, so a lazy
    # adjoint region would put a mid-circuit measurement inside an inverse.
    adjoint(lambda: _left_inequalities(M, N, mu_wires, nu_wires, work_wires), lazy=False)()
    RY(-angle, wires=work_wires[0])

    # 5. Reflection about the equal-superposition state (the amplification step).
    for wire in mu_wires + nu_wires:
        Hadamard(wire)

    # Fig. 3 has a typo; the correct reflection state is [0...0]
    FlipSign([0] * (2 * n + 1), mu_wires + nu_wires + work_wires[:1], work_wires=extra_work)
    GlobalPhase(np.pi)

    for wire in mu_wires + nu_wires:
        Hadamard(wire)

    # 6. Recompute the flags onto the output ancilla register (work_wires[5], work_wires[6]).
    _left_inequalities(M, N, mu_wires, nu_wires, work_wires)

    TemporaryAND(work_wires[3:6])
    X(wires=work_wires[5])
    ctrl(
        X(work_wires[6]),
        control=work_wires[1:3] + work_wires[5:6],
        work_wires=extra_work,
        work_wire_type="zeroed",
    )
    X(wires=work_wires[5])
    adjoint(TemporaryAND(work_wires[3:6]))

    # 7. Final uncomputation, keeping the diagonal (mu = nu) equality flag.
    # ``lazy=False`` for the same reason as in step 4.
    adjoint(
        lambda: _left_inequalities(M, N, mu_wires, nu_wires, work_wires, keep_eq=True), lazy=False
    )()  # The rotation that would clean work_wires[0] back to |0> is omitted (see note above).


add_decomps(SuperpositionTHC, _superposition_thc)
