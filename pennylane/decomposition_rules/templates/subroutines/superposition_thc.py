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

"""Built-in decomposition rules for templates.subroutines.superposition_thc."""

import numpy as np

from pennylane import math
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import RY, GlobalPhase, Hadamard, MultiControlledX, X, Z, adjoint, ctrl
from pennylane.templates.subroutines.arithmetic.left_classical_comparator import (
    LeftClassicalComparator,
)
from pennylane.templates.subroutines.arithmetic.left_quantum_comparator import LeftQuantumComparator
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.templates.subroutines.flip_sign import FlipSign
from pennylane.templates.subroutines.superposition_thc import SuperpositionTHC
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires, concatenate_wires


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
    work wires directly and therefore takes the whole pool from index ``7`` on.

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
        # TODO: Replace by TemporaryAND ladder if it does not cost the qubits for too long
        MultiControlledX(
            wires=nu_wires + work_wires[3:4],
            control_values=math.int_to_binary(M, n),
            work_wires=work_wires[7:],
            work_wire_type="zeroed",
        )

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


def _superposition_thc_resources(M, N, mu_wires, nu_wires, work_wires):
    # pylint: disable=unused-argument
    r"""Returns the exact gate counts of the SuperpositionTHC decomposition."""

    n = len(mu_wires)
    num_work_wires = len(work_wires)

    # Number of zeroed work wires available to each gate, spelled as in the decomposition:
    # extra_work = work_wires[3n+5:], the MCX of _left_inequalities takes work_wires[7:], the
    # reflection takes work_wires[1:] and the Controlled(Z) of step 3 one wire more than
    # extra_work. The Controlled(X) of step 6 is the only one left with extra_work alone.
    extra_work = max(0, num_work_wires - (3 * n + 5))

    lcc_le = LeftClassicalComparator(Wire[n], M, Wire[1], Wire[n - 1], comparator="<=")
    lcc_gt = LeftClassicalComparator(Wire[n], N // 2, Wire[1], Wire[n - 1], comparator=">=")
    lqc = LeftQuantumComparator(Wire[n], Wire[n], Wire[1], Wire[n], comparator="<=")
    mcx = MultiControlledX(Wire[n + 1], Bool[n], Wire[num_work_wires - 7], work_wire_type="zeroed")

    return {
        GlobalPhase: 1,
        Hadamard: 6 * n,
        X: 4,
        RY: 2,
        TemporaryAND: 2,
        FlipSign([0] * (2 * n + 1), Wire[2 * n + 1], work_wires=Wire[num_work_wires - 1]): 1,
        adjoint(TemporaryAND(Wire[3])): 2,
        MultiControlledX(Wire[4], work_wires=Wire[extra_work], work_wire_type="zeroed"): 1,
        ctrl(Z(Wire[1]), Wire[3], work_wires=Wire[extra_work + 1], work_wire_type="zeroed"): 1,
        # _left_inequalities applied twice in the forward direction
        lcc_le: 2,
        lcc_gt: 2,
        lqc: 2,
        # _left_inequalities applied twice as an adjoint.
        adjoint(lcc_le): 2,
        adjoint(lcc_gt): 2,
        adjoint(lqc): 2,
        mcx: 2,
        adjoint(mcx): 1,
    }


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
    # The comparators leave their scratch dirty until the adjoint of step 4, so the only
    # zeroed wires here are the not-yet-written success flag and whatever was passed in extra.
    ctrl(
        Z(work_wires[5]),
        control=work_wires[0:3],
        work_wires=work_wires[6:7] + extra_work,
        work_wire_type="zeroed",
    )
    X(wires=work_wires[5])
    adjoint(TemporaryAND(work_wires[3:6]))

    # 4. Uncompute the flags and the amplitude-marking rotation. The closure keeps ``M``
    # and ``N`` concrete; passing them as traced arguments breaks the comparators, whose
    # classical operands must be compile-time constants.
    # Keep the comparators symbolic so their adjoint rules can select measurement-based
    # uncomputation without reversing measurements in an expanded forward rule.
    adjoint(lambda: _left_inequalities(M, N, mu_wires, nu_wires, work_wires))()
    RY(-angle, wires=work_wires[0])

    # 5. Reflection about the equal-superposition state (the amplification step).
    for wire in mu_wires + nu_wires:
        Hadamard(wire)

    # Fig. 3 has a typo; the correct reflection state is [0...0]
    # Step 4 undid every flag and every comparator, so all of ``work_wires`` but the amplitude
    # amplification wire is zeroed and can absorb the controls of this reflection.
    FlipSign([0] * (2 * n + 1), mu_wires + nu_wires + work_wires[:1], work_wires=work_wires[1:])
    GlobalPhase(np.pi)

    for wire in mu_wires + nu_wires:
        Hadamard(wire)

    # 6. Recompute the flags onto the output ancilla register (work_wires[5], work_wires[6]).
    _left_inequalities(M, N, mu_wires, nu_wires, work_wires)

    TemporaryAND(work_wires[3:6])
    X(wires=work_wires[5])
    MultiControlledX(
        wires=concatenate_wires(work_wires[1:3], work_wires[5:7]),
        work_wires=extra_work,
        work_wire_type="zeroed",
    )
    X(wires=work_wires[5])
    adjoint(TemporaryAND(work_wires[3:6]))

    # 7. Final uncomputation, keeping the diagonal (mu = nu) equality flag.
    # The rotation that would clean work_wires[0] back to |0> is omitted (see note above).
    adjoint(lambda: _left_inequalities(M, N, mu_wires, nu_wires, work_wires, keep_eq=True))()


add_decomps(SuperpositionTHC, _superposition_thc)
