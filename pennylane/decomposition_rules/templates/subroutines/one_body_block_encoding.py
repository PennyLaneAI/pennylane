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

"""Built-in decomposition rules for templates.subroutines.one_body_block_encoding."""

import numpy as np

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.templates.subroutines.alias_sampling import _apply_hadamards
from pennylane.ops import GlobalPhase, Hadamard, Z, adjoint
from pennylane.templates.subroutines.alias_sampling import AliasSampling
from pennylane.templates.subroutines.arithmetic.left_classical_comparator import (
    LeftClassicalComparator,
)
from pennylane.templates.subroutines.one_body_block_encoding import OneBodyBlockEncoding
from pennylane.templates.subroutines.qchem.basis_rotation import BasisRotation
from pennylane.templates.subroutines.qrom import QROM
from pennylane.typing import Bool, Float, Wire


def _block_encoding_data(op_matrix):
    r"""Return the classical data the block-encoding needs from ``op_matrix``.

    Diagonalizes ``op_matrix`` as :math:`o = V \operatorname{diag}(\mu) V^T` and returns the
    PREP weights :math:`|\mu_p|`, the number of negative eigenvalues, and the orbital-rotation
    matrix handed to :class:`~.BasisRotation`. Negating the first column if needed keeps
    :math:`\operatorname{det} V = +1` so that the Givens decomposition of ``BasisRotation``
    carries no extra phase flip.

    ``eigh`` returns the eigenvalues in ascending order, so the negative ones are exactly the
    indices :math:`0, \dots, n_\text{neg} - 1`. That is what lets the decomposition apply the sign
    of :math:`\mu_p` with a single ``p < n_neg`` comparison instead of a per-index phase.

    Args:
        op_matrix (tuple[tuple[float]]): the validated real symmetric one-body matrix

    Returns:
        tuple[tuple[float], int, numpy.ndarray]: the weights ``absmu`` as a hashable tuple, the
        number of negative eigenvalues ``n_neg``, and the ``unitary_matrix`` for
        ``BasisRotation``
    """
    mat = np.asarray(op_matrix, dtype=float)

    mu, vmat = np.linalg.eigh(mat)  # o = vmat diag(mu) vmat.T

    if np.linalg.det(vmat) < 0:
        vmat[:, 0] *= -1

    absmu = tuple(float(value) for value in np.abs(mu))
    n_neg = int(np.sum(mu < 0))
    return absmu, n_neg, vmat


def _split_prep_wires(prep_wires, norbs):
    """Split ``prep_wires`` into the index, spin and garbage sub-registers."""
    n_index = math.ceil_log2(norbs)
    index_wires = prep_wires[:n_index]
    spin_wire = prep_wires[n_index]
    garbage_wires = prep_wires[n_index + 1 :]
    return index_wires, spin_wire, garbage_wires


def _one_body_block_encoding_resources(
    op_matrix, alias_sampling_nbits, prep_wires, system_wires, work_wires
):  # pylint: disable=too-many-arguments,unused-argument
    norbs = len(op_matrix)
    absmu, n_neg, _ = _block_encoding_data(op_matrix)
    n_prep, n_work, n_system = len(prep_wires), len(work_wires), len(system_wires)
    n_index = math.ceil_log2(norbs)
    n_garbage = n_prep - n_index - 1

    prep = AliasSampling(
        absmu,
        alias_sampling_nbits,
        target_wires=Wire[n_index],
        temp_wires=Wire[n_garbage],
        work_wires=Wire[n_work],
    )
    # NOTE: '_block_encoding_data' diagonalizes a real symmetric matrix, so the orbital rotation is always real.
    rotation = BasisRotation(Float[norbs, norbs], wires=Wire[norbs])
    qrom = QROM(
        Bool[2 * norbs, n_system],
        control_wires=Wire[n_index + 1],
        target_wires=Wire[n_system],
        work_wires=Wire[n_work],
    )

    resources = {
        prep: 1,
        adjoint(prep): 1,
        Hadamard: 2 + 2 * n_system,
        rotation: 4,
        qrom: 1,
        GlobalPhase: 1,
    }

    if n_neg > 0:
        compare = LeftClassicalComparator(
            x_wires=Wire[n_index],
            value=n_neg,
            target_wire=Wire[1],
            work_wires=Wire[max(n_index - 1, 0)],
            comparator="<",
        )
        resources[compare] = 1
        resources[adjoint(compare)] = 1
        resources[Z] = 1

    return resources


@register_resources(_one_body_block_encoding_resources)
def _one_body_block_encoding_decomp(
    op_matrix, alias_sampling_nbits, prep_wires, system_wires, work_wires, **_
):  # pylint: disable=too-many-arguments
    norbs = len(op_matrix)
    absmu, n_neg, unitary_matrix = _block_encoding_data(op_matrix)

    index_wires, spin_wire, garbage_wires = _split_prep_wires(prep_wires, norbs)

    # PREP
    AliasSampling(
        absmu,
        alias_sampling_nbits,
        target_wires=index_wires,
        temp_wires=garbage_wires,
        work_wires=work_wires,
    )
    Hadamard(spin_wire)

    reshaped_system_wires = [system_wires[:norbs], system_wires[norbs : 2 * norbs]]
    if compiler.active() or capture.enabled():
        reshaped_system_wires = math.array(reshaped_system_wires, like="jax")

    # SEL = V^dagger . (multiplexed Z_{p,sigma}) . V as a matrix product, so V is applied first.
    # The signs of mu_p are phased in before the multiplexed Z.

    @for_loop(2)
    def basis_rot_both_spins(i, mat):
        BasisRotation(unitary_matrix=mat, wires=reshaped_system_wires[i])
        return mat

    basis_rot_both_spins(unitary_matrix)  # pylint: disable=no-value-for-parameter

    # Carry the sign of mu_p as a -1 phase on |p> rather than scaling the multiplexed Z: Select
    # controls a bare Pauli far more cheaply than a scaled SProd. The negative eigenvalues occupy
    # the first n_neg indices because eigh sorts them ascending, so one `p < n_neg` test flags them.
    if n_neg > 0:
        n_index = len(index_wires)
        compare_kwargs = {
            "x_wires": index_wires,
            "value": n_neg,
            "target_wire": work_wires[0],
            "work_wires": work_wires[1:n_index],
            "comparator": "<",
        }
        LeftClassicalComparator(**compare_kwargs)
        Z(work_wires[0])
        adjoint(LeftClassicalComparator)(**compare_kwargs)

    # Apply Select([Z(system_wires[s * norbs + p]) for p in range(norbs) for s in (0, 1)])
    # by wrapping a QROM with one-hot bitstrings in Hadamard layers.
    _apply_hadamards(system_wires)
    ids = np.arange(2 * norbs).reshape((2, norbs)).T.reshape(-1)
    bitstrings = np.eye(2 * norbs, dtype=int)[ids]
    QROM(
        bitstrings,
        control_wires=list(index_wires) + [spin_wire],
        target_wires=system_wires,
        work_wires=work_wires,
    )
    _apply_hadamards(system_wires)

    basis_rot_both_spins(unitary_matrix.conj().T)  # pylint: disable=no-value-for-parameter

    # PREP^dagger
    adjoint(
        AliasSampling(
            absmu,
            alias_sampling_nbits,
            target_wires=index_wires,
            temp_wires=garbage_wires,
            work_wires=work_wires,
        )
    )
    Hadamard(spin_wire)

    # PREP^dagger . SEL . PREP encodes -O/lambda, because O carries the minus sign of
    # Z = 1 - 2n. The global phase flips it to +O/lambda at no gate cost.
    GlobalPhase(np.pi)


add_decomps(OneBodyBlockEncoding, _one_body_block_encoding_decomp)
