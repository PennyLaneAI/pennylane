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

"""Built-in decomposition rules for templates.subroutines.qchem.basis_rotation."""

import numpy as np

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.decomposition_rule import register_condition
from pennylane.ops import PauliZ, PhaseShift, SingleExcitation, cond
from pennylane.templates.subroutines.qchem.basis_rotation import BasisRotation
from pennylane.wires import Wires


def _qjit_or_capture():
    return compiler.active() or capture.enabled()


def _adjust_determinant(matrix):
    """Given an orthogonal (real-valued unitary) matrix, adjust its determinant to be 1.

    Args:
        matrix (array): orthogonal matrix to adjust the determinant of.

    Returns:
        tuple[bool, array]: Whether the determinant was negative, as well as the new
        matrix with determinant :math:`+1`.

    """
    det = math.linalg.det(matrix)

    mat = matrix
    if math.is_tracer(det):
        was_negative = det < 0
        mat = math.set_index(mat, 0, math.where(was_negative, -mat[0], mat[0]))
        return was_negative, mat

    was_negative = bool(det < 0)
    if was_negative:
        if math.get_interface(mat) != "jax":
            mat = math.toarray(mat).copy()
        mat = math.set_index(mat, 0, -mat[0])

    return was_negative, mat


def _is_jax_jit(U):
    return math.is_tracer(U) and not _qjit_or_capture()


def _prepare_args(unitary_matrix, wires):
    if isinstance(wires, Wires):
        wires = wires.labels

    if _qjit_or_capture():
        unitary_matrix, wires = math.array(unitary_matrix, like="jax"), math.array(
            wires, like="jax"
        )

    return unitary_matrix, wires


# pylint: disable=unused-argument
def _real_basis_rotation_resources(unitary_matrix, wires, check=False):
    """Upper bound on the gates emitted by '_real_basis_rotation_decomp'.

    - SingleExcitation: Exact count of N(N-1)/2 for Givens rotation.
    - PauliZ: *At most* one for determinant correction; skipped if det = +1. Under ``jax.jit``
      without qjit or program capture, this is a PhaseShift instead.

    """
    dim = math.shape(unitary_matrix)[0]
    correction = PhaseShift if _is_jax_jit(unitary_matrix) else PauliZ

    return {correction: 1, SingleExcitation: dim * (dim - 1) // 2}


@register_condition(lambda unitary_matrix, **_: not math.is_complex_dtype(unitary_matrix))
@register_resources(_real_basis_rotation_resources, exact=False)
def _real_basis_rotation_decomp(unitary_matrix, wires, **_):

    unitary, wires = _prepare_args(unitary_matrix, wires)

    was_negative, unitary = _adjust_determinant(unitary)

    if _is_jax_jit(was_negative):
        # Plain jax.jit cannot branch on the traced determinant, so apply PauliZ as PhaseShift(pi).
        PhaseShift(math.where(was_negative, np.pi, 0.0), wires[0])
    else:
        cond(was_negative, PauliZ)(wires[0])

    _, givens_list = math.decomposition.givens_decomposition(unitary)
    givens_matrices, givens_ids = zip(*givens_list, strict=True)

    if _qjit_or_capture():
        givens_ids = math.array(givens_ids, like="jax")
        givens_matrices = math.array(givens_matrices, like="jax")
        wires = math.array(wires, like="jax")

    @for_loop(len(givens_list))
    def givens_loop(idx):
        grot_mat = givens_matrices[idx]
        i, j = givens_ids[idx]
        theta = math.arctan2(math.real(grot_mat[1, 0]), math.real(grot_mat[0, 0]))
        SingleExcitation(2 * theta, wires=[wires[i], wires[j]])

    givens_loop()  # pylint: disable=no-value-for-parameter


# pylint: disable=unused-argument
def _complex_basis_rotation_resources(unitary_matrix, wires, check=False):
    """Upper bound on the gates emitted by '_complex_basis_rotation_decomp'.

    - SingleExcitation: Exact count of N(N-1)/2 for Givens rotation.
    - PhaseShift: *At most* N(N-1)/2 + N for the additional N diagonal phases; skipped if phase equal to zero.

    """
    dim = math.shape(unitary_matrix)[0]
    se_count = dim * (dim - 1) // 2

    return {PhaseShift: dim + se_count, SingleExcitation: se_count}


@register_condition(lambda unitary_matrix, **_: math.is_complex_dtype(unitary_matrix))
@register_resources(_complex_basis_rotation_resources, exact=False)
def _complex_basis_rotation_decomp(unitary_matrix, wires, **_):

    unitary, wires = _prepare_args(unitary_matrix, wires)

    phase_list, givens_list = math.decomposition.givens_decomposition(unitary)
    givens_matrices, givens_ids = zip(*givens_list, strict=True)

    if _qjit_or_capture():
        phase_list = math.array(phase_list, like="jax")
        givens_ids = math.array(givens_ids, like="jax")
        givens_matrices = math.array(givens_matrices, like="jax")

    @for_loop(len(phase_list))
    def phase_loop(idx):
        phase = phase_list[idx]
        PhaseShift(math.angle(phase), wires=wires[idx])

    phase_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(len(givens_matrices))
    def givens_loop(idx):
        grot_mat = givens_matrices[idx]
        i, j = givens_ids[idx]
        theta = math.arccos(math.real(grot_mat[1, 1]))
        # The first column of grot_mat is e^{i phi} (cos(theta), sin(theta)) with both entries
        # non-negative before the phase, so their sum always carries the phase.
        phi = math.angle(grot_mat[0, 0] + grot_mat[1, 0])
        SingleExcitation(2 * theta, wires=[wires[i], wires[j]])
        if _is_jax_jit(phi):
            PhaseShift(phi, wires[i])
        else:
            cond(math.logical_not(math.allclose(phi, 0.0)), PhaseShift)(phi, wires[i])

    givens_loop()  # pylint: disable=no-value-for-parameter


add_decomps(BasisRotation, _real_basis_rotation_decomp, _complex_basis_rotation_decomp)
