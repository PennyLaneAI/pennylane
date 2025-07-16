import math
from itertools import product

import numpy as np
import scipy as sp

from pennylane.labs.trotter_error import RealspaceMatrix, RealspaceSum
from pennylane.labs.trotter_error.realspace.matrix import _position_operator, _momentum_operator

from mode_selector import get_reduced_model
from pennylane.labs.trotter_error import ProductFormula, effective_hamiltonian, vibronic_fragments

def vibronic_norm(rs_mat: RealspaceMatrix, gridpoints: int, batch: list):
    if not _is_pow_2(gridpoints) or gridpoints <= 0:
        raise ValueError(
            f"Number of gridpoints must be a positive power of 2, got {gridpoints}."
        )

    padded = RealspaceMatrix(_next_pow_2(rs_mat.states), rs_mat.modes, rs_mat._blocks)

    norms = np.zeros(shape=(padded.states, padded.states))

    for i, j in batch:
        norms[i, j] = _block_norm(padded.block(i, j), gridpoints)

    return norms


def build_error_term(freqs, taylor_coeffs, modes):
    freqs, taylor_coeffs = get_reduced_model(freqs, taylor_coeffs, modes, strategy="PT")
    states = taylor_coeffs[0].shape[0]

    frags = vibronic_fragments(states, modes, np.array(freqs), taylor_coeffs.values())
    frags = dict(enumerate(frags))
    ham = sum(frags.values(), RealspaceMatrix.zero(states, modes))

    frag_labels = list(frags.keys()) + list(frags.keys())[::-1]
    second_order = ProductFormula(frag_labels, [1/2]*len(frag_labels))
    eff = effective_hamiltonian(second_order, frags, order=3, timestep=1)

    return (eff - 1j*ham)*(1/1j)

def _compute_norm(norm_mat: np.ndarray):
    if norm_mat.shape == (1, 1):
        return norm_mat[0, 0]

    half = norm_mat.shape[0] // 2

    top_left = norm_mat[0:half, 0:half]
    top_right = norm_mat[0:half, half:]
    bottom_left = norm_mat[half:, 0: half]
    bottom_right = norm_mat[half:, half:]

    norm1 = max(_compute_norm(top_left), _compute_norm(bottom_right))
    norm2 = math.sqrt(_compute_norm(top_right) * _compute_norm(bottom_left))

    return norm1 + norm2

def _block_norm(rs_sum: RealspaceSum, gridpoints: int):
    mode_groups = {}

    for op in rs_sum.ops:
        for index, coeff in op.coeffs.nonzero().items():
            group = frozenset(index)

            mode_mats = {i: sp.sparse.eye(gridpoints, dtype=np.complex128) for i in group}
            for i, mat in zip(index, op.ops):
                if mat == "P":
                    mode_mats[i] @= _momentum_operator(gridpoints, basis="harmonic", sparse=True)
                elif mat == "Q":
                    mode_mats[i] @= _position_operator(gridpoints, basis="harmonic", sparse=True)


            sorted_group = sorted(group)
            if len(sorted_group) == 0:
                mat = sp.sparse.eye(gridpoints)
            else:
                mat = mode_mats[sorted_group[0]]
                for i in sorted_group[1:]:
                    mat = sp.sparse.kron(mat, mode_mats[i], format="csr")

            mat = coeff * mat

            try:
                mode_groups[group] += mat
            except KeyError:
                mode_groups[group] = mat

    return sum(_get_eigenvalue(mat) for mat in mode_groups.values())

def _get_eigenvalue(mat):
    _, _, values = sp.sparse.find(mat)
    if np.allclose(values, 0):
        return 0

    eigvals, _ = sp.sparse.linalg.eigs(mat, k=1)
    return np.abs(eigvals[0])

def _is_pow_2(k: int) -> bool:
    """Test if k is a power of two"""
    return k & (k - 1) == 0

def _next_pow_2(k: int) -> int:
    """Return the smallest power of 2 greater than or equal to k"""
    return 2 ** (k - 1).bit_length()
