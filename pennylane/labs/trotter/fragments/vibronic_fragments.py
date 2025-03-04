"""Base class for Vibronic Hamiltonian"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from pennylane.labs.trotter.realspace import Node, RealspaceOperator, RealspaceSum, VibronicMatrix
from pennylane.labs.trotter.utils import next_pow_2


def vibronic_hamiltonian(states: int, modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]) -> VibronicMatrix:
    """Return a VibronicMatrix representation of a vibronic Hamiltonian"""
    _validate_input(states, modes, omegas, phis)

    ham = _momentum_fragment(states, modes, omegas)
    for i in range(states):
        ham += _position_fragment(i, states, modes, omegas, phis)

    return ham

def vibronic_fragments(states: int, modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]) -> List[VibronicMatrix]:
    """Return a list of VibronicMatrix fragments that sum to the vibronic Hamiltonian"""
    _validate_input(states, modes, omegas, phis)

    frags = [_position_fragment(i, states, modes, omegas, phis) for i in range(next_pow_2(states))]
    frags.append(_momentum_fragment(states, modes, omegas))

    return frags

def _position_fragment(i: int, states: int, modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]) -> VibronicMatrix:
    pow2 = next_pow_2(states)
    blocks = {(j, i ^ j): _realspace_sum(j, i ^ j, states, modes, omegas, phis) for j in range(pow2)}
    return VibronicMatrix(pow2, modes, blocks)

def _momentum_fragment(states: int, modes: int, omegas: np.ndarray) -> VibronicMatrix:
    pow2 = next_pow_2(states)
    term = RealspaceOperator(
        modes,
        ("P", "P"),
        Node.tensor_node(np.diag(omegas) / 2, label=("omegas", np.diag(omegas) / 2)),
    )
    word = RealspaceSum(modes, (term,))
    blocks = {(i, i): word for i in range(states)}

    return VibronicMatrix(pow2, modes, blocks)

#pylint: disable=too-many-arguments,too-many-positional-arguments
def _realspace_sum(i: int, j: int, states: int, modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]) -> RealspaceSum:
    if i > states - 1 or j > states - 1:
        return RealspaceSum.zero()

    realspace_ops = []
    for k, phi in enumerate(phis):
        op = ("Q",) * k
        realspace_op = RealspaceOperator(
            modes, op, Node.tensor_node(phi[i, j], label=(f"phis[{k}][{i}, {j}]", phis))
        )
        realspace_ops.append(realspace_op)

    if i == j:
        op = ("Q", "Q")
        coeffs = Node.tensor_node(
            np.diag(omegas) / 2, label=("omegas", np.diag(omegas) / 2)
        )
        realspace_ops.append(RealspaceOperator(modes, op, coeffs))

    return RealspaceSum(modes, realspace_ops)

def _validate_input(states: int, modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]) -> None:
    for i, phi in enumerate(phis):
        shape = (states, states) + (modes,) * i

        if phi.shape != shape:
            raise ValueError(
                f"{i}th order coefficient tensor must have shape {shape}, got shape {phi.shape}"
            )

    if omegas.shape != (modes,):
        raise TypeError(f"Omegas must have shape {(modes,)}, got shape {omegas.shape}.")
