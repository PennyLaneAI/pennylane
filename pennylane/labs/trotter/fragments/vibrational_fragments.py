"""The realspace vibrational Hamiltonian"""

from typing import List, Sequence

import numpy as np

from pennylane.labs.trotter.realspace import Node, RealspaceOperator, RealspaceSum

def vibrational_hamiltonian(modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]) -> RealspaceSum:
    """Return a RealspaceSum representing the vibrational Hamiltonian."""
    _validate_input(modes, omegas, phis)

    return harmonic_fragment(modes, omegas) + anharmonic_fragment(modes, phis)

def vibrational_fragments(modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray], frags="harmonic") -> List[RealspaceSum]:
    """Return a list of fragments"""

    if frags == "harmonic":
        return [harmonic_fragment(modes, omegas), anharmonic_fragment(modes, phis)]

    if frags == "kinetic":
        return [kinetic_fragment(modes, omegas), potential_fragment(modes, omegas, phis)]

    if frags == "position":
        return [position_fragment(modes, omegas, phis), momentum_fragment(modes, omegas)]

    raise ValueError(f"{frags} is not a valid fragmentation scheme.")

def harmonic_fragment(modes: int, omegas: np.ndarray) -> RealspaceSum:
    """Returns the fragment of the Hamiltonian corresponding to the harmonic part."""
    _validate_omegas(modes, omegas)

    coeffs = Node.tensor_node(omegas / 2, label=("omegas", omegas / 2))
    momentum = RealspaceOperator(("PP",), coeffs)
    position = RealspaceOperator(("QQ",), coeffs)

    return RealspaceSum([momentum, position])

def anharmonic_fragment(modes: int, phis: Sequence[np.ndarray]) -> RealspaceSum:
    """Returns the fragment of the Hamiltonian corresponding to the anharmonic part."""
    _validate_phis(modes, phis)

    ops = []
    for i, phi in enumerate(phis):
        op = ("Q",) * i
        coeffs = Node.tensor_node(phi, label=(f"phis[{i}]", phis))
        realspace_op = RealspaceOperator(op, coeffs)
        ops.append(realspace_op)

    return RealspaceSum(ops)

def kinetic_fragment(modes: int, omegas: np.ndarray) -> RealspaceSum:
    """Returns the fragment of the Hamiltonian corresponding to the kinetic part"""
    _validate_omegas(modes, omegas)

    coeffs = Node.tensor_node(omegas / 2, label=("omegas", omegas / 2))
    kinetic = RealspaceOperator(("PP",), coeffs)

    return RealspaceSum([kinetic])

def potential_fragment(modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]) -> RealspaceSum:
    """Returns the fragment of the Hamiltonian corresponding to the potential part"""
    _validate_input(modes, omegas, phis)

    ops = []
    for i, phi in enumerate(phis):
        op = ("Q",) * i
        coeffs = Node.tensor_node(phi, label=(f"phis[{i}]", phis))
        realspace_op = RealspaceOperator(op, coeffs)
        ops.append(realspace_op)

    op = ("Q", "Q")
    diag = np.diag(omegas / 2)
    coeffs = Node.tensor_node(diag, label=("omegas", diag))
    realspace_op = RealspaceOperator(op, coeffs)
    ops.append(realspace_op)

    return RealspaceSum(ops)

def position_fragment(modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]) -> RealspaceSum:
    """Return the position term of the Hamiltonian"""
    coeffs = Node.tensor_node(omegas / 2, label=("omegas", omegas / 2))
    position = RealspaceOperator(("QQ",), coeffs)

    return RealspaceSum([position]) + anharmonic_fragment(modes, phis)

def momentum_fragment(modes: int, omegas: np.ndarray) -> RealspaceSum:
    """Return the momentum term of the Hamiltonian"""
    _validate_omegas(modes, omegas)

    coeffs = Node.tensor_node(omegas / 2, label=("omegas", omegas / 2))
    momentum = RealspaceOperator(("PP",), coeffs)

    return RealspaceSum([momentum])

def _validate_phis(modes: int, phis: Sequence[np.ndarray]) -> None:
    for i, phi in enumerate(phis):
        shape = (modes,) * i

        if not phi.shape == shape:
            raise ValueError(f"Expected order {i} to be of shape {shape}, got {phi.shape}.")

def _validate_omegas(modes: int, omegas: np.ndarray) -> None:
    if not len(omegas) == modes:
        raise ValueError(f"Expected omegas to be of length {modes}, got {len(omegas)}.")

def _validate_input(modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]) -> None:
    _validate_phis(modes, phis)
    _validate_omegas(modes, omegas)
