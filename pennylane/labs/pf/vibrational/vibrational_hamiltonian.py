"""The realspace vibrational Hamiltonian"""

from typing import List, Sequence

import numpy as np

from pennylane.labs.pf.realspace import Node, RealspaceOperator, RealspaceSum


class VibrationalHamiltonian:
    """Class representing the realspace Vibrational Hamiltonian."""

    def __init__(self, modes: int, omegas: np.ndarray, phis: Sequence[np.ndarray]):
        if not len(omegas) == modes:
            raise ValueError(f"Expected omegas to be of length {modes}, got {len(omegas)}.")

        for i, phi in enumerate(phis):
            shape = (modes,) * i

            if not phi.shape == shape:
                raise ValueError(f"Expected order {i} to be of shape {shape}, got {phi.shape}.")

        self.modes = modes
        self.omegas = omegas
        self.phis = phis

    def fragments(self) -> List[RealspaceSum]:
        """Retrun a list of fragments"""
        return [self.harmonic_fragment(), self.anharmonic_fragment()]

    def position_term(self) -> RealspaceSum:
        """Return the position term of the harmonic fragment"""
        coeffs = Node.tensor_node(self.omegas / 2, label="omegas")
        position = RealspaceOperator(("QQ",), coeffs)

        return RealspaceSum([position])

    def momentum_term(self) -> RealspaceSum:
        """Return the momentum term of the harmonic fragment"""
        coeffs = Node.tensor_node(self.omegas / 2, label="omegas")
        momentum = RealspaceOperator(("PP",), coeffs)

        return RealspaceSum([momentum])

    def harmonic_fragment(self) -> RealspaceSum:
        """Returns the fragment of the Hamiltonian corresponding to the harmonic part."""
        coeffs = Node.tensor_node(self.omegas / 2, label="omegas")
        momentum = RealspaceOperator(("PP",), coeffs)
        position = RealspaceOperator(("QQ",), coeffs)

        return RealspaceSum([momentum, position])

    def anharmonic_fragment(self) -> RealspaceSum:
        """Returns the fragment of the Hamiltonian corresponding to the anharmonic part."""
        ops = []
        for i, phi in enumerate(self.phis):
            op = ("Q",) * i
            coeffs = Node.tensor_node(phi, label=f"phis[{i}]")
            realspace_op = RealspaceOperator(op, coeffs)
            ops.append(realspace_op)

        return RealspaceSum(ops)

    def kinetic_fragment(self) -> RealspaceSum:
        """Returns the fragment of the Hamiltonian corresponding to the kinetic part"""
        coeffs = Node.tensor_node(self.omegas / 2, label="omegas")
        kinetic = RealspaceOperator(("PP",), coeffs)

        return RealspaceSum([kinetic])

    def potential_fragment(self) -> RealspaceSum:
        """Returns the fragment of the Hamiltonian corresponding to the potential part"""
        ops = []
        for i, phi in enumerate(self.phis):
            op = ("Q",) * i
            coeffs = Node.tensor_node(phi, label=f"phis[{i}]")
            realspace_op = RealspaceOperator(op, coeffs)
            ops.append(realspace_op)

        op = ("Q", "Q")
        coeffs = Node.tensor_node(np.diag(self.omegas / 2), label="omegas")
        realspace_op = RealspaceOperator(op, coeffs)
        ops.append(realspace_op)

        return RealspaceSum(ops)

    def operator(self) -> RealspaceSum:
        """Returns a RealspaceSum representing the Hamiltonian"""

        return self.harmonic_fragment() + self.anharmonic_fragment()
