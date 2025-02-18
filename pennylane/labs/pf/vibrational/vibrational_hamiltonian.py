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
            shape = (modes,) * (i + 3)

            if not phi.shape == shape:
                raise ValueError(f"Expected order {i+3} to be of shape {shape}, got {phi.shape}.")

        self.modes = modes
        self.omegas = omegas
        self.phis = phis

    def fragments(self) -> List[RealspaceSum]:
        """Retrun a list of fragments"""
        return [self.harmonic_fragment(), self.anharmonic_fragment()]

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
            op = ("Q",) * (i + 3)
            coeffs = Node.tensor_node(phi, label=f"phis[{i}]")
            realspace_op = RealspaceOperator(op, coeffs)
            ops.append(realspace_op)

        return RealspaceSum(ops)

    def operator(self) -> RealspaceSum:
        """Returns a RealspaceSum representing the Hamiltonian"""

        return self.harmonic_fragment() + self.anharmonic_fragment()
