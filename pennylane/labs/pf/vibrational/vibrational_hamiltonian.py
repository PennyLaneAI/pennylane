"""The realspace vibrational Hamiltonian"""

from typing import List, Sequence

import numpy as np

from pennylane.labs.pf.realspace import Node, RealspaceOperator, RealspaceSum


class VibrationalHamiltonian:
    """Class representing the realspace Vibrational Hamiltonian."""

    def __init__(self, modes: int, order: int, omegas: np.ndarray, phis: Sequence[np.ndarray]):
        if not len(omegas) == modes:
            raise ValueError(f"Expected omegas to be of length {modes}, got {len(omegas)}.")

        if not len(phis) == order - 2:
            raise ValueError(f"Expected phis to be of length {order + 1}, got {len(phis)}.")

        for i, phi in enumerate(phis):
            shape = (modes,) * (i + 3)

            if not phi.shape == shape:
                raise ValueError(f"Expected order {i+3} to be of shape {shape}, got {phi.shape}.")

        self.modes = modes
        self.order = order
        self.omegas = omegas
        self.phis = phis

    def fragments(self) -> List[RealspaceSum]:
        """Retrun a list of fragments"""
        return [self.harmonic_fragment(), self.anharmonic_fragment()]

    def harmonic_fragment(self) -> RealspaceSum:
        """Returns the fragment of the Hamiltonian corresponding to the harmonic part."""
        coeffs = Node.tensor_node(self.omegas / 2, label="omegas")
        position = RealspaceOperator(("PP",), coeffs)
        momentum = RealspaceOperator(("QQ",), coeffs)

        return RealspaceSum([position, momentum])

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
