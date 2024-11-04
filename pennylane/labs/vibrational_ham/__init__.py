"""
This submodule provides the functionality to calculate vibrational Hamiltonians.
"""

from .vibrational_class import single_point, harmonic_analysis, optimize_geometry, VibrationalPES

from .localize_modes import localize_normal_modes

from .dipole import get_dipole

from .pes_generator import pes_onemode, pes_twomode, pes_threemode, vibrational_pes

from .taylorForm import (
    taylor_integrals,
    taylor_integrals_dipole,
    taylor_bosonic,
    taylor_anharmonic,
)

from .christiansenForm import christiansen_integrals, christiansen_integrals_dipole

from .christiansen_ops import christiansen_mapping, christiansen_bosonic, christiansen_hamiltonian

from .bosonic import BoseWord, BoseSentence

from .bosonic_mapping import binary_mapping
