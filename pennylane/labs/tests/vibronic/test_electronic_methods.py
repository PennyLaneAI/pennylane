# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for electronic structure methods in the vibronic module."""

import numpy as np
import pytest

try:
    from pyscf import scf
    from pyscf.dft import rks, uks

except ImportError:
    pytest.skip("Skipped, no pyscf support")

from pennylane.labs.vibronic.electronic_methods import (
    _run_eom_ccsd,
    _run_scf,
    _run_tddft,
    _setup_molecule,
    run_electronic_method,
)


class TestElectronicMethods:
    """Tests for electronic structure methods."""

    @pytest.mark.parametrize(
        "symbols, coords, basis, spin, point_group",
        [
            (["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]), "sto-3g", 0, None),
            (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]]), "6-31g", 0, "c2v"),
        ],
    )
    def test_setup_molecule(self, symbols, coords, basis, spin, point_group):
        """Test that _setup_molecule creates a valid PySCF Mole object."""
        mol = _setup_molecule(symbols, coords, basis, spin, point_group)

        assert mol.natm == len(symbols)
        assert mol.basis == basis
        assert mol.spin == spin
        if point_group:
            assert mol.symmetry.lower() == point_group.lower()

    @pytest.mark.parametrize(
        "symbols, coords, method_type, functional",
        [
            (["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]), "hf", None),
            (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]]), "dft", "b3lyp"),
        ],
    )
    def test_run_scf(self, symbols, coords, method_type, functional):
        """Test that _run_scf produces a converged SCF object."""
        mol = _setup_molecule(symbols, coords, basis="sto-3g")

        kwargs = {}
        if functional:
            kwargs["functional"] = functional

        mf = _run_scf(mol, method_type=method_type, **kwargs)

        assert mf.converged
        assert hasattr(mf, "e_tot")

        # Check that the correct SCF type was created
        if method_type == "dft":
            if mol.spin == 0:
                assert isinstance(mf, rks.RKS)
            else:
                assert isinstance(mf, uks.UKS)
        else:
            if mol.spin == 0:
                assert isinstance(mf, scf.hf.RHF)
            else:
                assert isinstance(mf, scf.hf.UHF)

    @pytest.mark.parametrize(
        "symbols, coords, functional, nroots",
        [
            (["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]), "b3lyp", 1),
            (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]]), "cam-b3lyp", 2),
        ],
    )
    def test_run_tddft(self, symbols, coords, functional, nroots):
        """Test that _run_tddft returns the expected number of energies."""
        energies = _run_tddft(symbols, coords, basis="sto-3g", functional=functional, nroots=nroots)

        # Should return ground state + excited states
        assert len(energies) == nroots + 1

        # Energies should be in ascending order
        assert all(energies[i] <= energies[i + 1] for i in range(len(energies) - 1))

        # All energies should be finite
        assert all(np.isfinite(e) for e in energies)

    @pytest.mark.parametrize(
        "symbols, coords, nroots",
        [
            (["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]), 1),
            (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]]), 2),
        ],
    )
    def test_run_eom_ccsd(self, symbols, coords, nroots):
        """Test that _run_eom_ccsd returns the expected number of energies."""
        energies = _run_eom_ccsd(symbols, coords, basis="sto-3g", nroots=nroots)

        # Should return ground state + excited states
        assert len(energies) == nroots + 1

        # Energies should be in ascending order
        assert all(energies[i] <= energies[i + 1] for i in range(len(energies) - 1))

        # All energies should be finite
        assert all(np.isfinite(e) for e in energies)

    def test_tddft_error_handling(self):
        """Test that _run_tddft raises appropriate errors."""
        # Test with invalid functional
        with pytest.raises((ValueError, KeyError)):
            _run_tddft(
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                functional="invalid_functional",
                max_cycle=1,  # Force non-convergence
            )

    def test_eom_ccsd_error_handling(self):
        """Test that _run_eom_ccsd raises appropriate errors."""
        # Test with non-converging parameters
        with pytest.raises(ValueError):
            _run_eom_ccsd(
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                max_cycle=1,  # Force non-convergence
            )

    @pytest.mark.parametrize(
        "method, extra_params, expected_energies",
        [
            ("casscf", {"ncas": 2, "nelecas": 2}, [-1.1372838344885021]),
            ("tddft", {"functional": "b3lyp"}, [-1.1654184105262062, -0.2125685780341241]),
            ("eom_ccsd", {}, [-1.13728399861044, -0.1683524329710634]),
        ],
    )
    def test_run_electronic_method(self, method, extra_params, expected_energies):
        """Test that run_electronic_method correctly dispatches to the appropriate method."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        # Common parameters
        common_params = {"basis": "sto-3g", "nroots": 1, **extra_params}

        # Run the dispatcher
        energies = run_electronic_method(method, symbols, coords, **common_params)

        # Check that we get the expected number of energies
        assert len(energies) == len(expected_energies)

        # Check that all energies are finite
        assert all(np.isfinite(e) for e in energies)

        # Compare with expected energies using a reasonable tolerance
        assert np.allclose(
            energies, expected_energies, rtol=1e-5, atol=1e-5
        ), f"Expected {expected_energies}, got {energies}"
