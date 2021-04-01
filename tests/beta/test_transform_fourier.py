# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the fourier qnode transforms.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane.beta.transforms.fourier import spectrum, _join_spectra, _get_spectrum, _simplify_tape


class TestUnits:

    @pytest.mark.parametrize("spectrum1, spectrum2, expected", [([-1, 0, 1], [-1, 0, 1], [-2, -1, 0, 1, 2]),
                                                                ([-3, 0, 3], [-5, 0, 5],
                                                                 [-8, -5, -3, -2, 0, 2, 3, 5, 8]),
                                                                ([-2, -1, 0, 1, 2], [-1, 0, 1],
                                                                 [-3, -2, -1, 0, 1, 2, 3]),
                                                                ([-0.5, 0, 0.5], [-1, 0, 1],
                                                                 [-1.5, -1, -0.5, 0, 0.5, 1., 1.5])
                                                                ])
    def test_join_spectra(self, spectrum1, spectrum2, expected, tol):
        """Test that spectra are joined correctly."""
        joined = _join_spectra(spectrum1, spectrum2)
        assert np.allclose(joined, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op, expected", [(qml.RX(0.1, wires=0), [-1, 0,  1]),
                                              (qml.RY(0.1, wires=0), [-1, 0, 1]),
                                              (qml.RZ(0.1, wires=0), [-1, 0, 1]),
                                              (qml.PhaseShift(0.5, wires=0), [0, 1, 2]),
                                              (qml.ControlledPhaseShift(0.5, wires=[0, 1]), [0, 1, 2])])
    def test_get_spectrum(self, op, expected, tol):
        """Test that the spectrum is correctly extracted from an operator."""
        spec = _get_spectrum(op)
        assert np.allclose(spec, expected, atol=tol, rtol=0)

    def test_get_spectrum_complains_wrong_op(self):
        """Test that an error is raised if the operator has no generator defined."""
        op = qml.Rot(0.1, 0.1, 0.1, wires=0)
        with pytest.raises(ValueError, match="no generator defined"):
            _get_spectrum(op)


class TestSimplifyTape:

    def test_already_simplified_tape(self):

        x = [0.1, 0.2, 0.3]

        with qml.tape.QuantumTape() as tape_already_simple:
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=2)
            qml.PhaseShift(x[2], wires=0)

        new_tape = _simplify_tape(tape_already_simple, original_inputs=x)
        assert new_tape is tape_already_simple



