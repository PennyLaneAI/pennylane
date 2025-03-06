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
Tests for the Fourier module helper functions.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.fourier.utils import format_nvec, get_spectrum, join_spectra


@pytest.mark.parametrize(
    "nvec, exp",
    [(1, "1"), (-20, "-20"), ((23,), " 23"), ((-1,), "-1"), ((2, -1, 42), " 2 -1  42")],
)
def test_format_nvec(nvec, exp):
    """Test formatting of a tuple of integers into a nice string."""
    assert format_nvec(nvec) == exp


@pytest.mark.parametrize(
    "spectrum1, spectrum2, expected",
    [
        ({0, 1}, {0, 1}, {0, 1, 2}),
        ({0, 3}, {0, 5}, {0, 2, 3, 5, 8}),
        ({0, 1, 2}, {0, 1}, {0, 1, 2, 3}),
        ({0, 0.5}, {0, 1}, {0, 0.5, 1.0, 1.5}),
        ({0, 0.5}, {0}, {0, 0.5}),
        ({0}, {0, 0.5}, {0, 0.5}),
    ],
)
def test_join_spectra(spectrum1, spectrum2, expected):
    """Test that spectra are joined correctly."""
    joined = join_spectra(spectrum1, spectrum2)
    assert joined == expected


@pytest.mark.parametrize(
    "op, expected",
    [
        (qml.RX(0.1, wires=0), [0, 1]),  # generator is a class
        (qml.RY(0.1, wires=0), [0, 1]),  # generator is a class
        (qml.RZ(0.1, wires=0), [0, 1]),  # generator is a class
        (qml.PhaseShift(0.5, wires=0), [0, 1]),  # generator is an array
        (qml.CRX(0.2, wires=[0, 1]), [0, 0.5, 1]),  # generator is an array
        (qml.ControlledPhaseShift(0.5, wires=[0, 1]), [0, 1]),  # generator is an array
    ],
)
def test_get_spectrum(op, expected):
    """Test that the spectrum is correctly extracted from an operator."""
    spec = get_spectrum(op, decimals=10)
    assert np.allclose(sorted(spec), expected, atol=1e-6, rtol=0)


def test_get_spectrum_complains_no_generator():
    """Test that an error is raised if the operator has no generator defined."""

    # pylint: disable=too-few-public-methods
    class CustomOp(qml.operation.Operation):
        num_wires = 1
        num_params = 1

    with pytest.raises(qml.operation.GeneratorUndefinedError, match="does not have a generator"):
        get_spectrum(CustomOp(0.5, wires=0), decimals=10)
