# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the math module init methods."""
import pennylane as qml


class TestNumpyMimicForFFT:
    """Test that the fft module is accessible via Autoray."""

    def test_find_fft_module_and_funcs(self):
        """Test that the FFT module can be accessed and contains functions."""
        fft_module = qml.math.fft
        for fn in ["fft", "ifft", "fft2", "ifft2"]:
            assert hasattr(fft_module, fn)

    def test_find_other_module_and_funcs(self):
        """Test that a module other than FFT can be accessed and contains functions."""
        linalg_module = qml.math.linalg
        for fn in ["expm", "eigvals"]:
            assert hasattr(linalg_module, fn)
