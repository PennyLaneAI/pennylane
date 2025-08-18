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
import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "data, dtype, exp_output",
    [
        [0.4, "float", True],
        [0.4, "complex", True],
        [0.4 + 0.2j, "complex", False],
        [0.4 + 1e-14j, "complex", True],
    ],
)
class TestIsRealObjOrClose:
    """Test that is_real_obj_or_close functions correctly."""

    def test_numpy(self, data, dtype, exp_output):
        """Test with numpy."""
        x = np.array(data, dtype=dtype)
        assert qml.math.is_real_obj_or_close(x) is exp_output

    @pytest.mark.autograd
    def test_autograd(self, data, dtype, exp_output):
        """Test with Autograd."""
        from pennylane import numpy as pnp

        x = pnp.array(data, dtype=dtype)
        assert qml.math.is_real_obj_or_close(x) is exp_output

    @pytest.mark.tf
    def test_tf(self, data, dtype, exp_output):
        """Test with TensorFlow."""
        import tensorflow as tf

        dtype = tf.float64 if dtype == "float" else tf.complex128
        x = tf.Variable(data, dtype=dtype)
        assert qml.math.is_real_obj_or_close(x) is exp_output

    @pytest.mark.torch
    def test_torch(self, data, dtype, exp_output):
        """Test with Torch."""
        import torch

        dtype = torch.float64 if dtype == "float" else torch.complex128
        x = torch.tensor(data, dtype=dtype)
        assert qml.math.is_real_obj_or_close(x) is exp_output

    @pytest.mark.jax
    def test_jax(self, data, dtype, exp_output):
        """Test with JAX."""
        import jax

        x = jax.numpy.array(data, dtype=dtype)
        assert qml.math.is_real_obj_or_close(x) is exp_output
