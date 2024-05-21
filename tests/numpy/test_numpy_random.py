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
Tests for ``pennylane.numpy.random`` wrapping.  Arrays generated should have an
additional property, ``requires_grad``, that marks them as trainable/ non-trainable.
"""
# pylint: disable=too-few-public-methods

import pytest

from pennylane import numpy as np
from pennylane.numpy import random

# distributions that require no extra parameters
distributions_no_extra_input = [
    "exponential",
    "gumbel",
    "laplace",
    "logistic",
    "lognormal",
    "normal",
    "poisson",
    "rayleigh",
    "standard_cauchy",
    "standard_exponential",
    "standard_normal",
    "uniform",
]

bit_generator_classes = [random.MT19937, random.PCG64, random.Philox, random.SFC64]

general_gen = random.default_rng()


@pytest.mark.unit
class TestGeneratorDistributions:
    @pytest.mark.parametrize("distribution", distributions_no_extra_input)
    def test_generator_distributions(self, distribution):
        """Tests the distributions from a generator object that don't require
        additional distribution-specific parameters to make sure they have a
        ``requires_grad`` attribute."""
        size = (3,)
        output = getattr(general_gen, distribution)(size=size)

        assert isinstance(output, np.tensor)

        assert output.shape == size
        assert output.requires_grad is True

        output.requires_grad = False
        assert output.requires_grad is False


@pytest.mark.unit
class Test_default_rng:
    def test_no_input(self):
        """Tests that np.random.default_rng() returns a generator object when
        no input is passed."""
        rng = random.default_rng()

        assert isinstance(rng, random.Generator)

        output = rng.random((3,))
        assert isinstance(output, np.tensor)

    @pytest.mark.parametrize("bitgen_cls", bit_generator_classes)
    def test_bit_generators(self, bitgen_cls):
        """Tests that np.random.default_rng() returns a generator when passed a bit
        generator.  Also checks that the bit generators are imported into the namespace."""

        bitgen = bitgen_cls()

        rng = random.default_rng(bitgen)

        assert isinstance(rng, random.Generator)
        assert isinstance(rng.bit_generator, bitgen_cls)

        output = rng.random((3,))
        assert isinstance(output, np.tensor)

    def test_generator_input(self):
        """Tests that ``np.random.default_rng`` passes through a Generator when its passed as input."""

        rng1 = random.default_rng()
        rng2 = random.default_rng(rng1)

        assert rng1 == rng2

    def test_seed_reproducible(self):
        """Tests that setting a seed to ``default_rng`` gives reproducible results."""

        seed = 42
        size = (3, 2)

        rng1 = random.default_rng(seed)
        rng2 = random.default_rng(seed)

        assert isinstance(rng1, random.Generator)
        assert isinstance(rng2, random.Generator)

        mat1 = rng1.random(size=size)
        mat2 = rng2.random(size=size)

        assert np.all(mat1 == mat2)

        mat1_2 = rng1.normal(size=size)
        mat2_2 = rng2.normal(size=size)

        assert np.all(mat1_2 == mat2_2)
