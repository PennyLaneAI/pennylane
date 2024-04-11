# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qutrit mixed."""
import pytest
import numpy as np

import pennylane as qml
from pennylane.devices.default_qutrit_mixed import DefaultQutritMixed


def test_name():
    """Tests the name of DefaultQutritMixed."""
    assert DefaultQutritMixed().name == "default.qutrit.mixed"


def test_debugger_attribute():
    """Test that DefaultQutritMixed has a debugger attribute and that it is `None`"""
    # pylint: disable=protected-access
    dev = DefaultQutritMixed()

    assert hasattr(dev, "_debugger")
    assert dev._debugger is None


class TestRandomSeed:
    """Test that the device behaves correctly when provided with a random seed"""

    def test_global_seed_no_device_seed_by_default(self):
        """Test that the global numpy seed initializes the rng if device seed is none."""
        np.random.seed(42)
        dev = DefaultQutritMixed()
        first_num = dev._rng.random()  # pylint: disable=protected-access

        np.random.seed(42)
        dev2 = DefaultQutritMixed()
        second_num = dev2._rng.random()  # pylint: disable=protected-access

        assert qml.math.allclose(first_num, second_num)

        np.random.seed(42)
        dev2 = DefaultQutritMixed(seed="global")
        third_num = dev2._rng.random()  # pylint: disable=protected-access

        assert qml.math.allclose(third_num, first_num)

    def test_None_seed_not_using_global_rng(self):
        """Test that if the seed is None, it is uncorrelated with the global rng."""
        np.random.seed(42)
        dev = DefaultQutritMixed(seed=None)
        first_nums = dev._rng.random(10)  # pylint: disable=protected-access

        np.random.seed(42)
        dev2 = DefaultQutritMixed(seed=None)
        second_nums = dev2._rng.random(10)  # pylint: disable=protected-access

        assert not qml.math.allclose(first_nums, second_nums)

    def test_rng_as_seed(self):
        """Test that a PRNG can be passed as a seed."""
        rng1 = np.random.default_rng(42)
        first_num = rng1.random()

        rng = np.random.default_rng(42)
        dev = DefaultQutritMixed(seed=rng)
        second_num = dev._rng.random()  # pylint: disable=protected-access

        assert qml.math.allclose(first_num, second_num)


@pytest.mark.jax
class TestPRNGKeySeed:
    """Test that the device behaves correctly when provided with a PRNG key and using the JAX interface"""

    # pylint: disable=too-few-public-methods

    def test_prng_key_as_seed(self):
        """Test that a jax PRNG can be passed as a seed."""
        from jax import random

        key1 = random.key(123)
        first_nums = random.uniform(key1, shape=(10,))

        key = random.key(123)
        dev = DefaultQutritMixed(seed=key)

        second_nums = random.uniform(dev._prng_key, shape=(10,))  # pylint: disable=protected-access
        assert np.all(first_nums == second_nums)
