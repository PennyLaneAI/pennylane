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


def test_seed():
    """Test that DefaultQutritMixed has a seed and _prng_key is None"""
    # pylint: disable=protected-access
    dev = DefaultQutritMixed()
    assert hasattr(dev, "_rng")
    assert hasattr(dev, "_prng_key")
    assert dev._prng_key is None


def test_seed_jax():
    """Test that DefaultQutritMixed has a _prng_key and _rng is None"""
    # pylint: disable=protected-access
    dev = DefaultQutritMixed()

    assert hasattr(dev, "_rng")
    assert hasattr(dev, "_prng_key")
    assert dev._debugger is not None
