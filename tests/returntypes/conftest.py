# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Pytest configuration file for return types test suite.
"""

import pytest
import pennylane as qml

try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError) as e:
    tf_available = False
else:
    tf_available = True

try:
    import torch
    from torch.autograd import Variable

    torch_available = True
except ImportError as e:
    torch_available = False

try:
    import jax
    import jax.numpy as jnp

    jax_available = True
except ImportError as e:
    jax_available = False


def pytest_sessionstart(session):
    qml.enable_return()


def pytest_sessionfinish(session, exitstatus):
    qml.disable_return()


def pytest_runtest_setup(item):
    """Automatically skip tests if interfaces are not installed"""
    # Autograd is assumed to be installed
    interfaces = {"tf_r", "torch_r", "jax_r"}
    available_interfaces = {
        "tf_r": tf_available,
        "torch_r": torch_available,
        "jax_r": jax_available,
    }

    allowed_interfaces = [
        allowed_interface
        for allowed_interface in interfaces
        if available_interfaces[allowed_interface] is True
    ]

    # load the marker specifying what the interface is
    all_interfaces = {"tf_r", "torch_r", "jax_r", "all_interfaces_r"}
    marks = {mark.name for mark in item.iter_markers() if mark.name in all_interfaces}

    for b in marks:
        if b == "all_interfaces_r":
            required_interfaces = {"tf_r", "torch_r", "jax_r"}
            for interface in required_interfaces:
                if interface not in allowed_interfaces:
                    pytest.skip(
                        f"\nTest {item.nodeid} only runs with {allowed_interfaces} interfaces(s) but {b} interface provided",
                    )
        else:
            if b not in allowed_interfaces:
                pytest.skip(
                    f"\nTest {item.nodeid} only runs with {allowed_interfaces} interfaces(s) but {b} interface provided",
                )
