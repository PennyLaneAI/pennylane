# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains shared fixtures for the device tests."""
import numpy as np
import pytest
import os

# ==========================================================
# pytest fixtures

np.random.seed(42)

TOL = 1e-3

# List of all devices that are included in PennyLane
# by default
LIST_CORE_DEVICES = {'default.gaussian', 'default.qubit', 'default.qubit.tf'}


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session")
def init_state():
    """Fixture to create an n-qubit initial state vector."""
    def _init_state(n):
        state = np.random.random([2 ** n]) + np.random.random([2 ** n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return


# Fixture to skip tests
@pytest.fixture(scope="session")
def skip_if():
    """Fixture to skip tests."""
    def _skip_if(condition):
        if condition:
            pytest.skip("Test does not apply to this device.")
    return _skip_if()

# ============================
# These functions are required to define the device name to run the tests for


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="device to run tests for")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".

    option_value = metafunc.config.option.device
    if 'device' in metafunc.fixturenames and option_value is not None:
        # if command line argument given, run tests on the requested device
        metafunc.parametrize("device_name", [option_value])
    else:
        # if no command line argument given, run tests on core devices
        metafunc.parametrize("device_name", LIST_CORE_DEVICES)
