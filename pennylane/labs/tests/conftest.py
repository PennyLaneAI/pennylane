# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
import argparse
import os

import numpy as np
import pytest
from _pytest.runner import pytest_runtest_makereport as orig_pytest_runtest_makereport

import pennylane as qml
from pennylane.exceptions import DeviceError

# ==========================================================
# pytest fixtures

# seed for random functions
np.random.seed(42)
# Tolerance for analytic tests
TOL = 1e-6
# Tolerance for non-analytic tests
TOL_STOCHASTIC = 0.05
# Number of shots to call the devices with
N_SHOTS = 1e6
# List of all devices that are included in PennyLane
LIST_CORE_DEVICES = {
    "default.qubit",
}


@pytest.fixture(scope="function")
def tol():
    """Numerical tolerance for equality tests. Returns a different tolerance for tests
    probing analytic or non-analytic devices, which allows us to define the
    standard for deterministic or stochastic test results dynamically."""

    def _tol(shots):
        if shots is None:
            return float(os.environ.get("TOL", TOL))
        return TOL_STOCHASTIC

    return _tol


@pytest.fixture(scope="session")
def init_state():
    """Fixture to create an n-qubit random initial state vector."""

    def _init_state(n):
        state = np.random.random([2**n]) + np.random.random([2**n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


def get_legacy_capabilities(dev):
    """Gets the capabilities dictionary of a device."""

    if isinstance(dev, qml.devices.LegacyDeviceFacade):
        return dev.target_device.capabilities()

    if isinstance(dev, qml.devices.LegacyDevice):
        return dev.capabilities()

    return {}


@pytest.fixture(scope="session")
def skip_if():
    """Fixture to skip tests."""

    def _skip_if(dev, capabilities):
        """Skip test if device has any of the given capabilities."""

        dev_capabilities = get_legacy_capabilities(dev)
        for capability, value in capabilities.items():
            # skip if capability not found, or if capability has specific value
            if capability not in dev_capabilities or dev_capabilities[capability] == value:
                pytest.skip(
                    f"Test skipped for {dev.name} device with capability {capability}:{value}."
                )

    return _skip_if


@pytest.fixture
def validate_diff_method(device, diff_method, device_kwargs):
    """Skip tests if a device does not support a diff_method"""
    if diff_method in {"parameter-shift", "hadamard"}:
        return
    if diff_method == "backprop" and device_kwargs.get("shots") is not None:
        pytest.skip(reason="test should only be run in analytic mode")
    dev = device(1)
    config = qml.devices.ExecutionConfig(gradient_method=diff_method)
    if not dev.supports_derivatives(execution_config=config):
        pytest.skip(reason="device does not support diff_method")


@pytest.fixture(scope="function", name="device")
def fixture_device(device_kwargs):
    """Fixture to create a device."""

    # internally used by pytest
    __tracebackhide__ = True  # pylint:disable=unused-variable

    def _device(wires):
        device_kwargs["wires"] = wires

        try:
            dev = qml.device(**device_kwargs)
        except DeviceError:
            dev_name = device_kwargs["name"]
            # exit the tests if the device cannot be created
            pytest.exit(
                f"Device {dev_name} cannot be created. To run the device tests on an external device, the "
                f"plugin and all of its dependencies must be installed."
            )

        return dev

    return _device


def pytest_runtest_setup(item):
    """Skip tests marked as broken."""

    # skip tests marked as broken
    for mark in item.iter_markers(name="broken"):
        if mark.args:
            pytest.skip(f"Broken test skipped: {mark.args}")
        else:
            pytest.skip("Test skipped as corresponding code base is currently broken!")


@pytest.fixture(scope="session", name="mpi4py_support")
def fixture_mpi4py_support():
    """Fixture to determine whether mpi4py is installed."""
    # pylint: disable=unused-import, import-outside-toplevel
    try:
        import mpi4py

        mpi4py_support = True
    except ModuleNotFoundError:
        mpi4py_support = False

    return mpi4py_support


@pytest.fixture()
def skip_if_no_mpi4py_support(mpi4py_support):
    """Fixture to skip a test if mpi4py is not installed."""
    if not mpi4py_support:
        pytest.skip("Skipped, mpi4py support")


@pytest.fixture(scope="session", name="pyscf_support")
def fixture_pyscf_support():
    """Fixture to determine whether pyscf is installed."""
    # pylint: disable=unused-import, import-outside-toplevel
    try:
        import pyscf

        pyscf_support = True
    except ModuleNotFoundError:
        pyscf_support = False

    return pyscf_support


@pytest.fixture()
def skip_if_no_pyscf_support(pyscf_support):
    """Fixture to skip a test if pyscf is not installed."""
    if not pyscf_support:
        pytest.skip("Skipped, pyscf support")


@pytest.fixture(scope="session", name="geometric_support")
def fixture_geometric_support():
    """Fixture to determine whether geometric is installed."""
    # pylint: disable=unused-import, import-outside-toplevel
    try:
        import geometric

        geometric_support = True
    except ModuleNotFoundError:
        geometric_support = False

    return geometric_support


@pytest.fixture()
def skip_if_no_geometric_support(geometric_support):
    """Fixture to skip a test if geometric is not installed."""
    if not geometric_support:
        pytest.skip("Skipped, geometric support")


# ============================
# These functions are required to define the device name to run the tests for


def _convert_to_int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


class StoreDictKeyPair(argparse.Action):
    """Argparse action for storing key-value pairs as a dictionary.

    For example, calling a CLI program with ``--mydict v1=k1 v2=5``:

    >>> parser.add_argument("--mydict", dest="my_dict", action=StoreDictKeyPair, nargs="+")
    >>> args = parser.parse()
    >>> args.my_dict
    {"v1": "k1", "v2": "5"}

    Note that strings will be converted to ints and floats if possible.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = _convert_to_int_or_float(v)
        setattr(namespace, self.dest, my_dict)


def pytest_runtest_makereport(item, call):
    """Post-processing test reports to exclude those known to be failing."""
    tr = orig_pytest_runtest_makereport(item, call)

    if "skip_unsupported" in item.keywords and item.config.option.skip_ops:
        if call.excinfo is not None:
            # Exclude failing test cases for unsupported operations/observables
            # and those using not implemented features
            if (
                call.excinfo.type == DeviceError
                and "supported" in str(call.excinfo.value)
                or call.excinfo.type == NotImplementedError
            ):
                tr.wasxfail = "reason:" + str(call.excinfo.value)
                tr.outcome = "skipped"

    return tr
