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
import argparse
import os

import numpy as np
import pytest
from _pytest.runner import pytest_runtest_makereport as orig_pytest_runtest_makereport

import pennylane as qml

# ==========================================================
# pytest fixtures

# seed for random functions
np.random.seed(42)
# Tolerance for analytic tests
TOL = 1e-6
# Tolerance for non-analytic tests
TOL_STOCHASTIC = 0.05
# Number of shots to call the devices with
N_SHOTS = 10000
# List of all devices that are included in PennyLane
LIST_CORE_DEVICES = {"default.qubit", "default.qubit.tf", "default.qubit.autograd"}
# TODO: add beta devices "default.tensor", "default.tensor.tf", which currently
# do not have an "analytic" attribute.


@pytest.fixture(scope="function")
def tol():
    """Numerical tolerance for equality tests. Returns a different tolerance for tests
    probing analytic or non-analytic devices (yielding deterministic or stochastic test results)."""

    def _tol(analytic):
        if analytic:
            return float(os.environ.get("TOL", TOL))
        return TOL_STOCHASTIC

    return _tol


@pytest.fixture(scope="session")
def init_state():
    """Fixture to create an n-qubit random initial state vector."""

    def _init_state(n):
        state = np.random.random([2 ** n]) + np.random.random([2 ** n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


@pytest.fixture(scope="session")
def skip_if():
    """Fixture to skip tests."""

    def _skip_if(dev, capabilities):
        """ Skip test if device has any of the given capabilities. """

        dev_capabilities = dev.capabilities()
        for capability, value in capabilities.items():
            # skip if capability not found, or if capability has specific value
            if capability not in dev_capabilities or dev_capabilities[capability] == value:
                pytest.skip(
                    "Test skipped for {} device with capability {}:{}.".format(
                        dev.name, capability, value
                    )
                )

    return _skip_if


@pytest.fixture(scope="function")
def device(device_kwargs):
    """Fixture to create a device."""

    __tracebackhide__ = True

    def _device(wires):
        device_kwargs["wires"] = wires

        try:
            dev = qml.device(**device_kwargs)
        except qml.DeviceError:
            # exit the tests if the device cannot be created
            pytest.exit(
                "Device {} cannot be created. To run the device tests on an external device, the "
                "plugin and all of its dependencies must be installed.".format(
                    device_kwargs["name"]
                )
            )

        capabilities = dev.capabilities()
        if not capabilities.get("supports_qubit", False):
            # exit the tests if device based on cv model (currently not supported)
            pytest.exit("The device test suite currently only runs on qubit-based devices.")

        return dev

    return _device


def pytest_runtest_setup(item):
    """Skip tests marked as broken."""

    # skip tests marked as broken
    for mark in item.iter_markers(name="broken"):
        if mark.args:
            pytest.skip("Broken test skipped: {}".format(*mark.args))
        else:
            pytest.skip("Test skipped as corresponding code base is currently broken!")


# ============================
# These functions are required to define the device name to run the tests for


class StoreDictKeyPair(argparse.Action):
    """Argparse action for storing key-value pairs as a dictionary.

    For example, calling a CLI program with ``--mydict v1=k1 v2=5``:

    >>> parser.add_argument("--mydict", dest="my_dict", action=StoreDictKeyPair, nargs="+")
    >>> args = parser.parse()
    >>> args.my_dict
    {"v1": "k1", "v2": "5"}

    Note that all keys will be strings.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def pytest_addoption(parser):
    """Add command line option to pytest."""

    if hasattr(parser, "add_argument"):
        # parser is a argparse.Parser object
        addoption = parser.add_argument
    else:
        # parser is a pytest.config.Parser object
        addoption = parser.addoption

    # The options are the three arguments every device takes
    addoption("--device", action="store", default=None, help="The device to test.")
    addoption(
        "--shots",
        action="store",
        default=None,
        type=int,
        help="Number of shots to use in stochastic mode.",
    )
    addoption(
        "--analytic",
        action="store",
        default=None,
        help="Whether to run the tests in stochastic or exact mode.",
    )
    addoption(
        "--skip-ops",
        action="store_true",
        default=False,
        help="Skip tests that use unsupported device operations.",
    )

    addoption(
        "--device-kwargs",
        dest="device_kwargs",
        action=StoreDictKeyPair,
        default={},
        nargs="+",
        metavar="KEY=VAL",
        help="Additional device kwargs.",
    )


def pytest_generate_tests(metafunc):
    """Set up fixtures from command line options. """
    opt = metafunc.config.option

    device_kwargs = {
        "name": opt.device,
        "shots": opt.shots,
        "analytic": opt.analytic,
        **opt.device_kwargs,
    }

    # ===========================================
    # some processing of the command line options

    if device_kwargs["shots"] is None:
        # use default value of device
        device_kwargs.pop("shots")
    if device_kwargs["analytic"] is None:
        # use default value of device
        device_kwargs.pop("analytic")
    else:
        if device_kwargs["analytic"].lower() == "false":
            # turn string into boolean
            device_kwargs["analytic"] = False
        else:
            device_kwargs["analytic"] = True

    # ==============================================

    # parametrize all functions that take device_kwargs as an argument
    # this is needed for the "device" fixture
    if "device_kwargs" in metafunc.fixturenames:
        if opt.device is None:
            # if no command line argument for device given, run tests on core devices
            list_device_kwargs = []
            for dev_name in LIST_CORE_DEVICES:
                core_dev_kwargs = device_kwargs.copy()
                core_dev_kwargs["name"] = dev_name
                list_device_kwargs.append(core_dev_kwargs)
            metafunc.parametrize("device_kwargs", list_device_kwargs)

        else:
            # run tests on specified device
            metafunc.parametrize("device_kwargs", [device_kwargs])


def pytest_runtest_makereport(item, call):
    """Post-processing test reports to exclude those known to be failing."""
    tr = orig_pytest_runtest_makereport(item, call)

    if "skip_unsupported" in item.keywords and item.config.option.skip_ops:
        if call.excinfo is not None:

            # Exclude failing test cases for unsupported operations/observables
            # and those using not implemented features
            if (
                call.excinfo.type == qml.DeviceError
                and "supported" in str(call.excinfo.value)
                or call.excinfo.type == NotImplementedError
            ):
                tr.wasxfail = "reason:" + str(call.excinfo.value)
                tr.outcome = "skipped"

    return tr
