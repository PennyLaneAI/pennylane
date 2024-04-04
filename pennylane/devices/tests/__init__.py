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
This subpackage provides integration tests for the devices with PennyLane's core
functionalities. At the moment, the tests only run on devices based on the 'qubit' model.

The tests require that ``pytest``, ``pytest-mock``, and ``flaky`` be installed.
These can be installed using ``pip``:

.. code-block:: console

    pip install pytest pytest-mock flaky

The tests can also be run on an external device from a PennyLane plugin, such as
``'qiskit.aer'``. For this, make sure you have the correct dependencies installed.

Most tests query the device's capabilities and only get executed if they apply to the device.
Both analytic devices (producing an exact probability distribution) and non-analytic devices (producing an estimated
probability distribution) are tested.

For non-analytic tests, the tolerance of the assert statements
is set to a high enough value to account for stochastic fluctuations. Flaky is used to automatically
repeat failed tests.

There are several methods for running the tests against a particular device (i.e., for
``'default.qubit'``), detailed below.

Using pytest
------------

.. code-block:: console

    pytest path_to_pennylane_src/devices/tests --device=default.qubit --shots=10000

The location of your PennyLane installation may differ depending on installation method and
operating system. To find the location, you can use the :func:`~.get_device_tests` function:

>>> from pennylane.devices.tests import get_device_tests
>>> get_device_tests()

The pl-device-test CLI
----------------------

Alternatively, PennyLane provides a command line interface for invoking the device tests.

.. code-block:: console

    pl-device-test --device default.qubit --shots 10000

Within Python
-------------

Finally, the tests can be invoked within a Python session via the :func:`~.test_device`
function:

>>> from pennylane.devices.tests import test_device
>>> test_device("default.qubit.legacy")

For more details on the available arguments, see the :func:`~.test_device` documentation.

Functions
---------
"""
# pylint: disable=import-outside-toplevel,too-many-arguments
import argparse
import pathlib
import subprocess
import sys


# determine if running in an interactive environment
import __main__

interactive = False

try:
    __main__.__file__
except AttributeError:
    interactive = True


def get_device_tests():
    """Returns the location of the device integration tests."""
    return str(pathlib.Path(__file__).parent.absolute())


def test_device(
    device_name,
    shots=0,
    skip_ops=True,
    flaky_report=False,
    pytest_args=None,
    **kwargs,
):
    """Run the device integration tests using an installed PennyLane device.

    Args:
        device_name (str): the name of the device to test
        shots (int or None): The number of shots/samples used to estimate
            expectation values and probability. If ``shots=None``, then the
            device is run in analytic mode (where expectation values and
            probabilities are computed exactly from the quantum state).
            If not provided, the device default is used.
        skip_ops (bool): whether to skip tests that use operations not supported
            by the device
        pytest_args (list[str]): additional PyTest arguments and flags
        **kwargs: Additional device keyword args

    **Example**

    >>> from pennylane.devices.tests import test_device
    >>> test_device("default.qubit.legacy")
    ================================ test session starts =======================================
    platform linux -- Python 3.7.7, pytest-5.4.2, py-1.8.1, pluggy-0.13.1
    rootdir: /home/josh/xanadu/pennylane/pennylane/devices/tests, inifile: pytest.ini
    devices: flaky-3.6.1, cov-2.8.1, mock-3.1.0
    collected 86 items
    xanadu/pennylane/pennylane/devices/tests/test_gates.py ..............................
    ...............................                                                       [ 70%]
    xanadu/pennylane/pennylane/devices/tests/test_measurements.py .......sss...sss..sss   [ 95%]
    xanadu/pennylane/pennylane/devices/tests/test_properties.py ....                      [100%]
    ================================= 77 passed, 9 skipped in 0.78s ============================

    """
    try:
        import pytest  # pylint: disable=unused-import
        import pytest_mock  # pylint: disable=unused-import
        import flaky  # pylint: disable=unused-import
    except ImportError as e:
        raise ImportError(
            "The device tests requires the following Python packages:"
            "\npytest pytest_mock flaky"
            "\nThese can be installed using pip."
        ) from e

    pytest_args = pytest_args or []
    test_dir = get_device_tests()

    cmds = ["pytest"]
    cmds.append(test_dir)
    cmds.append(f"--device={device_name}")

    # Note: None is a valid setting for shots
    if shots != 0:
        cmds.append(f"--shots={shots}")

    if skip_ops:
        cmds.append("--skip-ops")

    if not flaky_report:
        cmds.append("--no-flaky-report")

    if kwargs:
        device_kwargs = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        cmds += ["--device-kwargs", device_kwargs]

    try:
        subprocess.run(cmds + pytest_args, check=not interactive)
    except subprocess.CalledProcessError as e:
        # pytest return codes:
        #   Exit code 0:    All tests were collected and passed successfully
        #   Exit code 1:    Tests were collected and run but some of the tests failed
        #   Exit code 2:    Test execution was interrupted by the user
        #   Exit code 3:    Internal error happened while executing tests
        #   Exit code 4:    pytest command line usage error
        #   Exit code 5:    No tests were collected
        if e.returncode in range(1, 6):
            # If a known pytest error code is returned, exit gracefully without
            # an error message to avoid the user seeing duplicated tracebacks
            sys.exit(1)

        # otherwise raise the exception
        raise e


def cli():
    """The PennyLane device test command line interface.

    The ``pl-device-test`` CLI is a convenience wrapper that calls
    pytest for a particular device.

    .. code-block:: console

        $ pl-device-test --help
        usage: pl-device-test [-h] [--device DEVICE] [--shots SHOTS]
                              [--skip-ops]

        See below for available options and commands for working with the PennyLane
        device tests.

        General Options:
          -h, --help           show this help message and exit
          --device DEVICE      The device to test.
          --shots SHOTS        Number of shots to use in stochastic mode.
          --skip-ops           Skip tests that use unsupported device operations.
          --flaky-report       Show the flaky report in the terminal
          --device-kwargs KEY=VAL [KEY=VAL ...]
                               Additional device kwargs.

    Note that additional pytest command line arguments and flags can also be passed:

    .. code-block:: console

        $ pl-device-test --device default.qubit --shots 1234 --tb=short -x
    """
    from .conftest import pytest_addoption

    parser = argparse.ArgumentParser(
        description="See below for available options and commands for working with the PennyLane device tests."
    )
    parser._optionals.title = "General Options"  # pylint: disable=protected-access
    pytest_addoption(parser)
    args, pytest_args = parser.parse_known_args()

    flaky = False
    if "--flaky-report" in pytest_args:
        pytest_args.remove("--flaky-report")
        flaky = True

    test_device(
        device_name=args.device,
        shots=args.shots,
        skip_ops=args.skip_ops,
        flaky_report=flaky,
        pytest_args=pytest_args,
        **args.device_kwargs,
    )
