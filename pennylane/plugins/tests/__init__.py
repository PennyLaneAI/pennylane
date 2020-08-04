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

    pytest path_to_pennylane_src/plugins/tests --device=default.qubit --shots=10000 --analytic=False

The location of your PennyLane installation may differ depending on installation method and
operating system. To find the location, you can use the :func:`~.get_device_tests` function:

>>> from pennylane.plugins.tests import get_device_tests
>>> get_device_tests()

The pl-device-test CLI
----------------------

Alternatively, PennyLane provides a command line interface for invoking the device tests.

.. code-block:: console

    pl-device-test --device default.qubit --shots 10000 --analytic False

Within Python
-------------

Finally, the tests can be invoked within a Python session via the :func:`~.test_device`
function:

>>> from pennylane.plugins.tests import test_device
>>> test_device("default.qubit")

For more details on the available arguments, see the :func:`~.test_device` documentation.

Functions
---------
"""
# pylint: disable=import-outside-toplevel
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
    device, analytic=None, shots=None, skip_ops=True, flaky_report=False, pytest_args=None
):
    """Run the device integration tests using an installed PennyLane device.

    Args:
        device (str): the name of the device to test
        analytic (bool): Whether to run the device in analytic mode (where
            expectation values and probabilities are computed exactly from the quantum state)
            or non-analytic/"stochastic" mode (where probabilities and expectation
            values are *estimated* using a finite number of shots.)
            If not provided, the device default is used.
        shots (int): The number of shots/samples used to estimate expectation
            values and probability. Only takes affect if ``analytic=False``. If not
            provided, the device default is used.
        skip_ops (bool): whether to skip tests that use operations not supported
            by the device
        pytest_args (list[str]): additional PyTest arguments and flags

    **Example**

    >>> from pennylane.plugins.tests import test_device
    >>> test_device("default.qubit")
    ================================ test session starts =======================================
    platform linux -- Python 3.7.7, pytest-5.4.2, py-1.8.1, pluggy-0.13.1
    rootdir: /home/josh/xanadu/pennylane/pennylane/plugins/tests, inifile: pytest.ini
    plugins: flaky-3.6.1, cov-2.8.1, mock-3.1.0
    collected 86 items
    xanadu/pennylane/pennylane/plugins/tests/test_gates.py ..............................
    ...............................                                                       [ 70%]
    xanadu/pennylane/pennylane/plugins/tests/test_measurements.py .......sss...sss..sss   [ 95%]
    xanadu/pennylane/pennylane/plugins/tests/test_properties.py ....                      [100%]
    ================================= 77 passed, 9 skipped in 0.78s ============================

    """
    try:
        import pytest  # pylint: disable=unused-import
        import pytest_mock  # pylint: disable=unused-import
        import flaky  # pylint: disable=unused-import
    except ImportError:
        raise ImportError(
            "The device tests requires the following Python packages:"
            "\npytest pytest-cov pytest_mock flaky"
            "\nThese can be installed using pip."
        )

    pytest_args = pytest_args or []
    test_dir = get_device_tests()

    cmds = ["pytest"]
    cmds.append(test_dir)
    cmds.append(f"--device={device}")

    if shots is not None:
        cmds.append(f"--shots={shots}")

    if analytic is not None:
        cmds.append(f"--analytic={analytic}")

    if skip_ops:
        cmds.append("--skip-ops")

    if not flaky_report:
        cmds.append("--no-flaky-report")

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
            sys.exit(1)
        raise e


def cli():
    """The PennyLane device test command line interface.

    The ``pl-device-test`` CLI is a convenience wrapper that calls
    pytest for a particular device.

    .. code-block:: console

        $ pl-device-test --help
        usage: pl-device-test [-h] [--device DEVICE] [--shots SHOTS]
                              [--analytic ANALYTIC] [--skip-ops]

        See below for available options and commands for working with the PennyLane
        device tests.

        General Options:
          -h, --help           show this help message and exit
          --device DEVICE      The device to test.
          --shots SHOTS        Number of shots to use in stochastic mode.
          --analytic ANALYTIC  Whether to run the tests in stochastic or exact mode.
          --skip-ops           Skip tests that use unsupported device operations.
          --flaky-report       Show the flaky report in the terminal

    Note that additional pytest command line arguments and flags can also be passed:

    .. code-block:: console

        $ pl-device-test --device default.qubit --shots 1234 --analytic False --tb=short -x
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
        args.device,
        analytic=args.analytic,
        shots=args.shots,
        skip_ops=args.skip_ops,
        flaky_report=flaky,
        pytest_args=pytest_args,
    )
