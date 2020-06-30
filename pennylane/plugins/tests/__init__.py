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

To run the tests against a particular device (i.e., for 'default.qubit'):

.. code-block:: console

    python3 -m pytest path_to_pennylane/plugins/tests --device default.qubit --shots 1234 --analytic False

The location of your PennyLane installation may differ depending on installation method and operating
system. To find the location, you can execute the following Python code:

>>> import os
>>> import pennylane as qml
>>> print(os.path.dirname(qml.__file__))

The command line arguments are optional.

* If `--device` is not given, the tests are run on the qubit core devices that ship with PennyLane.

* If `--shots` is not given, a default of 50000 is used.

* If `--analytic` is not given, the device's default is used.

The tests can also be run on an external device from a PennyLane plugin, such as
``'qiskit.aer'``. For this, make sure you have the correct dependencies installed.

Most tests query the device's capabilities and only get executed if they apply to the device.
Both analytic devices (producing an exact probability distribution) and non-analytic devices (producing an estimated
probability distribution) are tested.

For non-analytic tests, the tolerance of the assert statements
is set to a high enough value to account for stochastic fluctuations. Flaky is used to automatically
repeat failed tests.
"""
