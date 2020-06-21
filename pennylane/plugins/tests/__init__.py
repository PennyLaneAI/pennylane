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
functionalities.

They can be run by navigating to the parent directory and running:

>>> python3 -m pytest tests/*

The tests can also be run on an external device from a PennyLane plugin, such as
``'qiskit.aer'``. For this, make sure you have the correct dependencies installed and
run

>>> python3 -m pytest tests/* --device qiskit.aer

(where ``qiskit.aer`` is replaced by the device to be tested).

Most tests query the device's capabilities and only get executed if they apply to the device.
Both analytic (with an exact probability distribution) and non-analytic devices (with an estimated
probability distribution) are tested.

For non-analytic tests, the tolerance of the assert statements
is set to a high enough value to account for stochastic fluctuations, and flaky is used to automatically
repeat failed tests.
"""



