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
"""
This file provides a dummy debugger for device tests.
"""

import pennylane as qml


# pylint: disable=too-few-public-methods
class Debugger:
    """A dummy debugger class"""

    def __init__(self):
        # Create a dummy object to act as the device
        # and add a dummy shots attribute to it
        self.device = type("", (), {})()
        self.device.shots = qml.measurements.Shots(None)

        self.active = True
        self.snapshots = {}
