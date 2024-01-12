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
Contains the set_shots context manager, which allows devices shots
to be temporarily modified.
"""
# pylint: disable=protected-access
import contextlib

import pennylane as qml
from pennylane.measurements import Shots


@contextlib.contextmanager
def set_shots(device, shots):
    """Context manager to temporarily change the shots
    of a device.

    This context manager can be used in two ways.

    As a standard context manager:

    >>> dev = qml.device("default.qubit.legacy", wires=2, shots=None)
    >>> with set_shots(dev, shots=100):
    ...     print(dev.shots)
    100
    >>> print(dev.shots)
    None

    Or as a decorator that acts on a function that uses the device:

    >>> set_shots(dev, shots=100)(lambda: dev.shots)()
    100
    """
    if isinstance(device, qml.devices.Device):
        raise ValueError(
            "The new device interface is not compatible with `set_shots`. "
            "Set shots when calling the qnode or put the shots on the QuantumTape."
        )
    if isinstance(shots, Shots):
        shots = shots.shot_vector if shots.has_partitioned_shots else shots.total_shots
    if shots == device.shots:
        yield
        return

    original_shots = device.shots
    original_shot_vector = device._shot_vector

    try:
        if shots is not False and device.shots != shots:
            device.shots = shots
        yield
    finally:
        device.shots = original_shots
        device._shot_vector = original_shot_vector
