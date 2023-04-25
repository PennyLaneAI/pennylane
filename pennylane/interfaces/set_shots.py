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
from functools import partial


def set_shots(device, shots):
    """Context manager to temporarily change the shots
    of a device.

    This context manager can be used in two ways.

    As a standard context manager:

    >>> dev = qml.device("default.qubit", wires=2, shots=None)
    >>> with set_shots(dev, shots=100):
    ...     print(dev.shots)
    100
    >>> print(dev.shots)
    None

    Or as a decorator that acts on a function that uses the device:

    >>> set_shots(dev, shots=100)(lambda: dev.shots)()
    100
    """
    return partial(SetShots, device=device, shots=shots)

class SetShots:
    def __init__(self, function, device, shots):
        self._function = function
        self._device = device
        self._shots = shots

    def __call__(self, *args, **kwargs):
        if self.shots == self.device.shots:
            result = self.wrapped_function(*args, **kwargs)
        else:
            original_shots = self.device.shots
            original_shot_vector = self.device._shot_vector

            try:
                if self.shots is not False and self.device.shots != self.shots:
                    self.device.shots = self.shots
                result = self.wrapped_function(*args, **kwargs)
            finally:
                self.device.shots = original_shots
                self.device._shot_vector = original_shot_vector

        return result

    @property
    def device(self):
        return self._device

    @property
    def shots(self):
        return self._shots

    @property
    def wrapped_function(self):
        return self._function

    def __hash__(self):
        return hash((self._function,  self._device, self._shots))

    def __eq__(self, o):
        if isinstance(o, SetShots):
            return (self._function, self._device, self._shots) == (o._function, o._device, o._shots)
        return False

    def __repr__(self):
        return f"SetShots(device={self.device}, shots={self.shots}, function={self.wrapped_function})"

