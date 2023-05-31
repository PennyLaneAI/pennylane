# Copyright 2023 Xanadu Quantum Technologies Inc.

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
This module contains the transform function, the transform dispatcher and the transform container.
"""
import pennylane
from .transform_dispatcher import TransformContainer, TransformError


class TransformProgram:
    """Class that contains a transform program and the method to interact with it. The order of execution is the order
    in the list containing the containers."""

    def __init__(self):
        self._transform_program = []

    def push_back(self, transform_container: TransformContainer):
        """Add a transform (container) to the program."""
        if not isinstance(transform_container, TransformContainer):
            raise TransformError("Only transform container can be added to the transform program.")
        if (
            not self.empty()
            and self.last_program().is_informative
            and transform_container.is_informative
        ):
            raise TransformError("The transform program already have an informative transform.")
        self._transform_program.append(transform_container)

    def insert_front(self, transform_container):
        self._transform_program.insert(0, transform_container)

    def pop_front(self):
        first_container = self._transform_program.pop(0)
        return first_container

    def add_device_pre_processing(self, device: pennylane.Device):
        device_preprocessing_container = TransformContainer(device.preprocessing)
        # Device pre-processing is first.
        self.insert_front(0, device_preprocessing_container)

    def last_program(self):
        if not self.empty():
            return self._transform_program[-1]
        else:
            raise TransformError(
                "The transform program is empty and you cannot get the last transform container."
            )

    def empty(self):
        return len(self._transform_program) == 0
