# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This file contains a utility for patching the new devices to work with the old pennylane workflow.

"""


def backward_patch_interface(dev):
    """This utility function patches the interface of a new device to match that currently
    expected by the PennyLane workflow.
    """
    # map the execute function
    dev.batch_execute = dev.execute

    # map the preprocessing steps
    dev.batch_transform = dev.preprocess
    dev.expand_fn = lambda circuit, max_expansion: circuit

    # give dummy shots. We will be moving these out of the class
    dev.shots = None
    dev._shot_vector = []  # pylint: disable=protected-access
    dev.shot_vector = None

    # short name needed for validation in one place
    dev.short_name = "testpython"
    dev.capabilities = lambda: {}
    return dev
