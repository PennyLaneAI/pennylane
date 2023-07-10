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
Contains the cache_execute decoratator, for adding caching to a function
that executes multiple tapes on a device.

Also contains the general execute function, for exectuting tapes on
devices with autodifferentiation support.
"""

import inspect
import warnings
from functools import wraps, partial
from typing import Callable, Sequence, Optional, Union, Tuple

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import ResultBatch
from pennylane.transforms.core import TransformProgram

device_type = Union[qml.Device, "qml.devices.experimental.Device"]

INTERFACE_MAP = {
    None: "Numpy",
    "auto": "auto",
    "autograd": "autograd",
    "numpy": "autograd",
    "scipy": "numpy",
    "jax": "jax",
    "jax-jit": "jax",
    "jax-python": "jax",
    "JAX": "jax",
    "torch": "torch",
    "pytorch": "torch",
    "tf": "tf",
    "tensorflow": "tf",
    "tensorflow-autograph": "tf",
    "tf-autograph": "tf",
}
"""dict[str, str]: maps an allowed interface specification to its canonical name."""

#: list[str]: allowed interface strings
SUPPORTED_INTERFACES = list(INTERFACE_MAP)
"""list[str]: allowed interface strings"""


def execute(
    tapes: Sequence[QuantumTape],
    device: device_type,
    transform_program: TransformProgram,
) -> ResultBatch:
    """New function to execute a batch of tapes on a device with a transform program in an
    autodifferentiable-compatible manner.
    """
    # Apply all transforms (device pre-processing, compilation)
    for transform in transform_program:
        transform(tapes)

    # The resulting batch of tapes is executed by the device
    res = device.batch_execute(tapes)

    # Apply post processing fns and classical co-tranforms