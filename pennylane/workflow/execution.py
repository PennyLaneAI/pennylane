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
    transform_program,
) -> ResultBatch:
    """New function to execute a batch of tapes on a device with a transform program.
    """
    # Apply all transforms (device pre-processing, compilation)
    if not transform_program.is_empty():
        tapes, processing_fns, classical_cotransforms = transform_program(tapes)

    # The resulting batch of tapes is executed by the device
    # Execution tapes
    if not transform_program.is_informative():
        with qml.tape.Unwrap(*tapes):
            res = device.batch_execute(tapes)

    # Apply postprocessing (apply classical cotransform and processing function)
    for p_fn, cotransform in zip(processing_fns, classical_cotransforms):
        if cotransform:
            res = cotransform(res)
        res = p_fn(res)
    return res