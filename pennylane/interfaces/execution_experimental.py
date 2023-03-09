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
Contains the cache_execute decoratator, for adding caching to a function
that executes multiple tapes on a device.

Also contains the general execute function, for exectuting tapes on
devices with autodifferentiation support.
"""

from functools import partial
from typing import Callable, Sequence

import pennylane as qml
from pennylane.tape import QuantumTape

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


def _adjoint_jacobian_expansion(
    tapes: Sequence[QuantumTape], mode: str, interface: str, max_expansion: int
):
    """Performs adjoint jacobian specific expansion.  Expands so that every
    trainable operation has a generator.

    TODO: Let the device specify any gradient-specific expansion logic.  This
    function will be removed once the device-support pipeline is improved.
    """
    if mode == "forward" and INTERFACE_MAP[interface] == "jax":
        # qml.math.is_trainable doesn't work with jax on the forward pass
        non_trainable = qml.operation.has_nopar
    else:
        non_trainable = ~qml.operation.is_trainable

    stop_at = ~qml.operation.is_measurement & (
        non_trainable  # pylint: disable=unsupported-binary-operation
        | qml.operation.has_unitary_gen
    )
    for i, tape in enumerate(tapes):
        if any(not stop_at(op) for op in tape.operations):
            tapes[i] = tape.expand(stop_at=stop_at, depth=max_expansion)

    return tapes


def execute_experimental(
    tapes: Sequence[QuantumTape],
    device,
    transforms_program,
):
    """New function to execute a batch of tapes on a device in an autodifferentiable-compatible manner. More cases will be added,
    during the project. The current version is supporting forward execution for Numpy and does not support shot vectors.
    """
    execute = device._batch_execute_new

    tape_process_fn = []
    qnode_process_fn = []
    # Apply transform queue and get a batch of tapes
    while len(transforms_program._transform_program) != 0:
        t_fn, targs, tkwargs, expand_fn, q_process = transforms_program.pop()
        tapes, t_process = qml.transforms.map_batch_transform(t_fn, tapes)
        tape_process_fn.append(t_process)
        qnode_process_fn.append(q_process)

    # Device expansion just before execution
    expand_fn = lambda tape: device.expand_fn(tape, max_expansion=10)
    tapes = [expand_fn(tape) for tape in tapes]

    # Execution tapes
    with qml.tape.Unwrap(*tapes):
        res = execute(tapes)

    # Postprocessing
    for fn in tape_process_fn[::-1]:
        res = fn(res)

    return res
    # return res
