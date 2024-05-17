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
"""Transform for adding a noise model to a quantum circuit or device"""
from functools import lru_cache

import pennylane as qml
from pennylane.transforms.core import transform


@transform
def add_noise(tape, noise_model, level=None):
    """Insert operations according to a provided noise model.

    Circuits passed through this transform will be updated to have the operation,
    specified by the ``noise_model`` argument.

    Args:
        tape (QNode or QuantumTape or Callable or pennylane.devices.Device): the input circuit to be transformed.
        noise_model (NoiseModel): noise model for
        level (None, str, int, slice): An indication of a stage in the transform program.
            * ``None``: expands the tape to have no ``Adjoint`` and ``Templates``.
            * ``str``: Acceptable keys are ``"top"``, ``"user"``, ``"device"``, and ``"gradient"``
            * ``int``: How many transforms to include, starting from the front of the program
            * ``slice``: a slice to select out components of the transform program.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function] or device (pennylane.devices.Device):

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ValueError: argument ``noise_model`` is not an instance of :class:`NoiseModel`.
    """
    if not isinstance(noise_model, qml.NoiseModel):
        raise ValueError(
            f"Argument noise_model must be an instance of NoiseModel, got {noise_model}."
        )

    if level is not None:
        raise NotImplementedError("Support for level argument is not currently present.")

    try:
        tape = tape.expand(
            stop_at=lambda op: not hasattr(qml.templates, op.name)
            and not isinstance(op, qml.ops.Adjoint)
        )
    except qml.QuantumFunctionError as e:
        raise qml.QuantumFunctionError(
            "The add_noise transform cannot transform a circuit measuring non-commuting observables. "
            "Consider wrapping the gates in their own function and transforming only that function."
        ) from e

    conditions, noises = [], []
    metadata = noise_model.metadata
    for condition, noise in noise_model.model_map.items():
        conditions.append(lru_cache(maxsize=512)(condition))
        noises.append(noise)

    with qml.queuing.AnnotatedQueue() as new_operations:
        for operation in tape.operations:
            qml.apply(operation)
            new_operations.append(operation)
            for condition, noise in zip(conditions, noises):
                if condition(operation):
                    noise(operation, **metadata)

    new_tape = type(tape)(new_operations.queue, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
