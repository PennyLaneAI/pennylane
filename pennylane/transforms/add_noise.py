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
        noise_model (NoiseModel): noise model according to which noise has to be inserted.
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

    # decompose templates and their adjoints
    def stop_at(obj):
        if not isinstance(obj, qml.operation.Operator):
            return True
        if not obj.has_decomposition:
            return True
        return not (hasattr(qml.templates, obj.name) or isinstance(obj, qml.ops.Adjoint))

    error_type = (qml.operation.DecompositionUndefinedError,)
    decompose = qml.devices.preprocess.decompose
    [tape], _ = decompose(tape, stopping_condition=stop_at, name="add_noise", error=error_type)

    conditions, noises = [], []
    metadata = noise_model.metadata
    for condition, noise in noise_model.model_map.items():
        conditions.append(lru_cache(maxsize=512)(condition))
        noises.append(qml.tape.make_qscript(noise))

    new_operations = []
    for operation in tape.operations:
        curr_ops = [operation]
        for condition, noise in zip(conditions, noises):
            if condition(operation):
                noise_ops = noise(operation, **metadata).operations
                if operation in noise_ops and _check_queue_op(operation, noise, metadata):
                    ops_indx = noise_ops.index(operation)
                    curr_ops = noise_ops[:ops_indx] + curr_ops + noise_ops[ops_indx + 1 :]
                else:
                    curr_ops.extend(noise_ops)
        new_operations.extend(curr_ops)

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


def _check_queue_op(operation, noise_func, metadata):
    """Performs a secondary check for existence of an operation in the queue using a randomized ID"""

    test_id = "f49968bfc4W0H86df3A733bf6e92904d21a_!$-T-@!_c131S549b169b061I25b85398bfd8ec1S3c"
    test_queue = noise_func(
        qml.noise.partial_wires(operation, id=test_id)(operation.wires), **metadata
    ).operations

    return test_id in [getattr(o, "id", "") for o in test_queue]
