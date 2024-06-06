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
from copy import copy
from functools import lru_cache

import pennylane as qml
from pennylane.transforms.core import TransformContainer, transform


@transform
def add_noise(tape, noise_model, level=None):
    """Insert operations according to a provided noise model.

    Circuits passed through this transform will be updated to have the operation,
    specified by the ``noise_model`` argument after appyling the transforms specified
    by the ``level`` keyword argument.

    Args:
        tape (QNode or QuantumTape or Callable or pennylane.devices.Device): the input circuit to be transformed.
        noise_model (~pennylane.NoiseModel): noise model according to which noise has to be inserted.
        level (None, str, int, slice): An indication of a stage in the transform program, when transforming a ``QNode``.
            The following are the permissible values -

            * ``None``: expands the tape to have no ``Adjoint`` and ``Templates``.
            * ``str``: Acceptable keys are ``"top"``, ``"user"``, ``"device"``, and ``"gradient"``
            * ``int``: How many transforms to include, starting from the front of the program
            * ``slice``: a slice to select out components of the transform program.

            Check :func:`~.workflow.get_transform_program` for more information on usage details of this argument.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function] or device (pennylane.devices.Device):
        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ValueError: argument ``noise_model`` is not an instance of :class:`NoiseModel`.

    .. note::

        For a given ``model_map``, if multiple ``conditionals`` in the ``model_map`` evaluates to
        ``True`` for an operation, then the noise operations defined via their respective ``noise_fns``
        will be added in the same order in which the ``conditionals`` appear in the ``model_map``.

    **Example:**

    The following QNode can be transformed to add noise to the circuit:

    .. code-block:: python3

        from functools import partial

        dev = qml.device("default.mixed", wires=2)

        fcond1 = qml.noise.op_eq(qml.RX) & qml.noise.wires_in([0, 1])
        noise1 = qml.noise.partial_wires(qml.PhaseDamping, 0.4)

        fcond2 = qml.noise.op_in([qml.RX, qml.RZ])
        def noise2(op, **kwargs):
            qml.ThermalRelaxationError(op.parameters[0] * 0.5, kwargs["t1"],  kwargs["t2"], 0.6, op.wires)

        noise_model = qml.NoiseModel({fcond1: noise1, fcond2: noise2}, t1=2.0, t2=0.2)

        @partial(qml.transforms.add_noise, noise_model=noise_model)
        @qml.qnode(dev)
        def circuit(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

    Executions of this circuit will differ from the noise-free value:

    >>> circuit(0.9, 0.4, 0.5, 0.6)
    tensor(0.60722291, requires_grad=True)
    >>> print(qml.draw(f)(0.9, 0.4, 0.5, 0.6))
    0: ──RX(0.9)──PhaseDamping(0.4)───────────────────────╭●──RY(0.5)───ThermalRelaxationError(0.2,2.0,0.2,0.6)─┤ ╭<Z@Z>
    1: ──RY(0.4)──ThermalRelaxationError(0.2,2.0,0.2,0.6)─╰X──RX(0.6)───PhaseDamping(0.4)───────────────────────┤ ╰<Z@Z>

    """
    if not isinstance(noise_model, qml.NoiseModel):
        raise ValueError(
            f"Argument noise_model must be an instance of NoiseModel, got {noise_model}."
        )

    if level is None or level == "user":  # decompose templates and their adjoints

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


# pylint:disable = protected-access
@add_noise.custom_qnode_transform
def custom_qnode_wrapper(self, qnode, targs, tkwargs):
    """QNode execution wrapper for supporting ``add_noise`` with levels"""
    qnode = copy(qnode)
    level = tkwargs.get("level", "user")

    transform_program = qml.workflow.get_transform_program(qnode, level=level)
    qnode._transform_program = transform_program

    qnode.add_transform(
        TransformContainer(
            self._transform,
            targs,
            {**tkwargs},
            self._classical_cotransform,
            self._is_informative,
            self._final_transform,
        )
    )

    return qnode
