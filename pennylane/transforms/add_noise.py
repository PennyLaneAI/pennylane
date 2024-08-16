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

    Circuits passed through this quantum transform will be updated to apply the
    insertion-based :class:`~.NoiseModel`, which contains a mapping
    ``{BooleanFn: Callable}`` from conditions to the corresponding noise
    gates. Each condition in the noise model will be evaluated on the
    operations contained within the given circuit. For conditions that
    evaluate to ``True``, the noisy gates contained within the ``Callable``
    will be inserted after the operation under consideration.

    Args:
        tape (QNode or QuantumTape or Callable or pennylane.devices.Device): the input circuit or
            device to be transformed.
        noise_model (~pennylane.NoiseModel): noise model according to which noise has to be inserted.
        level (None, str, int, slice): An indication of which stage in the transform program the
            noise model should be applied to. Only relevant when transforming a ``QNode``. More details
            on the following permissible values can be found in the :func:`~.workflow.get_transform_program` -

            * ``None``: expands the tape to have no ``Adjoint`` and ``Templates``.
            * ``str``: acceptable keys are ``"top"``, ``"user"``, ``"device"``, and ``"gradient"``.
            * ``int``: how many transforms to include, starting from the front of the program.
            * ``slice``: a slice to select out components of the transform program.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function] or device (pennylane.devices.Device):
        Transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ValueError: argument ``noise_model`` is not a valid noise model.

    .. note::

        For a given ``model_map`` within a ``NoiseModel``, if multiple conditionals in the ``model_map``
        evaluate to ``True`` for an operation, then the noise operations defined via their respective
        noisy quantum functions will be added in the same order in which the conditionals appear in the
        ``model_map``.

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

    .. code-block:: python

        >>> circuit(0.9, 0.4, 0.5, 0.6)
        array(0.544053)
        >>> print(qml.draw(circuit)(0.9, 0.4, 0.5, 0.6))
        0: ──RX(0.90)──PhaseDamping(0.40)──ThermalRelaxationError(0.45,2.00,0.20,0.60)─╭●──RY(0.50)
        1: ──RY(0.40)──────────────────────────────────────────────────────────────────╰X──RX(0.60)

        ───────────────────────────────────────────────────────────────────┤ ╭<Z@Z>
        ───PhaseDamping(0.40)──ThermalRelaxationError(0.30,2.00,0.20,0.60)─┤ ╰<Z@Z>

    .. details::
        :title: Tranform Levels
        :href: add-noise-levels

        When transforming an already constructed ``QNode``, the ``add_noise`` transform will be
        added at the end of the "user" transforms by default, i.e., after all the transforms
        that have been manually applied to the QNode up to that point.

        .. code-block:: python3

            dev = qml.device("default.mixed", wires=2)

            @qml.metric_tensor
            @qml.transforms.undo_swaps
            @qml.transforms.merge_rotations
            @qml.transforms.cancel_inverses
            @qml.qnode(dev)
            def circuit(w, x, y, z):
                qml.RX(w, wires=0)
                qml.RY(x, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(y, wires=0)
                qml.RX(z, wires=1)
                return qml.expval(qml.Z(0) @ qml.Z(1))

            noisy_circuit = qml.transforms.add_noise(circuit, noise_model)

        >>> qml.workflow.get_transform_program(circuit)
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, batch_transform, expand_fn, metric_tensor)

        >>> qml.workflow.get_transform_program(noisy_circuit)
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, add_noise, batch_transform, expand_fn, metric_tensor)

        However, one can request to insert the ``add_noise`` transform at any specific point in the transform program. By specifying the ``level`` keyword argument while
        transforming a ``QNode``, this transform can be added at a designated level within the transform program, as determined using the
        :func:`get_transform_program <pennylane.workflow.get_transform_program>`. For example, specifying ``None`` will add it at the end, ensuring that the tape is expanded to have no ``Adjoint`` and ``Templates``:

        >>> qml.transforms.add_noise(circuit, noise_model, level=None).transform_program
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, batch_transform, expand_fn, add_noise, metric_tensor)

        Other acceptable values for ``level`` are ``"top"``, ``"user"``, ``"device"``, and ``"gradient"``. Among these, `"top"` will allow addition
        to an empty transform program, `"user"` will allow addition at the end of user-specified transforms, `"device"` will allow addition at the
        end of device-specific transforms, and `"gradient"` will allow addition at the end of transforms that expand trainable operations. For example:

        >>> qml.transforms.add_noise(circuit, noise_model, level="top").transform_program
        TransformProgram(add_noise)

        >>> qml.transforms.add_noise(circuit, noise_model, level="user").transform_program
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, add_noise, metric_tensor)

        >>> qml.transforms.add_noise(circuit, noise_model, level="device").transform_program
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, batch_transform, expand_fn, add_noise, metric_tensor)

        Finally, more precise control over the insertion of the transform can be achieved by specifying an integer or slice for indexing when extracting the transform program. For example, one can do:

        >>> qml.transforms.add_noise(circuit, noise_model, level=2).transform_program
        TransformProgram(cancel_inverses, merge_rotations, add_noise)

        >>> qml.transforms.add_noise(circuit, noise_model, level=slice(1,3)).transform_program
        TransformProgram(merge_rotations, undo_swaps, add_noise)

    """
    if not hasattr(noise_model, "model_map") or not hasattr(noise_model, "metadata"):
        raise ValueError(
            f"Provided noise model object must define model_map and metatadata attributes, got {noise_model}."
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
    post_processing_fn = qml.devices.preprocess.null_postprocessing

    return [new_tape], post_processing_fn


def _check_queue_op(operation, noise_func, metadata):
    """Performs a secondary check for existence of an operation in the queue using a randomized ID"""

    test_id = "f49968bfc4W0H86df3A733bf6e92904d21a_!$-T-@!_c131S549b169b061I25b85398bfd8ec1S3c"
    test_queue = noise_func(
        qml.noise.partial_wires(operation, id=test_id)(operation.wires), **metadata
    ).operations

    return any(test_id == getattr(o, "id", "") for o in test_queue)


# pylint:disable = protected-access
@add_noise.custom_qnode_transform
def custom_qnode_wrapper(self, qnode, targs, tkwargs):
    """QNode execution wrapper for supporting ``add_noise`` with levels"""
    cqnode = copy(qnode)
    level = tkwargs.get("level", "user")

    transform_program = qml.workflow.get_transform_program(qnode, level=level)

    cqnode._transform_program = transform_program
    cqnode.add_transform(
        TransformContainer(
            self._transform,
            targs,
            {**tkwargs},
            self._classical_cotransform,
            self._is_informative,
            self._final_transform,
        )
    )

    return cqnode
