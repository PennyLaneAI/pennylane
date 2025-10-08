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

from pennylane import math, templates
from pennylane.devices.preprocess import decompose, null_postprocessing
from pennylane.operation import DecompositionUndefinedError, Operator
from pennylane.ops import Adjoint
from pennylane.tape import make_qscript
from pennylane.transforms.core import TransformContainer, transform
from pennylane.workflow import get_transform_program

from .conditionals import partial_wires


# pylint: disable=too-many-branches
@transform
def add_noise(tape, noise_model, level=None):
    """Insert operations according to a provided noise model.

    Circuits passed through this quantum transform will be updated to apply the
    insertion-based :class:`~.NoiseModel`, which contains mappings
    ``{BooleanFn: Callable}`` from conditions to the corresponding noise
    gates for circuit operations and measurements respectively. First, each condition
    in the first mapping of a noise model will be evaluated on the operations
    contained within the given circuit. For conditions that evaluate to ``True``,
    the noisy gates contained within the ``Callable`` will be inserted after the
    operation under consideration. Similar procedure will be followed for each
    measurement in the circuit, in case a second mapping is present in the
    noise model to indicate readout errors.

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

        For a given ``model_map`` and ``meas_map`` within a ``NoiseModel``, if multiple conditionals
        in the given maps evaluate to ``True`` for an operation or measurement process, then the
        noise operations defined via their respective noisy quantum functions will be added in the
        same order in which the conditionals appear in them.

    **Example:**

    The following QNode can be transformed to add noise to the circuit:

    .. code-block:: python

        from functools import partial

        dev = qml.device("default.mixed", wires=2)

        fcond1 = qml.noise.op_eq(qml.RX) & qml.noise.wires_in([0, 1])
        noise1 = qml.noise.partial_wires(qml.PhaseDamping, 0.4)

        fcond2 = qml.noise.op_in([qml.RX, qml.RZ])
        def noise2(op, **kwargs):
            qml.ThermalRelaxationError(op.parameters[0] * 0.5, kwargs["t1"],  kwargs["t2"], 0.6, op.wires)

        fcond3 = qml.noise.meas_eq(qml.expval) & qml.noise.wires_in([0, 1])
        noise3 = qml.noise.partial_wires(qml.PhaseFlip, 0.2)

        noise_model = qml.NoiseModel(
            {fcond1: noise1, fcond2: noise2}, {fcond3: noise3}, t1=2.0, t2=0.2
        )

        @partial(qml.noise.add_noise, noise_model=noise_model)
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
    np.float64(0.5440530007721438)
    >>> print(qml.draw(circuit)(0.9, 0.4, 0.5, 0.6))
    0: ──RX(0.90)──PhaseDamping(0.40)──ThermalRelaxationError(0.45,2.00,0.20,0.60)─╭●──RY(0.50) ···
    1: ──RY(0.40)──────────────────────────────────────────────────────────────────╰X──RX(0.60) ···
    <BLANKLINE>
    0: ··· ──PhaseFlip(0.20)──────────────────────────────────────────────────────────────────┤ ╭<Z@Z>
    1: ··· ──PhaseDamping(0.40)──ThermalRelaxationError(0.30,2.00,0.20,0.60)──PhaseFlip(0.20)─┤ ╰<Z@Z>

    .. details::
        :title: Tranform Levels
        :href: add-noise-levels

        When transforming an already constructed ``QNode``, the ``add_noise`` transform will be
        added at the end of the "user" transforms by default, i.e., after all the transforms
        that have been manually applied to the QNode up to that point.

        .. code-block:: python

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

            noisy_circuit = qml.noise.add_noise(circuit, noise_model)

        >>> qml.workflow.get_transform_program(circuit)
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, defer_measurements, decompose, no_sampling, validate_device_wires, validate_measurements, validate_observables, metric_tensor)

        >>> qml.workflow.get_transform_program(noisy_circuit)
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, add_noise, defer_measurements, decompose, no_sampling, validate_device_wires, validate_measurements, validate_observables, metric_tensor)

        However, one can request to insert the ``add_noise`` transform at any specific point in the transform program. By specifying the ``level`` keyword argument while
        transforming a ``QNode``, this transform can be added at a designated level within the transform program, as determined using the
        :func:`get_transform_program <pennylane.workflow.get_transform_program>`. For example, specifying ``None`` will add it at the end, ensuring that the tape is expanded to have no ``Adjoint`` and ``Templates``:

        >>> qml.noise.add_noise(circuit, noise_model, level="device").transform_program
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, defer_measurements, decompose, no_sampling, validate_device_wires, validate_measurements, validate_observables, add_noise, metric_tensor)

        Other acceptable values for ``level`` are ``"top"``, ``"user"``, ``"device"``, and ``"gradient"``. Among these, `"top"` will allow addition
        to an empty transform program, `"user"` will allow addition at the end of user-specified transforms, `"device"` will allow addition at the
        end of device-specific transforms, and `"gradient"` will allow addition at the end of transforms that expand trainable operations. For example:

        >>> qml.noise.add_noise(circuit, noise_model, level="top").transform_program
        TransformProgram(add_noise)

        >>> qml.noise.add_noise(circuit, noise_model, level="user").transform_program
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, add_noise, metric_tensor)

        >>> qml.noise.add_noise(circuit, noise_model, level="device").transform_program
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, defer_measurements, decompose, no_sampling, validate_device_wires, validate_measurements, validate_observables, add_noise, metric_tensor)

        Finally, more precise control over the insertion of the transform can be achieved by specifying an integer or slice for indexing when extracting the transform program. For example, one can do:

        >>> qml.noise.add_noise(circuit, noise_model, level=2).transform_program
        TransformProgram(cancel_inverses, merge_rotations, add_noise)

        >>> qml.noise.add_noise(circuit, noise_model, level=slice(1,3)).transform_program
        TransformProgram(merge_rotations, undo_swaps, add_noise)

    """
    if not hasattr(noise_model, "model_map") or not hasattr(noise_model, "metadata"):
        raise ValueError(
            f"Provided noise model object must define model_map and metatadata attributes, got {noise_model}."
        )

    if level is None or level == "user":  # decompose templates and their adjoints

        def stop_at(obj):
            if not isinstance(obj, Operator):
                return True
            if not obj.has_decomposition:
                return True
            return not (hasattr(templates, obj.name) or isinstance(obj, Adjoint))

        [tape], _ = decompose(
            tape, stopping_condition=stop_at, name="add_noise", error=DecompositionUndefinedError
        )

    conditions, noises = [], []
    metadata = noise_model.metadata
    for condition, noise in noise_model.model_map.items():
        conditions.append(lru_cache(maxsize=512)(condition))
        noises.append(make_qscript(noise))

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

    if not noise_model.meas_map:
        new_tape = tape.copy(operations=new_operations)
        return [new_tape], null_postprocessing

    meas_conds, meas_funcs = [], []
    for condition, noise in noise_model.meas_map.items():
        meas_conds.append(lru_cache(maxsize=512)(condition))
        meas_funcs.append(make_qscript(noise))

    split_operations, split_measurements = [], [[] for idx in tape.measurements]
    for midx, measurement in enumerate(tape.measurements):
        readout_operations = new_operations.copy()
        for condition, noise in zip(meas_conds, meas_funcs):
            if condition(measurement):
                noise_ops = noise(measurement, **metadata).operations
                readout_operations.extend(noise_ops)
        if readout_operations not in split_operations:
            split_operations.append(readout_operations)
        split_measurements[split_operations.index(readout_operations)].append((midx, measurement))

    split_measurements = split_measurements[: len(split_operations)]
    split_meas_indexes = math.argsort(
        [m_ for ms in ([m[0] for m in meas] for meas in split_measurements) for m_ in ms]
    )

    new_tapes = [
        tape.copy(operations=operations, measurements=[meas[1] for meas in measurements])
        for operations, measurements in zip(split_operations, split_measurements)
    ]

    def post_processing_fn(results):
        """A postprocessing function returned by a transform that converts the batch of results into a squeezed result."""
        split_results = []
        for result in results:
            getattr(split_results, "append" if not isinstance(result, tuple) else "extend")(result)
        final_res = [split_results[idx] for idx in split_meas_indexes]
        return tuple(final_res) if len(final_res) > 1 else final_res[0]

    return new_tapes, post_processing_fn


def _check_queue_op(operation, noise_func, metadata):
    """Performs a secondary check for existence of an operation in the queue using a randomized ID"""

    test_id = "f49968bfc4W0H86df3A733bf6e92904d21a_!$-T-@!_c131S549b169b061I25b85398bfd8ec1S3c"
    test_queue = noise_func(
        partial_wires(operation, id=test_id)(operation.wires), **metadata
    ).operations

    return any(test_id == getattr(o, "id", "") for o in test_queue)


# pylint:disable = protected-access
@add_noise.custom_qnode_transform
def custom_qnode_wrapper(self, qnode, targs, tkwargs):
    """QNode execution wrapper for supporting ``add_noise`` with levels"""
    cqnode = copy(qnode)
    level = tkwargs.get("level", "user")

    transform_program = get_transform_program(qnode, level=level)

    cqnode._transform_program = transform_program
    cqnode.transform_program.push_back(
        TransformContainer(
            self,
            targs,
            {**tkwargs},
        )
    )

    return cqnode
