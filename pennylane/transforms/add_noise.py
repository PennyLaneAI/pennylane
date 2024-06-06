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

    .. details::
        :title: Usage Details: Advanced Examples

        ``qml.add_noise`` can also be used to tranform quantum functions, devices or already constructed qnodes.
        To do this, one may construct noise models with various kinds of ``conditionals`` and ``noise_fns`` key-value pairs.
        For example, for a list of specified operations, one can functionally add noise operations for their every encounter
        in the circuit based on their wires:

        .. code-block:: python3

            fcond1 = qml.noise.op_in(["X", "Y"])
            noise1 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

        Moreover, a functional conditional construction could also be done in cases where more complex evaluations are needed
        than the ones done by the already provided conditional constructors in the :func:`qml.noise <pennylane.noise>`.

        .. code-block:: python3

            @qml.BooleanFn
            def fcond2(op, **kwargs):
                return isinstance(op, qml.RY) and op.parameters[0] >= 0.5
            noise2 = qml.noise.partial_wires(qml.PhaseDamping, 0.9)

        By default, for each operation for which a conditional evaluates to ``True``, the corresponding noise operations gets
        queued following an `iterative-insertion` approach. One could change this by including the operation itself, and
        queue it with :func:`~.pennylane.apply` as given below:

        .. code-block:: python3

            fcond3 = qml.noise.op_eq(qml.RX) & qml.noise.wires_in([0, 1])
            def noise3(op, **kwargs):
                qml.RZ(op.parameters[0] * 0.05, op.wires)
                qml.apply(op) # <-- sandwiched between two RZs
                qml.RZ(-op.parameters[0] * 0.05, op.wires)

        Additionally, as noted before in the example above, one can provide keyword arguments for building the noise operations
        based on the ``metadata`` based on properties like hardware topology, qubit relaxation times, etc.

        .. code-block:: python3

            fcond4 = qml.noise.op_in([qml.RX, qml.RZ])
            def noise4(op, **kwargs):
                qml.ThermalRelaxationError(op.parameters[0] * 0.5, kwargs["t1"],  kwargs["t2"], 0.6, op.wires)

        Finally, one could build a noise model and combine it with another noise model via addition.

        >>> noise_model = qml.NoiseModel({fcond1: noise1, fcond2: noise2, fcond3: noise3})
        >>> noise_model += qml.NoiseModel({fcond4: noise4}, t1=2.0, t2=0.2)
        >>> noise_model
        NoiseModel({
            OpIn(['PauliX', 'PauliY']): AmplitudeDamping(gamma=0.4)
            BooleanFn(fcond2): PhaseDamping(gamma=0.9)
            OpEq(RX) & WiresIn([0, 1]): noise3
            OpIn(['RX', 'RZ']): noise4
        }, t1 = 2.0, t2 = 0.2)

        Now, one can use this ``noise_model`` for the transform as below:

        .. code-block:: python3

            def f(w, x, y, z):
                qml.RX(w, wires=0)
                qml.RY(x, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(y, wires=0)
                qml.RX(z, wires=1)
                return qml.expval(qml.Z(0) @ qml.Z(1))

            dev = qml.device("default.mixed", wires=2)
            qfunc_qnode = qml.QNode(f, dev)
            noise_qnode = qml.transforms.add_noise(qfunc_qnode, noise_model)

        >>> print(qml.draw(qfunc_qnode, decimals=2)(0.1, 0.7, 0.8, 0.4))
        0: ──RX(0.10)─╭●──RY(0.80)─┤ ╭<Z@Z>
        1: ──RY(0.70)─╰X──RX(0.40)─┤ ╰<Z@Z>

        >>> print(qml.draw(noise_qnode, decimals=2)(0.1, 0.7, 0.8, 0.4))
        0: ──RZ(0.01)──RX(0.10)────────────RZ(-0.01)─╭●──RY(0.80)──PhaseDamping(0.90)────────────┤ ╭<Z@Z>
        1: ──RY(0.70)──PhaseDamping(0.90)────────────╰X──RZ(0.02)──RX(0.40)────────────RZ(-0.02)─┤ ╰<Z@Z>

        >>> noisy_dev = qml.transforms.add_noise(dev, noise_model)
        >>> noisy_qfn = qml.QNode(f, noisy_dev)
        >>> qml.math.isclose(noise_qnode(0.1, 0.7, 0.8, 0.4), noisy_qfn(0.1, 0.7, 0.8, 0.4))
        True

    .. details::
        :title: Usage Details: Tranform Levels

        By default, ``add_noise`` transform will be added to the end of the "user" transforms,
        i.e., after the transforms that are manually applied to the qnode.

        .. code-block:: python3

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

            dev = qml.device("default.mixed", wires=2)
            noisy_circuit = qml.transforms.add_noise(circuit, noise_model)

        >>> qml.workflow.get_transform_program(circuit)
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, batch_transform, expand_fn, metric_tensor)
        >>> qml.workflow.get_transform_program(noisy_circuit)
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, add_noise, batch_transform, expand_fn, metric_tensor)

        However, one can request inserting it at any specific point of the transform program. Using the ``level`` keyword argument,
        where the transform will be added at the end of the transform program extracted at a designated level via
        :func:`get_transform_program <pennylane.workflow.get_transform_program>`. For example, one could specify ``None`` to add it at the end,
        which will also ensure that the tape is expanded to have no ``Adjoint`` and ``Templates``:

        >>> qml.transforms.add_noise(circuit, noise_model, level=None).transform_program
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, batch_transform, expand_fn, add_noise, metric_tensor)

        Other, acceptable values for the level are ``"top"``, ``"user"``, ``"device"``, and ``"gradient"``. Among these, `"top"` will alow addition
        to an empty transform program, `"user"` will allow addition at the end of user specified transforms, `"device"` will allow addition at the
        end of device-specific transforms, and `"gradient"` will allow addition at the end of transform that expands trainable operations. For example:

        >>> qml.transforms.add_noise(circuit, noise_model, level="top").transform_program
        TransformProgram(add_noise)
        >>> qml.transforms.add_noise(circuit, noise_model, level="user").transform_program
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, add_noise)
        >>> qml.transforms.add_noise(circuit, noise_model, level="device").transform_program
        TransformProgram(cancel_inverses, merge_rotations, undo_swaps, _expand_metric_tensor, batch_transform, expand_fn, add_noise)

        Finally, more precise control over exctraction of the transform program at the end of which the transform is to be inserted can be achieved
        by specifying an integer or slice for indexing. For example:

        >>> qml.transforms.add_noise(circuit, noise_model, level=2).transform_program
        TransformProgram(cancel_inverses, merge_rotations, add_noise)
        >>> qml.transforms.add_noise(circuit, noise_model, level=slice(1,3)).transform_program
        TransformProgram(merge_rotations, undo_swaps, add_noise)

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
    cqnode = copy(qnode)
    level = tkwargs.get("level", "user")

    transform_program = qml.workflow.get_transform_program(qnode, level=level)
    if "level" not in tkwargs and qnode.transform_program.has_final_transform:
        transform_program.push_back(qnode.transform_program[-1])

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
