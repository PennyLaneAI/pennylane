# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains functions for preprocessing `QuantumTape` objects to ensure
that they are supported for execution by a device."""
# pylint: disable=protected-access, too-many-arguments

import os
import warnings
from collections.abc import Callable, Sequence
from copy import copy

from pennylane.decomposition import enabled_graph
from pennylane.exceptions import (
    AllocationError,
    DecompositionUndefinedError,
    DeviceError,
    PennyLaneDeprecationWarning,
    QuantumFunctionError,
    WireError,
)
from pennylane.math import is_abstract, requires_grad
from pennylane.measurements import (
    MeasurementProcess,
    SampleMeasurement,
    StateMeasurement,
    counts,
    sample,
)
from pennylane.operation import Operator, StatePrepBase
from pennylane.ops import Snapshot
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import (
    defer_measurements,
    diagonalize_measurements,
    dynamic_one_shot,
    resolve_dynamic_wires,
)
from pennylane.transforms.core import transform
from pennylane.transforms.decompose import (
    _construct_and_solve_decomp_graph,
    _operator_decomposition_gen,
)
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires

from .execution_config import MCMConfig


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


#######################


@transform
def no_sampling(
    tape: QuantumScript, name: str = "device"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Raises an error if the tape has finite shots.

    Args:
        tape (QuantumTape or .QNode or Callable): a quantum circuit
        name (str): name to use in error message.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qp.transform <pennylane.transform>`.


    This transform can be added to forbid finite shots. For example, ``default.qubit`` uses it for
    adjoint and backprop validation.
    """
    if tape.shots:
        raise DeviceError(f"Finite shots are not supported with {name}")
    return (tape,), null_postprocessing


@transform
def no_analytic(
    tape: QuantumScript, name: str = "device"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Raises an error if the tape does not have finite shots.
    Args:
        tape (QuantumTape or .QNode or Callable): a quantum circuit
        name (str): name to use in error message.
    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:
        The unaltered input circuit. The output type is explained in :func:`qp.transform <pennylane.transform>`.


    This transform can be added to forbid analytic results. This is relevant for devices
    that can only return samples and/or counts based results.
    """
    if not tape.shots:
        raise DeviceError(f"Analytic execution is not supported with {name}")
    return (tape,), null_postprocessing


@transform
def validate_device_wires(
    tape: QuantumScript, wires: Wires | None = None, name: str = "device"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Validates that all wires present in the tape are in the set of provided wires. Adds the
    device wires to measurement processes like :class:`~.measurements.StateMP` that are broadcasted
    across all available wires.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        wires=None (Optional[Wires]): the allowed wires. Wires of ``None`` allows any wires
            to be present in the tape.
        name="device" (str): the name of the device to use in error messages.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qp.transform <pennylane.transform>`.

    Raises:
        WireError: if the tape has a wire not present in the provided wires, or if abstract wires are present.
    """

    if any(is_abstract(w) for w in tape.wires):
        raise WireError(
            f"Cannot run circuit(s) on {name} as abstract wires are present in the tape: {tape.wires}. "
            f"Abstract wires are not yet supported."
        )

    if not wires:
        return (tape,), null_postprocessing

    if any(is_abstract(w) for w in wires):
        raise WireError(
            f"Cannot run circuit(s) on {name} as abstract wires are present in the device: {wires}. "
            f"Abstract wires are not yet supported."
        )

    if extra_wires := set(tape.wires) - set(wires):
        raise WireError(
            f"Cannot run circuit(s) on {name} as they contain wires "
            f"not found on the device: {extra_wires}"
        )

    modified = False
    new_ops = None
    for i, op in enumerate(tape.operations):
        if isinstance(op, Snapshot):
            mp = op.hyperparameters["measurement"]
            if not mp.wires:
                if not new_ops:
                    new_ops = list(tape.operations)
                modified = True
                new_mp = copy(mp)
                new_mp._wires = wires  # pylint:disable=protected-access
                new_ops[i] = Snapshot(
                    measurement=new_mp, tag=op.tag, shots=op.hyperparameters["shots"]
                )
    if not new_ops:
        new_ops = tape.operations  # no copy in this case

    measurements = tape.measurements.copy()
    for m_idx, mp in enumerate(measurements):
        if not mp.obs and not mp.wires:
            modified = True
            new_mp = copy(mp)
            new_mp._wires = wires  # pylint:disable=protected-access
            measurements[m_idx] = new_mp
    if modified:
        tape = tape.copy(ops=new_ops, measurements=measurements)

    return (tape,), null_postprocessing


@transform
def mid_circuit_measurements(
    tape: QuantumScript,
    device,
    mcm_config=MCMConfig(),
    **kwargs,  # pylint: disable=unused-argument
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Provide the transform to handle mid-circuit measurements.

    In the case where no method is specified, if the tape or device
    uses finite-shot, the ``qp.dynamic_one_shot`` transform will be
    applied, otherwise ``qp.defer_measurements`` is used instead.

    .. warning::

        This transform is deprecated and will be removed in a future release. Instead,
        the device should determine which mcm method to use, and explicitly include
        :func:`~pennylane.transforms.dynamic_one_shot` or :func:`~pennylane.transforms.defer_measurements`
        in its preprocess transforms if necessary. See :func:`DefaultQubit.setup_execution_config <pennylane.devices.DefaultQubit.setup_execution_config>`
        and :func:`DefaultQubit.preprocess_transforms <pennylane.devices.DefaultQubit.preprocess_transforms>` for an example.

    """

    warnings.warn(
        "The mid_circuit_measurements transform is deprecated. Instead, the device should "
        "determine the best mcm method, and explicitly include qp.transforms.dynamic_one_shot "
        "or qp.transforms.defer_measurements in the preprocess compile pileline if needed.",
        PennyLaneDeprecationWarning,
    )

    if isinstance(mcm_config, dict):
        mcm_config = MCMConfig(**mcm_config)
    mcm_method = mcm_config.mcm_method
    if mcm_method is None:
        mcm_method = "one-shot" if tape.shots else "deferred"

    if mcm_method == "one-shot":
        return dynamic_one_shot(tape, postselect_mode=mcm_config.postselect_mode)
    if mcm_method in ("tree-traversal", "device"):
        return (tape,), null_postprocessing
    return defer_measurements(tape, allow_postselect=False)


@transform
def validate_multiprocessing_workers(
    tape: QuantumScript, max_workers: int, device
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Validates the number of workers for multiprocessing.

    Checks that the CPU is not oversubscribed and warns user if it is,
    making suggestions for the number of workers and/or the number of
    threads per worker.

    Args:
        tape (QuantumTape or .QNode or Callable): a quantum circuit.
        max_workers (int): Maximal number of multiprocessing workers
        device (pennylane.devices.Device): The device to be checked.

    Returns:
        qnode (pennylane.QNode) or quantum function (callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qp.transform <pennylane.transform>`.

    """
    if max_workers is not None:
        threads_per_proc = os.cpu_count()  # all threads by default
        varname = "OMP_NUM_THREADS"
        varnames = ["MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
        for var in varnames:
            if os.getenv(var):  # pragma: no cover
                varname = var
                threads_per_proc = int(os.getenv(var))
                break
        num_threads = threads_per_proc * max_workers
        num_cpu = os.cpu_count()
        num_threads_suggest = max(1, os.cpu_count() // max_workers)
        num_workers_suggest = max(1, os.cpu_count() // threads_per_proc)
        if num_threads > num_cpu:
            warnings.warn(
                f"""The device requested {num_threads} threads ({max_workers} processes
                times {threads_per_proc} threads per process), but the processor only has
                {num_cpu} logical cores. The processor is likely oversubscribed, which may
                lead to performance deterioration. Consider decreasing the number of processes,
                setting the device or execution config argument `max_workers={num_workers_suggest}`
                for example, or decreasing the number of threads per process by setting the
                environment variable `{varname}={num_threads_suggest}`.""",
                UserWarning,
            )

        if device._debugger and device._debugger.active:
            raise DeviceError("Debugging with ``Snapshots`` is not available with multiprocessing.")

        if any(isinstance(op, Snapshot) for op in tape.operations):
            raise RuntimeError(
                """ProcessPoolExecutor cannot execute a QuantumScript with
                a ``Snapshot`` operation. Change the value of ``max_workers``
                to ``None`` or execute the QuantumScript separately."""
            )

    return (tape,), null_postprocessing


@transform
def validate_adjoint_trainable_params(
    tape: QuantumScript,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Raises a warning if any of the observables is trainable, and raises an error if any
    trainable parameters belong to state-prep operations. Can be used in validating circuits
    for adjoint differentiation.
    """

    for op in tape.operations[: tape.num_preps]:
        if any(requires_grad(d) for d in op.data):
            raise QuantumFunctionError(
                "Differentiating with respect to the input parameters of state-prep operations "
                "is not supported with the adjoint differentiation method."
            )
    for m in tape.measurements:
        if m.obs and any(requires_grad(d) for d in m.obs.data):
            warnings.warn(
                f"Differentiating with respect to the input parameters of {m.obs.name} "
                "is not supported with the adjoint differentiation method. Gradients are computed "
                "only with regards to the trainable parameters of the circuit.\n\n Mark the "
                "parameters of the measured observables as non-trainable to silence this warning.",
                UserWarning,
            )
    return (tape,), null_postprocessing


@transform
def decompose(  # pylint: disable = too-many-positional-arguments
    tape: QuantumScript,
    stopping_condition: Callable[[Operator], bool],
    stopping_condition_shots: Callable[[Operator], bool] | None = None,
    skip_initial_state_prep: bool = True,
    decomposer: Callable[[Operator], Sequence[Operator]] | None = None,
    device_wires: Wires | None = None,
    target_gates: set | None = None,
    name: str = "device",
    error: type[Exception] | None = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Decompose operations until the stopping condition is met.

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit.
        stopping_condition (Callable): a function from an operator to a boolean. If ``False``,
            the operator should be decomposed. If an operator cannot be decomposed and is not
            accepted by ``stopping_condition``, an ``Exception`` will be raised (of a type
            specified by the ``error`` keyword argument).

    Keyword Args:
        stopping_condition_shots (Callable): a function from an operator to a boolean. If
            ``False``, the operator should be decomposed. This replaces ``stopping_condition``
            if and only if the tape has shots.
        skip_initial_state_prep (bool): If ``True``, the first operator will not be decomposed if
            it inherits from :class:`~.StatePrepBase`. Defaults to ``True``.
        decomposer (Callable): an optional callable that takes an operator and implements the
            relevant decomposition. If ``None``, defaults to using a callable returning
            ``op.decomposition()`` for any :class:`~.Operator` .
        device_wires (Wires): The device wires. If provided along with ``target_gates``, will be
            used to automatically set up graph decomposition when enabled.
        target_gates (set): The target gate set for graph decomposition. If provided along with
            ``device_wires``, will automatically enable graph-based decomposition when available.
        name (str): The name of the transform, process or device using decompose. Used in the
            error message. Defaults to "device".
        error (type): An error type to raise if it is not possible to obtain a decomposition that
            fulfills the ``stopping_condition``. Defaults to ``DeviceError``.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:

        The decomposed circuit. The output type is explained in :func:`qp.transform <pennylane.transform>`.

    .. seealso:: This transform is intended for device developers. See :func:`qp.transforms.decompose <pennylane.transforms.decompose>` for a more user-friendly interface.

    Raises:
        Exception: Type defaults to ``DeviceError`` but can be modified via keyword argument.
            Raised if an operator is not accepted and does not define a decomposition, or if
            the decomposition enters an infinite loop and raises a ``RecursionError``.

    **Example:**

    >>> def stopping_condition(obj):
    ...     return obj.name in {"CNOT", "RX", "RZ"}
    >>> tape = qp.tape.QuantumScript([qp.IsingXX(1.2, wires=(0,1))], [qp.expval(qp.Z(0))])
    >>> batch, fn = decompose(tape, stopping_condition)
    >>> batch[0].circuit
    [CNOT(wires=[0, 1]),
    RX(1.2, wires=[0]),
    CNOT(wires=[0, 1]),
    expval(Z(0))]

    If an operator cannot be decomposed into a supported operation, an error is raised:

    >>> decompose(tape, lambda obj: obj.name == "S")
    DeviceError: Operator CNOT(wires=[0, 1]) not supported on device and does not provide a decomposition.

    The ``skip_initial_state_prep`` specifies whether the device supports state prep operations
    at the beginning of the circuit.

    >>> tape = qp.tape.QuantumScript([qp.BasisState([1], wires=0), qp.BasisState([1], wires=1)])
    >>> batch, fn = decompose(tape, stopping_condition)
    >>> batch[0].circuit
    [BasisState(array([1]), wires=[0]),
    RZ(1.5707963267948966, wires=[1]),
    RX(3.141592653589793, wires=[1]),
    RZ(1.5707963267948966, wires=[1])]
    >>> batch, fn = decompose(tape, stopping_condition, skip_initial_state_prep=False)
    >>> batch[0].circuit
    [RZ(1.5707963267948966, wires=[0]),
    RX(3.141592653589793, wires=[0]),
    RZ(1.5707963267948966, wires=[0]),
    RZ(1.5707963267948966, wires=[1]),
    RX(3.141592653589793, wires=[1]),
    RZ(1.5707963267948966, wires=[1])]

    """

    error = error or DeviceError

    if stopping_condition_shots is not None and tape.shots:
        stopping_condition = stopping_condition_shots

    # Compute parameters for graph decomposition if device_wires and target_gates are provided
    if device_wires is None:
        num_available_work_wires = device_wires  # no constraint on work wires
    else:
        # Calculate work wires as device wires that are not used by the tape
        num_available_work_wires = len(set(device_wires) - set(tape.wires))

    graph_solution = None
    if target_gates is not None and enabled_graph():
        # Filter out MeasurementProcess instances that shouldn't be decomposed
        decomposable_ops = [op for op in tape.operations if not isinstance(op, MeasurementProcess)]

        # Construct and solve the decomposition graph
        graph_solution = _construct_and_solve_decomp_graph(
            operations=decomposable_ops,
            target_gates=target_gates,
            num_work_wires=num_available_work_wires,
            minimize_work_wires=False,
            fixed_decomps=None,
            alt_decomps=None,
        )

    if tape.operations and isinstance(tape[0], StatePrepBase) and skip_initial_state_prep:
        prep_op = [tape[0]]
    else:
        prep_op = []

    if all(stopping_condition(op) for op in tape.operations[len(prep_op) :]):
        return (tape,), null_postprocessing

    try:
        new_ops = [
            final_op
            for op in tape.operations[len(prep_op) :]
            for final_op in _operator_decomposition_gen(
                op,
                stopping_condition,
                num_work_wires=num_available_work_wires,
                graph_solution=graph_solution,
                custom_decomposer=decomposer,
                strict=True,
            )
        ]
    except RecursionError as e:
        raise error(
            "Reached recursion limit trying to decompose operations. "
            "Operator decomposition may have entered an infinite loop."
        ) from e

    except DecompositionUndefinedError as e:
        message = str(e).replace("not supported", f"not supported with {name}")
        raise error(message) from e

    tape = tape.copy(operations=prep_op + new_ops)

    return (tape,), null_postprocessing


@transform
def validate_observables(
    tape: QuantumScript,
    stopping_condition: Callable[[Operator], bool],
    stopping_condition_shots: Callable[[Operator], bool] | None = None,
    name: str = "device",
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Validates the observables and measurements for a circuit.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        stopping_condition (callable): a function that specifies whether an observable is accepted.
        stopping_condition_shots (callable): a function that specifies whether an observable is
            accepted in finite-shots mode. This replaces ``stopping_condition`` if and only if the
            tape has shots.
        name (str): the name of the device to use in error messages.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qp.transform <pennylane.transform>`.

    Raises:
        ~pennylane.DeviceError: if an observable is not supported

    **Example:**

    >>> def accepted_observable(obj):
    ...    return obj.name in {"PauliX", "PauliY", "PauliZ"}
    >>> tape = qp.tape.QuantumScript([], [qp.expval(qp.Z(0) + qp.Y(0))])
    >>> validate_observables(tape, accepted_observable)
    DeviceError: Observable Z(0) + Y(0) not supported on device

    """
    if bool(tape.shots) and stopping_condition_shots is not None:
        stopping_condition = stopping_condition_shots

    for m in tape.measurements:
        if m.obs is not None and not stopping_condition(m.obs):
            raise DeviceError(f"Observable {repr(m.obs)} not supported on {name}")

    return (tape,), null_postprocessing


@transform
def validate_measurements(
    tape: QuantumScript, analytic_measurements=None, sample_measurements=None, name="device"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Validates the supported state and sample based measurement processes.

    Args:
        tape (QuantumTape, .QNode, Callable): a quantum circuit.
        analytic_measurements (Callable[[MeasurementProcess], bool]): a function from a measurement process
            to whether or not it is accepted in analytic simulations.
        sample_measurements (Callable[[MeasurementProcess], bool]): a function from a measurement process
            to whether or not it accepted for finite shot simulations.
        name (str): the name to use in error messages.

    Returns:
        qnode (pennylane.QNode) or quantum function (callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qp.transform <pennylane.transform>`.

    Raises:
        ~pennylane.DeviceError: if a measurement process is not supported.

    >>> def analytic_measurements(m):
    ...     return isinstance(m, qp.measurements.StateMP)
    >>> def shots_measurements(m):
    ...     return isinstance(m, qp.measurements.CountsMP)
    >>> tape = qp.tape.QuantumScript([], [qp.expval(qp.Z(0))])
    >>> validate_measurements(tape, analytic_measurements, shots_measurements)
    DeviceError: Measurement expval(Z(0)) not accepted for analytic simulation on device.
    >>> tape = qp.tape.QuantumScript([], [qp.sample()], shots=10)
    >>> validate_measurements(tape, analytic_measurements, shots_measurements)
    DeviceError: Measurement sample(wires=[]) not accepted with finite shots on device

    """
    if analytic_measurements is None:

        def analytic_measurements(m):
            return isinstance(m, StateMeasurement)

    if sample_measurements is None:

        def sample_measurements(m):
            return isinstance(m, SampleMeasurement)

    if tape.shots:
        for m in tape.measurements:
            if not sample_measurements(m):
                raise DeviceError(f"Measurement {m} not accepted with finite shots on {name}")
    else:
        for m in tape.measurements:
            if not analytic_measurements(m):
                raise DeviceError(
                    f"Measurement {m} not accepted for analytic simulation on {name}."
                )

    _validate_snapshot_shots(tape, sample_measurements, analytic_measurements, name)

    return (tape,), null_postprocessing


def _validate_snapshot_shots(tape, sample_measurements, analytic_measurements, name):
    for op in tape.operations:
        if isinstance(op, Snapshot):
            shots = (
                tape.shots
                if op.hyperparameters["shots"] == "workflow"
                else op.hyperparameters["shots"]
            )
            m = op.hyperparameters["measurement"]
            if shots:
                if not sample_measurements(m):
                    raise DeviceError(f"Measurement {m} not accepted with finite shots on {name}")
            else:
                if not analytic_measurements(m):
                    raise DeviceError(
                        f"Measurement {m} not accepted for analytic simulation on {name}."
                    )


@transform
def measurements_from_samples(tape):
    """Quantum function transform that replaces all terminal measurements from a tape with a single
    :func:`pennylane.sample` measurement, and adds postprocessing functions for each original measurement.

    This transform can be used to make tapes compatible with device backends that only return
    :func:`pennylane.sample`. The final output will return the initial requested measurements, calculated instead from
    the raw samples returned immediately after execution.

    The transform is only applied if the tape is being executed with shots.

    .. note::
        This transform diagonalizes all the operations on the tape. An error will
        be raised if non-commuting terms are encountered. To avoid non-commuting
        terms in circuit measurements, the :func:`split_non_commuting <pennylane.transforms.split_non_commuting>`
        transform can be applied.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qp.transform <pennylane.transform>`.

    **Example**

    Consider the tape:

    >>> ops = [qp.X(0), qp.RY(1.23, 1)]
    >>> measurements = [qp.expval(qp.Y(0)), qp.probs(wires=[1])]
    >>> tape = qp.tape.QuantumScript(ops, measurements, shots=10)

    We can apply the transform to diagonalize and convert the two measurements to a single `sample` measurement:

    >>> (new_tape, ), fn = qp.devices.preprocess.measurements_from_samples(tape)
    >>> new_tape.measurements
    [sample(wires=[0, 1])]

    The tape operations now include diagonalizing gates.

    >>> new_tape.operations
    [X(0), RY(1.23, wires=[1]), RX(1.5707963267948966, wires=[0])]

    Executing the tape returns samples that can be post-processed to get the originally requested measurements:

    >>> dev = qp.device("default.qubit")
    >>> res = dev.execute(new_tape)
    >>> res
    array([[1, 0],
           [0, 0],
           [0, 1],
           [1, 1],
           [0, 1],
           [1, 0],
           [0, 1],
           [1, 0],
           [1, 0],
           [1, 0]])

    >>> fn((res,))
    [-0.2, array([0.6, 0.4])]
    """
    if not tape.shots:
        return (tape,), null_postprocessing

    for mp in tape.measurements:
        if not mp.obs and not mp.wires:
            raise RuntimeError(
                "Please apply validate_device_wires transform before measurements_from_samples"
            )

    diagonalized_tape, measured_wires = _get_diagonalized_tape_and_wires(tape)
    new_tape = diagonalized_tape.copy(measurements=[sample(wires=measured_wires)])

    def postprocessing_fn(results):
        """A processing function to get measurement values from samples."""
        samples = results[0]
        if tape.shots.has_partitioned_shots:
            results_processed = []
            for s in samples:
                res = [m.process_samples(s, measured_wires) for m in tape.measurements]
                if len(tape.measurements) == 1:
                    res = res[0]
                results_processed.append(res)
        else:
            results_processed = [
                m.process_samples(samples, measured_wires) for m in tape.measurements
            ]
            if len(tape.measurements) == 1:
                results_processed = results_processed[0]

        return results_processed

    return [new_tape], postprocessing_fn


@transform
def measurements_from_counts(tape):
    r"""Quantum function transform that replaces all terminal measurements from a tape with a single
    :func:`pennylane.counts` measurement, and adds postprocessing functions for each original measurement.

    This transform can be used to make tapes compatible with device backends that only return
    :func:`pennylane.counts`. The final output will return the initial requested measurements, calculated instead from
    the raw counts returned immediately after execution.

    The transform is only applied if the tape is being executed with shots.

    .. note::
        This transform diagonalizes all the operations on the tape. An error will
        be raised if non-commuting terms are encountered. To avoid non-commuting
        terms in circuit measurements, the :func:`split_non_commuting <pennylane.transforms.split_non_commuting>`
        transform can be applied.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qp.transform <pennylane.transform>`.

    **Example**

    Consider the tape:

    >>> ops = [qp.X(0), qp.RY(1.23, 1)]
    >>> measurements = [qp.expval(qp.Y(0)), qp.probs(wires=[1])]
    >>> tape = qp.tape.QuantumScript(ops, measurements, shots=10)

    We can apply the transform to diagonalize and convert the two measurements to a single `counts` measurement:

    >>> (new_tape, ), fn = qp.devices.preprocess.measurements_from_counts(tape)
    >>> new_tape.measurements
    [CountsMP(wires=[0, 1], all_outcomes=False)]

    The tape operations now include diagonalizing gates.

    >>> new_tape.operations
    [X(0), RY(1.23, wires=[1]), RX(1.5707963267948966, wires=[0])]

    The tape is now compatible with a device backend that only supports counts. Executing the
    tape returns the raw counts:

    >>> dev = qp.device("default.qubit")
    >>> res = dev.execute(new_tape)
    >>> res
    {'00': 4, '01': 2, '10': 2, '11': 2}

    And these can be post-processed to get the originally requested measurements:

    >>> fn((res,))
    [-0.19999999999999996, array([0.7, 0.3])]
    """
    if tape.shots.total_shots is None:
        return (tape,), null_postprocessing

    for mp in tape.measurements:
        if not mp.obs and not mp.wires:
            raise RuntimeError(
                "Please apply validate_device_wires transform before measurements_from_counts"
            )

    diagonalized_tape, measured_wires = _get_diagonalized_tape_and_wires(tape)
    new_tape = diagonalized_tape.copy(measurements=[counts(wires=measured_wires)])

    def postprocessing_fn(results):
        """A processing function to get measurement values from counts."""
        counts_res = results[0]

        if tape.shots.has_partitioned_shots:
            results_processed = []
            for c in counts_res:
                res = [m.process_counts(c, measured_wires) for m in tape.measurements]
                if len(tape.measurements) == 1:
                    res = res[0]
                results_processed.append(res)
        else:
            results_processed = [
                m.process_counts(counts_res, measured_wires) for m in tape.measurements
            ]
            if len(tape.measurements) == 1:
                results_processed = results_processed[0]

        return results_processed

    return [new_tape], postprocessing_fn


def _get_diagonalized_tape_and_wires(tape):
    """Apply the diagonalize_measurements transform to the tape and extract a list of
    all the wires present in the measurements"""

    (diagonalized_tape,), _ = diagonalize_measurements(tape)

    measured_wires = set()
    for m in diagonalized_tape.measurements:
        measured_wires.update(
            m.wires.tolist()
        )  # ToDo: add test confirming that the wire order can be weird and this still works
    measured_wires = list(measured_wires)

    return diagonalized_tape, measured_wires


@transform
def device_resolve_dynamic_wires(
    tape: QuantumScript, wires: None | Wires, allow_resets: bool = True
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Allocate dynamic wires in a manner consistent with the provided device wires.

    Args:
        tape (QuantumScript): a circuit that may contain dynamic wire allocation
        wires (None| Wires): the device wires

    If device wires are provided, possible values for dynamic wires are determined from
    device wires not present in the tape.

    >>> from pennylane.devices.preprocess import device_resolve_dynamic_wires
    >>> def f():
    ...     qp.H(0)
    ...     with qp.allocation.allocate(1) as wires:
    ...         qp.X(wires)
    ...     with qp.allocation.allocate(1) as wires:
    ...         qp.X(wires)

    >>> transformed = device_resolve_dynamic_wires(f, wires=qp.wires.Wires((0, "a", "b")))
    >>> print(qp.draw(transformed)())
    0: ──H─┤
    b: ──X─┤
    a: ──X─┤

    If the device has no wires, then wires are allocated starting at the smallest
    integer that is larger than all integer wires present in the ``tape``.

    >>> transformed_None = device_resolve_dynamic_wires(f, wires=None)
    >>> print(qp.draw(transformed_None)())
    0: ──H──────────────┤
    1: ──X──┤↗│  │0⟩──X─┤

    See :func:`~.resolve_dynamic_wires` for a more detailed description.

    """
    if wires:
        zeroed = reversed(list(set(wires) - set(tape.wires)))
        min_int = None
    else:
        zeroed = ()
        min_int = max((i for i in tape.wires if isinstance(i, int)), default=-1) + 1
    try:
        return resolve_dynamic_wires(
            tape, zeroed=zeroed, min_int=min_int, allow_resets=allow_resets
        )
    except AllocationError as e:
        raise AllocationError(
            f"Not enough available wires on device with wires {wires} for requested dynamic wires."
        ) from e
