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
"""
Contains the batch dimension transform.
"""

import itertools

# pylint: disable=import-outside-toplevel
from collections import Counter
from typing import Callable, Sequence

import numpy as np

import pennylane as qml
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MeasurementValue,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)
from pennylane.typing import TensorLike

from .core import transform

fill_in_value = np.iinfo(np.int32).min


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def dynamic_one_shot(
    tape: qml.tape.QuantumTape, **kwargs
) -> tuple[Sequence[qml.tape.QuantumTape], Callable]:
    """Transform a QNode to into several one-shot tapes to support dynamic circuit execution.

    Args:
        tape (QNode or QuantumTape or Callable): a quantum circuit to add a batch dimension to

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.
        This circuit will provide the results of a dynamic execution.


    **Example**

    Consider the following circuit:

    .. code-block:: python

        dev = qml.device("default.qubit", shots=100)
        params = np.pi / 4 * np.ones(2)

        @qml.dynamic_one_shot
        @qml.qnode(dev)
        def func(x, y):
            qml.RX(x, wires=0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.RY)(y, wires=1)
            return qml.expval(op=m0)

    The ``qml.dynamic_one_shot`` decorator prompts the QNode to perform a hundred one-shot
    calculations, where in each calculation the ``qml.measure`` operations dynamically
    measures the 0-wire and collapse the state vector stochastically. This transforms
    contrasts with ``qml.defer_measurements``, which instead introduces an extra wire
    for each mid-circuit measurement. The ``qml.dynamic_one_shot`` transform is favorable in the
    few-shots several-mid-circuit-measurement limit, whereas ``qml.defer_measurements`` is favorable
    in the opposite limit.
    """
    if not any(is_mcm(o) for o in tape.operations):
        return (tape,), null_postprocessing

    for m in tape.measurements:
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise TypeError(
                f"Native mid-circuit measurement mode does not support {type(m).__name__} "
                "measurements."
            )
    _ = kwargs.get("device", None)

    if not tape.shots:
        raise qml.QuantumFunctionError("dynamic_one_shot is only supported with finite shots.")

    samples_present = any(isinstance(mp, SampleMP) for mp in tape.measurements)
    postselect_present = any(op.postselect is not None for op in tape.operations if is_mcm(op))
    if postselect_present and samples_present and tape.batch_size is not None:
        raise ValueError(
            "Returning qml.sample is not supported when postselecting mid-circuit "
            "measurements with broadcasting"
        )

    if (batch_size := tape.batch_size) is not None:
        tapes, broadcast_fn = qml.transforms.broadcast_expand(tape)
    else:
        tapes = [tape]
        broadcast_fn = None

    aux_tapes = [init_auxiliary_tape(t) for t in tapes]

    def processing_fn(results, has_partitioned_shots=None, batched_results=None):
        if batched_results is None and batch_size is not None:
            # If broadcasting, recursively process the results for each batch. For each batch
            # there are tape.shots.total_shots results. The length of the first axis of final_results
            # will be batch_size.
            final_results = []
            for result in results:
                final_results.append(processing_fn((result,), batched_results=False))
            return broadcast_fn(final_results)

        if has_partitioned_shots is None and tape.shots.has_partitioned_shots:
            # If using shot vectors, recursively process the results for each shot bin. The length
            # of the first axis of final_results will be the length of the shot vector.
            results = list(results[0])
            final_results = []
            for s in tape.shots:
                final_results.append(
                    processing_fn(results[0:s], has_partitioned_shots=False, batched_results=False)
                )
                del results[0:s]
            return tuple(final_results)
        if not tape.shots.has_partitioned_shots:
            results = results[0]
        return parse_native_mid_circuit_measurements(tape, aux_tapes, results)

    return aux_tapes, processing_fn


@dynamic_one_shot.custom_qnode_transform
def _dynamic_one_shot_qnode(self, qnode, targs, tkwargs):
    """Custom qnode transform for ``dynamic_one_shot``."""
    if tkwargs.get("device", None):
        raise ValueError(
            "Cannot provide a 'device' value directly to the dynamic_one_shot decorator "
            "when transforming a QNode."
        )
    if qnode.device is not None:
        support_mcms = hasattr(qnode.device, "capabilities") and qnode.device.capabilities().get(
            "supports_mid_measure", False
        )
        support_mcms = support_mcms or qnode.device.name in ("default.qubit", "lightning.qubit")
        if not support_mcms:
            raise TypeError(
                f"Device {qnode.device.name} does not support mid-circuit measurements "
                "natively, and hence it does not support the dynamic_one_shot transform. "
                "'default.qubit' and 'lightning.qubit' currently support mid-circuit "
                "measurements and the dynamic_one_shot transform."
            )
    tkwargs.setdefault("device", qnode.device)
    return self.default_qnode_transform(qnode, targs, tkwargs)


def is_mcm(operation):
    """Returns True if the operation is a mid-circuit measurement and False otherwise."""
    mcm = isinstance(operation, MidMeasureMP)
    return mcm or "MidCircuitMeasure" in str(type(operation))


def init_auxiliary_tape(circuit: qml.tape.QuantumScript):
    """Creates an auxiliary circuit to perform one-shot mid-circuit measurement calculations.

    Measurements are replaced by SampleMP measurements on wires and observables found in the
    original measurements.

    Args:
        circuit (QuantumTape): The original QuantumScript

    Returns:
        QuantumScript: A copy of the circuit with modified measurements
    """
    new_measurements = []
    for m in circuit.measurements:
        if not m.mv:
            if isinstance(m, VarianceMP):
                new_measurements.append(SampleMP(obs=m.obs))
            else:
                new_measurements.append(m)
    for op in circuit:
        if is_mcm(op):
            new_measurements.append(qml.sample(MeasurementValue([op], lambda res: res)))

    return qml.tape.QuantumScript(
        circuit.operations,
        new_measurements,
        shots=[1] * circuit.shots.total_shots,
        trainable_params=circuit.trainable_params,
    )


def parse_native_mid_circuit_measurements(
    circuit: qml.tape.QuantumScript, aux_tapes: qml.tape.QuantumScript, results: TensorLike
):
    """Combines, gathers and normalizes the results of native mid-circuit measurement runs.

    Args:
        circuit (QuantumTape): Initial ``QuantumScript``
        aux_tapes (List[QuantumTape]): List of auxilary ``QuantumScript`` objects
        results (TensorLike): Array of measurement results

    Returns:
        tuple(TensorLike): The results of the simulation
    """

    def measurement_with_no_shots(measurement):
        return (
            np.nan * np.ones_like(measurement.eigvals())
            if isinstance(measurement, ProbabilityMP)
            else np.nan
        )

    interface = qml.math.get_deep_interface(circuit.data)
    interface = "numpy" if interface == "builtins" else interface

    all_mcms = [op for op in aux_tapes[0].operations if is_mcm(op)]
    n_mcms = len(all_mcms)
    post_process_tape = qml.tape.QuantumScript(
        aux_tapes[0].operations,
        aux_tapes[0].measurements[0:-n_mcms],
        shots=aux_tapes[0].shots,
        trainable_params=aux_tapes[0].trainable_params,
    )
    single_measurement = (
        len(post_process_tape.measurements) == 0 and len(aux_tapes[0].measurements) == 1
    )
    mcm_samples = qml.math.array(
        [[res] if single_measurement else res[-n_mcms::] for res in results], like=interface
    )
    # Can't use boolean dtype array with tf, hence why conditionally setting items to 0 or 1
    has_postselect = qml.math.array(
        [[int(op.postselect is not None) for op in all_mcms]], like=interface
    )
    postselect = qml.math.array(
        [[0 if op.postselect is None else op.postselect for op in all_mcms]], like=interface
    )
    is_valid = qml.math.all(mcm_samples * has_postselect == postselect, axis=1)
    has_valid = qml.math.any(is_valid)
    mid_meas = [op for op in circuit.operations if is_mcm(op)]
    mcm_samples = [mcm_samples[:, i : i + 1] for i in range(n_mcms)]
    mcm_samples = dict((k, v) for k, v in zip(mid_meas, mcm_samples))

    normalized_meas = []
    m_count = 0
    for m in circuit.measurements:
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise TypeError(
                f"Native mid-circuit measurement mode does not support {type(m).__name__} measurements."
            )
        if interface != "jax" and m.mv and not has_valid:
            meas = measurement_with_no_shots(m)
        elif m.mv:
            meas = gather_mcm(m, mcm_samples, is_valid)
        elif interface != "jax" and not has_valid:
            meas = measurement_with_no_shots(m)
            m_count += 1
        else:
            result = [res[m_count] for res in results]
            if not isinstance(m, CountsMP):
                # We don't need to cast to arrays when using qml.counts. qml.math.array is not viable
                # as it assumes all elements of the input are of builtin python types and not belonging
                # to any particular interface
                result = qml.math.stack(result, like=interface)
            meas = gather_non_mcm(m, result, is_valid)
            m_count += 1
        if isinstance(m, SampleMP):
            meas = qml.math.squeeze(meas)
        normalized_meas.append(meas)

    return tuple(normalized_meas) if len(normalized_meas) > 1 else normalized_meas[0]


def gather_non_mcm(measurement, samples, is_valid):
    """Combines, gathers and normalizes several measurements with trivial measurement values.

    Args:
        measurement (MeasurementProcess): measurement
        samples (TensorLike): Post-processed measurement samples
        is_valid (TensorLike): Boolean array with the same shape as ``samples`` where the value at
            each index specifies whether or not the respective sample is valid.

    Returns:
        TensorLike: The combined measurement outcome
    """
    if isinstance(measurement, CountsMP):
        tmp = Counter()
        for i, d in enumerate(samples):
            tmp.update(
                dict((k if isinstance(k, str) else float(k), v * is_valid[i]) for k, v in d.items())
            )
        tmp = Counter({k: v for k, v in tmp.items() if v > 0})
        return dict(sorted(tmp.items()))
    if isinstance(measurement, ExpectationMP):
        return qml.math.sum(samples * is_valid) / qml.math.sum(is_valid)
    if isinstance(measurement, ProbabilityMP):
        return qml.math.sum(samples * is_valid.reshape((-1, 1)), axis=0) / qml.math.sum(is_valid)
    if isinstance(measurement, SampleMP):
        is_interface_jax = qml.math.get_deep_interface(is_valid) == "jax"
        if is_interface_jax and samples.ndim == 2:
            is_valid = is_valid.reshape((-1, 1))
        return (
            qml.math.where(is_valid, samples, fill_in_value)
            if is_interface_jax
            else samples[is_valid]
        )
    # VarianceMP
    expval = qml.math.sum(samples * is_valid) / qml.math.sum(is_valid)
    return qml.math.sum((samples - expval) ** 2 * is_valid) / qml.math.sum(is_valid)


def gather_mcm(measurement, samples, is_valid):
    """Combines, gathers and normalizes several measurements with non-trivial measurement values.

    Args:
        measurement (MeasurementProcess): measurement
        samples (List[dict]): Mid-circuit measurement samples
        is_valid (TensorLike): Boolean array with the same shape as ``samples`` where the value at
            each index specifies whether or not the respective sample is valid.

    Returns:
        TensorLike: The combined measurement outcome
    """
    interface = qml.math.get_deep_interface(is_valid)
    mv = measurement.mv
    # The following block handles measurement value lists, like ``qml.counts(op=[mcm0, mcm1, mcm2])``.
    if isinstance(measurement, (CountsMP, ProbabilityMP, SampleMP)) and isinstance(mv, Sequence):
        mcm_samples = [m.concretize(samples) for m in mv]
        mcm_samples = qml.math.concatenate(mcm_samples, axis=1)
        if isinstance(measurement, ProbabilityMP):
            values = [list(m.branches.values()) for m in mv]
            values = list(itertools.product(*values))
            values = [qml.math.array([v], like=interface) for v in values]
            counts = [
                qml.math.sum(qml.math.all(mcm_samples == v, axis=1) * is_valid) for v in values
            ]
            counts = qml.math.array(counts, like=interface)
            return counts / qml.math.sum(counts)
        if isinstance(measurement, CountsMP):
            mcm_samples = [{"".join(str(int(v)) for v in tuple(s)): 1} for s in mcm_samples]
        return gather_non_mcm(measurement, mcm_samples, is_valid)
    mcm_samples = qml.math.ravel(qml.math.array(mv.concretize(samples), like=interface))
    if isinstance(measurement, ProbabilityMP):
        counts = [qml.math.sum((mcm_samples == v) * is_valid) for v in list(mv.branches.values())]
        counts = qml.math.array(counts, like=interface)
        return counts / qml.math.sum(counts)
    if isinstance(measurement, CountsMP):
        mcm_samples = [{float(s): 1} for s in mcm_samples]
    return gather_non_mcm(measurement, mcm_samples, is_valid)
