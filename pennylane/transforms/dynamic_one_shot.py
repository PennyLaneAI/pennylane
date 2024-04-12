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

from .core import transform


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def dynamic_one_shot(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
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
            return measure_f(op=m0)

    The ``qml.dynamic_one_shot`` decorator prompts the QNode to perform a hundred one-shot
    calculations, where in each calculation the ``qml.measure`` operations dynamically
    measures the 0-wire and collapse the state vector stochastically. This transforms
    contrasts with ``qml.defer_measurements``, which instead introduces an extra wire
    for each mid-circuit measurement. The ``qml.dynamic_one_shot`` transform is favorable in the few-shots
    several-mid-circuit-measurement limit, whereas ``qml.defer_measurements`` is favorable
    in the opposite limit.
    """

    if not any(isinstance(o, MidMeasureMP) for o in tape.operations):
        return (tape,), null_postprocessing

    for m in tape.measurements:
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise TypeError(
                f"Native mid-circuit measurement mode does not support {type(m).__name__} measurements."
            )

    aux_tape = init_auxiliary_tape(tape)
    output_tapes = [aux_tape] * tape.shots.total_shots

    def processing_fn(results, has_partitioned_shots=None):
        if has_partitioned_shots is None and tape.shots.has_partitioned_shots:
            results = list(results)
            final_results = []
            for s in tape.shots:
                final_results.append(processing_fn(results[0:s], has_partitioned_shots=False))
                del results[0:s]
            return tuple(final_results)
        return parse_native_mid_circuit_measurements(tape, aux_tape, results)

    return output_tapes, processing_fn


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
        support_mcms = support_mcms or isinstance(
            qnode.device, qml.devices.default_qubit.DefaultQubit
        )
        if not support_mcms:
            raise TypeError(
                f"Device {qnode.device.name} does not support mid-circuit measurements natively, and hence it does not support the dynamic_one_shot transform. `default.qubit` and `lightning.qubit` currently support mid-circuit measurements and the dynamic_one_shot transform."
            )
    tkwargs.setdefault("device", qnode.device)
    return self.default_qnode_transform(qnode, targs, tkwargs)


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
        if isinstance(op, MidMeasureMP):
            new_measurements.append(qml.sample(MeasurementValue([op], lambda res: res)))

    return qml.tape.QuantumScript(
        circuit.operations, new_measurements, shots=1, trainable_params=circuit.trainable_params
    )


def accumulate_native_mcm(circuit: qml.tape.QuantumScript, all_shot_meas, one_shot_meas):
    """Incorporates new measurements in current measurement sequence.

    Args:
        circuit (QuantumTape): A one-shot (auxiliary) QuantumScript
        all_shot_meas (Sequence[Any]): List of accumulated measurement results
        one_shot_meas (Sequence[Any]): List of measurement results

    Returns:
        tuple(TensorLike): The results of the simulation
    """
    if len(circuit.measurements) == 1:
        one_shot_meas = [one_shot_meas]
    if all_shot_meas is None:
        new_shot_meas = list(one_shot_meas)
        for i, (m, s) in enumerate(zip(circuit.measurements, new_shot_meas)):
            if isinstance(m, SampleMP) and isinstance(s, np.ndarray):
                new_shot_meas[i] = [s]
        return new_shot_meas
    new_shot_meas = all_shot_meas
    for i, m in enumerate(circuit.measurements):
        if isinstance(m, CountsMP):
            tmp = Counter(all_shot_meas[i])
            tmp.update(Counter(one_shot_meas[i]))
            new_shot_meas[i] = tmp
        elif isinstance(m, (ExpectationMP, ProbabilityMP)):
            new_shot_meas[i] = all_shot_meas[i] + one_shot_meas[i]
        elif isinstance(m, SampleMP):
            new_shot_meas[i].append(one_shot_meas[i])
        else:
            raise TypeError(
                f"Native mid-circuit measurement mode does not support {type(m).__name__} measurements."
            )
    return new_shot_meas


def parse_native_mid_circuit_measurements(
    circuit: qml.tape.QuantumScript, aux_circuit: qml.tape.QuantumScript, results
):
    """Combines, gathers and normalizes the results of native mid-circuit measurement runs.

    Args:
        circuit (QuantumTape): A one-shot (auxiliary) QuantumScript
        all_shot_meas (Sequence[Any]): List of accumulated measurement results
        mcm_shot_meas (Sequence[dict]): List of dictionaries containing the mid-circuit measurement results of each shot

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

    n_mcms = sum(isinstance(op, MidMeasureMP) for op in circuit.operations)
    sca_results = len(aux_circuit.measurements) - n_mcms == 0 and len(aux_circuit.measurements) == 1
    mcm_samples = qml.math.array(
        [[res] if sca_results else res[-n_mcms::] for res in results], like=interface
    )
    is_valid = qml.math.all(mcm_samples != -1, axis=1)
    has_valid = qml.math.any(is_valid)
    mid_meas = [op for op in circuit.operations if isinstance(op, MidMeasureMP)]
    mcm_samples = [mcm_samples[:, i : i + 1] for i in range(n_mcms)]
    mcm_samples = dict((k, v) for k, v in zip(mid_meas, mcm_samples))

    normalized_meas = []
    for i, m in enumerate(circuit.measurements):
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise TypeError(
                f"Native mid-circuit measurement mode does not support {type(m).__name__} measurements."
            )
        if m.mv and not has_valid:
            meas = measurement_with_no_shots(m)
        elif m.mv:
            meas = gather_mcm(m, mcm_samples, is_valid)
        elif interface != "jax" and not has_valid:
            meas = measurement_with_no_shots(m)
        else:
            result = qml.math.array([res[i] for res in results], like=interface)
            meas = gather_non_mcm(m, result, is_valid)
        if isinstance(m, SampleMP):
            meas = qml.math.squeeze(meas)
        normalized_meas.append(meas)

    return tuple(normalized_meas) if len(normalized_meas) > 1 else normalized_meas[0]


def gather_non_mcm(circuit_measurement, measurement, is_valid):
    """Combines, gathers and normalizes several measurements with trivial measurement values.

    Args:
        circuit_measurement (MeasurementProcess): measurement
        measurement (TensorLike): measurement results
        samples (List[dict]): Mid-circuit measurement samples

    Returns:
        TensorLike: The combined measurement outcome
    """
    if isinstance(circuit_measurement, CountsMP):
        tmp = Counter()
        for i, d in enumerate(measurement):
            tmp.update(dict((k, v * is_valid[i]) for k, v in d.items()))
        tmp = Counter({k: v for k, v in tmp.items() if v > 0})
        return dict(sorted(tmp.items()))
    if isinstance(circuit_measurement, ExpectationMP):
        return qml.math.sum(measurement * is_valid) / qml.math.sum(is_valid)
    if isinstance(circuit_measurement, ProbabilityMP):
        return qml.math.sum(measurement * is_valid.reshape((-1, 1)), axis=0) / qml.math.sum(
            is_valid
        )
    if isinstance(circuit_measurement, SampleMP):
        if measurement.ndim == 2:
            is_valid = is_valid.reshape((-1, 1))
        return qml.math.where(is_valid, measurement, -123456)
    # VarianceMP
    expval = qml.math.sum(measurement * is_valid) / qml.math.sum(is_valid)
    return qml.math.sum((measurement - expval) ** 2 * is_valid) / qml.math.sum(is_valid)


def gather_mcm(measurement, samples, is_valid):
    """Combines, gathers and normalizes several measurements with non-trivial measurement values.

    Args:
        measurement (MeasurementProcess): measurement
        samples (List[dict]): Mid-circuit measurement samples

    Returns:
        TensorLike: The combined measurement outcome
    """
    mv = measurement.mv
    # The following block handles measurement value lists, like ``qml.counts(op=[mcm0, mcm1, mcm2])``.
    if isinstance(measurement, (CountsMP, ProbabilityMP, SampleMP)) and isinstance(mv, Sequence):
        wires = qml.wires.Wires(range(len(mv)))
        mcm_samples = [m.concretize(samples) for m in mv]
        mcm_samples = qml.math.concatenate(mcm_samples, axis=1)
        meas_tmp = measurement.__class__(wires=wires)
        return meas_tmp.process_samples(mcm_samples, wire_order=wires)
    if isinstance(measurement, ProbabilityMP):
        mcm_samples = qml.math.array([samples[mv.measurements[0]]])
        wires, meas_tmp = mv.wires, measurement
        probs = meas_tmp.process_samples(mcm_samples, wire_order=wires)
        return probs / qml.math.sum(is_valid) * is_valid.size
    mcm_samples = qml.math.array([mv.concretize(samples)]).ravel()
    if isinstance(measurement, CountsMP):
        mcm_samples = [{s: 1} for s in mcm_samples]
    return gather_non_mcm(measurement, mcm_samples, is_valid)
