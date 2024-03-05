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
import warnings

import numpy as np

import pennylane as qml
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
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
        all_shot_meas, list_mcm_values_dict, valid_shots = None, [], 0
        for res in results:
            one_shot_meas, mcm_values_dict = res
            if one_shot_meas is None:
                continue
            valid_shots += 1
            all_shot_meas = accumulate_native_mcm(aux_tape, all_shot_meas, one_shot_meas)
            list_mcm_values_dict.append(mcm_values_dict)
        if not valid_shots:
            warnings.warn(
                "All shots were thrown away as invalid. This can happen for example when post-selecting the 1-branch of a 0-state. Make sure your circuit has some probability of producing a valid shot.",
                UserWarning,
            )
        return parse_native_mid_circuit_measurements(tape, all_shot_meas, list_mcm_values_dict)

    return output_tapes, processing_fn


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
    circuit: qml.tape.QuantumScript, all_shot_meas, mcm_shot_meas
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

    normalized_meas = []
    for i, m in enumerate(circuit.measurements):
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise TypeError(
                f"Native mid-circuit measurement mode does not support {type(m).__name__} measurements."
            )
        if m.mv and not mcm_shot_meas:
            meas = measurement_with_no_shots(m)
        elif m.mv:
            meas = gather_mcm(m, mcm_shot_meas)
        elif not all_shot_meas:
            meas = measurement_with_no_shots(m)
        else:
            meas = gather_non_mcm(m, all_shot_meas[i], mcm_shot_meas)
        if isinstance(m, SampleMP):
            meas = qml.math.squeeze(meas)
        normalized_meas.append(meas)

    return tuple(normalized_meas) if len(normalized_meas) > 1 else normalized_meas[0]


def gather_non_mcm(circuit_measurement, measurement, samples):
    """Combines, gathers and normalizes several measurements with trivial measurement values.

    Args:
        circuit_measurement (MeasurementProcess): measurement
        measurement (TensorLike): measurement results
        samples (List[dict]): Mid-circuit measurement samples

    Returns:
        TensorLike: The combined measurement outcome
    """
    if isinstance(circuit_measurement, CountsMP):
        return dict(sorted(measurement.items()))
    if isinstance(circuit_measurement, (ExpectationMP, ProbabilityMP)):
        return measurement / len(samples)
    if isinstance(circuit_measurement, SampleMP):
        return np.squeeze(np.concatenate(tuple(s.reshape(1, -1) for s in measurement)))
    # VarianceMP
    return qml.math.var(np.concatenate(tuple(s.ravel() for s in measurement)))


def gather_mcm(measurement, samples):
    """Combines, gathers and normalizes several measurements with non-trivial measurement values.

    Args:
        measurement (MeasurementProcess): measurement
        samples (List[dict]): Mid-circuit measurement samples

    Returns:
        TensorLike: The combined measurement outcome
    """
    mv = measurement.mv
    if isinstance(measurement, (CountsMP, ProbabilityMP, SampleMP)) and isinstance(mv, Sequence):
        wires = qml.wires.Wires(range(len(mv)))
        mcm_samples = list(
            np.array([m.concretize(dct) for dct in samples]).reshape((-1, 1)) for m in mv
        )
        mcm_samples = np.concatenate(mcm_samples, axis=1)
        meas_tmp = measurement.__class__(wires=wires)
        return meas_tmp.process_samples(mcm_samples, wire_order=wires)
    mcm_samples = np.array([mv.concretize(dct) for dct in samples]).reshape((-1, 1))
    use_as_is = len(mv.measurements) == 1
    if use_as_is:
        wires, meas_tmp = mv.wires, measurement
    else:
        # For composite measurements, `mcm_samples` has one column but
        # `mv.wires` usually includes several wires. We therefore need to create a
        # single-wire measurement for `process_samples` to handle the conversion
        # correctly.
        if isinstance(measurement, (ExpectationMP, VarianceMP)):
            mcm_samples = mcm_samples.ravel()
        wires = qml.wires.Wires(0)
        meas_tmp = measurement.__class__(wires=wires)
    new_measurement = meas_tmp.process_samples(mcm_samples, wire_order=wires)
    if isinstance(measurement, CountsMP) and not use_as_is:
        new_measurement = dict(sorted((int(x, 2), y) for x, y in new_measurement.items()))
    return new_measurement
