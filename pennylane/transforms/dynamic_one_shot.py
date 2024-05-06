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
import warnings

# pylint: disable=import-outside-toplevel
from collections import Counter
from itertools import compress
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
            return qml.expval(op=m0)

    The ``qml.dynamic_one_shot`` decorator prompts the QNode to perform a hundred one-shot
    calculations, where in each calculation the ``qml.measure`` operations dynamically
    measures the 0-wire and collapse the state vector stochastically. This transforms
    contrasts with ``qml.defer_measurements``, which instead introduces an extra wire
    for each mid-circuit measurement. The ``qml.dynamic_one_shot`` transform is favorable in the
    few-shots several-mid-circuit-measurement limit, whereas ``qml.defer_measurements`` is favorable
    in the opposite limit.
    """

    if not any(isinstance(o, MidMeasureMP) for o in tape.operations):
        return (tape,), null_postprocessing

    for m in tape.measurements:
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise TypeError(
                f"Native mid-circuit measurement mode does not support {type(m).__name__} "
                "measurements."
            )

    if not tape.shots:
        raise qml.QuantumFunctionError("dynamic_one_shot is only supported with finite shots.")

    samples_present = any(isinstance(mp, SampleMP) for mp in tape.measurements)
    postselect_present = any(
        op.postselect is not None for op in tape.operations if isinstance(op, MidMeasureMP)
    )
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
    # Shape of output_tapes is (batch_size * total_shots,) with broadcasting,
    # and (total_shots,) otherwise
    output_tapes = [at for at in aux_tapes for _ in range(tape.shots.total_shots)]

    def processing_fn(results, has_partitioned_shots=None, batched_results=None):
        if batched_results is None and batch_size is not None:
            # If broadcasting, recursively process the results for each batch. For each batch
            # there are tape.shots.total_shots results. The length of the first axis of final_results
            # will be batch_size.
            results = list(results)
            final_results = []
            for _ in range(batch_size):
                final_results.append(
                    processing_fn(results[0 : tape.shots.total_shots], batched_results=False)
                )
                del results[0 : tape.shots.total_shots]
            return broadcast_fn(final_results)

        if has_partitioned_shots is None and tape.shots.has_partitioned_shots:
            # If using shot vectors, recursively process the results for each shot bin. The length
            # of the first axis of final_results will be the length of the shot vector.
            results = list(results)
            final_results = []
            for s in tape.shots:
                final_results.append(
                    processing_fn(results[0:s], has_partitioned_shots=False, batched_results=False)
                )
                del results[0:s]
            return tuple(final_results)
        all_mcms = [op for op in aux_tapes[0].operations if isinstance(op, MidMeasureMP)]
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
        mcm_samples = np.zeros((len(results), n_mcms), dtype=np.int64)
        for i, res in enumerate(results):
            mcm_samples[i, :] = [res] if single_measurement else res[-n_mcms::]
        mcm_mask = qml.math.all(mcm_samples != -1, axis=1)
        mcm_samples = mcm_samples[mcm_mask, :]
        results = list(compress(results, mcm_mask))

        # The following code assumes no broadcasting and no shot vectors. The above code should
        # handle those cases
        all_shot_meas, list_mcm_values_dict, valid_shots = None, [], 0
        for i, res in enumerate(results):
            samples = [res] if single_measurement else res[-n_mcms::]
            valid_shots += 1
            mcm_values_dict = dict((k, v) for k, v in zip(all_mcms, samples))
            if len(post_process_tape.measurements) == 0:
                one_shot_meas = []
            elif len(post_process_tape.measurements) == 1:
                one_shot_meas = res[0]
            else:
                one_shot_meas = res[0:-n_mcms]
            all_shot_meas = accumulate_native_mcm(post_process_tape, all_shot_meas, one_shot_meas)
            list_mcm_values_dict.append(mcm_values_dict)
        if not valid_shots:
            warnings.warn(
                "All shots were thrown away as invalid. This can happen for example when "
                "post-selecting the 1-branch of a 0-state. Make sure your circuit has some "
                "probability of producing a valid shot.",
                UserWarning,
            )
        return parse_native_mid_circuit_measurements(tape, all_shot_meas, list_mcm_values_dict)

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
    # The following block handles measurement value lists, like ``qml.counts(op=[mcm0, mcm1, mcm2])``.
    if isinstance(measurement, (CountsMP, ProbabilityMP, SampleMP)) and isinstance(mv, Sequence):
        wires = qml.wires.Wires(range(len(mv)))
        mcm_samples = list(
            np.array([m.concretize(dct) for dct in samples]).reshape((-1, 1)) for m in mv
        )
        mcm_samples = np.concatenate(mcm_samples, axis=1)
        meas_tmp = measurement.__class__(wires=wires)
        return meas_tmp.process_samples(mcm_samples, wire_order=wires)
    if isinstance(measurement, ProbabilityMP):
        mcm_samples = [dct[mv.measurements[0]] for dct in samples]
        use_as_is = True
    else:
        mcm_samples = [mv.concretize(dct) for dct in samples]
        use_as_is = mv.branches == {(0,): 0, (1,): 1}
    mcm_samples = np.array(mcm_samples).reshape((len(samples), 1))
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
        keys = np.array(list(new_measurement.keys())).astype(mcm_samples.dtype)
        new_measurement = dict(sorted((x, y) for x, y in zip(keys, new_measurement.values())))
    return new_measurement
