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
"""Transform to diagonalize measurements on a tape and transform the measurements
to come from readout in the computational basis returned as samples or counts."""

from functools import partial

import pennylane as qml
from pennylane.transforms import transform


@transform
def measurements_from_computational_basis(tape, from_counts=False):
    r"""Replace all measurements from a tape with sample measurements, and adds postprocessing
    functions for each original measurement.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        from_counts (bool): whether to convert to and from counts (instead of samples).
            Defaults to False.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.
    """

    for mp in tape.measurements:
        if not mp.obs and not mp.wires:
            raise RuntimeError(
                "Please apply validate_device_wires transform before measurements_from_computational_basis"
            )

    new_operations, measured_wires = _diagonalize_measurements(tape)

    if from_counts:
        new_tape = type(tape)(new_operations, [qml.counts(wires=measured_wires)], shots=tape.shots)
        return [new_tape], partial(postprocessing_counts, tape=tape, measured_wires=measured_wires)

    new_tape = type(tape)(new_operations, [qml.sample(wires=measured_wires)], shots=tape.shots)
    return [new_tape], partial(postprocessing_samples, tape=tape, measured_wires=measured_wires)


def _diagonalize_measurements(tape):
    """Takes a tape and returns the information needed to create a new tape based on
    diagonalization and readout in the measurement basis.

    Args:
        tape (QuantumTape): A quantum circuit.

    Returns:
        new_operations (list): The original operations, plus the diagonalizing gates for the circuit
        measured_wires (list): A list of all wires that are measured on the tape

    """

    (diagonalized_tape,), _ = qml.transforms.diagonalize_measurements(tape)

    measured_wires = set()
    for m in diagonalized_tape.measurements:
        measured_wires.update(m.wires.tolist())

    return diagonalized_tape.operations, list(measured_wires)


def postprocessing_samples(results, tape, measured_wires):
    """A processing function to get expecation values from samples."""
    samples = results[0]
    results_processed = []
    for m in tape.measurements:
        if len(tape.shots.shot_vector) > 1:
            res = tuple(m.process_samples(_unsqueezed(s), measured_wires) for s in samples)
        else:
            res = m.process_samples(_unsqueezed(samples), measured_wires)
        results_processed.append(res)

    if len(tape.measurements) == 1:
        results_processed = results_processed[0]
    else:
        results_processed = tuple(results_processed)
    return results_processed


def _unsqueezed(samples):
    """If the samples have been squeezed to remove the 'extra' dimension in the case where
    shots=1 or wires=1, unsqueeze to restore the raw samples format expected by mp.process_samples
    """

    # Before we start post-processing the transforms, we squeeze out the extra dimension in samples
    # where wire=1 or shots=1 (this is not done in Catalyst). This makes it incompatible with
    # the process_samples method on the measurement processes.
    # this would be fixed by waiting to squeeze until the very end before returning
    # so that we don't have to handle special cases regarding shapes in our processing pipeline

    if len(samples.shape) == 1:
        samples = qml.math.array([[s] for s in samples], like=samples)
    return samples


def postprocessing_counts(results, tape, measured_wires):
    """A processing function to get expecation values from samples."""
    samples = results[0]
    results_processed = []
    for m in tape.measurements:
        if len(tape.shots.shot_vector) > 1:
            res = tuple(m.process_counts(s, measured_wires) for s in samples)
        else:
            res = m.process_counts(samples, measured_wires)
        results_processed.append(res)

    if len(tape.measurements) == 1:
        results_processed = results_processed[0]
    else:
        results_processed = tuple(results_processed)
    return results_processed
