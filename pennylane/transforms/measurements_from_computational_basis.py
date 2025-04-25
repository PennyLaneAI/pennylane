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

import pennylane as qml
from pennylane.transforms import transform


@transform
def measurements_from_samples(tape):
    r"""Quantum function transform that replaces all terminal measurements with a single sample 
    measurement.

    .. note::
        This transform diagonalizes all the operations on the tape. An error will
        be raised if non-commuting terms are encountered. To avoid non-commuting
        terms in circuit measurements, the :func:`split_non_commuting <pennylane.transforms.split_non_commuting>`
        transform can be applied.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    Consider the tape:

    >>> ops = [qml.X(0), qml.RY(1.23, 1)]
    >>> measurements = [qml.expval(qml.Y(0)), qml.probs(wires=[1])]
    >>> tape = qml.tape.QuantumScript(ops, measurements, shots=10)

    We can apply the transform to diagonalize and convert the two measurements to a single sample:

    >>> (new_tape, ), fn = qml.transforms.measurements_from_samples(tape)
    >>> new_tape.measurements
    [sample(wires=[0, 1])]

    The tape operations now include diagonalizing gates.

    >>> new_tape.operations
    [X(0), RY(1.23, wires=[1]), RX(1.5707963267948966, wires=[0])]

    Executing the tape returns samples that can be post-processed to get the originally requested measurements:

    >>> dev = qml.device("default.qubit")
    >>> dev.execute(new_tape)
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
    (-0.2, array([0.6, 0.4]))
    """

    for mp in tape.measurements:
        if not mp.obs and not mp.wires:
            raise RuntimeError(
                "Please apply validate_device_wires transform before measurements_from_samples"
            )

    diagonalized_tape, measured_wires = _get_diagonalized_tape_and_wires(tape)
    new_tape = diagonalized_tape.copy(measurements=[qml.sample(wires=measured_wires)])

    def postprocessing_fn(results):
        """A processing function to get measurement values from samples."""
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

    return [new_tape], postprocessing_fn


@transform
def measurements_from_counts(tape):
    r"""Quantum function transform that replaces all terminal measurements with a single counts 
    measurement.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    Consider the tape:

    >>> ops = [qml.X(0), qml.RY(1.23, 1)]
    >>> measurements = [qml.expval(qml.Y(0)), qml.probs(wires=[1])]
    >>> tape = qml.tape.QuantumScript(ops, measurements, shots=10)

    We can apply the transform to diagonalize and convert the two measurements to a single sample:

    >>> (new_tape, ), fn = qml.transforms.measurements_from_counts(tape)
    >>> new_tape.measurements
    [CountsMP(wires=[0, 1], all_outcomes=False)]

    The tape operations now include diagonalizing gates.

    >>> new_tape.operations
    [X(0), RY(1.23, wires=[1]), RX(1.5707963267948966, wires=[0])]

    Executing the tape returns samples that can be post-processed to get the originally requested measurements:

    >>> dev = qml.device("default.qubit")
    >>> dev.execute(new_tape)
    {'00': 4, '01': 2, '10': 2, '11': 2}

    >>> fn((res,))
    (-0.19999999999999996, array([0.7, 0.3]))
    """

    for mp in tape.measurements:
        if not mp.obs and not mp.wires:
            raise RuntimeError(
                "Please apply validate_device_wires transform before measurements_from_samples"
            )

    diagonalized_tape, measured_wires = _get_diagonalized_tape_and_wires(tape)
    new_tape = diagonalized_tape.copy(measurements=[qml.counts(wires=measured_wires)])

    def postprocessing_fn(results):
        """A processing function to get measurement values from counts."""
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

    return [new_tape], postprocessing_fn


def _get_diagonalized_tape_and_wires(tape):
    """Apply the diagonalize_measurements transform to the tape and extract a list of
    all the wires present in the measurements"""

    (diagonalized_tape,), _ = qml.transforms.diagonalize_measurements(tape)

    measured_wires = set()
    for m in diagonalized_tape.measurements:
        measured_wires.update(
            m.wires.tolist()
        )  # ToDo: add test confirming that the wire order can be weird and this still works
    measured_wires = list(measured_wires)

    return diagonalized_tape, measured_wires


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
