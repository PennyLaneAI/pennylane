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
"""Simulate a quantum script."""
# pylint: disable=protected-access
from collections import Counter
from typing import Optional, Sequence

from numpy.random import default_rng
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
from pennylane.ops import Conditional
from pennylane.typing import Result

from .initialize_state import create_initial_state
from .apply_operation import apply_operation, apply_mid_measure
from .measure import measure
from .sampling import measure_with_samples


INTERFACE_TO_LIKE = {
    # map interfaces known by autoray to themselves
    None: None,
    "numpy": "numpy",
    "autograd": "autograd",
    "jax": "jax",
    "torch": "torch",
    "tensorflow": "tensorflow",
    # map non-standard interfaces to those known by autoray
    "auto": None,
    "scipy": "numpy",
    "jax-jit": "jax",
    "jax-python": "jax",
    "JAX": "jax",
    "pytorch": "torch",
    "tf": "tensorflow",
    "tensorflow-autograph": "tensorflow",
    "tf-autograph": "tensorflow",
}


class _FlexShots(qml.measurements.Shots):
    """Shots class that allows zero shots."""

    # pylint: disable=super-init-not-called
    def __init__(self, shots=None):
        if isinstance(shots, int):
            self.total_shots = shots
            self.shot_vector = (qml.measurements.ShotCopies(shots, 1),)
        else:
            self.__all_tuple_init__([s if isinstance(s, tuple) else (s, 1) for s in shots])

        self._frozen = True


def _postselection_postprocess(state, is_state_batched, shots):
    """Update state after projector is applied."""
    if is_state_batched:
        raise ValueError(
            "Cannot postselect on circuits with broadcasting. Use the "
            "qml.transforms.broadcast_expand transform to split a broadcasted "
            "tape into multiple non-broadcasted tapes before executing if "
            "postselection is used."
        )

    # The floor function is being used here so that a norm very close to zero becomes exactly
    # equal to zero so that the state can become invalid. This way, execution can continue, and
    # bad postselection gives results that are invalid rather than results that look valid but
    # are incorrect.
    norm = qml.math.norm(state)

    if not qml.math.is_abstract(state) and qml.math.allclose(norm, 0.0):
        norm = 0.0

    if shots:
        # Clip the number of shots using a binomial distribution using the probability of
        # measuring the postselected state.
        postselected_shots = (
            [np.random.binomial(s, float(norm**2)) for s in shots]
            if not qml.math.is_abstract(norm)
            else shots
        )

        # _FlexShots is used here since the binomial distribution could result in zero
        # valid samples
        shots = _FlexShots(postselected_shots)

    state = state / norm
    return state, shots


def get_final_state(circuit, debugger=None, interface=None):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(sorted(circuit.op_wires), prep, like=INTERFACE_TO_LIKE[interface])

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = bool(prep and prep.batch_size is not None)
    measurement_values = {}
    for op in circuit.operations[bool(prep) :]:
        if isinstance(op, Conditional):
            if not op.meas_val.concretize(measurement_values):
                continue
            op = op.then_op
        if isinstance(op, MidMeasureMP):
            state, measurement_values[op.hash] = apply_mid_measure(
                op, state, is_state_batched=is_state_batched, debugger=debugger
            )
        else:
            state = apply_operation(op, state, is_state_batched=is_state_batched, debugger=debugger)
        # Handle postselection on mid-circuit measurements
        if isinstance(op, qml.Projector):
            state, circuit._shots = _postselection_postprocess(
                state, is_state_batched, circuit.shots
            )

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or (op.batch_size is not None)

    for _ in range(len(circuit.wires) - len(circuit.op_wires)):
        # if any measured wires are not operated on, we pad the state with zeros.
        # We know they belong at the end because the circuit is in standard wire-order
        state = qml.math.stack([state, qml.math.zeros_like(state)], axis=-1)

    return state, is_state_batched, measurement_values


def measure_final_state(circuit, state, is_state_batched, rng=None, prng_key=None) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
            will be for sampling.

    Returns:
        Tuple[TensorLike]: The measurement results
    """

    circuit = circuit.map_to_standard_wires()

    if not circuit.shots:
        # analytic case

        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state, is_state_batched=is_state_batched)

        return tuple(
            measure(mp, state, is_state_batched=is_state_batched) for mp in circuit.measurements
        )

    # finite-shot case

    rng = default_rng(rng)
    results = measure_with_samples(
        circuit.measurements,
        state,
        shots=circuit.shots,
        is_state_batched=is_state_batched,
        rng=rng,
        prng_key=prng_key,
    )

    if len(circuit.measurements) == 1:
        if circuit.shots.has_partitioned_shots:
            return tuple(res[0] for res in results)

        return results[0]

    return results


# pylint: disable=too-many-arguments
def simulate(
    circuit: qml.tape.QuantumScript,
    rng=None,
    prng_key=None,
    debugger=None,
    interface=None,
    state_cache: Optional[dict] = None,
) -> Result:
    """Simulate a single quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        state_cache=None (Optional[dict]): A dictionary mapping the hash of a circuit to the pre-rotated state. Used to pass the state between forward passes and vjp calculations.

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    if circuit.shots and has_mid_circuit_measurements(circuit):
        return simulate_native_mcm(
            circuit, rng=rng, prng_key=prng_key, debugger=debugger, interface=interface
        )
    state, is_state_batched, _ = get_final_state(circuit, debugger=debugger, interface=interface)
    if state_cache is not None:
        state_cache[circuit.hash] = state
    return measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=prng_key)


# pylint: disable=too-many-arguments
def simulate_native_mcm(
    circuit: qml.tape.QuantumScript,
    rng=None,
    prng_key=None,
    debugger=None,
    interface=None,
) -> Result:
    """Simulate a single quantum script with native mid-circuit measurements.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        state_cache=None (Optional[dict]): A dictionary mapping the hash of a circuit to the pre-rotated state. Used to pass the state between forward passes and vjp calculations.

    Returns:
        tuple(TensorLike): The results of the simulation
    """
    if circuit.shots.has_partitioned_shots:
        results = []
        for s in circuit.shots:
            aux_circuit = circuit.copy()
            aux_circuit._shots = qml.measurements.Shots(s)
            results.append(simulate(aux_circuit, rng, prng_key, debugger, interface))
        return tuple(results)
    aux_circuit = init_auxiliary_circuit(circuit)
    all_shot_meas, count = None, 0
    while all_shot_meas is None:
        count += 1
        all_shot_meas, mcm_values_dict = simulate_one_shot_native_mcm(
            aux_circuit, rng, prng_key, debugger, interface
        )
    list_mcm_values_dict = [mcm_values_dict]
    for _ in range(circuit.shots.total_shots - count):
        one_shot_meas, mcm_values_dict = simulate_one_shot_native_mcm(
            aux_circuit, rng, prng_key, debugger, interface
        )
        if one_shot_meas is None:
            continue
        all_shot_meas = accumulate_native_mcm(aux_circuit, all_shot_meas, one_shot_meas)
        list_mcm_values_dict.append(mcm_values_dict)
    return parse_native_mid_circuit_measurements(circuit, all_shot_meas, list_mcm_values_dict)


def init_auxiliary_circuit(circuit: qml.tape.QuantumScript):
    """Creates an auxiliary circuit to perform one-shot mid-circuit measurement calculations.

    Measurements are replaced by SampleMP measurements on wires and observables found in the
    original measurements.

    Args:
        circuit (QuantumTape): The original QuantumScript

    Returns:
        QuantumScript: A copy of the circuit with modified measurements
    """
    aux_circuit = circuit.copy()
    aux_circuit._shots = qml.measurements.Shots(1)
    idx_sample = find_measurement_values(circuit)
    for i in reversed(idx_sample):
        aux_circuit._measurements.pop(i)
    for i, m in enumerate(circuit.measurements):
        if isinstance(m, VarianceMP) and m.mv is None:
            aux_circuit._measurements[i] = SampleMP(obs=m.obs)
    return aux_circuit


def simulate_one_shot_native_mcm(
    circuit: qml.tape.QuantumScript,
    rng=None,
    prng_key=None,
    debugger=None,
    interface=None,
) -> Result:
    """Simulate a single shot of a single quantum script with native mid-circuit measurements.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation
        dict: The mid-circuit measurement results of the simulation
    """
    state, is_state_batched, mcm_dict = get_final_state(
        circuit, debugger=debugger, interface=interface
    )
    if not np.allclose(np.linalg.norm(state), 1.0):
        return None, mcm_dict
    return (
        measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=prng_key),
        mcm_dict,
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
    new_shot_meas = [None] * len(circuit.measurements)
    if not isinstance(all_shot_meas, Sequence):
        return accumulate_native_mcm(circuit, [all_shot_meas], one_shot_meas)
    if not isinstance(one_shot_meas, Sequence):
        return accumulate_native_mcm(circuit, all_shot_meas, [one_shot_meas])
    for i, m in enumerate(circuit.measurements):
        if isinstance(m, CountsMP):
            tmp = Counter(all_shot_meas[i])
            tmp.update(Counter(one_shot_meas[i]))
            new_shot_meas[i] = tmp
        elif isinstance(m, (ExpectationMP, ProbabilityMP)):
            new_shot_meas[i] = all_shot_meas[i] + one_shot_meas[i]
        elif isinstance(m, SampleMP):
            if not isinstance(all_shot_meas[i], (list, tuple)):
                new_shot_meas[i] = [all_shot_meas[i]]
            else:
                new_shot_meas[i] = all_shot_meas[i]
            new_shot_meas[i].append(one_shot_meas[i])
        else:
            raise TypeError(f"Unsupported measurement of {type(m)}.")
    return new_shot_meas


def has_mid_circuit_measurements(
    circuit: qml.tape.QuantumScript,
):
    """Returns True if the circuit contains a MidMeasureMP object and False otherwise.

    Args:
        circuit (QuantumTape): A QuantumScript

    Returns:
        bool: Whether the circuit contains a MidMeasureMP object
    """
    return any(isinstance(op, MidMeasureMP) for op in circuit._ops)


def has_measurement_values(measurement):
    """Returns True if a measurement has a non-trivial measurement value and False otherwise.

    Args:
        measurement (MeasurementProcess): A QuantumScript

    Returns:
        bool: Whether the measurement contains a non-trivial measurement value
    """
    return measurement.mv is not None


def find_measurement_values(
    circuit: qml.tape.QuantumScript,
):
    """Returns the indices of measurements with a non-trivial measurement value.

    Args:
        circuit (QuantumTape): A QuantumScript

    Returns:
        List[int]: Indices of measurements with a non-trivial measurement value.
    """
    return [i for i, m in enumerate(circuit.measurements) if has_measurement_values(m)]


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
    idx_sample = find_measurement_values(circuit)
    normalized_meas = [None] * len(circuit.measurements)
    for i, m in enumerate(circuit.measurements):
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise ValueError(
                f"Native mid-circuit measurement mode does not support {type(m)} measurements."
            )
        if i in idx_sample:
            normalized_meas[i] = gather_mcm(m, mcm_shot_meas)
        else:
            normalized_meas[i] = gather_non_mcm(m, all_shot_meas[i], mcm_shot_meas)
        if isinstance(m, SampleMP):
            normalized_meas[i] = qml.math.squeeze(normalized_meas[i])
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
        new_meas = dict(sorted(measurement.items()))
    elif isinstance(circuit_measurement, (ExpectationMP, ProbabilityMP)):
        new_meas = measurement / len(samples)
    elif isinstance(circuit_measurement, SampleMP):
        new_meas = np.squeeze(np.concatenate(tuple(s.reshape(1, -1) for s in measurement)))
    elif isinstance(circuit_measurement, VarianceMP):
        new_meas = qml.math.var(np.concatenate(tuple(s.ravel() for s in measurement)))
    else:
        raise ValueError(
            f"Native mid-circuit measurement mode does not support {type(circuit_measurement)} measurements."
        )
    return new_meas


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
        if isinstance(measurement, (ExpectationMP, VarianceMP)):
            mcm_samples = mcm_samples.ravel()
        wires = qml.wires.Wires(0)
        meas_tmp = measurement.__class__(wires=wires)
    new_measurement = meas_tmp.process_samples(mcm_samples, wire_order=wires)
    if isinstance(measurement, CountsMP) and not use_as_is:
        new_measurement = dict((int(x), y) for x, y in new_measurement.items())
    return new_measurement
