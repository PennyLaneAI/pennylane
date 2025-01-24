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
Code relevant for performing measurements on a qubit mixed state.
"""

from collections.abc import Callable

from pennylane import math, queuing
from pennylane.measurements import (
    ExpectationMP,
    MeasurementProcess,
    ProbabilityMP,
    StateMeasurement,
    StateMP,
    VarianceMP,
)
from pennylane.ops import Sum
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .apply_operation import _get_num_wires, apply_operation


def _reshape_state_as_matrix(state, num_wires):
    """Given a non-flat, potentially batched state, flatten it to square matrix or matrices if batched.

    Args:
        state (TensorLike): A state that needs to be reshaped to a square matrix or matrices if batched
        num_wires (int): The number of wires the state represents

    Returns:
        Tensorlike: A reshaped, square state, with an extra batch dimension if necessary
    """
    dim = 2**num_wires
    batch_size = math.get_batch_size(state, ((2,) * (num_wires * 2)), dim**2)
    shape = (batch_size, dim, dim) if batch_size is not None else (dim, dim)
    return math.reshape(state, shape)


def calculate_expval(
    measurementprocess: ExpectationMP,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Measure the expectation value of an observable.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state.
        state (TensorLike): the state to measure.
        is_state_batched (bool): whether the state is batched or not.
        readout_errors (List[Callable]): List of chanels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        TensorLike: expectation value of observable wrt the state.
    """
    probs = calculate_probability(measurementprocess, state, is_state_batched, readout_errors)
    eigvals = math.asarray(measurementprocess.eigvals(), dtype="float64")
    # In case of broadcasting, `probs` has two axes and these are a matrix-vector products
    return math.dot(probs, eigvals)


# pylint: disable=unused-argument
def calculate_reduced_density_matrix(
    measurementprocess: StateMeasurement,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Get the state or reduced density matrix.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state.
        state (TensorLike): state to apply the measurement to.
        is_state_batched (bool): whether the state is batched or not.
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
            to simulate readout errors. These are not applied on this type of measurement.

    Returns:
        TensorLike: state or reduced density matrix.
    """
    wires = measurementprocess.wires
    state_reshaped_as_matrix = _reshape_state_as_matrix(
        state, _get_num_wires(state, is_state_batched)
    )
    if not wires:
        return state_reshaped_as_matrix

    return math.reduce_dm(state_reshaped_as_matrix, wires)


def calculate_probability(
    measurementprocess: StateMeasurement,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Find the probability of measuring states.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state.
        state (TensorLike): state to apply the measurement to.
        is_state_batched (bool): whether the state is batched or not.
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        TensorLike: the probability of the state being in each measurable state.
    """
    for op in measurementprocess.diagonalizing_gates():
        state = apply_operation(op, state, is_state_batched=is_state_batched)

    wires = measurementprocess.wires
    num_state_wires = _get_num_wires(state, is_state_batched)
    wire_order = Wires(range(num_state_wires))

    if readout_errors is not None:
        with queuing.QueuingManager.stop_recording():
            for wire in wires:
                for m_error in readout_errors:
                    state = apply_operation(m_error(wire), state, is_state_batched=is_state_batched)

    # probs are diagonal elements
    # stacking list since diagonal function axis selection parameter names
    # are not consistent across interfaces
    reshaped_state = _reshape_state_as_matrix(state, num_state_wires)
    probs = ProbabilityMP().process_density_matrix(reshaped_state, wire_order)
    # Convert the interface from numpy to whgatever from the state
    probs = math.convert_like(probs, state)

    # !NOTE: unclear if this whole post-processing here below is that much necessary
    # if a probability is very small it may round to negative, undesirable.
    # math.clip with None bounds breaks with tensorflow, using this instead:
    probs = math.where(probs < 0, 0, probs)
    if wires == Wires([]):
        # no need to marginalize
        return probs

    # !NOTE: one thing we can check in the future is if the following code is replacable with first calc rdm and then do probs
    # determine which subsystems are to be summed over
    inactive_wires = Wires.unique_wires([wire_order, wires])

    # translate to wire labels used by device
    wire_map = dict(zip(wire_order, range(len(wire_order))))
    mapped_wires = [wire_map[w] for w in wires]
    inactive_wires = [wire_map[w] for w in inactive_wires]

    # reshape the probability so that each axis corresponds to a wire
    num_device_wires = len(wire_order)
    shape = [2] * num_device_wires
    desired_axes = math.argsort(math.argsort(mapped_wires))
    flat_shape = (-1,)
    expected_size = 2**num_device_wires
    batch_size = math.get_batch_size(probs, (expected_size,), expected_size)
    if batch_size is not None:
        # prob now is reshaped to have self.num_wires+1 axes in the case of broadcasting
        shape.insert(0, batch_size)
        inactive_wires = [idx + 1 for idx in inactive_wires]
        desired_axes = math.insert(desired_axes + 1, 0, 0)
        flat_shape = (batch_size, -1)

    prob = math.reshape(probs, shape)
    # sum over all inactive wires
    prob = math.sum(prob, axis=tuple(inactive_wires))
    # rearrange wires if necessary
    prob = math.transpose(prob, desired_axes)
    # flatten and return probabilities
    return math.reshape(prob, flat_shape)


def calculate_variance(
    measurementprocess: VarianceMP,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Find variance of observable.

    Args:
        measurementprocess (VarianceMP): measurement to apply to the state.
        state (TensorLike): state to apply the measurement to.
        is_state_batched (bool): whether the state is batched or not.
        readout_errors (List[Callable]): List of operators to apply to each wire being measured
        to simulate readout errors.

    Returns:
        TensorLike: the variance of the observable with respect to the state.
    """
    probs = calculate_probability(measurementprocess, state, is_state_batched, readout_errors)
    eigvals = math.asarray(measurementprocess.eigvals(), dtype="float64")
    # In case of broadcasting, `probs` has two axes and these are a matrix-vector products
    return math.dot(probs, (eigvals**2)) - math.dot(probs, eigvals) ** 2


def calculate_expval_sum_of_terms(
    measurementprocess: ExpectationMP,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Measure the expectation value of the state when the measured observable is a ``Hamiltonian`` or ``Sum``
    and it must be backpropagation compatible.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state.
        state (TensorLike): the state to measure.
        is_state_batched (bool): whether the state is batched or not.
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        TensorLike: the expectation value of the sum of Hamiltonian observable with respect to the state.
    """
    # Recursively call measure on each term, so that the best measurement method can
    # be used for each term
    return sum(
        measure(
            ExpectationMP(term),
            state,
            is_state_batched=is_state_batched,
            readout_errors=readout_errors,
        )
        for term in measurementprocess.obs
    )


# pylint: disable=too-many-return-statements
def get_measurement_function(
    measurementprocess: MeasurementProcess,
) -> Callable[[MeasurementProcess, TensorLike, bool, list[Callable]], TensorLike]:
    """Get the appropriate method for performing a measurement.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state.
        state (TensorLike): the state to measure.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        Callable: function that returns the measurement result.
    """
    if isinstance(measurementprocess, StateMeasurement):
        if isinstance(measurementprocess, ExpectationMP):
            if isinstance(measurementprocess.obs, Sum):
                return calculate_expval_sum_of_terms
            if measurementprocess.obs.has_matrix:
                return calculate_expval
        if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
            if isinstance(measurementprocess, StateMP):
                return calculate_reduced_density_matrix
            if isinstance(measurementprocess, ProbabilityMP):
                return calculate_probability
            if isinstance(measurementprocess, VarianceMP):
                return calculate_variance

    raise NotImplementedError


def measure(
    measurementprocess: MeasurementProcess,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Apply a measurement process to a state.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state.
        state (TensorLike): the state to measure.
        is_state_batched (bool): whether the state is batched or not.
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        Tensorlike: the result of the measurement process being applied to the state.
    """
    measurement_function = get_measurement_function(measurementprocess)
    return measurement_function(
        measurementprocess, state, is_state_batched=is_state_batched, readout_errors=readout_errors
    )
