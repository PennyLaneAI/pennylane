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
Code relevant for performing measurements on a qutrit mixed state.
"""

from typing import Callable
from string import ascii_letters as alphabet
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
    StateMeasurement,
    MeasurementProcess,
    ExpectationMP,
    StateMP,
    ProbabilityMP,
    VarianceMP,
)
from pennylane.typing import TensorLike

from .utils import (
    reshape_state_as_matrix,
    get_num_wires,
    get_eigvals,
    get_diagonalizing_gates,
    QUDIT_DIM,
)
from .apply_operation import apply_operation


def calculate_expval(
    measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Measure the expectation value of an observable.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state.
        state (TensorLike): the state to measure.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: expectation value of observable wrt the state.
    """
    probs = calculate_probability(measurementprocess, state, is_state_batched)
    eigvals = math.asarray(get_eigvals(measurementprocess.obs), dtype="float64")
    # In case of broadcasting, `probs` has two axes and these are a matrix-vector products
    return math.dot(probs, eigvals)


def calculate_reduced_density_matrix(
    measurementprocess: StateMeasurement, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Get the state or reduced density matrix.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state.
        state (TensorLike): state to apply the measurement to.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: state or reduced density matrix.
    """
    wires = measurementprocess.wires
    if not wires:
        return reshape_state_as_matrix(state, get_num_wires(state, is_state_batched))

    num_obs_wires = len(wires)
    num_state_wires = get_num_wires(state, is_state_batched)
    state_wire_indices_list = list(alphabet[:num_state_wires] * 2)
    final_state_wire_indices_list = [""] * (2 * num_obs_wires)

    for i, wire in enumerate(wires):
        col_index = wire + num_state_wires
        state_wire_indices_list[col_index] = alphabet[col_index]
        final_state_wire_indices_list[i] = alphabet[wire]
        final_state_wire_indices_list[i + num_obs_wires] = alphabet[col_index]

    state_wire_indices = "".join(state_wire_indices_list)
    final_state_wire_indices = "".join(final_state_wire_indices_list)

    state = math.einsum(f"...{state_wire_indices}->...{final_state_wire_indices}", state)

    return reshape_state_as_matrix(state, len(wires))


def calculate_probability(
    measurementprocess: StateMeasurement, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Find the probability of measuring states.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state.
        state (TensorLike): state to apply the measurement to.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: the probability of the state being in each measurable state.
    """
    for op in get_diagonalizing_gates(measurementprocess.obs):
        state = apply_operation(op, state, is_state_batched=is_state_batched)

    num_state_wires = get_num_wires(state, is_state_batched)

    # probs are diagonal elements
    # stacking list since diagonal function axis selection parameter names
    # are not consistent across interfaces
    reshaped_state = reshape_state_as_matrix(state, num_state_wires)
    if is_state_batched:
        probs = math.real(math.stack([math.diagonal(dm) for dm in reshaped_state]))
    else:
        probs = math.real(math.diagonal(reshaped_state))

    # if a probability is very small it may round to negative, undesirable.
    # math.clip with None bounds breaks with tensorflow, using this instead:
    probs = math.where(probs < 0, 0, probs)

    if mp_wires := measurementprocess.wires:
        expanded_shape = [QUDIT_DIM] * num_state_wires
        new_shape = [QUDIT_DIM ** len(mp_wires)]
        if is_state_batched:
            batch_size = probs.shape[0]
            expanded_shape.insert(0, batch_size)
            new_shape.insert(0, batch_size)
        wires_to_trace = tuple(
            x + is_state_batched for x in range(num_state_wires) if x not in mp_wires
        )

        expanded_probs = math.reshape(probs, expanded_shape)
        summed_probs = math.sum(expanded_probs, axis=wires_to_trace)
        return math.reshape(summed_probs, new_shape)

    return probs


def calculate_variance(
    measurementprocess: StateMeasurement, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Find variance of observable.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state.
        state (TensorLike): state to apply the measurement to.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: the variance of the observable wrt the state.
    """
    probs = calculate_probability(measurementprocess, state, is_state_batched)
    eigvals = math.asarray(get_eigvals(measurementprocess.obs), dtype="float64")
    # In case of broadcasting, `probs` has two axes and these are a matrix-vector products
    return math.dot(probs, (eigvals**2)) - math.dot(probs, eigvals) ** 2


def calculate_expval_sum_of_terms(
    measurementprocess: ExpectationMP,
    state: TensorLike,
    is_state_batched: bool = False,
) -> TensorLike:
    """Measure the expectation value of the state when the measured observable is a ``Hamiltonian`` or ``Sum``
    and it must be backpropagation compatible.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state.
        state (TensorLike): the state to measure.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: the expectation value of the sum of Hamiltonian observable wrt the state.
    """
    if isinstance(measurementprocess.obs, Sum):
        # Recursively call measure on each term, so that the best measurement method can
        # be used for each term
        return sum(
            measure(ExpectationMP(term), state, is_state_batched=is_state_batched)
            for term in measurementprocess.obs
        )
    # else Hamiltonian
    return sum(
        c * measure(ExpectationMP(t), state, is_state_batched=is_state_batched)
        for c, t in zip(*measurementprocess.obs.terms())
    )


# pylint: disable=too-many-return-statements
def get_measurement_function(
    measurementprocess: MeasurementProcess,
) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
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
            # TODO add faster methods
            # TODO add support for sparce Hamiltonians
            if isinstance(measurementprocess.obs, (Hamiltonian, Sum)):
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
    measurementprocess: MeasurementProcess, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Apply a measurement process to a state.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state.
        state (TensorLike): the state to measure.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        Tensorlike: the result of the measurement process being applied to the state.
    """
    return get_measurement_function(measurementprocess)(measurementprocess, state, is_state_batched)
