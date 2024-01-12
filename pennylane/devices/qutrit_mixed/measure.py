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
import functools
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
    StateMeasurement,
    MeasurementProcess,
    ExpectationMP,
)
from pennylane.operation import Observable
from pennylane.typing import TensorLike
from string import ascii_letters as alphabet
from pennylane.wires import Wires

from .apply_operation import apply_operation
from .utils import get_einsum_indices

alphabet_array = math.asarray(list(alphabet))

qudit_dim = 3  # specifies qudit dimension


def resquare_state(state, num_wires):
    """
    Given a non-flat, potentially batched state, flatten it to a square matrix.

    Args:
        state (TensorLike): A state that needs flattening
        num_wires (int): The number of wires the state represents

    Returns:
        A squared state, with an extra batch dimension if necessary
    """
    dim = qudit_dim**num_wires
    batch_size = math.get_batch_size(state, ((qudit_dim,) * (num_wires * 2)), dim**2)
    shape = (batch_size, dim, dim) if batch_size is not None else (dim, dim)
    return math.reshape(state, shape)


def get_einsum_indices(obs: Observable, state, is_state_batched: bool = False):
    r"""Finds the indices for einsum TODO when it's merged over
    TODO remove this and abstract the apply_ops to do this

    Args:
        obs (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        dict: indices used for einsum
    """
    num_ch_wires = len(obs.wires)
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)
    rho_dim = 2 * num_wires

    # Tensor indices of the state. For each qutrit, need an index for rows *and* columns
    state_indices = alphabet[:rho_dim]

    # row indices of the quantum state affected by this operation
    row_wires_list = obs.wires.tolist()
    row_indices = "".join(alphabet_array[row_wires_list].tolist())

    # column indices are shifted by the number of wires
    col_wires_list = [w + num_wires for w in row_wires_list]
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    # indices in einsum must be replaced with new ones
    new_row_indices = alphabet[rho_dim : rho_dim + num_ch_wires]
    new_col_indices = alphabet[rho_dim + num_ch_wires : rho_dim + 2 * num_ch_wires]

    # index for summation over Kraus operators
    kraus_index = alphabet[rho_dim + 2 * num_ch_wires : rho_dim + 2 * num_ch_wires + 1]

    # new state indices replace row and column indices with new ones
    new_state_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(col_indices + row_indices, new_col_indices + new_row_indices),
        state_indices,
    )
    operation_1_indices = f"{kraus_index}{new_row_indices}{row_indices}"
    return {"op1": operation_1_indices, "state": state_indices, "new_state": new_state_indices}


def apply_observable_einsum(obs: Observable, state, is_state_batched: bool = False):
    r"""Applies an observable to a density matrix rho, giving obs@state
    TODO remove this and abstract the apply_ops to do this

    Args:
        obs (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        TensorLike: the result of obs@state
    """
    indices = get_einsum_indices(obs, state, is_state_batched)
    einsum_indices = f"{indices['op1']},...{indices['state']},->...{indices['new_state']}"

    obs_mat = math.cast(obs.matrix(), complex)
    return math.einsum(einsum_indices, obs_mat, state)


def trace_method(
    measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Measure the expectation value of an observable by finding the trace of obs@rho.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not
    Returns:
        TensorLike: the result of the measurement
    """
    obs = measurementprocess.obs
    rho_mult_obs = apply_observable_einsum(obs, state, is_state_batched)

    return math.real(math.trace(rho_mult_obs))


def sum_of_terms_method(  # TODO this is copied code, should this borrow from qubit?
    measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Measure the expecation value of the state when the measured observable is a ``Hamiltonian`` or ``Sum``
    and it must be backpropagation compatible.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    if isinstance(measurementprocess.obs, Sum):
        # Recursively call measure on each term, so that the best measurement method can
        # be used for each term
        return sum(
            measure(ExpectationMP(term), state, is_state_batched=is_state_batched)
            for term in measurementprocess.obs
        )
    # else hamiltonian
    return sum(
        c * measure(ExpectationMP(t), state, is_state_batched=is_state_batched)
        for c, t in zip(*measurementprocess.obs.terms())
    )


def state_diagonalizing_gates(
    measurementprocess: StateMeasurement, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state
        state (TensorLike): state to apply the measurement to
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    # for op in measurementprocess.diagonalizing_gates(): TODO re-add
    #     state = apply_operation(op, state, is_state_batched=is_state_batched)

    total_indices = len(state.shape) - is_state_batched
    wires = Wires(range(total_indices))
    resquared_state = resquare_state(state, total_indices)
    return measurementprocess.process_state(resquared_state, wires)


def get_measurement_function(
    measurementprocess: MeasurementProcess, state: TensorLike
) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
    """Get the appropriate method for performing a measurement.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        Callable: function that returns the measurement result
    """
    if isinstance(measurementprocess, StateMeasurement):
        if isinstance(measurementprocess, ExpectationMP):
            # TODO add faster methods
            if isinstance(measurementprocess.obs, Hamiltonian):
                return sum_of_terms_method
            if isinstance(measurementprocess.obs, Sum):
                return sum_of_terms_method
            return trace_method

        if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
            return state_diagonalizing_gates

    raise NotImplementedError


def measure(
    measurementprocess: MeasurementProcess, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Apply a measurement process to a state.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        Tensorlike: the result of the measurement
    """
    return get_measurement_function(measurementprocess, state)(
        measurementprocess, state, is_state_batched
    )