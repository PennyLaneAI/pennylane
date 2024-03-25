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
"""
Code relevant for performing measurements on a state.
"""
from typing import Callable

from scipy.sparse import csr_matrix

from pennylane import math
from pennylane.ops import Sum, Hamiltonian, LinearCombination
from pennylane.measurements import (
    StateMeasurement,
    MeasurementProcess,
    MeasurementValue,
    ExpectationMP,
)
from pennylane.pauli.conversion import is_pauli_sentence, pauli_sentence
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .apply_operation import apply_operation


def flatten_state(state, num_wires):
    """
    Given a non-flat, potentially batched state, flatten it.

    Args:
        state (TensorLike): A state that needs flattening
        num_wires (int): The number of wires the state represents

    Returns:
        A flat state, with an extra batch dimension if necessary
    """
    dim = 2**num_wires
    batch_size = math.get_batch_size(state, (2,) * num_wires, dim)
    shape = (batch_size, dim) if batch_size is not None else (dim,)
    return math.reshape(state, shape)


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
    for op in measurementprocess.diagonalizing_gates():
        state = apply_operation(op, state, is_state_batched=is_state_batched)

    total_indices = len(state.shape) - is_state_batched
    wires = Wires(range(total_indices))
    flattened_state = flatten_state(state, total_indices)
    return measurementprocess.process_state(flattened_state, wires)


def csr_dot_products(
    measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Measure the expectation value of an observable using dot products between ``scipy.csr_matrix``
    representations.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    total_wires = len(state.shape) - is_state_batched

    if is_pauli_sentence(measurementprocess.obs):
        state = math.toarray(state)
        if is_state_batched:
            state = state.reshape(math.shape(state)[0], -1)
        else:
            state = state.reshape(1, -1)
        bra = math.conj(state)
        ps = pauli_sentence(measurementprocess.obs)
        new_ket = ps.dot(state, wire_order=list(range(total_wires)))
        res = (bra * new_ket).sum(axis=1)
    elif is_state_batched:
        Hmat = measurementprocess.obs.sparse_matrix(wire_order=list(range(total_wires)))
        state = math.toarray(state).reshape(math.shape(state)[0], -1)

        bra = csr_matrix(math.conj(state))
        ket = csr_matrix(state)
        new_bra = bra.dot(Hmat)
        res = new_bra.multiply(ket).sum(axis=1).getA()
    else:
        Hmat = measurementprocess.obs.sparse_matrix(wire_order=list(range(total_wires)))
        state = math.toarray(state).flatten()

        # Find the expectation value using the <\psi|H|\psi> matrix contraction
        bra = csr_matrix(math.conj(state))
        ket = csr_matrix(state[..., None])
        new_ket = csr_matrix.dot(Hmat, ket)
        res = csr_matrix.dot(bra, new_ket).toarray()[0]

    return math.real(math.squeeze(res))


def full_dot_products(
    measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Measure the expectation value of an observable using the dot product between full matrix
    representations.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    ket = apply_operation(measurementprocess.obs, state, is_state_batched=is_state_batched)
    dot_product = math.sum(
        math.conj(state) * ket, axis=tuple(range(int(is_state_batched), math.ndim(state)))
    )
    return math.real(dot_product)


def sum_of_terms_method(
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


# pylint: disable=too-many-return-statements
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
        if isinstance(measurementprocess.mv, MeasurementValue):
            return state_diagonalizing_gates

        if isinstance(measurementprocess, ExpectationMP):
            if measurementprocess.obs.name == "SparseHamiltonian":
                return csr_dot_products

            if measurementprocess.obs.name == "Hermitian":
                return full_dot_products

            backprop_mode = math.get_interface(state, *measurementprocess.obs.data) != "numpy"
            if isinstance(measurementprocess.obs, (Hamiltonian, LinearCombination)):
                # need to work out thresholds for when its faster to use "backprop mode" measurements
                return sum_of_terms_method if backprop_mode else csr_dot_products

            if isinstance(measurementprocess.obs, Sum):
                if backprop_mode:
                    # always use sum_of_terms_method for Sum observables in backprop mode
                    return sum_of_terms_method
                if (
                    measurementprocess.obs.has_overlapping_wires
                    and len(measurementprocess.obs.wires) > 7
                ):
                    # Use tensor contraction for `Sum` expectation values with non-commuting summands
                    # and 8 or more wires as it's faster than using eigenvalues.

                    return csr_dot_products

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
