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
from pennylane.ops import Sum, prod
from pennylane.measurements import StateMeasurement, MeasurementProcess, ExpectationMP
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .apply_operation import apply_operation


def state_diagonalizing_gates(
    measurementprocess: StateMeasurement, state: TensorLike
) -> TensorLike:
    """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state
        state (TensorLike): state to apply the measurement to

    Returns:
        TensorLike: the result of the measurement
    """
    for op in measurementprocess.diagonalizing_gates():
        state = apply_operation(op, state)

    total_indices = len(state.shape)
    wires = Wires(range(total_indices))
    return measurementprocess.process_state(math.flatten(state), wires)


def state_hamiltonian_expval(measurementprocess: ExpectationMP, state: TensorLike) -> TensorLike:
    """Measure the expectation value of the state when the measured observable is a ``Hamiltonian`` or
    ``SparseHamiltonian``.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure

    Returns:
        TensorLike: the result of the measurement
    """
    total_wires = len(state.shape)
    Hmat = measurementprocess.obs.sparse_matrix(wire_order=list(range(total_wires)))
    _state = math.toarray(state).flatten()

    # Find the expectation value using the <\psi|H|\psi> matrix contraction
    bra = csr_matrix(math.conj(_state))
    ket = csr_matrix(_state[..., None])
    new_ket = csr_matrix.dot(Hmat, ket)
    res = csr_matrix.dot(bra, new_ket).toarray()[0]

    return math.real(math.squeeze(res))


def state_hamiltonian_expval_backprop(
    measurementprocess: ExpectationMP, state: TensorLike
) -> TensorLike:
    """Measure the expecation value of the state when the measured observable is a ``Hamiltonian`` or ``Sum``
    and it must be backpropagation compatible.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure

    Returns:
        TensorLike: the result of the measurement
    """
    obs = measurementprocess.obs
    if isinstance(obs, Sum):
        terms_iterator = obs
    else:
        terms_iterator = [prod(*o.obs) * c for c, o in zip(*obs.terms())]

    # Recursively call measure on each term, so that the best measurement method can
    # be used for each term
    return sum(measure(ExpectationMP(term), state) for term in terms_iterator)


def get_measurement_function(
    measurementprocess: MeasurementProcess, backprop_mode: bool = False
) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
    """Get the appropriate method for performing a measurement.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state

    Returns:
        Callable: function that returns the measurement result
    """
    if isinstance(measurementprocess, StateMeasurement):

        if isinstance(measurementprocess, ExpectationMP):
            if measurementprocess.obs.name == "SparseHamiltonian":
                return state_hamiltonian_expval

            if measurementprocess.obs.name == "Hamiltonian" or (
                isinstance(measurementprocess.obs, Sum)
                and measurementprocess.obs.has_overlapping_wires
                and len(measurementprocess.obs.wires) > 7
            ):
                # Use tensor contraction for `Sum` expectation values with non-commuting summands
                # and 8 or more wires as it's faster than using eigenvalues.

                # need to work out thresholds for when its faster to use "backprop mode" measurements
                return (
                    state_hamiltonian_expval_backprop if backprop_mode else state_hamiltonian_expval
                )

        if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
            return state_diagonalizing_gates

    raise NotImplementedError


def measure(measurementprocess: MeasurementProcess, state: TensorLike) -> TensorLike:
    """Apply a measurement process to a state.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (TensorLike): the state to measure

    Returns:
        Tensorlike: the result of the measurement
    """
    backprop_mode = math.get_interface(state) != "numpy"
    return get_measurement_function(measurementprocess, backprop_mode)(measurementprocess, state)
