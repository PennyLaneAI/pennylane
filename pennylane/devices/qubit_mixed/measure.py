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
# pylint:disable=too-many-branches, import-outside-toplevel, unused-argument

from collections.abc import Callable

from scipy.sparse import csr_matrix

from pennylane import math
from pennylane.measurements import (
    DensityMatrixMP,
    ExpectationMP,
    MeasurementProcess,
    MeasurementValue,
    StateMeasurement,
    StateMP,
)
from pennylane.ops import LinearCombination, Sum
from pennylane.pauli.conversion import is_pauli_sentence, pauli_sentence
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


def state_diagonalizing_gates(
    measurementprocess: StateMeasurement,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state
        state (TensorLike): state to apply the measurement to
        is_state_batched (bool): whether the state is batched or not
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        TensorLike: the result of the measurement
    """
    for op in measurementprocess.diagonalizing_gates():
        state = apply_operation(op, state, is_state_batched=is_state_batched)

    if readout_errors is not None and measurementprocess.wires is not None:
        for err_channel_fn in readout_errors:
            for w in measurementprocess.wires:
                # Here, we assume err_channel_fn(w) returns a quantum operation/channel like qml.BitFlip(...)
                error_op = err_channel_fn(w)
                state = apply_operation(error_op, state, is_state_batched=is_state_batched)

    num_wires = _get_num_wires(state, is_state_batched)
    wires = Wires(range(num_wires))
    flattened_state = _reshape_state_as_matrix(state, num_wires)
    is_StateMP = isinstance(measurementprocess, StateMP)
    is_DensityMatrixMP = isinstance(measurementprocess, DensityMatrixMP)
    if is_StateMP and not is_DensityMatrixMP:  # a pure qml.state()
        raw_wires = measurementprocess.raw_wires or wires  # incase the None raw_wires case
        measurementprocess = DensityMatrixMP(wires=raw_wires)
    res = measurementprocess.process_density_matrix(flattened_state, wires)

    return res


def csr_dot_products_density_matrix(
    measurementprocess: ExpectationMP,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Measure the expectation value of an observable from a density matrix using dot products between
    ``scipy.csr_matrix`` representations.

    For a density matrix :math:`\rho` and observable :math:`O`, the expectation value is: .. math:: \text{Tr}(\rho O),

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the density matrix state
        state (TensorLike): the density matrix, reshaped to (dim, dim) if not batched,
            or (batch, dim, dim) if batched. Use _reshape_state_as_matrix for that.
        num_wires (int): the number of wires the state represents
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    # Reshape the state into a density matrix form
    num_wires = _get_num_wires(state, is_state_batched)
    rho = _reshape_state_as_matrix(state, num_wires)  # shape (dim, dim) or (batch, dim, dim)
    rho_np = math.toarray(rho)  # convert to NumPy for stable sparse ops

    # Obtain the operator O in CSR form. If it's a Pauli sentence, we use its built-in method.
    if is_pauli_sentence(measurementprocess.obs):
        ps = pauli_sentence(measurementprocess.obs)
        # Create a CSR matrix representation of the operator
        O = ps.to_mat(wire_order=range(num_wires), format="csr")
    else:
        # For non-Pauli observables, just get their sparse matrix representation directly.
        O = measurementprocess.obs.sparse_matrix(wire_order=list(range(num_wires)))

    # Compute Tr(rho O)
    # !NOTE: please do NOT try use ps.dot here; in 0.40 somehow the ps.dot wrongly calculates the product with density matrix
    if is_state_batched:
        # handle batch case
        results = []
        for i in range(rho_np.shape[0]):
            rho_i_csr = csr_matrix(rho_np[i])
            rhoO = rho_i_csr.dot(O).toarray()
            results.append(math.trace(rhoO))
        res = math.stack(results)
    else:
        # single state case
        rho_csr = csr_matrix(rho_np)
        rhoO = rho_csr.dot(O).toarray()
        res = math.trace(rhoO)

    # Convert back to the same interface and return the real part
    res = math.real(math.squeeze(res))
    return math.convert_like(res, state)


def full_dot_products_density_matrix(
    measurementprocess: ExpectationMP,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Measure the expectation value of an observable from a density matrix using full matrix
    multiplication.

    For a density matrix ρ and observable O, the expectation value is:
    .. math:: \text{Tr}(\rho O).

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the density matrix state
        state (TensorLike): the density matrix, reshaped via _reshape_state_as_matrix to
            (dim, dim) if not batched, or (batch, dim, dim) if batched.
        num_wires (int): the number of wires the state represents
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    # Reshape the state into a density matrix form
    num_wires = _get_num_wires(state, is_state_batched)
    rho = _reshape_state_as_matrix(state, num_wires)
    dim = 2**num_wires

    # Obtain the operator matrix O
    O = measurementprocess.obs.matrix(wire_order=list(range(num_wires)))
    O = math.convert_like(O, rho)

    # Compute ρ O
    rhoO = math.matmul(rho, O)  # shape: (batch, dim, dim) if batched, else (dim, dim)

    # Take the diagonal and sum to get the trace
    if (
        math.get_interface(rhoO) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        import tensorflow as tf

        diag_elements = tf.linalg.diag_part(rhoO)
    else:
        # fallback to a math.diagonal approach or indexing for other interfaces
        dim = math.shape(rhoO)[-1]
        diag_indices = math.arange(dim, like=rhoO)
        diag_elements = rhoO[..., diag_indices, diag_indices]
    # If batched, diag_elements shape: (batch, dim); if single: (dim,)

    res = math.sum(diag_elements, axis=-1 if is_state_batched else 0)
    return math.real(res)


def sum_of_terms_method(
    measurementprocess: ExpectationMP,
    state: TensorLike,
    is_state_batched: bool = False,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Measure the expectation value of the state when the measured observable is a ``Hamiltonian`` or ``Sum``
    and it must be backpropagation compatible.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the expectation value of the sum of Hamiltonian observable with respect to the state.
    """
    # Recursively call measure on each term, so that the best measurement method can
    # be used for each term
    return math.sum(
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
    measurementprocess: MeasurementProcess, state: TensorLike
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
        if isinstance(measurementprocess.mv, MeasurementValue):
            return state_diagonalizing_gates
        if isinstance(measurementprocess, ExpectationMP):
            if measurementprocess.obs.name == "SparseHamiltonian":
                return csr_dot_products_density_matrix

            if measurementprocess.obs.name == "Hermitian":
                return full_dot_products_density_matrix

            backprop_mode = math.get_interface(state, *measurementprocess.obs.data) != "numpy"
            if isinstance(measurementprocess.obs, LinearCombination):

                # need to work out thresholds for when it's faster to use "backprop mode"
                if backprop_mode:
                    return sum_of_terms_method

                if not all(obs.has_sparse_matrix for obs in measurementprocess.obs.terms()[1]):
                    return sum_of_terms_method

                return csr_dot_products_density_matrix

            if isinstance(measurementprocess.obs, Sum):
                if backprop_mode:
                    # always use sum_of_terms_method for Sum observables in backprop mode
                    return sum_of_terms_method

                if not all(obs.has_sparse_matrix for obs in measurementprocess.obs):
                    return sum_of_terms_method

                if (
                    measurementprocess.obs.has_overlapping_wires
                    and len(measurementprocess.obs.wires) > 7
                ):
                    # Use tensor contraction for `Sum` expectation values with non-commuting summands
                    # and 8 or more wires as it's faster than using eigenvalues.

                    return csr_dot_products_density_matrix
        if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
            return state_diagonalizing_gates

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
    measurement_function = get_measurement_function(measurementprocess, state)
    res = measurement_function(
        measurementprocess, state, is_state_batched=is_state_batched, readout_errors=readout_errors
    )
    return res
