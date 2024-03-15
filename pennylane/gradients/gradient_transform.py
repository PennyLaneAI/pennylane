# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains utilities for defining custom gradient transforms,
including a decorator for specifying gradient expansions."""
# pylint: disable=too-few-public-methods
from functools import partial
import warnings

import pennylane as qml
from pennylane.measurements import (
    MutualInfoMP,
    StateMP,
    VarianceMP,
    VnEntropyMP,
    ProbabilityMP,
)

SUPPORTED_GRADIENT_KWARGS = [
    "approx_order",
    "argnum",
    "aux_wire",
    "broadcast",  # [TODO: This is in param_shift. Unify with use_broadcasting in stoch_pulse_grad
    "device_wires",
    "diagonal_shifts",
    "f0",
    "force_order2",
    "gradient_recipes",
    "gradient_kwargs",
    "h",
    "n",
    "num",
    "num_directions",
    "num_split_times",
    "off_diagonal_shifts",
    "order",
    "reduction",
    "sampler",
    "sampler_rng",
    "sampler_seed",
    "shifts",
    "shots",
    "strategy",
    "use_broadcasting",
    "validate_params",
]


def assert_multimeasure_not_broadcasted(measurements, broadcast):
    """Assert that there are not simultaneously multiple measurements and
    broadcasting activated.Otherwise raises an error."""
    if broadcast and len(measurements) > 1:
        raise NotImplementedError(
            "Broadcasting with multiple measurements is not supported yet. "
            f"Set broadcast to False instead. The tape measurements are {measurements}."
        )


def assert_no_state_returns(measurements, transform_name):
    """Check whether a set of measurements contains a measurement process that returns the quantum
    state and raise an error if this is the case.

    Args:
        measurements (list[MeasurementProcess]): measurements to analyze
        transform_name (str): Name of the gradient transform that queries the measurements

    Currently, the measurement processes that are considered to return the state are
    ``~.measurements.StateMP``, ``~.measurements.VnEntropyMP``, and ``~.measurements.MutualInfoMP``.
    """
    if any(isinstance(m, (StateMP, VnEntropyMP, MutualInfoMP)) for m in measurements):
        raise ValueError(
            f"Computing the gradient of circuits that return the state with the {transform_name} "
            "gradient transform is not supported, as it is a hardware-compatible method."
        )


def assert_no_variance(measurements, transform_name):
    """Check whether a set of measurements contains a variance measurement
    raise an error if this is the case.

    Args:
        measurements (list[MeasurementProcess]): measurements to analyze
        transform_name (str): Name of the gradient transform that queries the measurements
    """
    if any(isinstance(m, VarianceMP) for m in measurements):
        raise ValueError(
            f"Computing the gradient of variances with the {transform_name} "
            "gradient transform is not supported."
        )


def assert_no_tape_batching(tape, transform_name):
    """Check whether a tape is broadcasted and raise an error if this is the case.

    Args:
        tape (`~.QuantumScript`): measurements to analyze
        transform_name (str): Name of the gradient transform that queries the tape
    """
    if tape.batch_size is not None:
        raise NotImplementedError(
            f"Computing the gradient of broadcasted tapes with the {transform_name} "
            "gradient transform is currently not supported. See #4462 for details."
        )


def choose_trainable_params(tape, argnum=None):
    """Returns a list of trainable parameter indices in the tape.

    Chooses the subset of trainable parameters to compute the Jacobian for. The function
    returns a list of indices with respect to the list of trainable parameters. If argnum
    is not provided, all trainable parameters are considered.

    Args:
        tape (`~.QuantumScript`): the tape to analyze
        argnum (int, list(int), None): Indices for trainable parameters(s)
            to compute the Jacobian for.

    Returns:
        list: list of the trainable parameter indices

    """

    if argnum is None:
        return [idx for idx, _ in enumerate(tape.trainable_params)]

    if isinstance(argnum, int):
        argnum = [argnum]

    if len(argnum) == 0:
        warnings.warn(
            "No trainable parameters were specified for computing the Jacobian.",
            UserWarning,
        )

    return argnum


def _try_zero_grad_from_graph_or_get_grad_method(tape, param_index, use_graph=True):
    """Gets the gradient method of a parameter. If use_graph=True, analyze the
    circuit graph to find if the parameter has zero gradient.

    Args:
        tape (`~.QuantumScript`): the tape to analyze
        param_index (int): the index of the parameter to analyze
        use_graph (bool): whether to use the circuit graph to find if
            a parameter has zero gradient

    """

    # pylint:disable=protected-access
    par_info = tape._par_info[param_index]

    if use_graph:
        op_or_mp = tape[par_info["op_idx"]]
        if not any(tape.graph.has_path(op_or_mp, mp) for mp in tape.measurements):
            # there is no influence of this operation on any of the observables
            return "0"

    return getattr(par_info["op"], "grad_method", None)


def _find_gradient_methods(tape, trainable_param_indices, use_graph=True):
    """Returns a dictionary with gradient information of each trainable parameter."""

    return {
        idx: _try_zero_grad_from_graph_or_get_grad_method(
            tape, tape.trainable_params[idx], use_graph
        )
        for idx in trainable_param_indices
    }


def _validate_gradient_methods(tape, method, diff_methods):
    """Validates if the gradient method requested is supported by the trainable
    parameters of a tape, and returns the allowed parameter gradient methods."""

    # check and raise an error if any parameters are non-differentiable
    nondiff_params = [tape.trainable_params[idx] for idx, m in diff_methods.items() if m is None]
    if nondiff_params:
        raise ValueError(f"Cannot differentiate with respect to parameter(s) {nondiff_params}")

    # If explicitly using analytic mode, ensure that all
    # parameters support analytic differentiation.
    numeric_params = [tape.trainable_params[idx] for idx, m in diff_methods.items() if m == "F"]
    if method == "analytic" and numeric_params:
        raise ValueError(
            f"The analytic gradient method cannot be used with the parameter(s) {numeric_params}."
        )


def find_and_validate_gradient_methods(tape, method, trainable_param_indices, use_graph=True):
    """Returns a dictionary of gradient methods for each trainable parameter after
    validating if the gradient method requested is supported by the trainable parameters

    Parameter gradient methods include:

    * ``None``: the parameter does not support differentiation.

    * ``"0"``: the variational circuit output does not depend on this
      parameter (the partial derivative is zero).

    In addition, the operator might define its own grad method
    via :attr:`.Operator.grad_method`.

    Args:
        tape (`~.QuantumScript`): the tape to analyze
        method (str): the gradient method to use
        trainable_param_indices (list[int]): the indices of the trainable parameters
            to compute the Jacobian for
        use_graph (bool): whether to use the circuit graph to find if
            a parameter has zero gradient

    Returns:
        dict: dictionary of the gradient methods for each trainable parameter

    Raises:
        ValueError: If there exist non-differentiable trainable parameters on the tape.
        ValueError: If the Jacobian method is ``"analytic"`` but there exist some trainable
            parameters on the tape that only support numeric differentiation.

    """
    diff_methods = _find_gradient_methods(tape, trainable_param_indices, use_graph=use_graph)
    _validate_gradient_methods(tape, method, diff_methods)
    return diff_methods


def _all_zero_grad(tape):
    """Auxiliary function to return zeros for the all-zero gradient case."""
    list_zeros = []

    par_shapes = [qml.math.shape(p) for p in tape.get_parameters()]
    for m in tape.measurements:
        # TODO: Update shape for CV variables
        shape = (2 ** len(m.wires),) if isinstance(m, ProbabilityMP) else ()
        if len(tape.trainable_params) == 1:
            sub_list_zeros = qml.math.zeros(par_shapes[0] + shape)
        else:
            sub_list_zeros = tuple(qml.math.zeros(sh + shape) for sh in par_shapes)

        list_zeros.append(sub_list_zeros)

    if tape.shots.has_partitioned_shots:
        if len(tape.measurements) == 1:
            return [], lambda _: tuple(list_zeros[0] for _ in range(tape.shots.num_copies))
        return [], lambda _: tuple(tuple(list_zeros) for _ in range(tape.shots.num_copies))

    if len(tape.measurements) == 1:
        return [], lambda _: list_zeros[0]

    return [], lambda _: tuple(list_zeros)


_no_trainable_grad_warning = (
    "Attempted to compute the gradient of a tape with no trainable parameters. "
    "If this is unintended, please mark trainable parameters in accordance with the "
    "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
)


def _no_trainable_grad(tape):
    """Auxiliary function that returns correctly formatted gradients when there
    are no trainable parameters."""
    warnings.warn(_no_trainable_grad_warning)
    if tape.shots.has_partitioned_shots:
        if len(tape.measurements) == 1:
            return [], lambda _: tuple(qml.math.zeros([0]) for _ in range(tape.shots.num_copies))
        return [], lambda _: tuple(
            tuple(qml.math.zeros([0]) for _ in range(len(tape.measurements)))
            for _ in range(tape.shots.num_copies)
        )

    if len(tape.measurements) == 1:
        return [], lambda _: qml.math.zeros([0])
    return [], lambda _: tuple(qml.math.zeros([0]) for _ in range(len(tape.measurements)))


def _swap_first_two_axes(grads, first_axis_size, second_axis_size):
    """Transpose the first two axes of an iterable of iterables, returning
    a tuple of tuples."""
    if first_axis_size == 1:
        return tuple(grads[0][i] for i in range(second_axis_size))
    return tuple(
        tuple(grads[j][i] for j in range(first_axis_size)) for i in range(second_axis_size)
    )


def _move_first_axis_to_third_pos(grads, first_axis_size, second_axis_size, third_axis_size):
    """Transpose the first two axes of an iterable of iterables, returning
    a tuple of tuples."""
    if first_axis_size == 1:
        return tuple(
            tuple(grads[0][i][j] for j in range(third_axis_size)) for i in range(second_axis_size)
        )
    return tuple(
        tuple(tuple(grads[k][i][j] for k in range(first_axis_size)) for j in range(third_axis_size))
        for i in range(second_axis_size)
    )


def reorder_grads(grads, tape_specs):
    """Reorder the axes of tape gradients according to the original tape specifications.

    Args:
        grads (list[tensorlike] or list[tuple[tensorlike]] or list[tuple[tuple[tensorlike]]]:
            Gradient entries with leading parameter axis to be reordered.
        tape_specs (tuple): Information about the differentiated original tape in the order
            ``(bool: single_measure, int: num_params, int: num_measurements, Shots: shots)``.

    Returns:
        tensor_like or tuple[tensor_like] or tuple[tuple[tensor_like]]: The reordered gradient
            entries. Consider the details below for the ordering of the axes.

    The order of axes of the gradient output matches the structure outputted by jax.jacobian for
    a tuple-valued function. Internally, this may not be the case when computing the gradients,
    so the axes are reordered here.

    The axes of the input are assumed to be in the following order:

        1. Number of trainable parameters (Num params)
        2. Shot vector (if ``shots`` is a ``list`` or ``list[tuple]``. Skipped otherwise)
        3. Measurements (if there are multiple measurements. Skipped otherwise)
        4. Measurement shape
        5. Broadcasting dimension (for broadcasted tapes, skipped otherwise)

    The final order of axes of gradient results should be:

        1. Shot vector [1]
        2. Measurements [1]
        3. Number of trainable parameters (Num params) [1]
        4. Broadcasting dimension [2]
        5. Measurement shape

    [1] These axes are skipped in the output if they have length one. For shot vector and
        measurements, this already is true for the input. For num params, the axis is skipped
        "in addition", compared to the input.
    [2] Parameter broadcasting doesn't yet support multiple measurements, hence such cases are not
        dealt with at the moment by this function.

    The above reordering requires the following operations:

        1. In all cases, remove the parameter axis if it has length one.
        2. For a single measurement and no shot vector: Do nothing (but cast to ``tuple``)
        3. For a single measurement and shot vector: Swap first two axes (shots and parameters)
        4. For multiple measurements and no shot vector: Swap first two axes
           (measurements and parameters)
        5. For multiple measurements and shot vector: Move parameter axis from first to third
           position.

    In all cases the output will be a ``tuple``, except for single-measurement, single-parameter
    tapes, which will return a single measurement-like shaped output (no shot vector), or a list
    thereof (shot vector).
    """
    single_measure, num_params, num_measurements, shots = tape_specs
    if single_measure:
        if num_params == 1:
            return grads[0]
        if not shots.has_partitioned_shots:
            return tuple(grads)
        return _swap_first_two_axes(grads, num_params, shots.num_copies)

    if not shots.has_partitioned_shots:
        return _swap_first_two_axes(grads, num_params, num_measurements)
    return _move_first_axis_to_third_pos(grads, num_params, shots.num_copies, num_measurements)


# pylint: disable=too-many-return-statements,too-many-branches
def _contract_qjac_with_cjac(qjac, cjac, tape):
    """Contract a quantum Jacobian with a classical preprocessing Jacobian.
    Essentially, this function computes the generalized version of
    ``tensordot(qjac, cjac)`` over the tape parameter axis, adapted to the new
    return type system. This function takes the measurement shapes and different
    QNode arguments into account.
    """
    num_measurements = len(tape.measurements)
    has_partitioned_shots = tape.shots.has_partitioned_shots

    if isinstance(qjac, tuple) and len(qjac) == 1:
        qjac = qjac[0]

    if isinstance(cjac, tuple) and len(cjac) == 1:
        cjac = cjac[0]

    cjac_is_tuple = isinstance(cjac, tuple)
    if not cjac_is_tuple:
        is_square = cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1]

        if not qml.math.is_abstract(cjac) and (
            is_square and qml.math.allclose(cjac, qml.numpy.eye(cjac.shape[0]))
        ):
            # Classical Jacobian is the identity. No classical processing is present in the QNode
            return qjac

    multi_meas = num_measurements > 1

    if cjac_is_tuple:
        multi_params = True
    else:
        _qjac = qjac
        if multi_meas:
            _qjac = _qjac[0]
        if has_partitioned_shots:
            _qjac = _qjac[0]
        multi_params = isinstance(_qjac, tuple)

    tdot = partial(qml.math.tensordot, axes=[[0], [0]])

    if not multi_params:
        # Without dimension (e.g. expval) or with dimension (e.g. probs)
        def _reshape(x):
            return qml.math.reshape(x, (1,) if x.shape == () else (1, -1))

        if not (multi_meas or has_partitioned_shots):
            # Single parameter, single measurements
            return tdot(_reshape(qjac), cjac)

        if not (multi_meas and has_partitioned_shots):
            return tuple(tdot(_reshape(q), cjac) for q in qjac)

        # Single parameter, multiple measurements
        return tuple(tuple(tdot(_reshape(_q), cjac) for _q in q) for q in qjac)

    if not multi_meas:
        # Multiple parameters, single measurement
        qjac = qml.math.stack(qjac)
        if not cjac_is_tuple:
            return tdot(qjac, qml.math.stack(cjac))
        return tuple(tdot(qjac, c) for c in cjac if c is not None)

    # Multiple parameters, multiple measurements
    if not cjac_is_tuple:
        return tuple(tdot(qml.math.stack(q), qml.math.stack(cjac)) for q in qjac)
    return tuple(tuple(tdot(qml.math.stack(q), c) for c in cjac if c is not None) for q in qjac)
