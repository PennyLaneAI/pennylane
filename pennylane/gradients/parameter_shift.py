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
"""
This module contains functions for computing the parameter-shift gradient
of a qubit-based quantum tape.
"""
# pylint: disable=protected-access,too-many-arguments,too-many-statements,unused-argument
from typing import Sequence, Callable
from functools import partial

import numpy as np

import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac

from .finite_difference import finite_diff
from .general_shift_rules import (
    _iterate_shift_rule,
    frequencies_to_period,
    generate_shifted_tapes,
    process_shifts,
)
from .gradient_transform import (
    _all_zero_grad,
    assert_no_state_returns,
    assert_no_tape_batching,
    assert_multimeasure_not_broadcasted,
    choose_grad_methods,
    gradient_analysis_and_validation,
    _no_trainable_grad,
    reorder_grads,
)

NONINVOLUTORY_OBS = {
    "Hermitian": lambda obs: obs.__class__(obs.matrix() @ obs.matrix(), wires=obs.wires),
    "SparseHamiltonian": lambda obs: obs.__class__(obs.matrix() @ obs.matrix(), wires=obs.wires),
    "Projector": lambda obs: obs,
}
"""Dict[str, callable]: mapping from a non-involutory observable name
to a callable that accepts an observable object, and returns the square
of that observable.
"""


def _square_observable(obs):
    """Returns the square of an observable."""

    if isinstance(obs, qml.operation.Tensor):
        # Observable is a tensor, we must consider its
        # component observables independently. Note that
        # we assume all component observables are on distinct wires.

        components_squared = []

        for comp in obs.obs:
            try:
                components_squared.append(NONINVOLUTORY_OBS[comp.name](comp))
            except KeyError:
                # component is involutory
                pass

        return qml.operation.Tensor(*components_squared)

    return NONINVOLUTORY_OBS[obs.name](obs)


def _process_op_recipe(op, p_idx, order):
    """Process an existing recipe of an operation."""
    recipe = op.grad_recipe[p_idx]
    if recipe is None:
        return None

    recipe = qml.math.array(recipe)
    if order == 1:
        return process_shifts(recipe, batch_duplicates=False)

    # Try to obtain the period of the operator frequencies for iteration of custom recipe
    try:
        period = frequencies_to_period(op.parameter_frequencies[p_idx])
    except qml.operation.ParameterFrequenciesUndefinedError:
        period = None

    # Iterate the custom recipe to obtain the second-order recipe
    if qml.math.allclose(recipe[:, 1], qml.math.ones_like(recipe[:, 1])):
        # If the multipliers are ones, we do not include them in the iteration
        # but keep track of them manually
        iter_c, iter_s = process_shifts(_iterate_shift_rule(recipe[:, ::2], order, period)).T
        return qml.math.stack([iter_c, qml.math.ones_like(iter_c), iter_s]).T

    return process_shifts(_iterate_shift_rule(recipe, order, period))


def _choose_recipe(argnum, idx, gradient_recipes, shifts, tape):
    """Obtain the gradient recipe for an indicated parameter from provided
    ``gradient_recipes``. If none is provided, use the recipe of the operation instead."""
    arg_idx = argnum.index(idx)
    recipe = gradient_recipes[arg_idx]
    if recipe is not None:
        recipe = process_shifts(np.array(recipe))
    else:
        op_shifts = None if shifts is None else shifts[arg_idx]
        recipe = _get_operation_recipe(tape, idx, shifts=op_shifts)
    return recipe


def _extract_unshifted(recipe, at_least_one_unshifted, f0, gradient_tapes, tape):
    """Exctract the unshifted term from a gradient recipe, if it is present.

    Returns:
        array_like[float]: The reduced recipe without the unshifted term.
        bool: The updated flag whether an unshifted term was found for any of the recipes.
        float or None: The coefficient of the unshifted term. None if no such term was present.

    This assumes that there will be at most one unshifted term in the recipe (others simply are
    not extracted) and that it comes first if there is one.
    """
    first_c, first_m, first_s = recipe[0]
    # Extract zero-shift term if present (if so, it will always be the first)
    if first_s == 0 and first_m == 1:
        # Gradient recipe includes a term with zero shift.
        if not at_least_one_unshifted and f0 is None:
            # Append the unshifted tape to the gradient tapes, if not already present
            gradient_tapes.insert(0, tape)

        # Store the unshifted coefficient. It is always the first coefficient due to processing
        unshifted_coeff = first_c
        at_least_one_unshifted = True
        recipe = recipe[1:]
    else:
        unshifted_coeff = None

    return recipe, at_least_one_unshifted, unshifted_coeff


def _single_meas_grad(result, coeffs, unshifted_coeff, r0):
    """Compute the gradient for a single measurement by taking the linear combination of
    the coefficients and the measurement result.

    If an unshifted term exists, its contribution is added to the gradient.
    """
    if isinstance(result, list) and result == []:
        if unshifted_coeff is None:
            raise ValueError(
                "This gradient component neither has a shifted nor an unshifted component. "
                "It should have been identified to have a vanishing gradient earlier on."
            )  # pragma: no cover
        # return the unshifted term, which is the only contribution
        return qml.math.array(unshifted_coeff * r0)
    result = qml.math.stack(result)
    coeffs = qml.math.convert_like(coeffs, result)
    g = qml.math.tensordot(result, coeffs, [[0], [0]])
    if unshifted_coeff is not None:
        # add the unshifted term
        g = g + unshifted_coeff * r0
        g = qml.math.array(g)
    return g


def _multi_meas_grad(res, coeffs, r0, unshifted_coeff, num_measurements):
    """Compute the gradient for multiple measurements by taking the linear combination of
    the coefficients and each measurement result."""
    g = []
    if r0 is None:
        r0 = [None] * num_measurements
    for meas_idx in range(num_measurements):
        # Gather the measurement results
        meas_result = [param_result[meas_idx] for param_result in res]
        g_component = _single_meas_grad(meas_result, coeffs, unshifted_coeff, r0[meas_idx])
        g.append(g_component)

    return tuple(g)


def _evaluate_gradient(tape, res, data, r0):
    """Use shifted tape evaluations and parameter-shift rule coefficients to evaluate
    a gradient result. If res is an empty list, ``r0`` and ``data[3]``, which is the
    coefficient for the unshifted term, must be given and not None.
    """

    _, coeffs, fn, unshifted_coeff, _ = data

    # individual post-processing of e.g. Hamiltonian grad tapes
    if fn is not None:
        res = fn(res)

    num_measurements = len(tape.measurements)

    if num_measurements == 1:
        if not tape.shots.has_partitioned_shots:
            return _single_meas_grad(res, coeffs, unshifted_coeff, r0)
        g = []
        len_shot_vec = tape.shots.num_copies
        # Res has order of axes:
        # 1. Number of parameters
        # 2. Shot vector
        if r0 is None:
            r0 = [None] * int(len_shot_vec)
        for i in range(len_shot_vec):
            shot_comp_res = [r[i] for r in res]
            shot_comp_res = _single_meas_grad(shot_comp_res, coeffs, unshifted_coeff, r0[i])
            g.append(shot_comp_res)
        return tuple(g)

    g = []
    if not tape.shots.has_partitioned_shots:
        return _multi_meas_grad(res, coeffs, r0, unshifted_coeff, num_measurements)

    # Res has order of axes:
    # 1. Number of parameters
    # 2. Shot vector
    # 3. Number of measurements
    for idx_shot_comp in range(tape.shots.num_copies):
        single_shot_component_result = [
            result_for_each_param[idx_shot_comp] for result_for_each_param in res
        ]
        multi_meas_grad = _multi_meas_grad(
            single_shot_component_result, coeffs, r0, unshifted_coeff, num_measurements
        )
        g.append(multi_meas_grad)

    return tuple(g)


def _get_operation_recipe(tape, t_idx, shifts, order=1):
    """Utility function to return the parameter-shift rule
    of the operation corresponding to trainable parameter
    t_idx on tape.

    Args:
        tape (.QuantumTape): Tape containing the operation to differentiate
        t_idx (int): Parameter index of the operation to differentiate within the tape
        shifts (Sequence[float or int]): Shift values to use if no static ``grad_recipe`` is
            provided by the operation to differentiate
        order (int): Order of the differentiation

    This function performs multiple attempts to obtain the recipe:

    - If the operation has a custom :attr:`~.grad_recipe` defined, it is used.

    - If :attr:`.parameter_frequencies` yields a result, the frequencies are
      used to construct the general parameter-shift rule via
      :func:`.generate_shift_rule`.
      Note that by default, the generator is used to compute the parameter frequencies
      if they are not provided by a custom implementation.

    That is, the order of precedence is :meth:`~.grad_recipe`, custom
    :attr:`~.parameter_frequencies`, and finally :meth:`.generator` via the default
    implementation of the frequencies.

    If order is set to 2, the rule for the second-order derivative is obtained instead.
    """
    if order not in {1, 2}:
        raise NotImplementedError("_get_operation_recipe only is implemented for orders 1 and 2.")

    op, _, p_idx = tape.get_operation(t_idx)

    # Try to use the stored grad_recipe of the operation
    op_recipe = _process_op_recipe(op, p_idx, order)
    if op_recipe is not None:
        return op_recipe

    # Try to obtain frequencies, either via custom implementation or from generator eigvals
    try:
        frequencies = op.parameter_frequencies[p_idx]
    except qml.operation.ParameterFrequenciesUndefinedError as e:
        raise qml.operation.OperatorPropertyUndefined(
            f"The operation {op.name} does not have a grad_recipe, parameter_frequencies or "
            "a generator defined. No parameter shift rule can be applied."
        ) from e

    # Create shift rule from frequencies with given shifts
    coeffs, shifts = qml.gradients.generate_shift_rule(frequencies, shifts=shifts, order=order).T
    # The generated shift rules do not include a rescaling of the parameter, only shifts.
    mults = np.ones_like(coeffs)

    return qml.math.stack([coeffs, mults, shifts]).T


def _make_zero_rep(g, single_measure, has_partitioned_shots, par_shapes=None):
    """Create a zero-valued gradient entry adapted to the measurements and shot_vector
    of a gradient computation, where g is a previously computed non-zero gradient entry.

    Args:
        g (tensor_like): Gradient entry that was computed for a different parameter, from which
            we inherit the shape and data type of the zero-valued entry to create
        single_measure (bool): Whether the differentiated function returned a single measurement.
        has_partitioned_shots (bool): Whether the differentiated function used a shot vector.
        par_shapes (tuple(tuple)): Shapes of the parameter for which ``g`` is the gradient entry,
            and of the parameter for which to create a zero-valued gradient entry, in this order.

    Returns:
        tensor_like or tuple(tensor_like) or tuple(tuple(tensor_like)): Zero-valued gradient entry
        similar to the non-zero gradient entry ``g``, potentially adapted to differences between
        parameter shapes if ``par_shapes`` were provided.

    """
    cut_dims, par_shape = (len(par_shapes[0]), par_shapes[1]) if par_shapes else (0, ())

    if par_shapes is None:
        zero_entry = qml.math.zeros_like
    else:

        def zero_entry(grad_entry):
            """Create a gradient entry that is zero and has the correctly modified shape."""
            new_shape = par_shape + qml.math.shape(grad_entry)[cut_dims:]
            return qml.math.zeros(new_shape, like=grad_entry)

    if single_measure and not has_partitioned_shots:
        return zero_entry(g)
    if single_measure or not has_partitioned_shots:
        return tuple(map(zero_entry, g))
    return tuple(tuple(map(zero_entry, shot_comp_g)) for shot_comp_g in g)


def expval_param_shift(
    tape, argnum=None, shifts=None, gradient_recipes=None, f0=None, broadcast=False
):
    r"""Generate the parameter-shift tapes and postprocessing methods required
        to compute the gradient of a gate parameter with respect to an
        expectation value.

        The returned post-processing function will output tuples instead of
    stacking resaults.

        Args:
            tape (.QuantumTape): quantum tape to differentiate
            argnum (int or list[int] or None): Trainable parameter indices to differentiate
                with respect to. If not provided, the derivatives with respect to all
                trainable indices are returned.
            shifts (list[tuple[int or float]]): List containing tuples of shift values.
                If provided, one tuple of shifts should be given per trainable parameter
                and the tuple should match the number of frequencies for that parameter.
                If unspecified, equidistant shifts are assumed.
            gradient_recipes (tuple(list[list[float]] or None)): List of gradient recipes
                for the parameter-shift method. One gradient recipe must be provided
                per trainable parameter.
            f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
                and the gradient recipe contains an unshifted term, this value is used,
                saving a quantum evaluation.
            broadcast (bool): Whether or not to use parameter broadcasting to create the
                a single broadcasted tape per operation instead of one tape per shift angle.

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing a
            list of generated tapes, together with a post-processing
            function to be applied to the results of the evaluated tapes
            in order to obtain the Jacobian matrix.
    """

    argnum = argnum or tape.trainable_params

    gradient_tapes = []
    # Each entry for gradient_data will be a tuple with entries
    # (num_tapes, coeffs, fn, unshifted_coeff, batch_size)
    gradient_data = []
    # Keep track of whether there is at least one unshifted term in all the parameter-shift rules
    at_least_one_unshifted = False

    for idx, _ in enumerate(tape.trainable_params):
        if idx not in argnum:
            # parameter has zero gradient
            gradient_data.append((0, [], None, None, 0))
            continue

        op, *_ = tape.get_operation(idx)

        if op.name == "Hamiltonian":
            # operation is a Hamiltonian
            if op.return_type is not qml.measurements.Expectation:
                raise ValueError(
                    "Can only differentiate Hamiltonian "
                    f"coefficients for expectations, not {op.return_type.value}"
                )

            g_tapes, h_fn = qml.gradients.hamiltonian_grad(tape, idx)
            gradient_tapes.extend(g_tapes)
            # hamiltonian_grad always returns a list with a single tape!
            gradient_data.append((1, np.array([1.0]), h_fn, None, g_tapes[0].batch_size))
            continue

        recipe = _choose_recipe(argnum, idx, gradient_recipes, shifts, tape)
        recipe, at_least_one_unshifted, unshifted_coeff = _extract_unshifted(
            recipe, at_least_one_unshifted, f0, gradient_tapes, tape
        )
        coeffs, multipliers, op_shifts = recipe.T

        g_tapes = generate_shifted_tapes(tape, idx, op_shifts, multipliers, broadcast)
        gradient_tapes.extend(g_tapes)
        # If broadcast=True, g_tapes only contains one tape. If broadcast=False, all returned
        # tapes will have the same batch_size=None. Thus we only use g_tapes[0].batch_size here.
        # If no gradient tapes are returned (e.g. only unshifted term in recipe), batch_size=None
        batch_size = g_tapes[0].batch_size if broadcast and g_tapes else None
        gradient_data.append((len(g_tapes), coeffs, None, unshifted_coeff, batch_size))

    num_measurements = len(tape.measurements)
    single_measure = num_measurements == 1
    num_params = len(tape.trainable_params)
    tape_specs = (single_measure, num_params, num_measurements, tape.shots)

    def processing_fn(results):
        start, r0 = (1, results[0]) if at_least_one_unshifted and f0 is None else (0, f0)
        grads = []
        for data in gradient_data:
            num_tapes, *_, unshifted_coeff, batch_size = data
            if num_tapes == 0:
                if unshifted_coeff is None:
                    # parameter has zero gradient. We don't know the output shape yet, so just
                    # memorize that this gradient will be set to zero, via grad = None
                    grads.append(None)
                    continue
                # The gradient for this parameter is computed from r0 alone.
                g = _evaluate_gradient(tape, [], data, r0)
                grads.append(g)
                continue

            res = results[start : start + num_tapes] if batch_size is None else results[start]
            start = start + num_tapes

            g = _evaluate_gradient(tape, res, data, r0)
            grads.append(g)

        # g will have been defined at least once (because otherwise all gradients would have
        # been zero), providing a representative for a zero gradient to emulate its type/shape.
        zero_rep = _make_zero_rep(g, single_measure, tape.shots.has_partitioned_shots)

        # Fill in zero-valued gradients
        grads = [zero_rep if g is None else g for g in grads]

        return reorder_grads(grads, tape_specs)

    processing_fn.first_result_unshifted = at_least_one_unshifted

    return gradient_tapes, processing_fn


def _get_var_with_second_order(pdA2, f0, pdA):
    """Auxiliary function to compute d(var(A))/dp = d<A^2>/dp -2 * <A> *
    d<A>/dp for the variances.

    The result is converted to an array-like object as some terms (specifically f0) may not be array-like in every case.

    Args:
        pdA (tensor_like[float]): The analytic derivative of <A>.
        pdA2 (tensor_like[float]): The analytic derivatives of the <A^2> observables.
        f0 (tensor_like[float]): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.
    """
    # Only necessary for numpy array with shape () not to be float
    if any(isinstance(term, np.ndarray) for term in [pdA2, f0, pdA]):
        # It breaks differentiability for Torch
        return qml.math.array(pdA2 - 2 * f0 * pdA)
    return pdA2 - 2 * f0 * pdA


def _put_zeros_in_pdA2_involutory(tape, pdA2, involutory_indices):
    """Auxiliary function for replacing parts of the partial derivative of <A^2>
    with the zero gradient of involutory observables.

    Involutory observables in the partial derivative of <A^2> have zero gradients.
    For the pdA2_tapes, we have replaced non-involutory observables with their
    square (A -> A^2). However, involutory observables have been left as-is
    (A), and have not been replaced by their square (A^2 = I). As a result,
    components of the gradient vector will not be correct. We need to replace
    the gradient value with 0 (the known, correct gradient for involutory
    variables).
    """
    new_pdA2 = []
    for i in range(len(tape.measurements)):
        if i in involutory_indices:
            num_params = len(tape.trainable_params)
            item = (
                qml.math.array(0)
                if num_params == 1
                else tuple(qml.math.array(0) for _ in range(num_params))
            )
        else:
            item = pdA2[i]
        new_pdA2.append(item)

    return tuple(new_pdA2)


def _get_pdA2(results, tape, pdA2_fn, non_involutory_indices, var_indices):
    """The main auxiliary function to get the partial derivative of <A^2>."""
    pdA2 = 0

    if non_involutory_indices:
        # compute the second derivative of non-involutory observables
        pdA2 = pdA2_fn(results)

        # For involutory observables (A^2 = I) we have d<A^2>/dp = 0.
        if involutory := set(var_indices) - set(non_involutory_indices):
            if tape.shots.has_partitioned_shots:
                pdA2 = tuple(
                    _put_zeros_in_pdA2_involutory(tape, pdA2_shot_comp, involutory)
                    for pdA2_shot_comp in pdA2
                )
            else:
                pdA2 = _put_zeros_in_pdA2_involutory(tape, pdA2, involutory)
    return pdA2


def _single_variance_gradient(tape, var_mask, pdA2, f0, pdA):
    """Auxiliary function to return the derivative of variances.

    return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances (var_mask==True)
    d<A>/dp for plain expectations (var_mask==False)

    Note: if isinstance(pdA2, int) and pdA2 != 0, then len(pdA2) == len(pdA)

    """
    num_params = len(tape.trainable_params)
    num_measurements = len(tape.measurements)
    if num_measurements > 1:
        if num_params == 1:
            var_grad = []

            for m_idx in range(num_measurements):
                m_res = pdA[m_idx]
                if var_mask[m_idx]:
                    pdA2_comp = pdA2[m_idx] if pdA2 != 0 else pdA2
                    f0_comp = f0[m_idx]
                    m_res = _get_var_with_second_order(pdA2_comp, f0_comp, m_res)

                var_grad.append(m_res)
            return tuple(var_grad)

        var_grad = []
        for m_idx in range(num_measurements):
            m_res = []
            if var_mask[m_idx]:
                for p_idx in range(num_params):
                    pdA2_comp = pdA2[m_idx][p_idx] if pdA2 != 0 else pdA2
                    f0_comp = f0[m_idx]
                    pdA_comp = pdA[m_idx][p_idx]
                    r = _get_var_with_second_order(pdA2_comp, f0_comp, pdA_comp)
                    m_res.append(r)
                m_res = tuple(m_res)
            else:
                m_res = tuple(pdA[m_idx][p_idx] for p_idx in range(num_params))
            var_grad.append(m_res)
        return tuple(var_grad)

    # Single measurement case, meas has to be variance
    if num_params == 1:
        return _get_var_with_second_order(pdA2, f0, pdA)

    var_grad = []

    for p_idx in range(num_params):
        pdA2_comp = pdA2[p_idx] if pdA2 != 0 else pdA2
        r = _get_var_with_second_order(pdA2_comp, f0, pdA[p_idx])
        var_grad.append(r)

    return tuple(var_grad)


def _create_variance_proc_fn(
    tape, var_mask, var_indices, pdA_fn, pdA2_fn, tape_boundary, non_involutory_indices
):
    """Auxiliary function to define the processing function for computing the
    derivative of variances using the parameter-shift rule.

    Args:
        var_mask (list): The mask of variance measurements in the measurement queue.
        var_indices (list): The indices of variance measurements in the measurement queue.
        pdA_fn (callable): The function required to evaluate the analytic derivative of <A>.
        pdA2_fn (callable): If not None, non-involutory observables are
            present; the partial derivative of <A^2> may be non-zero. Here, we
            calculate the analytic derivatives of the <A^2> observables.
        tape_boundary (callable): the number of first derivative tapes used to
            determine the number of results to post-process later
        non_involutory_indices (list): the indices in the measurement queue of all non-involutory
            observables
    """

    def processing_fn(results):
        f0 = results[0]

        # analytic derivative of <A>
        pdA = pdA_fn(results[int(not pdA_fn.first_result_unshifted) : tape_boundary])

        # analytic derivative of <A^2>
        pdA2 = _get_pdA2(
            results[tape_boundary:],
            tape,
            pdA2_fn,
            non_involutory_indices,
            var_indices,
        )

        # The logic follows:
        # variances (var_mask==True): return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp
        # plain expectations (var_mask==False): return d<A>/dp
        # Note: if pdA2 != 0, then len(pdA2) == len(pdA)
        if tape.shots.has_partitioned_shots:
            final_res = []
            for idx_shot_comp in range(tape.shots.num_copies):
                f0_comp = f0[idx_shot_comp]

                pdA_comp = pdA[idx_shot_comp]

                pdA2_comp = pdA2 if isinstance(pdA2, int) else pdA2[idx_shot_comp]
                r = _single_variance_gradient(tape, var_mask, pdA2_comp, f0_comp, pdA_comp)
                final_res.append(r)

            return tuple(final_res)

        return _single_variance_gradient(tape, var_mask, pdA2, f0, pdA)

    return processing_fn


def var_param_shift(tape, argnum, shifts=None, gradient_recipes=None, f0=None, broadcast=False):
    r"""Generate the parameter-shift tapes and postprocessing methods required
    to compute the gradient of a gate parameter with respect to a
    variance value.

    The post-processing function returns a tuple instead of stacking results.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivative with respect to all
            trainable indices are returned.
        shifts (list[tuple[int or float]]): List containing tuples of shift values.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are assumed.
        gradient_recipes (tuple(list[list[float]] or None)): List of gradient recipes
            for the parameter-shift method. One gradient recipe must be provided
            per trainable parameter.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.
        broadcast (bool): Whether or not to use parameter broadcasting to create the
            a single broadcasted tape per operation instead of one tape per shift angle.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
        list of generated tapes, together with a post-processing
        function to be applied to the results of the evaluated tapes
        in order to obtain the Jacobian matrix.
    """

    argnum = argnum or tape.trainable_params

    # Determine the locations of any variance measurements in the measurement queue.
    var_mask = [isinstance(m, VarianceMP) for m in tape.measurements]
    var_indices = np.where(var_mask)[0]

    # Get <A>, the expectation value of the tape with unshifted parameters.
    expval_tape = tape.copy(copy_operations=True)

    # Convert all variance measurements on the tape into expectation values
    for i in var_indices:
        obs = expval_tape.measurements[i].obs
        expval_tape._measurements[i] = qml.expval(op=obs)

    # evaluate the analytic derivative of <A>
    pdA_tapes, pdA_fn = expval_param_shift(
        expval_tape, argnum, shifts, gradient_recipes, f0, broadcast
    )
    gradient_tapes = [] if pdA_fn.first_result_unshifted else [expval_tape]
    gradient_tapes.extend(pdA_tapes)

    # Store the number of first derivative tapes, so that we know
    # the number of results to post-process later.
    tape_boundary = len(gradient_tapes)

    # If there are non-involutory observables A present, we must compute d<A^2>/dp.
    # Get the indices in the measurement queue of all non-involutory
    # observables.
    non_involutory_indices = []

    for i in var_indices:
        obs_name = tape.observables[i].name

        if isinstance(obs_name, list):
            # Observable is a tensor product, we must investigate all constituent observables.
            if any(name in NONINVOLUTORY_OBS for name in obs_name):
                non_involutory_indices.append(i)

        elif obs_name in NONINVOLUTORY_OBS:
            non_involutory_indices.append(i)

    pdA2_fn = None
    if non_involutory_indices:
        tape_with_obs_squared_expval = tape.copy(copy_operations=True)

        for i in non_involutory_indices:
            # We need to calculate d<A^2>/dp; to do so, we replace the
            # involutory observables A in the queue with A^2.
            obs = _square_observable(tape_with_obs_squared_expval.measurements[i].obs)
            tape_with_obs_squared_expval._measurements[i] = qml.expval(op=obs)

        # Non-involutory observables are present; the partial derivative of <A^2>
        # may be non-zero. Here, we calculate the analytic derivatives of the <A^2>
        # observables.
        pdA2_tapes, pdA2_fn = expval_param_shift(
            tape_with_obs_squared_expval, argnum, shifts, gradient_recipes, f0, broadcast
        )
        gradient_tapes.extend(pdA2_tapes)

    processing_fn = _create_variance_proc_fn(
        tape, var_mask, var_indices, pdA_fn, pdA2_fn, tape_boundary, non_involutory_indices
    )
    return gradient_tapes, processing_fn


def _expand_transform_param_shift(
    tape: qml.tape.QuantumTape,
    argnum=None,
    shifts=None,
    gradient_recipes=None,
    fallback_fn=finite_diff,
    f0=None,
    broadcast=False,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Expand function to be applied before parameter shift."""
    expanded_tape = expand_invalid_trainable(tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [expanded_tape], null_postprocessing


@partial(
    transform,
    expand_transform=_expand_transform_param_shift,
    classical_cotransform=_contract_qjac_with_cjac,
    final_transform=True,
)
def param_shift(
    tape: qml.tape.QuantumTape,
    argnum=None,
    shifts=None,
    gradient_recipes=None,
    fallback_fn=finite_diff,
    f0=None,
    broadcast=False,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    r"""Transform a circuit to compute the parameter-shift gradient of all gate
    parameters with respect to its inputs.

    Args:
        tape (QNode or QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivative with respect to all
            trainable indices are returned.
        shifts (list[tuple[int or float]]): List containing tuples of shift values.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are assumed.
        gradient_recipes (tuple(list[list[float]] or None)): List of gradient recipes
            for the parameter-shift method. One gradient recipe must be provided
            per trainable parameter.

            This is a tuple with one nested list per parameter. For
            parameter :math:`\phi_k`, the nested list contains elements of the form
            :math:`[c_i, a_i, s_i]` where :math:`i` is the index of the
            term, resulting in a gradient recipe of

            .. math:: \frac{\partial}{\partial\phi_k}f = \sum_{i} c_i f(a_i \phi_k + s_i).

            If ``None``, the default gradient recipe containing the two terms
            :math:`[c_0, a_0, s_0]=[1/2, 1, \pi/2]` and :math:`[c_1, a_1,
            s_1]=[-1/2, 1, -\pi/2]` is assumed for every parameter.
        fallback_fn (None or Callable): a fallback gradient function to use for
            any parameters that do not support the parameter-shift rule.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.
        broadcast (bool): Whether or not to use parameter broadcasting to create the
            a single broadcasted tape per operation instead of one tape per shift angle.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Jacobian in the form of a tensor, a tuple, or a nested tuple depending upon the nesting
        structure of measurements in the original circuit.

    For a variational evolution :math:`U(\mathbf{p}) \vert 0\rangle` with
    :math:`N` parameters :math:`\mathbf{p}`,
    consider the expectation value of an observable :math:`O`:

    .. math::

        f(\mathbf{p})  = \langle \hat{O} \rangle(\mathbf{p}) = \langle 0 \vert
        U(\mathbf{p})^\dagger \hat{O} U(\mathbf{p}) \vert 0\rangle.


    The gradient of this expectation value can be calculated via the parameter-shift rule:

    .. math::

        \frac{\partial f}{\partial \mathbf{p}} = \sum_{\mu=1}^{2R}
        f\left(\mathbf{p}+\frac{2\mu-1}{2R}\pi\right)
        \frac{(-1)^{\mu-1}}{4R\sin^2\left(\frac{2\mu-1}{4R}\pi\right)}

    Here, :math:`R` is the number of frequencies with which the parameter :math:`\mathbf{p}`
    enters the function :math:`f` via the operation :math:`U`, and we assumed that these
    frequencies are equidistant.
    For more general shift rules, both regarding the shifts and the frequencies, and
    for more technical details, see
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ and
    `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`_.

    **Gradients of variances**

    For a variational evolution :math:`U(\mathbf{p}) \vert 0\rangle` with
    :math:`N` parameters :math:`\mathbf{p}`,
    consider the variance of an observable :math:`O`:

    .. math::

        g(\mathbf{p})=\langle \hat{O}^2 \rangle (\mathbf{p}) - [\langle \hat{O}
        \rangle(\mathbf{p})]^2.

    We can relate this directly to the parameter-shift rule by noting that

    .. math::

        \frac{\partial g}{\partial \mathbf{p}}= \frac{\partial}{\partial
        \mathbf{p}} \langle \hat{O}^2 \rangle (\mathbf{p})
        - 2 f(\mathbf{p}) \frac{\partial f}{\partial \mathbf{p}}.

    The derivatives in the expression on the right hand side can be computed via
    the shift rule as above, allowing for the computation of the variance derivative.

    In the case where :math:`O` is involutory (:math:`\hat{O}^2 = I`), the first
    term in the above expression vanishes, and we are simply left with

    .. math::

      \frac{\partial g}{\partial \mathbf{p}} = - 2 f(\mathbf{p})
      \frac{\partial f}{\partial \mathbf{p}}.

    **Example**

    This transform can be registered directly as the quantum gradient transform
    to use during autodifferentiation:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.jacobian(circuit)(params)
    array([-0.3875172 , -0.18884787, -0.38355704])

    When differentiating QNodes with multiple measurements using Autograd or TensorFlow, the outputs of the QNode first
    need to be stacked. The reason is that those two frameworks only allow differentiating functions with array or
    tensor outputs, instead of functions that output sequences. In contrast, Jax and Torch require no additional
    post-processing.

    >>> import jax
    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev, interface="jax", diff_method="parameter-shift")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
    >>> params = jax.numpy.array([0.1, 0.2, 0.3])
    >>> jax.jacobian(circuit)(params)
    (Array([-0.38751727, -0.18884793, -0.3835571 ], dtype=float32), Array([0.6991687 , 0.34072432, 0.6920237 ], dtype=float32))

    .. note::

        ``param_shift`` performs multiple attempts to obtain the gradient recipes for
        each operation:

        - If an operation has a custom :attr:`~.operation.Operation.grad_recipe` defined,
          it is used.

        - If :attr:`~.operation.Operation.parameter_frequencies` yields a result, the frequencies
          are used to construct the general parameter-shift rule via
          :func:`.generate_shift_rule`.
          Note that by default, the generator is used to compute the parameter frequencies
          if they are not provided via a custom implementation.

        That is, the order of precedence is :attr:`~.operation.Operation.grad_recipe`, custom
        :attr:`~.operation.Operation.parameter_frequencies`, and finally
        :meth:`~.operation.Operation.generator` via the default implementation of the frequencies.

    .. warning::

        Note that using parameter broadcasting via ``broadcast=True`` is not supported for tapes
        with multiple return values or for evaluations with shot vectors.
        As the option ``broadcast=True`` adds a broadcasting dimension, it is not compatible
        with circuits that already are broadcasted.
        Finally, operations with trainable parameters are required to support broadcasting.
        One way of checking this is the `Attribute` `supports_broadcasting`:

        >>> qml.RX in qml.ops.qubit.attributes.supports_broadcasting
        True

    .. details::
        :title: Usage Details

        This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>` objects.
        However, for performance reasons, we recommend providing the gradient transform as the ``diff_method`` argument
        of the QNode decorator, and differentiating with your preferred machine learning framework.

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
        >>> qml.gradients.param_shift(circuit)(params)
        ((tensor(-0.38751724, requires_grad=True),
          tensor(-0.18884792, requires_grad=True),
          tensor(-0.38355709, requires_grad=True)),
         (array(0.69916862), array(0.34072424), array(0.69202359)))

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> ops = [qml.RX(params[0], 0), qml.RY(params[1], 0), qml.RX(params[2], 0)]
        >>> measurements = [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))]
        >>> tape = qml.tape.QuantumTape(ops, measurements)
        >>> gradient_tapes, fn = qml.gradients.param_shift(tape)
        >>> gradient_tapes
        [<QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[0, 1], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        ((tensor(-0.3875172, requires_grad=True),
          tensor(-0.18884787, requires_grad=True),
          tensor(-0.38355704, requires_grad=True)),
         (array(0.69916862), array(0.34072424), array(0.69202359)))

        This gradient transform is compatible with devices that use shot vectors for execution.

        >>> shots = (10, 100, 1000)
        >>> dev = qml.device("default.qubit", wires=2, shots=shots)
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.param_shift(circuit)(params)
        (((array(-0.6), array(-0.1), array(-0.1)),
          (array(1.2), array(0.2), array(0.2))),
         ((array(-0.39), array(-0.24), array(-0.49)),
          (array(0.7488), array(0.4608), array(0.9408))),
         ((array(-0.36), array(-0.191), array(-0.37)),
          (array(0.65808), array(0.349148), array(0.67636))))

        The outermost tuple contains results corresponding to each element of the shot vector.

        When setting the keyword argument ``broadcast`` to ``True``, the shifted
        circuit evaluations for each operation are batched together, resulting in
        broadcasted tapes:

        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> ops = [qml.RX(p, wires=0) for p in params]
        >>> measurements = [qml.expval(qml.PauliZ(0))]
        >>> tape = qml.tape.QuantumTape(ops, measurements)
        >>> gradient_tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        >>> len(gradient_tapes)
        3
        >>> [t.batch_size for t in gradient_tapes]
        [2, 2, 2]

        The postprocessing function will know that broadcasting is used and handle the results accordingly:

        >>> fn(qml.execute(gradient_tapes, dev, None))
        (array(-0.3875172), array(-0.18884787), array(-0.38355704))

        An advantage of using ``broadcast=True`` is a speedup:

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.PauliZ(0))
        >>> number = 100
        >>> serial_call = "qml.gradients.param_shift(circuit, broadcast=False)(params)"
        >>> timeit.timeit(serial_call, globals=globals(), number=number) / number
        0.020183045039993887
        >>> broadcasted_call = "qml.gradients.param_shift(circuit, broadcast=True)(params)"
        >>> timeit.timeit(broadcasted_call, globals=globals(), number=number) / number
        0.01244492811998498

        This speedup grows with the number of shifts and qubits until all preprocessing and
        postprocessing overhead becomes negligible. While it will depend strongly on the details
        of the circuit, at least a small improvement can be expected in most cases.
        Note that ``broadcast=True`` requires additional memory by a factor of the largest
        batch_size of the created tapes.
    """

    transform_name = "parameter-shift rule"
    assert_no_state_returns(tape.measurements, transform_name)
    assert_multimeasure_not_broadcasted(tape.measurements, broadcast)
    assert_no_tape_batching(tape, transform_name)

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad(tape)

    method = "analytic" if fallback_fn is None else "best"
    diff_methods = gradient_analysis_and_validation(tape, method, grad_fn=param_shift)

    if all(g == "0" for g in diff_methods):
        return _all_zero_grad(tape)

    method_map = choose_grad_methods(diff_methods, argnum)

    # If there are unsupported operations, call the fallback gradient function
    unsupported_params = {idx for idx, g in method_map.items() if g == "F"}
    argnum = [i for i, dm in method_map.items() if dm == "A"]
    gradient_tapes = []

    if unsupported_params:
        if not argnum:
            return fallback_fn(tape)

        g_tapes, fallback_proc_fn = fallback_fn(tape, argnum=unsupported_params)
        gradient_tapes.extend(g_tapes)
        fallback_len = len(g_tapes)

        # remove finite difference parameters from the method map
        method_map = {t_idx: dm for t_idx, dm in method_map.items() if dm != "F"}

    # Generate parameter-shift gradient tapes

    if gradient_recipes is None:
        gradient_recipes = [None] * len(argnum)

    if any(isinstance(m, VarianceMP) for m in tape.measurements):
        g_tapes, fn = var_param_shift(tape, argnum, shifts, gradient_recipes, f0, broadcast)
    else:
        g_tapes, fn = expval_param_shift(tape, argnum, shifts, gradient_recipes, f0, broadcast)

    gradient_tapes.extend(g_tapes)

    if unsupported_params:

        def _single_shot_batch_grad(unsupported_grads, supported_grads):
            """Auxiliary function for post-processing one batch of supported and unsupported gradients corresponding to
            finite shot execution.

            If the device used a shot vector, gradients corresponding to a single component of the shot vector should be
            passed to this aux function.
            """
            multi_measure = len(tape.measurements) > 1
            if not multi_measure:
                res = []
                for i, j in zip(unsupported_grads, supported_grads):
                    component = qml.math.array(i + j)
                    res.append(component)
                return tuple(res)

            combined_grad = []
            for meas_res1, meas_res2 in zip(unsupported_grads, supported_grads):
                meas_grad = []
                for param_res1, param_res2 in zip(meas_res1, meas_res2):
                    component = qml.math.array(param_res1 + param_res2)
                    meas_grad.append(component)

                meas_grad = tuple(meas_grad)
                combined_grad.append(meas_grad)
            return tuple(combined_grad)

        # If there are unsupported parameters, we must process
        # the quantum results separately, once for the fallback
        # function and once for the parameter-shift rule, and recombine.
        def processing_fn(results):
            unsupported_res = results[:fallback_len]
            supported_res = results[fallback_len:]

            if not tape.shots.has_partitioned_shots:
                unsupported_grads = fallback_proc_fn(unsupported_res)
                supported_grads = fn(supported_res)
                return _single_shot_batch_grad(unsupported_grads, supported_grads)

            supported_grads = fn(supported_res)
            unsupported_grads = fallback_proc_fn(unsupported_res)

            final_grad = []
            for idx in range(tape.shots.num_copies):
                u_grads = unsupported_grads[idx]
                sup = supported_grads[idx]
                final_grad.append(_single_shot_batch_grad(u_grads, sup))
            return tuple(final_grad)

        return gradient_tapes, processing_fn

    return gradient_tapes, fn
