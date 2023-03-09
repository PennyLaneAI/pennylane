import warnings
from collections.abc import Sequence
from functools import partial

import numpy as np

import pennylane as qml
from pennylane._device import _get_num_copies
from pennylane.measurements import MutualInfoMP, StateMP, VarianceMP, VnEntropyMP
from pennylane.gradients.general_shift_rules import (
    _iterate_shift_rule,
    frequencies_to_period,
    generate_shifted_tapes,
    process_shifts,
)

from pennylane.transforms.experimental.transforms import transform

SUPPORTED_GRADIENT_KWARGS = [
    "approx_order",
    "argnum",
    "aux_wire",
    "broadcast",
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
    "off_diagonal_shifts",
    "order",
    "reduction",
    "sampler",
    "sampler_seed",
    "shifts",
    "shots",
    "strategy",
    "validate_params",
]


def gradient_analysis(tape, use_graph=True, grad_fn=None):
    """Update the parameter information dictionary of the tape with
    gradient information of each parameter.

    Parameter gradient methods include:

    * ``None``: the parameter does not support differentiation.

    * ``"0"``: the variational circuit output does not depend on this
      parameter (the partial derivative is zero).

    In addition, the operator might define its own grad method
    via :attr:`.Operator.grad_method`.

    Note that this function modifies the input tape in-place.

    Args:
        tape (.QuantumTape): the quantum tape to analyze
        use_graph (bool): whether to use a directed-acyclic graph to determine
            if the parameter has a gradient of 0
        grad_fn (None or callable): The gradient transform performing the analysis.
            This is an optional argument; if provided, and the tape has already
            been analyzed for the gradient information by the same gradient transform,
            the cached gradient analysis will be used.
    """
    # pylint:disable=protected-access
    if grad_fn is not None:
        if getattr(tape, "_gradient_fn", None) is grad_fn:
            # gradient analysis has already been performed on this tape
            return

        tape._gradient_fn = grad_fn

    for idx, info in enumerate(tape._par_info):
        if idx not in tape.trainable_params:
            # non-trainable parameters do not require a grad_method
            info["grad_method"] = None
        else:
            op = tape._par_info[idx]["op"]

            if not qml.operation.has_grad_method(op):
                # no differentiation method is registered for this operation
                info["grad_method"] = None

            elif (tape._graph is not None) or use_graph:
                if not any(tape.graph.has_path(op, ob) for ob in tape.observables):
                    # there is no influence of this operation on any of the observables
                    info["grad_method"] = "0"
                    continue

            info["grad_method"] = op.grad_method


def grad_method_validation(method, tape):
    """Validates if the gradient method requested is supported by the trainable
    parameters of a tape, and returns the allowed parameter gradient methods.

    This method will generate parameter gradient information for the given tape if it
    has not already been generated, and then proceed to validate the gradient method.
    In particular:

    * An exception will be raised if there exist non-differentiable trainable
      parameters on the tape.

    * An exception will be raised if the Jacobian method is ``"analytic"`` but there
      exist some trainable parameters on the tape that only support numeric differentiation.

    If all validations pass, this method will return a tuple containing the allowed parameter
    gradient methods for each trainable parameter.

    Args:
        method (str): the overall Jacobian differentiation method
        tape (.QuantumTape): the tape with associated parameter information

    Returns:
        tuple[str, None]: the allowed parameter gradient methods for each trainable parameter
    """
    diff_methods = {
        idx: info["grad_method"]
        for idx, info in enumerate(tape._par_info)  # pylint: disable=protected-access
        if idx in tape.trainable_params
    }

    # check and raise an error if any parameters are non-differentiable
    nondiff_params = {idx for idx, g in diff_methods.items() if g is None}

    if nondiff_params:
        raise ValueError(f"Cannot differentiate with respect to parameter(s) {nondiff_params}")

    numeric_params = {idx for idx, g in diff_methods.items() if g == "F"}

    # If explicitly using analytic mode, ensure that all parameters
    # support analytic differentiation.
    if method == "analytic" and numeric_params:
        raise ValueError(
            f"The analytic gradient method cannot be used with the parameter(s) {numeric_params}."
        )

    return tuple(diff_methods.values())


def choose_grad_methods(diff_methods, argnum):
    """Chooses the trainable parameters to use for computing the Jacobian
    by returning a map of their indices and differentiation methods.

    When there are fewer parameters specified than the total number of
    trainable parameters, the Jacobian is estimated by using the parameters
    specified using the ``argnum`` keyword argument.

    Args:
        diff_methods (list): the ordered list of differentiation methods
            for each parameter
        argnum (int, list(int), None): Indices for argument(s) with respect
            to which to compute the Jacobian.

    Returns:
        dict: map of the trainable parameter indices and
        differentiation methods
    """
    if argnum is None:
        return dict(enumerate(diff_methods))

    if isinstance(argnum, int):
        argnum = [argnum]

    num_params = len(argnum)

    if num_params == 0:
        warnings.warn(
            "No trainable parameters were specified for computing the Jacobian.",
            UserWarning,
        )
        return {}

    return {idx: diff_methods[idx] for idx in argnum}


NONINVOLUTORY_OBS = {
    "Hermitian": lambda obs: obs.__class__(obs.matrix() @ obs.matrix(), wires=obs.wires),
    "SparseHamiltonian": lambda obs: obs.__class__(obs.matrix() @ obs.matrix(), wires=obs.wires),
    "Projector": lambda obs: obs,
}


def expval_param_shift(
    tape, argnum=None, shifts=None, gradient_recipes=None, f0=None, broadcast=False, shots=None
):
    r"""Generate the parameter-shift tapes and postprocessing methods required
    to compute the gradient of a gate parameter with respect to an
    expectation value.
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

        op, *_ = tape.get_operation(idx, return_op_index=True)

        if op.name == "Hamiltonian":
            # operation is a Hamiltonian
            if op.return_type is not qml.measurements.Expectation:
                raise ValueError(
                    "Can only differentiate Hamiltonian "
                    f"coefficients for expectations, not {op.return_type.value}"
                )

            g_tapes, h_fn = qml.gradients.hamiltonian_grad(tape, idx)
            # hamiltonian_grad always returns a list with a single tape
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

    def processing_fn(results):
        start, r0 = (1, results[0]) if at_least_one_unshifted and f0 is None else (0, f0)
        single_measure = len(tape.measurements) == 1
        single_param = len(tape.trainable_params) == 1
        shot_vector = isinstance(shots, Sequence)

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
                g = _evaluate_gradient_new(tape, [], data, r0, shots)
                grads.append(g)
                continue

            res = results[start : start + num_tapes] if batch_size is None else results[start]
            start = start + num_tapes

            g = _evaluate_gradient_new(tape, res, data, r0, shots)
            grads.append(g)

        # g will have been defined at least once (because otherwise all gradients would have
        # been zero), providing a representative for a zero gradient to emulate its type/shape.
        if single_measure and not shot_vector:
            zero_rep = qml.math.zeros_like(g)
        elif single_measure:
            zero_rep = tuple(qml.math.zeros_like(shot_comp_g) for shot_comp_g in g)
        elif not shot_vector:
            zero_rep = tuple(qml.math.zeros_like(meas_g) for meas_g in g)
        else:
            zero_rep = tuple(
                tuple(qml.math.zeros_like(grad_component) for grad_component in shot_comp_g)
                for shot_comp_g in g
            )

        # Fill in zero-valued gradients
        grads = [g if g is not None else zero_rep for i, g in enumerate(grads)]

        if single_measure and single_param:
            return grads[0]

        num_params = len(tape.trainable_params)
        len_shot_vec = _get_num_copies(shots) if shot_vector else None
        if single_measure and shot_vector:
            return _reorder_grad_axes_single_measure_shot_vector(grads, num_params, len_shot_vec)
        if not single_measure:
            shot_vector_multi_measure = not single_measure and shot_vector
            num_measurements = len(tape.measurements)
            grads = _reorder_grad_axes_multi_measure(
                grads,
                num_params,
                num_measurements,
                len_shot_vec,
                shot_vector_multi_measure,
            )

        return tuple(grads)

    processing_fn.first_result_unshifted = at_least_one_unshifted

    return gradient_tapes, processing_fn


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


def _evaluate_gradient_new(tape, res, data, r0, shots):
    """Use shifted tape evaluations and parameter-shift rule coefficients to evaluate
    a gradient result. If res is an empty list, ``r0`` and ``data[3]``, which is the
    coefficient for the unshifted term, must be given and not None.

    This is a helper function for the new return type system.
    """

    _, coeffs, fn, unshifted_coeff, _ = data

    shot_vector = isinstance(shots, Sequence)

    # individual post-processing of e.g. Hamiltonian grad tapes
    if fn is not None:
        res = fn(res)

    num_measurements = len(tape.measurements)

    if num_measurements == 1:
        if not shot_vector:
            return _single_meas_grad(res, coeffs, unshifted_coeff, r0)
        g = []
        len_shot_vec = _get_num_copies(shots)
        # Res has order of axes:
        # 1. Number of parameters
        # 2. Shot vector
        if r0 is None:
            r0 = [None] * len_shot_vec
        for i in range(len_shot_vec):
            shot_comp_res = [r[i] for r in res]
            shot_comp_res = _single_meas_grad(shot_comp_res, coeffs, unshifted_coeff, r0[i])
            g.append(shot_comp_res)
        return tuple(g)

    g = []
    if not shot_vector:
        return _multi_meas_grad(res, coeffs, r0, unshifted_coeff, num_measurements)

    len_shot_vec = _get_num_copies(shots)
    # Res has order of axes:
    # 1. Number of parameters
    # 2. Shot vector
    # 3. Number of measurements
    for idx_shot_comp in range(len_shot_vec):
        single_shot_component_result = [
            result_for_each_param[idx_shot_comp] for result_for_each_param in res
        ]
        multi_meas_grad = _multi_meas_grad(
            single_shot_component_result, coeffs, r0, unshifted_coeff, num_measurements
        )
        g.append(multi_meas_grad)

    return tuple(g)


def _evaluate_gradient(res, data, broadcast, r0, scalar_qfunc_output):
    """Use shifted tape evaluations and parameter-shift rule coefficients to evaluate
    a gradient result. If res is an empty list, ``r0`` and ``data[3]``, which is the
    coefficient for the unshifted term, must be given and not None."""

    _, coeffs, fn, unshifted_coeff, batch_size = data

    if isinstance(res, list) and len(res) == 0:
        # No shifted evaluations are present, just the unshifted one.
        return r0 * unshifted_coeff

    # individual post-processing of e.g. Hamiltonian grad tapes
    if fn is not None:
        res = fn(res)

    # compute the linear combination of results and coefficients
    axis = 0
    if not broadcast:
        res = qml.math.stack(res)
    elif (
        qml.math.get_interface(res[0]) != "torch"
        and batch_size is not None
        and not scalar_qfunc_output
    ):
        # If the original output is not scalar and broadcasting is used, the second axis
        # (index 1) needs to be contracted. For Torch, this is not true because the
        # output of the broadcasted tape is flat due to the behaviour of the Torch device.
        axis = 1
    g = qml.math.tensordot(res, qml.math.convert_like(coeffs, res), [[axis], [0]])

    if unshifted_coeff is not None:
        # add the unshifted term
        g = g + unshifted_coeff * r0

    return g


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

    op, _, p_idx = tape.get_operation(t_idx, return_op_index=True)

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


def _swap_two_axes(grads, first_axis_size, second_axis_size):
    if first_axis_size == 1:
        return tuple(grads[0][i] for i in range(second_axis_size))
    return tuple(
        tuple(grads[j][i] for j in range(first_axis_size)) for i in range(second_axis_size)
    )


def _reorder_grad_axes_single_measure_shot_vector(grads, num_params, len_shot_vec):
    """Reorder the axes for gradient results obtained for a tape with a single measurement from a device that defined a
    shot vector.

    The order of axes of the gradient output matches the structure outputted by jax.jacobian for a tuple-valued
    function. Internally, this may not be the case when computing the gradients, so the axes are reordered here.

    The first axis always corresponds to the number of trainable parameters because the parameter-shift transform
    defines multiple tapes each of which corresponds to a trainable parameter. Those tapes are then executed using a
    device, which at the moment outputs results with the first axis corresponding to each tape output.

    The final order of axes of gradient results should be:
    1. Shot vector
    2. Measurements
    3. Number of trainable parameters (Num params)
    4. Broadcasting dimension
    5. Measurement shape

    According to the order above, the following reordering is done:

    Shot vectors:

        Go from
        1. Num params
        2. Shot vector
        3. Measurement shape

        To
        1. Shot vector
        2. Num params
        3. Measurement shape
    """
    return _swap_two_axes(grads, num_params, len_shot_vec)


def _reorder_grad_axes_multi_measure(
    grads, num_params, num_measurements, len_shot_vec, shot_vector_multi_measure
):
    """Reorder the axes for gradient results obtained for a tape with multiple measurements.

    The order of axes of the gradient output matches the structure outputted by jax.jacobian for a tuple-valued
    function. Internally, this may not be the case when computing the gradients, so the axes are reordered here.

    The first axis always corresponds to the number of trainable parameters because the parameter-shift transform
    defines multiple tapes each of which corresponds to a trainable parameter. Those tapes are then executed using a
    device, which at the moment outputs results with the first axis corresponding to each tape output.

    The final order of axes of gradient results should be:
    1. Shot vector
    2. Measurements
    3. Number of trainable parameters (Num params)
    4. Broadcasting dimension
    5. Measurement shape

    Parameter broadcasting doesn't yet support multiple measurements, hence such cases are not dealt with at the moment
    by this function.

    According to the order above, the following reorderings are done:

    A) Analytic (``shots=None``) or finite shots:

        Go from
        1. Num params
        2. Measurements
        3. Measurement shape

        To
        1. Measurements
        2. Num params
        3. Measurement shape

    B) Shot vectors:

        Go from
        1. Num params
        2. Shot vector
        3. Measurements
        4. Measurement shape

        To
        1. Shot vector
        2. Measurements
        3. Num params
        4. Measurement shape
    """
    multi_param = num_params > 1
    if not shot_vector_multi_measure:
        new_grad = _swap_two_axes(grads, num_params, num_measurements)
    else:
        new_grad = []
        for i in range(len_shot_vec):
            shot_vec_grad = []
            for j in range(num_measurements):
                measurement_grad = []
                for k in range(num_params):
                    measurement_grad.append(grads[k][i][j])

                measurement_grad = tuple(measurement_grad) if multi_param else measurement_grad[0]
                shot_vec_grad.append(measurement_grad)
            new_grad.append(tuple(shot_vec_grad))

    return new_grad


@transform
def param_shift_experimental(
    tape,
    argnum=None,
    shifts=None,
    gradient_recipes=None,
    f0=None,
    broadcast=False,
    shots=None,
):
    r"""Transform a QNode to compute the parameter-shift gradient of all gate
    parameters with respect to its inputs.
    """

    if any(isinstance(m, (StateMP, VnEntropyMP, MutualInfoMP)) for m in tape.measurements):
        raise ValueError(
            "Computing the gradient of circuits that return the state is not supported."
        )

    if broadcast and len(tape.measurements) > 1:
        raise NotImplementedError(
            "Broadcasting with multiple measurements is not supported yet. "
            f"Set broadcast to False instead. The tape measurements are {tape.measurements}."
        )

    gradient_analysis(tape, grad_fn=param_shift_experimental)
    method = "best"
    diff_methods = grad_method_validation(method, tape)

    method_map = choose_grad_methods(diff_methods, argnum)

    # If there are unsupported operations, call the fallback gradient function
    gradient_tapes = []
    argnum = [i for i, dm in method_map.items() if dm == "A"]

    # Generate parameter-shift gradient tapes

    if gradient_recipes is None:
        gradient_recipes = [None] * len(argnum)

    g_tapes, fn = expval_param_shift(tape, argnum, shifts, gradient_recipes, f0, broadcast, shots)

    gradient_tapes.extend(g_tapes)

    return gradient_tapes, fn
