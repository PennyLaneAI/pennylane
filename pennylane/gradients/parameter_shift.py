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
# pylint: disable=protected-access,too-many-arguments,too-many-statements
import warnings
import numpy as np

import pennylane as qml

from .gradient_transform import (
    gradient_transform,
    grad_method_validation,
    choose_grad_methods,
)
from .finite_difference import finite_diff, generate_shifted_tapes
from .general_shift_rules import process_shifts


NONINVOLUTORY_OBS = {
    "Hermitian": lambda obs: obs.__class__(obs.get_matrix() @ obs.get_matrix(), wires=obs.wires),
    "SparseHamiltonian": lambda obs: obs.__class__(
        obs.get_matrix() @ obs.get_matrix(), wires=obs.wires
    ),
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


def _get_operation_recipe(tape, t_idx, shifts):
    """Utility function to return the parameter-shift rule
    of the operation corresponding to trainable parameter
    t_idx on tape.

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
    """
    op, p_idx = tape.get_operation(t_idx)

    # Try to use the stored grad_recipe of the operation
    recipe = op.grad_recipe[p_idx]
    if recipe is not None:
        return process_shifts(np.array(recipe).T, check_duplicates=False)

    # Try to obtain frequencies, either via custom implementation or from generator eigvals
    try:
        frequencies = op.parameter_frequencies[p_idx]
    except qml.operation.ParameterFrequenciesUndefinedError as e:
        raise qml.operation.OperatorPropertyUndefined(
            f"The operation {op.name} does not have a grad_recipe, parameter_frequencies or "
            "a generator defined. No parameter shift rule can be applied."
        ) from e

    # Create shift rule from frequencies with given shifts
    coeffs, shifts = qml.gradients.generate_shift_rule(frequencies, shifts=shifts, order=1)
    # The generated shift rules do not include a rescaling of the parameter, only shifts.
    mults = np.ones_like(coeffs)

    return coeffs, mults, shifts


def _gradient_analysis(tape, use_graph=True):
    """Update the parameter information dictionary of the tape with
    gradient information of each parameter."""

    if getattr(tape, "_gradient_fn", None) is param_shift:
        # gradient analysis has already been performed on this tape
        return

    tape._gradient_fn = param_shift

    for idx, info in tape._par_info.items():

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


def expval_param_shift(tape, argnum=None, shifts=None, gradient_recipes=None, f0=None):
    r"""Generate the parameter-shift tapes and postprocessing methods required
    to compute the gradient of a gate parameter with respect to an
    expectation value.

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

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
        list of generated tapes, in addition to a post-processing
        function to be applied to the results of the evaluated tapes.
    """
    argnum = argnum or tape.trainable_params

    gradient_tapes = []
    gradient_coeffs = []
    shapes = []
    unshifted_coeffs = []

    fns = []

    for idx, _ in enumerate(tape.trainable_params):

        if idx not in argnum:
            # parameter has zero gradient
            shapes.append(0)
            gradient_coeffs.append([])
            fns.append(None)
            continue

        op, _ = tape.get_operation(idx)

        if op.name == "Hamiltonian":
            # operation is a Hamiltonian
            if op.return_type is not qml.operation.Expectation:
                raise ValueError(
                    "Can only differentiate Hamiltonian "
                    f"coefficients for expectations, not {op.return_type.value}"
                )

            g_tapes, h_fn = qml.gradients.hamiltonian_grad(tape, idx)
            gradient_tapes.extend(g_tapes)
            shapes.append(1)
            gradient_coeffs.append(np.array([1.0]))
            fns.append(h_fn)
            continue

        # get the gradient recipe for the trainable parameter
        arg_idx = argnum.index(idx)
        recipe = gradient_recipes[arg_idx]
        if recipe is not None:
            recipe = process_shifts(np.array(recipe).T)
        else:
            op_shifts = None if shifts is None else shifts[arg_idx]
            recipe = _get_operation_recipe(tape, idx, shifts=op_shifts)
        coeffs, multipliers, op_shifts = recipe
        fns.append(None)

        # Extract zero-shift term if present (if so, it will always be the first)
        if op_shifts[0] == 0 and multipliers[0] == 1:
            # Gradient recipe includes a term with zero shift.

            if not unshifted_coeffs and f0 is None:
                # Ensure that the unshifted tape is appended
                # to the gradient tapes, if not already.
                gradient_tapes.append(tape)

            # Store the unshifted coefficient. We know that
            # it will always be the first coefficient due to processing.
            unshifted_coeffs.append(coeffs[0])
            coeffs, multipliers, op_shifts = recipe[:, 1:]

        # generate the gradient tapes
        gradient_coeffs.append(coeffs)
        g_tapes = generate_shifted_tapes(tape, idx, op_shifts, multipliers)

        gradient_tapes.extend(g_tapes)
        shapes.append(len(g_tapes))

    def processing_fn(results):
        grads = []
        start = 1 if unshifted_coeffs and f0 is None else 0
        r0 = f0 or results[0]

        for i, (s, f) in enumerate(zip(shapes, fns)):

            if s == 0:
                # parameter has zero gradient
                g = qml.math.zeros_like(results[0])
                grads.append(g)
                continue

            res = results[start : start + s]
            start = start + s

            if f is not None:
                res = f(res)

            # compute the linear combination of results and coefficients
            res = qml.math.stack(res)
            g = qml.math.tensordot(res, qml.math.convert_like(gradient_coeffs[i], res), [[0], [0]])

            if unshifted_coeffs:
                # add on the unshifted term
                g = g + unshifted_coeffs[i] * r0

            grads.append(g)

        # The following is for backwards compatibility; currently,
        # the device stacks multiple measurement arrays, even if not the same
        # size, resulting in a ragged array.
        # In the future, we might want to change this so that only tuples
        # of arrays are returned.
        for i, g in enumerate(grads):
            g = qml.math.convert_like(g, res[0])
            if hasattr(g, "dtype") and g.dtype is np.dtype("object"):
                grads[i] = qml.math.hstack(g)

        return qml.math.T(qml.math.stack(grads))

    return gradient_tapes, processing_fn


def var_param_shift(tape, argnum, shifts=None, gradient_recipes=None, f0=None):
    r"""Generate the parameter-shift tapes and postprocessing methods required
    to compute the gradient of a gate parameter with respect to a
    variance value.

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

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
        list of generated tapes, in addition to a post-processing
        function to be applied to the results of the evaluated tapes.
    """
    argnum = argnum or tape.trainable_params

    # Determine the locations of any variance measurements in the measurement queue.
    var_mask = [m.return_type is qml.operation.Variance for m in tape.measurements]
    var_idx = np.where(var_mask)[0]

    # Get <A>, the expectation value of the tape with unshifted parameters.
    expval_tape = tape.copy(copy_operations=True)

    # Convert all variance measurements on the tape into expectation values
    for i in var_idx:
        obs = expval_tape._measurements[i].obs
        expval_tape._measurements[i] = qml.measurements.MeasurementProcess(
            qml.operation.Expectation, obs=obs
        )

    gradient_tapes = [expval_tape]

    # evaluate the analytic derivative of <A>
    pdA_tapes, pdA_fn = expval_param_shift(expval_tape, argnum, shifts, gradient_recipes, f0)
    gradient_tapes.extend(pdA_tapes)

    # Store the number of first derivative tapes, so that we know
    # the number of results to post-process later.
    tape_boundary = len(pdA_tapes) + 1

    # If there are non-involutory observables A present, we must compute d<A^2>/dp.
    # Get the indices in the measurement queue of all non-involutory
    # observables.
    non_involutory = []

    for i in var_idx:
        obs_name = tape.observables[i].name

        if isinstance(obs_name, list):
            # Observable is a tensor product, we must investigate all constituent observables.
            if any(name in NONINVOLUTORY_OBS for name in obs_name):
                non_involutory.append(i)

        elif obs_name in NONINVOLUTORY_OBS:
            non_involutory.append(i)

    # For involutory observables (A^2 = I) we have d<A^2>/dp = 0.
    involutory = set(var_idx) - set(non_involutory)

    if non_involutory:
        expval_sq_tape = tape.copy(copy_operations=True)

        for i in non_involutory:
            # We need to calculate d<A^2>/dp; to do so, we replace the
            # involutory observables A in the queue with A^2.
            obs = _square_observable(expval_sq_tape._measurements[i].obs)
            expval_sq_tape._measurements[i] = qml.measurements.MeasurementProcess(
                qml.operation.Expectation, obs=obs
            )

        # Non-involutory observables are present; the partial derivative of <A^2>
        # may be non-zero. Here, we calculate the analytic derivatives of the <A^2>
        # observables.
        pdA2_tapes, pdA2_fn = expval_param_shift(
            expval_sq_tape, argnum, shifts, gradient_recipes, f0
        )
        gradient_tapes.extend(pdA2_tapes)

    def processing_fn(results):
        # We need to expand the dimensions of the variance mask,
        # and convert it to be the same type as the results.
        res = results[0]
        ragged = getattr(results[0], "dtype", None) is np.dtype("object")

        mask = []
        for m, r in zip(var_mask, results[0]):
            array_func = np.ones if m else np.zeros
            shape = qml.math.shape(r)
            shape = (1,) if shape == tuple() else shape
            mask.append(array_func(shape, dtype=bool))

        if ragged:
            res = qml.math.hstack(res)
            mask = qml.math.hstack(mask)

        mask = qml.math.convert_like(qml.math.reshape(mask, [-1, 1]), res)
        f0 = qml.math.expand_dims(res, -1)

        pdA = pdA_fn(results[1:tape_boundary])
        pdA2 = 0

        if non_involutory:
            # compute the second derivative of non-involutory observables
            pdA2 = pdA2_fn(results[tape_boundary:])

            if involutory:
                # if involutory observables are present, ensure they have zero gradient.
                #
                # For the pdA2_tapes, we have replaced non-involutory
                # observables with their square (A -> A^2). However,
                # involutory observables have been left as-is (A), and have
                # not been replaced by their square (A^2 = I). As a result,
                # components of the gradient vector will not be correct. We
                # need to replace the gradient value with 0 (the known,
                # correct gradient for involutory variables).

                m = [tape.observables[i].name not in NONINVOLUTORY_OBS for i in var_idx]
                m = qml.math.convert_like(m, pdA2)
                pdA2 = qml.math.where(qml.math.reshape(m, [-1, 1]), 0, pdA2)

        # return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances (mask==True)
        # d<A>/dp for plain expectations (mask==False)
        return qml.math.where(mask, pdA2 - 2 * f0 * pdA, pdA)

    return gradient_tapes, processing_fn


@gradient_transform
def param_shift(
    tape, argnum=None, shifts=None, gradient_recipes=None, fallback_fn=finite_diff, f0=None
):
    r"""Transform a QNode to compute the parameter-shift gradient of all gate
    parameters with respect to its inputs.

    Args:
        qnode (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
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

    Returns:
        tensor_like or tuple[list[QuantumTape], function]:

        - If the input is a QNode, a tensor
          representing the output Jacobian matrix of size ``(number_outputs, number_gate_parameters)``
          is returned.

        - If the input is a tape, a tuple containing a list of generated tapes,
          in addition to a post-processing function to be applied to the
          evaluated tapes.

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
    `Wierichs et al. (2021) <https://arxiv.org/abs/2107.12390>`_.

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
    >>> @qml.qnode(dev, gradient_fn=qml.gradients.param_shift)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.jacobian(circuit)(params)
    tensor([[-0.38751725, -0.18884792, -0.38355708],
            [ 0.69916868,  0.34072432,  0.69202365]], requires_grad=True)

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

    .. UsageDetails::

        This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>` objects:

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
        >>> qml.gradients.param_shift(circuit)(params)
        tensor([[-0.38751725, -0.18884792, -0.38355708],
                [ 0.69916868,  0.34072432,  0.69202365]], requires_grad=True)

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> with qml.tape.JacobianTape() as tape:
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     qml.expval(qml.PauliZ(0))
        ...     qml.var(qml.PauliZ(0))
        >>> gradient_tapes, fn = qml.gradients.param_shift(tape)
        >>> gradient_tapes
        [<JacobianTape: wires=[0, 1], params=3>,
         <JacobianTape: wires=[0, 1], params=3>,
         <JacobianTape: wires=[0, 1], params=3>,
         <JacobianTape: wires=[0, 1], params=3>,
         <JacobianTape: wires=[0, 1], params=3>,
         <JacobianTape: wires=[0, 1], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        [[-0.38751721 -0.18884787 -0.38355704]
         [ 0.69916862  0.34072424  0.69202359]]
    """

    if any(m.return_type is qml.operation.State for m in tape.measurements):
        raise ValueError(
            "Computing the gradient of circuits that return the state is not supported."
        )

    if argnum is None and not tape.trainable_params:
        warnings.warn(
            "Attempted to compute the gradient of a tape with no trainable parameters. "
            "If this is unintended, please mark trainable parameters in accordance with the "
            "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
        )
        return [], lambda _: ()

    _gradient_analysis(tape)
    method = "analytic" if fallback_fn is None else "best"
    diff_methods = grad_method_validation(method, tape)

    if all(g == "0" for g in diff_methods):
        return [], lambda _: np.zeros([tape.output_dim, len(tape.trainable_params)])

    method_map = choose_grad_methods(diff_methods, argnum)

    # If there are unsupported operations, call the fallback gradient function
    gradient_tapes = []
    unsupported_params = {idx for idx, g in method_map.items() if g == "F"}
    argnum = [i for i, dm in method_map.items() if dm == "A"]

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

    if any(m.return_type is qml.operation.Variance for m in tape.measurements):
        g_tapes, fn = var_param_shift(tape, argnum, shifts, gradient_recipes, f0)
    else:
        g_tapes, fn = expval_param_shift(tape, argnum, shifts, gradient_recipes, f0)

    gradient_tapes.extend(g_tapes)

    if unsupported_params:
        # If there are unsupported parameters, we must process
        # the quantum results separately, once for the fallback
        # function and once for the parameter-shift rule, and recombine.

        def processing_fn(results):
            unsupported_grads = fallback_proc_fn(results[:fallback_len])
            supported_grads = fn(results[fallback_len:])
            return unsupported_grads + supported_grads

        return gradient_tapes, processing_fn

    return gradient_tapes, fn
