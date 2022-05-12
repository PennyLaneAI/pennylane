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
of a CV-based quantum tape.
"""
# pylint: disable=protected-access,too-many-arguments,too-many-statements,too-many-branches
import itertools
import warnings
from collections.abc import Sequence

import numpy as np

import pennylane as qml

from .gradient_transform import (
    gradient_transform,
    grad_method_validation,
    choose_grad_methods,
)
from .finite_difference import finite_diff
from .parameter_shift import expval_param_shift, _get_operation_recipe
from .general_shift_rules import process_shifts, generate_shifted_tapes


def _grad_method(tape, idx):
    """Determine the best CV parameter-shift gradient recipe for a given
    parameter index of a tape.

    Args:
        tape (.QuantumTape): input tape
        idx (int): positive integer corresponding to the parameter location
            on the tape to inspect

    Returns:
        str: a string containing either ``"A"`` (for first-order analytic method),
            ``"A2"`` (second-order analytic method), ``"F"`` (finite differences),
            or ``"0"`` (constant parameter).
    """

    op = tape._par_info[idx]["op"]

    if op.grad_method in (None, "F"):
        return op.grad_method

    if op.grad_method != "A":
        raise ValueError(f"Operation {op} has unknown gradient method {op.grad_method}")

    # Operation supports the CV parameter-shift rule.
    # Create an empty list to store the 'best' partial derivative method
    # for each observable
    best = []

    for m in tape.measurements:

        if (m.return_type is qml.measurements.Probability) or (m.obs.ev_order not in (1, 2)):
            # Higher-order observables (including probability) only support finite differences.
            best.append("F")
            continue

        # get the set of operations betweens the operation and the observable
        ops_between = tape.graph.nodes_between(op, m.obs)

        if not ops_between:
            # if there is no path between the operation and the observable,
            # the operator has a zero gradient.
            best.append("0")
            continue

        # For parameter-shift compatible CV gates, we need to check both the
        # intervening gates, and the type of the observable.
        best_method = "A"

        if any(not k.supports_heisenberg for k in ops_between):
            # non-Gaussian operators present in-between the operation
            # and the observable. Must fallback to numeric differentiation.
            best_method = "F"

        elif m.obs.ev_order == 2:

            if m.return_type is qml.measurements.Expectation:
                # If the observable is second-order, we must use the second-order
                # CV parameter shift rule
                best_method = "A2"

            elif m.return_type is qml.measurements.Variance:
                # we only support analytic variance gradients for
                # first-order observables
                best_method = "F"

        best.append(best_method)

    if all(k == "0" for k in best):
        # if the operation is independent of *all* observables
        # in the circuit, the gradient will be 0
        return "0"

    if "F" in best:
        # one non-analytic observable path makes the whole operation
        # gradient method fallback to finite-difference
        return "F"

    if "A2" in best:
        # one second-order observable makes the whole operation gradient
        # require the second-order parameter-shift rule
        return "A2"

    return "A"


def _gradient_analysis_cv(tape):
    """Update the parameter information dictionary of the tape with
    gradient information of each parameter."""

    if getattr(tape, "_gradient_fn", None) is param_shift_cv:
        # gradient analysis has already been performed on this tape
        return

    tape._gradient_fn = param_shift_cv

    for idx, info in tape._par_info.items():
        info["grad_method"] = _grad_method(tape, idx)


def _transform_observable(obs, Z, device_wires):
    """Apply a Gaussian linear transformation to an observable.

    Args:
        obs (.Observable): observable to transform
        Z (array[float]): Heisenberg picture representation of the linear transformation
        device_wires (.Wires): wires on the device the transformed observable is to be
            measured on

    Returns:
        .Observable: the transformed observable
    """
    # Get the Heisenberg representation of the observable
    # in the position/momentum basis. The returned matrix/vector
    # will have been expanded to act on the entire device.
    if obs.ev_order > 2:
        raise NotImplementedError("Transforming observables of order > 2 not implemented.")

    A = obs.heisenberg_obs(device_wires)

    if A.ndim != obs.ev_order:
        raise ValueError(
            "Mismatch between the polynomial order of observable and its Heisenberg representation"
        )

    # transform the observable by the linear transformation Z
    A = A @ Z

    if A.ndim == 2:
        A = A + A.T

    # TODO: if the A matrix corresponds to a known observable in PennyLane,
    # for example qml.X, qml.P, qml.NumberOperator, we should return that
    # instead. This will allow for greater device compatibility.
    return qml.PolyXP(A, wires=device_wires)


def var_param_shift(tape, dev_wires, argnum=None, shifts=None, gradient_recipes=None, f0=None):
    r"""Partial derivative using the first-order or second-order parameter-shift rule of a tape
    consisting of a mixture of expectation values and variances of observables.

    Expectation values may be of first- or second-order observables,
    but variances can only be taken of first-order variables.

    .. warning::

        This method can only be executed on devices that support the
        :class:`~.PolyXP` observable.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        dev_wires (.Wires): wires on the device the parameter-shift method is computed on
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
        function to be applied to the evaluated tapes.
    """
    argnum = argnum or tape.trainable_params

    # Determine the locations of any variance measurements in the measurement queue.
    var_mask = [m.return_type is qml.measurements.Variance for m in tape.measurements]
    var_idx = np.where(var_mask)[0]

    # Get <A>, the expectation value of the tape with unshifted parameters.
    expval_tape = tape.copy(copy_operations=True)

    # Convert all variance measurements on the tape into expectation values
    for i in var_idx:
        obs = expval_tape._measurements[i].obs
        expval_tape._measurements[i] = qml.measurements.MeasurementProcess(
            qml.measurements.Expectation, obs=obs
        )

    gradient_tapes = [expval_tape]

    # evaluate the analytic derivative of <A>
    pdA_tapes, pdA_fn = expval_param_shift(expval_tape, argnum, shifts, gradient_recipes, f0)
    gradient_tapes.extend(pdA_tapes)

    # Store the number of first derivative tapes, so that we know
    # the number of results to post-process later.
    tape_boundary = len(pdA_tapes) + 1
    expval_sq_tape = tape.copy(copy_operations=True)

    for i in var_idx:
        # We need to calculate d<A^2>/dp; to do so, we replace the
        # observables A in the queue with A^2.
        obs = expval_sq_tape._measurements[i].obs

        # CV first-order observable
        # get the heisenberg representation
        # This will be a real 1D vector representing the
        # first-order observable in the basis [I, x, p]
        A = obs._heisenberg_rep(obs.parameters)

        # take the outer product of the heisenberg representation
        # with itself, to get a square symmetric matrix representing
        # the square of the observable
        obs = qml.PolyXP(np.outer(A, A), wires=obs.wires)
        expval_sq_tape._measurements[i] = qml.measurements.MeasurementProcess(
            qml.measurements.Expectation, obs=obs
        )

    # Non-involutory observables are present; the partial derivative of <A^2>
    # may be non-zero. Here, we calculate the analytic derivatives of the <A^2>
    # observables.
    pdA2_tapes, pdA2_fn = second_order_param_shift(
        expval_sq_tape, dev_wires, argnum, shifts, gradient_recipes
    )
    gradient_tapes.extend(pdA2_tapes)

    def processing_fn(results):
        # HOTFIX: Apply the same squeezing as in qml.QNode to make the transform output consistent.
        # pylint: disable=protected-access
        if tape._qfunc_output is not None and not isinstance(tape._qfunc_output, Sequence):
            results = [qml.math.squeeze(res) for res in results]

        mask = qml.math.convert_like(qml.math.reshape(var_mask, [-1, 1]), results[0])
        f0 = qml.math.expand_dims(results[0], -1)

        pdA = pdA_fn(results[1:tape_boundary])
        pdA2 = pdA2_fn(results[tape_boundary:])

        # return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances (mask==True)
        # d<A>/dp for plain expectations (mask==False)
        return qml.math.where(mask, pdA2 - 2 * f0 * pdA, pdA)

    return gradient_tapes, processing_fn


def second_order_param_shift(tape, dev_wires, argnum=None, shifts=None, gradient_recipes=None):
    r"""Generate the second-order CV parameter-shift tapes and postprocessing methods required
    to compute the gradient of a gate parameter with respect to an
    expectation value.

    .. note::

        The 2nd order method can handle also first-order observables, but
        1st order method may be more efficient unless it's really easy to
        experimentally measure arbitrary 2nd order observables.

    .. warning::

        The 2nd order method can only be executed on devices that support the
        :class:`~.PolyXP` observable.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        dev_wires (.Wires): wires on the device the parameter-shift method is computed on
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

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
        list of generated tapes, in addition to a post-processing
        function to be applied to the evaluated tapes.
    """
    argnum = argnum or list(tape.trainable_params)
    gradient_recipes = gradient_recipes or [None] * len(argnum)

    gradient_tapes = []
    shapes = []
    obs_indices = []
    gradient_values = []

    for idx, _ in enumerate(tape.trainable_params):
        t_idx = list(tape.trainable_params)[idx]
        op = tape._par_info[t_idx]["op"]

        if idx not in argnum:
            # parameter has zero gradient
            shapes.append(0)
            obs_indices.append([])
            gradient_values.append([])
            continue

        shapes.append(1)

        # get the gradient recipe for the trainable parameter
        arg_idx = argnum.index(idx)
        recipe = gradient_recipes[arg_idx]
        if recipe is not None:
            recipe = process_shifts(np.array(recipe))
        else:
            op_shifts = None if shifts is None else shifts[arg_idx]
            recipe = _get_operation_recipe(tape, idx, shifts=op_shifts)
        coeffs, multipliers, op_shifts = recipe.T

        if len(op_shifts) != 2:
            # The 2nd order CV parameter-shift rule only accepts two-term shifts
            raise NotImplementedError(
                "Taking the analytic gradient for order-2 operators is "
                f"unsupported for operation {op} which has a "
                "gradient recipe of more than two terms."
            )

        shifted_tapes = generate_shifted_tapes(tape, idx, op_shifts, multipliers)

        # evaluate transformed observables at the original parameter point
        # first build the Heisenberg picture transformation matrix Z
        Z0 = op.heisenberg_tr(dev_wires, inverse=True)
        Z2 = shifted_tapes[0]._par_info[t_idx]["op"].heisenberg_tr(dev_wires)
        Z1 = shifted_tapes[1]._par_info[t_idx]["op"].heisenberg_tr(dev_wires)

        # derivative of the operation
        Z = Z2 * coeffs[0] + Z1 * coeffs[1]
        Z = Z @ Z0

        # conjugate Z with all the descendant operations
        B = np.eye(1 + 2 * len(dev_wires))
        B_inv = B.copy()

        succ = tape.graph.descendants_in_order((op,))
        operation_descendents = itertools.filterfalse(qml.circuit_graph._is_observable, succ)
        observable_descendents = filter(qml.circuit_graph._is_observable, succ)

        for BB in operation_descendents:
            if not BB.supports_heisenberg:
                # if the descendant gate is non-Gaussian in parameter-shift differentiation
                # mode, then there must be no observable following it.
                continue

            B = BB.heisenberg_tr(dev_wires) @ B
            B_inv = B_inv @ BB.heisenberg_tr(dev_wires, inverse=True)

        Z = B @ Z @ B_inv  # conjugation

        g_tape = tape.copy(copy_operations=True)
        constants = []

        # transform the descendant observables into their derivatives using Z
        transformed_obs_idx = []

        for obs in observable_descendents:
            # get the index of the descendent observable
            idx = tape.observables.index(obs)
            transformed_obs_idx.append(idx)

            transformed_obs = _transform_observable(obs, Z, dev_wires)

            A = transformed_obs.parameters[0]
            constant = None

            # Check if the transformed observable corresponds to a constant term.
            if len(A.nonzero()[0]) == 1:
                if A.ndim == 2 and A[0, 0] != 0:
                    constant = A[0, 0]

                elif A.ndim == 1 and A[0] != 0:
                    constant = A[0]

            constants.append(constant)

            g_tape._measurements[idx] = qml.measurements.MeasurementProcess(
                qml.measurements.Expectation, _transform_observable(obs, Z, dev_wires)
            )

        if not any(i is None for i in constants):
            # Check if *all* transformed observables corresponds to a constant
            # term. If this is the case for all transformed observables on the
            # tape, then <psi|A|psi> = A<psi|psi> = A, and we can avoid the
            # device execution.
            shapes[-1] = 0
            obs_indices.append(transformed_obs_idx)
            gradient_values.append(constants)
            continue

        gradient_tapes.append(g_tape)
        obs_indices.append(transformed_obs_idx)
        gradient_values.append(None)

    def processing_fn(results):
        # HOTFIX: Apply the same squeezing as in qml.QNode to make the transform output consistent.
        # pylint: disable=protected-access
        if tape._qfunc_output is not None and not isinstance(tape._qfunc_output, Sequence):
            results = [qml.math.squeeze(res) for res in results]

        grads = []
        start = 0

        if not results:
            results = [np.squeeze(np.zeros([tape.output_dim]))]

        interface = qml.math.get_interface(results[0])
        iterator = enumerate(zip(shapes, gradient_values, obs_indices))

        for i, (shape, grad_value, obs_ind) in iterator:

            if shape == 0:
                # parameter has zero gradient
                isscalar = qml.math.ndim(results[0]) == 0
                g = qml.math.zeros_like(qml.math.atleast_1d(results[0]), like=interface)

                if grad_value:
                    g = qml.math.scatter_element_add(g, obs_ind, grad_value, like=interface)

                grads.append(g[0] if isscalar else g)
                continue

            obs_result = results[start : start + shape]
            start = start + shape

            # compute the linear combination of results and coefficients
            isscalar = qml.math.ndim(obs_result[0]) == 0
            obs_result = qml.math.stack(qml.math.atleast_1d(obs_result[0]))
            g = qml.math.zeros_like(obs_result, like=interface)

            if qml.math.get_interface(g) not in ("tensorflow", "autograd"):
                obs_ind = (obs_ind,)

            g = qml.math.scatter_element_add(g, obs_ind, obs_result[obs_ind], like=interface)
            grads.append(g[0] if isscalar else g)

        # The following is for backwards compatibility; currently,
        # the device stacks multiple measurement arrays, even if not the same
        # size, resulting in a ragged array.
        # In the future, we might want to change this so that only tuples
        # of arrays are returned.
        for i, g in enumerate(grads):
            g = qml.math.convert_like(g, results[0])
            if hasattr(g, "dtype") and g.dtype is np.dtype("object"):
                grads[i] = qml.math.hstack(g)

        return qml.math.T(qml.math.stack(grads))

    return gradient_tapes, processing_fn


@gradient_transform
def param_shift_cv(
    tape,
    dev,
    argnum=None,
    shifts=None,
    gradient_recipes=None,
    fallback_fn=finite_diff,
    f0=None,
    force_order2=False,
):
    r"""Transform a continuous-variable QNode to compute the parameter-shift gradient of all gate
    parameters with respect to its inputs.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        dev (.Device): device the parameter-shift method is to be computed on
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
        fallback_fn (None or Callable): a fallback grdient function to use for
            any parameters that do not support the parameter-shift rule.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.
        force_order2 (bool): if True, use the order-2 method even if not necessary

    Returns:
        tensor_like or tuple[list[QuantumTape], function]:

        - If the input is a QNode, a tensor
          representing the output Jacobian matrix of size ``(number_outputs, number_gate_parameters)``
          is returned.

        - If the input is a tape, a tuple containing a list of generated tapes,
          in addition to a post-processing function to be applied to the
          evaluated tapes.

    This transform supports analytic gradients of Gaussian CV operations using
    the parameter-shift rule. This gradient method returns *exact* gradients,
    and can be computed directly on quantum hardware.

    Analytic gradients of photonic circuits that satisfy
    the following constraints with regards to measurements are supported:

    * Expectation values are restricted to observables that are first- and
      second-order in :math:`\hat{x}` and :math:`\hat{p}` only.
      This includes :class:`~.X`, :class:`~.P`, :class:`~.QuadOperator`,
      :class:`~.PolyXP`, and :class:`~.NumberOperator`.

      For second-order observables, the device **must support** :class:`~.PolyXP`.

    * Variances are restricted to observables that are first-order
      in :math:`\hat{x}` and :math:`\hat{p}` only. This includes :class:`~.X`, :class:`~.P`,
      :class:`~.QuadOperator`, and *some* parameter values of :class:`~.PolyXP`.

      The device **must support** :class:`~.PolyXP`.

    .. warning::

        Fock state probabilities (tapes that return :func:`~pennylane.probs` or
        expectation values of :class:`~.FockStateProjector`) are not supported.

    In addition, the operations must fulfill the following requirements:

    * Only Gaussian operations are differentiable.

    * Non-differentiable Fock states and Fock operations may *precede* all differentiable Gaussian,
      operations. For example, the following is permissible:

      .. code-block:: python

          @qml.qnode(dev)
          def circuit(weights):
              # Non-differentiable Fock operations
              qml.FockState(np.array(2, requires_grad=False), wires=0)
              qml.Kerr(np.array(0.654, requires_grad=False), wires=1)

              # differentiable Gaussian operations
              qml.Displacement(weights[0], weights[1], wires=0)
              qml.Beamsplitter(weights[2], weights[3], wires=[0, 1])

              return qml.expval(qml.NumberOperator(0))

    * If a Fock operation succeeds a Gaussian operation, the Fock operation must
      not contribute to any measurements. For example, the following is allowed:

      .. code-block:: python

          @qml.qnode(dev)
          def circuit(weights):
              qml.Displacement(weights[0], weights[1], wires=0)
              qml.Beamsplitter(weights[2], weights[3], wires=[0, 1])
              qml.Kerr(np.array(0.654, requires_grad=False), wires=1)  # there is no measurement on wire 1
              return qml.expval(qml.NumberOperator(0))

    If any of the above constraints are not followed, the tape cannot be differentiated
    via the CV parameter-shift rule. Please use numerical differentiation instead.

    **Example**

    This transform can be registered directly as the quantum gradient transform
    to use during autodifferentiation:

    >>> dev = qml.device("default.gaussian", wires=2)
    >>> @qml.qnode(dev, diff_method="parameter-shift")
    ... def circuit(params):
    ...     qml.Squeezing(params[0], params[1], wires=[0])
    ...     qml.Squeezing(params[2], params[3], wires=[0])
    ...     return qml.expval(qml.NumberOperator(0))
    >>> params = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    >>> qml.jacobian(circuit)(params)
    array([ 0.87516064,  0.01273285,  0.88334834, -0.01273285])

    .. details::
        :title: Usage Details

        This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>` objects:

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.Squeezing(params[0], params[1], wires=[0])
        ...     qml.Squeezing(params[2], params[3], wires=[0])
        ...     return qml.expval(qml.NumberOperator(0))
        >>> params = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        >>> qml.gradients.param_shift_cv(circuit, dev)(params)
        tensor([[ 0.87516064,  0.01273285,  0.88334834, -0.01273285]], requires_grad=True)

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> r0, phi0, r1, phi1 = [0.4, -0.3, -0.7, 0.2]
        >>> with qml.tape.QuantumTape() as tape:
        ...     qml.Squeezing(r0, phi0, wires=[0])
        ...     qml.Squeezing(r1, phi1, wires=[0])
        ...     qml.expval(qml.NumberOperator(0))  # second-order
        >>> gradient_tapes, fn = qml.gradients.param_shift_cv(tape, dev)
        >>> gradient_tapes
        [<QuantumTape: wires=[0], params=4>,
         <QuantumTape: wires=[0], params=4>,
         <QuantumTape: wires=[0], params=4>,
         <QuantumTape: wires=[0], params=4>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.gaussian", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        array([[-0.32487113, -0.4054074 , -0.87049853,  0.4054074 ]])
    """

    # perform gradient method validation
    if any(m.return_type is qml.measurements.State for m in tape.measurements):
        raise ValueError(
            "Computing the gradient of circuits that return the state is not supported."
        )

    _gradient_analysis_cv(tape)

    if argnum is None and not tape.trainable_params:
        warnings.warn(
            "Attempted to compute the gradient of a tape with no trainable parameters. "
            "If this is unintended, please mark trainable parameters in accordance with the "
            "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
        )
        return [], lambda _: qml.math.zeros((tape.output_dim, 0))

    gradient_tapes = []
    shapes = []
    fns = []

    def _update(data):
        """Utility function to update the list of gradient tapes,
        the corresponding number of gradient tapes, and the processing functions"""
        gradient_tapes.extend(data[0])
        shapes.append(len(data[0]))
        fns.append(data[1])

    method = "analytic" if fallback_fn is None else "best"
    diff_methods = grad_method_validation(method, tape)
    if all(g == "0" for g in diff_methods):
        return [], lambda _: np.zeros([tape.output_dim, len(tape.trainable_params)])

    method_map = choose_grad_methods(diff_methods, argnum)
    var_present = any(m.return_type is qml.measurements.Variance for m in tape.measurements)

    unsupported_params = []
    first_order_params = []
    second_order_params = []

    for idx, g in method_map.items():
        if g == "F":
            unsupported_params.append(idx)

        elif g == "A":
            first_order_params.append(idx)

        elif g == "A2":
            second_order_params.append(idx)

    if force_order2:
        # all analytic parameters should be computed using the second-order method
        second_order_params += first_order_params
        first_order_params = []

    if "PolyXP" not in dev.observables and (second_order_params or var_present):
        warnings.warn(
            f"The device {dev.short_name} does not support "
            "the PolyXP observable. The analytic parameter-shift cannot be used for "
            "second-order observables; falling back to finite-differences.",
            UserWarning,
        )

        if var_present:
            unsupported_params += first_order_params
            first_order_params = []

        unsupported_params += second_order_params
        second_order_params = []

    # If there are unsupported operations, call the fallback gradient function
    if unsupported_params:
        _update(fallback_fn(tape, argnum=unsupported_params))

    # collect all the analytic parameters
    argnum = first_order_params + second_order_params

    if not argnum:
        # No analytic parameters. Return the existing fallback tapes/fn
        return gradient_tapes, fns[-1]

    gradient_recipes = gradient_recipes or [None] * len(argnum)

    if var_present:
        _update(var_param_shift(tape, dev.wires, argnum, shifts, gradient_recipes, f0))

    else:
        # Only expectation values were specified
        if first_order_params:
            _update(expval_param_shift(tape, first_order_params, shifts, gradient_recipes, f0))

        if second_order_params:
            _update(
                second_order_param_shift(
                    tape, dev.wires, second_order_params, shifts, gradient_recipes
                )
            )

    def processing_fn(results):
        start = 0
        grads = []

        for s, f in zip(shapes, fns):
            grads.append(f(results[start : start + s]))
            start += s

        return sum(grads)

    return gradient_tapes, processing_fn
