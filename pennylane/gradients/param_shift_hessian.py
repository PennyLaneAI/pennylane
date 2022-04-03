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
This module contains functions for computing the parameter-shift hessian
of a qubit-based quantum tape.
"""
import warnings
import itertools as it
from collections.abc import Sequence

import pennylane as qml
from pennylane import numpy as np

from .gradient_transform import (
    gradient_analysis,
    grad_method_validation,
    choose_grad_methods,
)
from .hessian_transform import hessian_transform
from .parameter_shift import _get_operation_recipe
from .general_shift_rules import (
    _combine_shift_rules_with_multipliers,
    generate_shifted_tapes,
    generate_multishifted_tapes,
)


def _collect_recipes(tape, argnum, diff_methods, diagonal_shifts, off_diagonal_shifts):
    r"""Extract second order recipes for the tape operations for the diagonal of the Hessian
    as well as the first-order derivative recipes for the off-diagonal entries.
    """
    diag_recipes = []
    partial_offdiag_recipes = []
    for i in range(tape.num_params):
        if i not in argnum or diff_methods[i] == "0":
            # hessian will be set to 0 for this row/column
            diag_recipes.append(None)
            partial_offdiag_recipes.append((None, None, None))
            continue

        # Get the diagonal second-order derivative recipe
        idx = argnum.index(i)
        diag_shifts = None if diagonal_shifts is None else diagonal_shifts[idx]
        diag_recipes.append(_get_operation_recipe(tape, i, diag_shifts, order=2))

        # Create the first-order gradient recipes per parameter
        _shifts = None if off_diagonal_shifts is None else off_diagonal_shifts[idx]
        partial_offdiag_recipes.append(_get_operation_recipe(tape, i, _shifts, order=1))

    return diag_recipes, partial_offdiag_recipes


def _generate_off_diag_tapes(tape, idx, recipe_i, recipe_j):
    r"""Combine two univariate first order recipes and create
    multi-shifted tapes to compute the off-diagonal entry of the Hessian."""

    if recipe_j[0] is None:
        return [], None, None

    # The rows of combined_rulesT contain the coefficients (1), the multipliers (2) and the
    # shifts (2) in that order, with the number in brackets indicating the number of columns
    combined_rulesT = _combine_shift_rules_with_multipliers(
        [qml.math.stack(recipe_i).T, qml.math.stack(recipe_j).T]
    )
    # If there are unmultiplied, unshifted tapes, the coefficient is memorized and the term
    # removed from the list of tapes to create
    if np.allclose(combined_rulesT[1:3, 0], 1.0) and np.allclose(combined_rulesT[3:5, 0], 0.0):
        unshifted_coeff = combined_rulesT[0, 0]
        combined_rulesT = combined_rulesT[:, 1:]
    else:
        unshifted_coeff = None

    h_tapes = generate_multishifted_tapes(tape, idx, combined_rulesT[3:5].T, combined_rulesT[1:3].T)

    return h_tapes, combined_rulesT[0], unshifted_coeff


def expval_hessian_param_shift(
    tape, argnum, diff_methods, diagonal_shifts, off_diagonal_shifts, f0
):
    r"""Generate the Hessian tapes that are used in the computation of the second derivative of a
    quantum tape, using analytical parameter-shift rules to do so exactly. Also define a
    post-processing function to combine the results of evaluating the tapes into the Hessian.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        argnum (int or list[int] or None): Parameter indices to differentiate
            with respect to. If not provided, the Hessian with respect to all
            trainable indices is returned.
        diff_methods (list[string]): The differentiation method to use for each trainable parameter.
            Can be "A" or "0", where "A" is the analytical parameter shift rule and "0" indicates
            a 0 derivative (that is the parameter does not affect the tape's output).
        diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift values
            for the Hessian diagonal.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple length should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are used.
        off_diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift
            values for the off-diagonal entries of the Hessian.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            The combination of shifts into bivariate shifts is performed automatically.
            If unspecified, equidistant shifts are used.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the Hessian tapes include the original input tape, the 'f0' value is used
            instead of evaluating the input tape, reducing the number of device invocations.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a list of generated tapes, in
            addition to a post-processing function to be applied to the results of the evaluated
            tapes.
    """
    # pylint: disable=too-many-arguments, too-many-statements
    argnum = tape.trainable_params if argnum is None else argnum
    h_dim = tape.num_params

    unshifted_coeffs = {}
    # Marks whether we will need to add the unshifted tape to all Hessian tapes.
    add_unshifted = f0 is None

    # Assemble all univariate recipes for the diagonal and as partial components for the
    # off-diagonal entries.
    diag_recipes, partial_offdiag_recipes = _collect_recipes(
        tape, argnum, diff_methods, diagonal_shifts, off_diagonal_shifts
    )

    hessian_tapes = []
    hessian_coeffs = []
    for i in range(h_dim):
        # The diagonal recipe is None if the parameter is not trainable or not in argnum
        if diag_recipes[i] is None:
            hessian_coeffs.extend([None] * (h_dim - i))
            continue

        # Obtain the recipe for the diagonal.
        dc_i, dm_i, ds_i = diag_recipes[i]
        # Add the unshifted tape if it is required for this diagonal, it has not been
        # added yet, and it is required because f0 was not provided.
        if ds_i[0] == 0 and dm_i[0] == 1.0:
            if add_unshifted:
                hessian_tapes.insert(0, tape)
                add_unshifted = False
            unshifted_coeffs[(i, i)] = dc_i[0]
            dc_i, dm_i, ds_i = dc_i[1:], dm_i[1:], ds_i[1:]

        # Create the shifted tapes for the diagonal entry and store them along with coefficients
        diag_tapes = generate_shifted_tapes(tape, i, ds_i, dm_i)
        hessian_tapes.extend(diag_tapes)
        hessian_coeffs.append(dc_i)

        recipe_i = partial_offdiag_recipes[i]
        for j in range(i + 1, h_dim):
            recipe_j = partial_offdiag_recipes[j]

            # Create tapes and coefficients for the off-diagonal entry by combining
            # the two univariate first-order derivative recipes.
            off_diag_data = _generate_off_diag_tapes(tape, (i, j), recipe_i, recipe_j)
            hessian_tapes.extend(off_diag_data[0])
            hessian_coeffs.append(off_diag_data[1])
            # It should not be possible to obtain an unshifted tape for the off-diagonal
            # terms if there hasn't already been one for the diagonal terms.
            # TODO: This will depend on the decision on how diagonal_shifts are formatted.
            # TODO: If this is confirmed, remove the following safety check
            if off_diag_data[2] is not None:
                raise ValueError(
                    "A tape without parameter shifts was created unexpectedly during "
                    "the computation of the Hessian. Please submit a bug report "
                    "at https://github.com/PennyLaneAI/pennylane/issues"
                )  # pragma: no cover

    def processing_fn(results):
        # Apply the same squeezing as in qml.QNode to make the transform output consistent.
        # pylint: disable=protected-access
        if tape._qfunc_output is not None and not isinstance(tape._qfunc_output, Sequence):
            results = qml.math.squeeze(qml.math.stack(results))

        # The first results dimension is the number of terms/tapes in the parameter-shift
        # rule, the remaining ones are the QNode output dimensions.
        out_dim = qml.math.shape(results)[1:]
        # The desired shape of the Hessian is:
        #       (QNode output dimensions, # trainable gate args, # trainable gate args),
        # but first we accumulate all elements into a list, since no array assignment is possible.
        hessian = []
        # Keep track of tape results already consumed.
        start = 1 if unshifted_coeffs and f0 is None else 0
        # Results of the unshifted tape.
        r0 = results[0] if start == 1 else f0

        for i, j in it.product(range(h_dim), repeat=2):
            if j < i:
                hessian.append(hessian[j * h_dim + i])
                continue
            k = i * h_dim + j - i * (i + 1) // 2
            coeffs = hessian_coeffs[k]
            if coeffs is None or len(coeffs) == 0:
                hessian.append(qml.math.zeros(out_dim))
                continue

            res = results[start : start + len(coeffs)]
            start = start + len(coeffs)

            res = qml.math.stack(res)
            coeffs = qml.math.cast(qml.math.convert_like(coeffs, res), res.dtype)
            hess = qml.math.tensordot(res, coeffs, [[0], [0]])
            if (i, j) in unshifted_coeffs:
                hess = hess + unshifted_coeffs[(i, j)] * r0

            hessian.append(hess)

        # Reshape the Hessian to have the QNode output dimensions on the outside, that is:
        #    (h_dim*h_dim, *out_dims) -> (h_dim, h_dim, *out_dims) -> (*out_dims, h_dim, h_dim)
        # Remember: h_dim = num_gate_args
        hessian = qml.math.reshape(qml.math.stack(hessian), (h_dim, h_dim) + out_dim)
        reordered_axes = list(range(2, len(out_dim) + 2)) + [0, 1]
        return qml.math.transpose(hessian, axes=reordered_axes)

    return hessian_tapes, processing_fn


@hessian_transform
def param_shift_hessian(tape, argnum=None, diagonal_shifts=None, off_diagonal_shifts=None, f0=None):
    r"""Transform a QNode to compute the parameter-shift Hessian with respect to its trainable
    parameters.

    Use this transform to explicitly generate and explore parameter-shift circuits for computing
    the Hessian of QNodes directly, without computing first derivatives.

    For second-order derivatives of more complicated cost functions, please consider using your
    chosen autodifferentiation framework directly, by chaining gradient computations:

    >>> qml.jacobian(qml.grad(cost))(weights)

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
        argnum (int or list[int] or None): Parameter indices to differentiate
            with respect to. If not provided, the Hessian with respect to all
            trainable indices is returned. Note that the indices refer to tape
            parameters both if ``tape`` is a tape, and if it is a QNode.
        diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift values
            for the Hessian diagonal. The shifts are understood as first-order derivative
            shifts and are iterated to obtain the second-order derivative.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple length should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are used.
        off_diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift
            values for the off-diagonal entries of the Hessian.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            The combination of shifts into bivariate shifts is performed automatically.
            If unspecified, equidistant shifts are used.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the Hessian tapes include the original input tape, the 'f0' value is used
            instead of evaluating the input tape, reducing the number of device invocations.

    Returns:
        tensor_like or tuple[tensor_like] or tuple[list[QuantumTape], function]:

        - If the input is a QNode with a single trainable argument, a tensor representing the
          Hessian of size ``(*QNode output dimensions, *QNode input dimensions, *QNode input dimensions)``
          is returned.

        - If the input is a QNode with multiple trainable arguments, a tuple of Hessian tensors is
          returned, one for each argument.

        - If the input is a tape, a tuple containing the list of parameter-shifted tapes, and a
          post-processing function to be applied to the evaluated tapes, is returned.

        Note: By default a QNode with the keyword ``hybrid=True`` computes derivates with respect to
        QNode arguments, which can include classical computations on those arguments before they are
        passed to quantum operations. The "purely quantum" Hessian can instead be obtained with
        ``hybrid=False``, which is then computed with respect to the gate arguments and produces a
        result of shape ``(*QNode output dimensions, # gate arguments, # gate arguments)``.

    **Example**

    Applying the Hessian transform to a QNode computes its Hessian tensor:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... def circuit(x):
    ...     qml.RX(x[0], wires=0)
    ...     qml.CRY(x[1], wires=[0, 1])
    ...     return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

    >>> x = np.array([0.5, 0.2], requires_grad=True)
    >>> qml.gradients.param_shift_hessian(circuit)(x)
    tensor([[-0.86883595,  0.04762358],
            [ 0.04762358,  0.05998862]], requires_grad=True)

    .. UsageDetails::

        The Hessian transform can also be applied to a quantum tape instead of a QNode, producing
        the parameter-shifted tapes and a post-processing function to combine the execution
        results of these tapes into the Hessian:

        >>> circuit(x)  # generate the QuantumTape inside the QNode
        >>> tape = circuit.qtape
        >>> hessian_tapes, postproc_fn = qml.gradients.param_shift_hessian(tape)
        >>> len(hessian_tapes)
        13
        >>> all(isinstance(tape, qml.tape.QuantumTape) for tape in hessian_tapes)
        True
        >>> postproc_fn(qml.execute(hessian_tapes, dev, None))
        array([[-0.86883595,  0.04762358],
               [ 0.04762358,  0.05998862]])

        The Hessian tapes can be inspected via their draw function, which reveals the different
        gate arguments generated from parameter-shift rules (we only draw the first four out of
        all 13 tapes here):

        >>> for h_tape in hessian_tapes:
        ...     print(qml.drawer.tape_text(h_tape, decimals=1))
        0: ──RX(0.5)─╭C───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(-2.6)─╭C───────┤ ╭<Z@Z>
        1: ───────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(2.1)─╭C───────┤ ╭<Z@Z>
        1: ──────────╰RY(1.8)─┤ ╰<Z@Z>
        0: ──RX(2.1)─╭C────────┤ ╭<Z@Z>
        1: ──────────╰RY(-1.4)─┤ ╰<Z@Z>

        To enable more detailed control over the parameter shifts, shift values can be provided
        per parameter, and separately for the diagonal and the off-diagonal terms.
        Here we choose them based on the parameters ``x`` themselves, mostly yielding multiples of
        the original parameters in the shifted tapes.

        >>> diag_shifts = [(x[0] / 2,), (x[1] / 2, x[1])]
        >>> offdiag_shifts = [(x[0],), (x[1], 2 * x[1])]
        >>> hessian_tapes, postproc_fn = qml.gradients.param_shift_hessian(
        ...     tape, diagonal_shifts=diag_shifts, off_diagonal_shifts=offdiag_shifts
        ... )
        >>> for h_tape in hessian_tapes:
        ...     print(qml.drawer.tape_text(h_tape, decimals=1))
        0: ──RX(0.5)─╭C───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(0.0)─╭C───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(1.0)─╭C───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(1.0)─╭C───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.4)─┤ ╰<Z@Z>

        .. note::

            Note that the ``diagonal_shifts`` are interpreted as *first-order* derivative
            shift values. That means they are used to generate a first-order derivative
            recipe, which then is iterated in order to obtain the second-order derivative
            for the diagonal Hessian entry. Explicit control over the used second-order
            shifts is not implemented.

        Finally, the `argnum` argument can be used to compute the Hessian only for some of the
        variational parameters. This refers to QNode input arguments if ``tape`` is a QNode
        and to trainable tape parameters if ``tape`` is a tape.

        >>> hessian_tapes, postproc_fn = qml.gradients.param_shift_hessian(tape, argnum=(1,))
        >>> postproc_fn(qml.execute(hessian_tapes, dev, None))
        array([[0.        , 0.        ],
               [0.        , 0.05998862]])

    """

    # Perform input validation before generating tapes.
    if any(m.return_type is qml.measurements.State for m in tape.measurements):
        raise ValueError(
            "Computing the Hessian of circuits that return the state is not supported."
        )

    # The parameter-shift Hessian implementation currently doesn't support variance measurements.
    # TODO: Support variances similar to how param_shift does it
    if any(m.return_type is qml.measurements.Variance for m in tape.measurements):
        raise ValueError(
            "Computing the Hessian of circuits that return variances is currently not supported."
        )

    if argnum is None and not tape.trainable_params:
        warnings.warn(
            "Attempted to compute the hessian of a tape with no trainable parameters. "
            "If this is unintended, please mark trainable parameters in accordance with the "
            "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
        )
        return [], lambda _: qml.math.zeros((tape.output_dim, 0, 0))

    if argnum is None:
        compare_to = tape.num_params
        compare_to_str = f"trainable tape parameters ({compare_to})"
    else:
        compare_to = len(argnum)
        compare_to_str = f"provided arguments to differentiate ({compare_to})"

    if diagonal_shifts is not None and len(diagonal_shifts) != compare_to:
        raise ValueError(
            "The number of provided sets of shift values for diagonal entries "
            f"({len(diagonal_shifts)}) does not match the number of {compare_to_str}."
        )
    if off_diagonal_shifts is not None and len(off_diagonal_shifts) != compare_to:
        raise ValueError(
            "The number of provided sets of shift values for off-diagonal entries "
            f"({len(off_diagonal_shifts)}) does not match the number of {compare_to_str}."
        )

    gradient_analysis(tape, grad_fn=qml.gradients.param_shift)
    # If argnum is given, the grad_method_validation may allow parameters with
    # finite-difference as method. If they are among the requested argnum, we catch this
    # further below (as no fallback function analogue to `parameter_shift` is used currently).
    method = "analytic" if argnum is None else "best"
    diff_methods = grad_method_validation(method, tape)

    if all(g == "0" for g in diff_methods):
        par_dim = len(tape.trainable_params)
        return [], lambda _: qml.math.zeros([tape.output_dim, par_dim, par_dim])

    method_map = choose_grad_methods(diff_methods, argnum)

    unsupported_params = {idx for idx, g in method_map.items() if g == "F"}
    if unsupported_params:
        raise ValueError(
            "The parameter-shift Hessian currently does not support the operations "
            f"for parameter(s) {unsupported_params}."
        )

    argnum = list(method_map.keys())

    return expval_hessian_param_shift(
        tape, argnum, diff_methods, diagonal_shifts, off_diagonal_shifts, f0
    )
