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
)
from .hessian_transform import hessian_transform
from .parameter_shift import _get_operation_recipe
from .general_shift_rules import (
    _combine_shift_rules,
    generate_shifted_tapes,
    generate_multishifted_tapes,
)


def _process_argnum(argnum, tape):
    """Process the argnum keyword argument to ``param_shift_hessian`` from any of ``None``,
    ``int``, ``Sequence[int]``, ``array_like[bool]`` to an ``array_like[bool]``."""
    _trainability_note = (
        "This may be caused by attempting to differentiate with respect to parameters "
        "that are not marked as trainable."
    )
    if argnum is None:
        # All trainable tape parameters are considered
        argnum = list(range(tape.num_params))
    elif isinstance(argnum, int):
        if argnum >= tape.num_params:
            raise ValueError(
                f"The index {argnum} exceeds the number of trainable tape parameters "
                f"({tape.num_params}). " + _trainability_note
            )
        # Make single marked parameter an iterable
        argnum = [argnum]

    if len(qml.math.shape(argnum)) == 1:
        # If the iterable is 1D, consider all combinations of all marked parameters
        if not qml.math.array(argnum).dtype == bool:
            # If the 1D iterable contains indices, make sure it contains valid indices...
            if qml.math.max(argnum) >= tape.num_params:
                raise ValueError(
                    f"The index {qml.math.max(argnum)} exceeds the number of "
                    f"trainable tape parameters ({tape.num_params})." + _trainability_note
                )
            # ...and translate it to Boolean 1D iterable
            argnum = [i in argnum for i in range(tape.num_params)]
        elif len(argnum) != tape.num_params:
            # If the 1D iterable already is Boolean, check its length
            raise ValueError(
                "One-dimensional Boolean array argnum is expected to have as many entries as the "
                f"tape has trainable parameters ({tape.num_params}), but got {len(argnum)}."
                + _trainability_note
            )
        # Finally mark all combinations using the outer product
        argnum = qml.math.tensordot(argnum, argnum, axes=0)

    elif not (
        qml.math.shape(argnum) == (tape.num_params,) * 2
        and qml.math.array(argnum).dtype == bool
        and qml.math.allclose(qml.math.transpose(argnum), argnum)
    ):
        # If the iterable is 2D, make sure it is Boolean, symmetric and of the correct size
        raise ValueError(
            f"Expected a symmetric 2D Boolean array with shape {(tape.num_params,) * 2} "
            f"for argnum, but received {argnum}." + _trainability_note
        )
    return argnum


def _collect_recipes(tape, argnum, diff_methods, diagonal_shifts, off_diagonal_shifts):
    r"""Extract second order recipes for the tape operations for the diagonal of the Hessian
    as well as the first-order derivative recipes for the off-diagonal entries.
    """
    diag_argnum = qml.math.diag(argnum)
    offdiag_argnum = qml.math.any(argnum ^ qml.math.diag(qml.math.diag(argnum)), axis=0)

    diag_recipes = []
    partial_offdiag_recipes = []
    diag_shifts_idx = offdiag_shifts_idx = 0
    for i, (d, od) in enumerate(zip(diag_argnum, offdiag_argnum)):
        if not d or diff_methods[i] == "0":
            # hessian will be set to 0 for this row/column
            diag_recipes.append(None)
        else:
            # Get the diagonal second-order derivative recipe
            diag_shifts = None if diagonal_shifts is None else diagonal_shifts[diag_shifts_idx]
            diag_recipes.append(_get_operation_recipe(tape, i, diag_shifts, order=2))
            diag_shifts_idx += 1

        if not od or diff_methods[i] == "0":
            # hessian will be set to 0 for this row/column
            partial_offdiag_recipes.append((None, None, None))
        else:
            # Create the first-order gradient recipes per parameter for off-diagonal entries
            offdiag_shifts = (
                None if off_diagonal_shifts is None else off_diagonal_shifts[offdiag_shifts_idx]
            )
            partial_offdiag_recipes.append(_get_operation_recipe(tape, i, offdiag_shifts, order=1))
            offdiag_shifts_idx += 1

    return diag_recipes, partial_offdiag_recipes


def _generate_offdiag_tapes(tape, idx, first_order_recipes, add_unshifted, tapes, coeffs):
    r"""Combine two univariate first order recipes and create
    multi-shifted tapes to compute the off-diagonal entry of the Hessian."""
    # pylint: disable=too-many-arguments

    recipe_i = first_order_recipes[idx[0]]
    recipe_j = first_order_recipes[idx[1]]
    # The columns of combined_rules contain the coefficients (1), the multipliers (2) and the
    # shifts (2) in that order, with the number in brackets indicating the number of columns
    combined_rules = _combine_shift_rules([recipe_i, recipe_j])
    # If there are unmultiplied, unshifted tapes, the coefficient is memorized and the term
    # removed from the list of tapes to create
    if np.allclose(combined_rules[0, 1:3], 1.0) and np.allclose(combined_rules[0, 3:5], 0.0):
        # Extract the unshifted coefficient, if the first shifts (multipliers) equal 0 (1).
        if add_unshifted:
            # Add the unshifted tape if it has not been added yet and is required
            # because f0 was not provided (both captured by add_unshifted).
            tapes.insert(0, tape)
            add_unshifted = False
        unshifted_coeff = combined_rules[0, 0]
        combined_rules = combined_rules[1:]
    else:
        unshifted_coeff = None

    s = combined_rules[:, 3:5]
    m = combined_rules[:, 1:3]
    new_tapes = generate_multishifted_tapes(tape, idx, s, m)
    tapes.extend(new_tapes)
    coeffs.append(combined_rules[:, 0])

    return add_unshifted, unshifted_coeff


def _generate_diag_tapes(tape, idx, diag_recipes, add_unshifted, tapes, coeffs):
    """Create the required parameter-shifted tapes for a single diagonal entry of
    the Hessian using precomputed second-order shift rules."""
    # pylint: disable=too-many-arguments
    # Obtain the recipe for the diagonal.
    c, m, s = diag_recipes[idx].T
    if s[0] == 0 and m[0] == 1.0:
        # Extract the unshifted coefficient, if the first shift (multiplier) equals 0 (1).
        if add_unshifted:
            # Add the unshifted tape if it has not been added yet and is required
            # because f0 was not provided (both captured by add_unshifted).
            tapes.insert(0, tape)
            add_unshifted = False
        unshifted_coeff = c[0]
        c, m, s = c[1:], m[1:], s[1:]
    else:
        unshifted_coeff = None

    # Create the shifted tapes for the diagonal entry and store them along with coefficients
    new_tapes = generate_shifted_tapes(tape, idx, s, m)
    tapes.extend(new_tapes)
    coeffs.append(c)

    return add_unshifted, unshifted_coeff


def _no_trainable_grad_new(tape):
    if len(tape.measurements) == 1:
        return [], lambda _: qml.math.zeros((0,))

    return [], lambda _: tuple(qml.math.zeros((0,)) for _ in tape.measurements)


def _all_zero_grad_new(tape):
    num_params = len(tape.trainable_params)

    zeros_list = []
    for m in tape.measurements:
        if m.return_type is qml.measurements.Probability:
            shape = 2 ** len(m.wires)
        else:
            shape = ()

        zeros = tuple(
            tuple(qml.math.zeros(shape) for _ in range(num_params)) for _ in range(num_params)
        )
        if num_params == 1:
            zeros = zeros[0][0]

        zeros_list.append(zeros)

    if len(tape.measurements) == 1:
        return [], lambda _: zeros_list[0]

    return [], lambda _: tuple(zeros_list)


def expval_hessian_param_shift(
    tape, argnum, diff_methods, diagonal_shifts, off_diagonal_shifts, f0
):
    r"""Generate the Hessian tapes that are used in the computation of the second derivative of a
    quantum tape, using analytical parameter-shift rules to do so exactly. Also define a
    post-processing function to combine the results of evaluating the tapes into the Hessian.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        argnum (array_like[bool]): Parameter indices to differentiate
            with respect to, in form of a two-dimensional boolean ``array_like`` mask.
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
    for i, j in it.combinations_with_replacement(range(h_dim), r=2):
        if not argnum[i, j]:
            # The (i, j) entry of the Hessian is not to be computed
            hessian_coeffs.append(None)
            continue

        if i == j:
            add_unshifted, unshifted_coeffs[(i, i)] = _generate_diag_tapes(
                tape, i, diag_recipes, add_unshifted, hessian_tapes, hessian_coeffs
            )
        else:
            # Create tapes and coefficients for the off-diagonal entry by combining
            # the two univariate first-order derivative recipes.
            add_unshifted, unshifted_coeffs[(i, j)] = _generate_offdiag_tapes(
                tape, (i, j), partial_offdiag_recipes, add_unshifted, hessian_tapes, hessian_coeffs
            )
    unshifted_coeffs = {key: val for key, val in unshifted_coeffs.items() if val is not None}

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
        # Keep track of tape results already consumed. Start with 1 if the unshifted tape was
        # included in the tapes for the Hessian.
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
            unshifted_coeff = unshifted_coeffs.get((i, j), None)
            if unshifted_coeff is not None:
                hess = hess + unshifted_coeff * r0

            hessian.append(hess)

        # Reshape the Hessian to have the QNode output dimensions on the outside, that is:
        #    (h_dim*h_dim, *out_dims) -> (h_dim, h_dim, *out_dims) -> (*out_dims, h_dim, h_dim)
        # Remember: h_dim = num_gate_args
        hessian = qml.math.reshape(qml.math.stack(hessian), (h_dim, h_dim) + out_dim)
        reordered_axes = list(range(2, len(out_dim) + 2)) + [0, 1]
        return qml.math.transpose(hessian, axes=reordered_axes)

    return hessian_tapes, processing_fn


def _expval_hessian_param_shift_tuple(
    tape, argnum, diff_methods, diagonal_shifts, off_diagonal_shifts, f0
):
    r"""Generate the Hessian tapes that are used in the computation of the second derivative of a
    quantum tape, using analytical parameter-shift rules to do so exactly. Also define a
    post-processing function to combine the results of evaluating the tapes into the Hessian.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        argnum (array_like[bool]): Parameter indices to differentiate
            with respect to, in form of a two-dimensional boolean ``array_like`` mask.
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
    for i, j in it.combinations_with_replacement(range(h_dim), r=2):
        if not argnum[i, j]:
            # The (i, j) entry of the Hessian is not to be computed
            hessian_coeffs.append(None)
            continue

        if i == j:
            add_unshifted, unshifted_coeffs[(i, i)] = _generate_diag_tapes(
                tape, i, diag_recipes, add_unshifted, hessian_tapes, hessian_coeffs
            )
        else:
            # Create tapes and coefficients for the off-diagonal entry by combining
            # the two univariate first-order derivative recipes.
            add_unshifted, unshifted_coeffs[(i, j)] = _generate_offdiag_tapes(
                tape, (i, j), partial_offdiag_recipes, add_unshifted, hessian_tapes, hessian_coeffs
            )
    unshifted_coeffs = {key: val for key, val in unshifted_coeffs.items() if val is not None}

    def processing_fn(results):
        # Apply the same squeezing as in qml.QNode to make the transform output consistent.
        # pylint: disable=protected-access

        num_measurements = len(tape.measurements)
        if num_measurements == 1:
            results = tuple((r,) for r in results)

        hessians = []
        start = 1 if unshifted_coeffs and f0 is None else 0
        r0 = results[0] if start == 1 else f0

        for i, j in it.product(range(h_dim), repeat=2):

            if j < i:
                hessians.append(hessians[j * h_dim + i])
                continue

            k = i * h_dim + j - i * (i + 1) // 2
            coeffs = hessian_coeffs[k]

            if coeffs is None or len(coeffs) == 0:
                hessian = []
                for m in range(len(tape.measurements)):
                    hessian.append(qml.math.zeros_like(results[0][m]))

                hessians.append(tuple(hessian))
                continue

            res = results[start : start + len(coeffs)]
            start = start + len(coeffs)

            hessian = []
            for m in range(len(tape.measurements)):
                measure_res = qml.math.stack([r[m] for r in res])
                coeffs = qml.math.convert_like(coeffs, measure_res)
                hess = qml.math.tensordot(measure_res, coeffs, [[0], [0]])
                unshifted_coeff = unshifted_coeffs.get((i, j), None)

                if unshifted_coeff is not None:
                    hess = hess + unshifted_coeff * r0[m]

                hess = qml.math.array(hess, like=measure_res)
                hessian.append(hess)

            hessians.append(tuple(hessian))

        # reshape the hessian so that the num_measurements axis precedes the num_params axes
        hessians = tuple(tuple(h[i] for h in hessians) for i in range(num_measurements))

        # replace the num_params ** 2 axis with two num_params axes
        hessians = tuple(
            tuple(tuple(hess[i * h_dim + j] for j in range(h_dim)) for i in range(h_dim))
            for hess in hessians
        )

        # squeeze every axis with size 1
        if h_dim == 1:
            hessians = tuple(hess[0][0] for hess in hessians)

        if num_measurements == 1:
            hessians = hessians[0]

        return hessians

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
        argnum (int or list[int] or array_like[bool] or None): Parameter indices to differentiate
            with respect to. If not provided, the Hessian with respect to all
            trainable indices is returned. Note that the indices refer to tape
            parameters both if ``tape`` is a tape, and if it is a QNode. If an ``array_like``
            is provided, it is expected to be a symmetric two-dimensional Boolean mask with
            shape ``(n, n)`` where ``n`` is the number of trainable tape parameters.
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

    .. details::
        :title: Usage Details

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

        >>> for h_tape in hessian_tapes[0:4]:
        ...     print(qml.drawer.tape_text(h_tape, decimals=1))
        0: ──RX(0.5)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(-2.6)─╭●───────┤ ╭<Z@Z>
        1: ───────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(2.1)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(1.8)─┤ ╰<Z@Z>
        0: ──RX(2.1)─╭●────────┤ ╭<Z@Z>
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
        >>> for h_tape in hessian_tapes[0:4]:
        ...     print(qml.drawer.tape_text(h_tape, decimals=1))
        0: ──RX(0.5)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(0.0)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(1.0)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(1.0)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.4)─┤ ╰<Z@Z>

        .. note::

            Note that the ``diagonal_shifts`` are interpreted as *first-order* derivative
            shift values. That means they are used to generate a first-order derivative
            recipe, which then is iterated in order to obtain the second-order derivative
            for the diagonal Hessian entry. Explicit control over the used second-order
            shifts is not implemented.

        Finally, the ``argnum`` argument can be used to compute the Hessian only for some of the
        variational parameters. Note that this indexing refers to trainable tape parameters both
        if ``tape`` is a ``QNode`` and if it is a ``QuantumTape``.

        >>> hessian_tapes, postproc_fn = qml.gradients.param_shift_hessian(tape, argnum=(1,))
        >>> postproc_fn(qml.execute(hessian_tapes, dev, None))
        array([[0.        , 0.        ],
               [0.        , 0.05998862]])

    """
    if qml.active_return():
        return _param_shift_hessian_tuple(
            tape,
            argnum=argnum,
            diagonal_shifts=diagonal_shifts,
            off_diagonal_shifts=off_diagonal_shifts,
            f0=f0,
        )

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

    bool_argnum = _process_argnum(argnum, tape)

    compare_diag_to = qml.math.sum(qml.math.diag(bool_argnum))
    offdiag = bool_argnum ^ qml.math.diag(qml.math.diag(bool_argnum))
    compare_offdiag_to = qml.math.sum(qml.math.any(offdiag, axis=0))

    if diagonal_shifts is not None and len(diagonal_shifts) != compare_diag_to:
        raise ValueError(
            "The number of provided sets of shift values for diagonal entries "
            f"({len(diagonal_shifts)}) does not match the number of marked arguments "
            f"to compute the diagonal for ({compare_diag_to})."
        )
    if off_diagonal_shifts is not None and len(off_diagonal_shifts) != compare_offdiag_to:
        raise ValueError(
            "The number of provided sets of shift values for off-diagonal entries "
            f"({len(off_diagonal_shifts)}) does not match the number of marked arguments "
            f"for which to compute at least one off-diagonal entry ({compare_offdiag_to})."
        )

    gradient_analysis(tape, grad_fn=qml.gradients.param_shift)
    # If argnum is given, the grad_method_validation may allow parameters with
    # finite-difference as method. If they are among the requested argnum, we catch this
    # further below (as no fallback function in analogy to `param_shift` is used currently).
    method = "analytic" if argnum is None else "best"
    diff_methods = grad_method_validation(method, tape)

    for i, g in enumerate(diff_methods):
        if g == "0":
            bool_argnum[i] = bool_argnum[:, i] = False
    if qml.math.all(~bool_argnum):  # pylint: disable=invalid-unary-operand-type
        par_dim = len(tape.trainable_params)
        return [], lambda _: qml.math.zeros([tape.output_dim, par_dim, par_dim])

    # Find all argument indices that appear in at least one derivative that was requested
    choose_argnum = qml.math.where(qml.math.any(bool_argnum, axis=0))[0]
    # If any of these argument indices correspond to a finite difference
    # derivative (diff_methods[idx]="F"), raise an error.
    unsupported_params = {idx for idx in choose_argnum if diff_methods[idx] == "F"}
    if unsupported_params:
        raise ValueError(
            "The parameter-shift Hessian currently does not support the operations "
            f"for parameter(s) {unsupported_params}."
        )

    return expval_hessian_param_shift(
        tape, bool_argnum, diff_methods, diagonal_shifts, off_diagonal_shifts, f0
    )


def _param_shift_hessian_tuple(
    tape, argnum=None, diagonal_shifts=None, off_diagonal_shifts=None, f0=None
):
    r"""Transform a QNode to compute the parameter-shift Hessian with respect to its trainable
    parameters. This is the Hessian transform to replace the old one in the new return types system

    Use this transform to explicitly generate and explore parameter-shift circuits for computing
    the Hessian of QNodes directly, without computing first derivatives.

    For second-order derivatives of more complicated cost functions, please consider using your
    chosen autodifferentiation framework directly, by chaining gradient computations:

    >>> qml.jacobian(qml.grad(cost))(weights)

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
        argnum (int or list[int] or array_like[bool] or None): Parameter indices to differentiate
            with respect to. If not provided, the Hessian with respect to all
            trainable indices is returned. Note that the indices refer to tape
            parameters both if ``tape`` is a tape, and if it is a QNode. If an ``array_like``
            is provided, it is expected to be a symmetric two-dimensional Boolean mask with
            shape ``(n, n)`` where ``n`` is the number of trainable tape parameters.
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

    .. details::
        :title: Usage Details

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

        >>> for h_tape in hessian_tapes[0:4]:
        ...     print(qml.drawer.tape_text(h_tape, decimals=1))
        0: ──RX(0.5)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(-2.6)─╭●───────┤ ╭<Z@Z>
        1: ───────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(2.1)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(1.8)─┤ ╰<Z@Z>
        0: ──RX(2.1)─╭●────────┤ ╭<Z@Z>
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
        >>> for h_tape in hessian_tapes[0:4]:
        ...     print(qml.drawer.tape_text(h_tape, decimals=1))
        0: ──RX(0.5)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(0.0)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(1.0)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.2)─┤ ╰<Z@Z>
        0: ──RX(1.0)─╭●───────┤ ╭<Z@Z>
        1: ──────────╰RY(0.4)─┤ ╰<Z@Z>

        .. note::

            Note that the ``diagonal_shifts`` are interpreted as *first-order* derivative
            shift values. That means they are used to generate a first-order derivative
            recipe, which then is iterated in order to obtain the second-order derivative
            for the diagonal Hessian entry. Explicit control over the used second-order
            shifts is not implemented.

        Finally, the ``argnum`` argument can be used to compute the Hessian only for some of the
        variational parameters. Note that this indexing refers to trainable tape parameters both
        if ``tape`` is a ``QNode`` and if it is a ``QuantumTape``.

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
        return _no_trainable_grad_new(tape)

    bool_argnum = _process_argnum(argnum, tape)

    compare_diag_to = qml.math.sum(qml.math.diag(bool_argnum))
    offdiag = bool_argnum ^ qml.math.diag(qml.math.diag(bool_argnum))
    compare_offdiag_to = qml.math.sum(qml.math.any(offdiag, axis=0))

    if diagonal_shifts is not None and len(diagonal_shifts) != compare_diag_to:
        raise ValueError(
            "The number of provided sets of shift values for diagonal entries "
            f"({len(diagonal_shifts)}) does not match the number of marked arguments "
            f"to compute the diagonal for ({compare_diag_to})."
        )
    if off_diagonal_shifts is not None and len(off_diagonal_shifts) != compare_offdiag_to:
        raise ValueError(
            "The number of provided sets of shift values for off-diagonal entries "
            f"({len(off_diagonal_shifts)}) does not match the number of marked arguments "
            f"for which to compute at least one off-diagonal entry ({compare_offdiag_to})."
        )

    gradient_analysis(tape, grad_fn=qml.gradients.param_shift)
    # If argnum is given, the grad_method_validation may allow parameters with
    # finite-difference as method. If they are among the requested argnum, we catch this
    # further below (as no fallback function in analogy to `param_shift` is used currently).
    method = "analytic" if argnum is None else "best"
    diff_methods = grad_method_validation(method, tape)

    for i, g in enumerate(diff_methods):
        if g == "0":
            bool_argnum[i] = bool_argnum[:, i] = False
    if qml.math.all(~bool_argnum):  # pylint: disable=invalid-unary-operand-type
        return _all_zero_grad_new(tape)

    # Find all argument indices that appear in at least one derivative that was requested
    choose_argnum = qml.math.where(qml.math.any(bool_argnum, axis=0))[0]
    # If any of these argument indices correspond to a finite difference
    # derivative (diff_methods[idx]="F"), raise an error.
    unsupported_params = {idx for idx in choose_argnum if diff_methods[idx] == "F"}
    if unsupported_params:
        raise ValueError(
            "The parameter-shift Hessian currently does not support the operations "
            f"for parameter(s) {unsupported_params}."
        )

    return _expval_hessian_param_shift_tuple(
        tape, bool_argnum, diff_methods, diagonal_shifts, off_diagonal_shifts, f0
    )
