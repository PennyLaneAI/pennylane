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
import numpy as np

import pennylane as qml

from .parameter_shift import _gradient_analysis, _get_operation_recipe
from .gradient_transform import grad_method_validation, choose_grad_methods
from .hessian_transform import hessian_transform
from .finite_difference import generate_shifted_tapes
from .general_shift_rules import _combine_shift_rules


def _process_hessian_recipe(hessian_recipe, tol=1e-10):
    """Utility function to process Hessian recipes."""

    hessian_recipe = np.array(hessian_recipe).T
    # set all small coefficients, shifts, and multipliers to 0
    hessian_recipe[np.abs(hessian_recipe) < tol] = 0
    # remove columns where the coefficients are 0
    hessian_recipe = hessian_recipe[:, :, ~(hessian_recipe[0, 0] == 0)]
    # sort columns according to abs(shift2) then abs(shift1)
    hessian_recipe = hessian_recipe[:, :, np.lexsort(np.abs(hessian_recipe)[:, -1])]

    return hessian_recipe[0, 0], hessian_recipe[0, 1], hessian_recipe[:, 2].T


def generate_multishifted_tapes(tape, idx, shifts):
    r"""Generate a list of tapes where the corresponding trainable parameter
    indices have been shifted by the values given.

    Args:
        tape (.QuantumTape): input quantum tape
        idx (list[int]): trainable parameter indices to shift the parameters of
        shifts (list[list[float or int]]): nested list of shift values, each
            list containing a value for each index

    Returns:
        list[QuantumTape]: List of quantum tapes. Each tape has multiple parameters
            (indicated by ``idx``) shifted by the values of ``shifts``. The length
            of the returned list of tapes will match the length of ``shifts``.
    """
    params = list(tape.get_parameters())
    tapes = []

    for shift in shifts:
        new_params = params.copy()
        shifted_tape = tape.copy(copy_operations=True)

        for id_, s in zip(idx, shift):
            dtype = new_params[id_].dtype
            new_params[id_] = new_params[id_] + qml.math.convert_like(s, new_params[id_])
            new_params[id_] = qml.math.cast(new_params[id_], dtype)

        shifted_tape.set_parameters(new_params)
        tapes.append(shifted_tape)

    return tapes


def expval_hessian_param_shift(tape, argnum, diff_methods, diagonal_shifts, off_diagonal_shifts, f0):
    r"""Generate the Hessian tapes that are used in the computation of the second derivative of a
    quantum tape, using analytical parameter-shift rules to do so exactly. Also define a
    post-processing function to combine the results of evaluating the Hessian tapes.

    Args:
        tape (.QuantumTape): input quantum tape
        diff_methods (list[string]): The differentiation method to use for each trainable parameter.
            Can be "A" or "0", where "A" is the analytical parameter shift rule and "0" indicates
            a 0 derivative (that is the parameter does not affect the tape's output).
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the Hessian tapes include the original input tape, the 'f0' value is used
            instead of evaluating the input tape, reducing the number of device invocations.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a list of generated tapes, in
            addition to a post-processing function to be applied to the results of the evaluated
            tapes.
    """
    argnum = tape.trainable_params if argnum is None else argnum

    #TODO: ASSERT len(diagonal_shits)=="len(off_diagonal_shifts)"==len(argnum)

    hessian_tapes = []
    hessian_coeffs = []
    num_tapes = []
    unshifted_coeffs = {}
    add_unshifted = f0 is None

    # The Hessian for a 2-term parameter-shift rule can be expressed via the following recipes.
    # Off-diagonal elements of the Hessian require shifts to two different parameter indices.
    # A recipe can thus be expressed via the tape patterns:
    #       [[coeff, dummy], [mult, dummy], [shift1, shift2]]    (dummy values for ndarray creation)
    # Each corresponding to one term in the parameter-shift formula:
    #       didj f(x) = coeff * f(mult*x + shift1*ei + shift2*ej) + ...

    hess_recipes = {}

    diag_recipes = []
    partial_offdiag_recipes = []
    h_dim = tape.num_params
    for i in range(h_dim):

        if i not in argnum or diff_methods[i] == "0":
            # hessian will be set to 0 for this row/column
            diag_recipes.append(None)
            partial_offdiag_recipes.append((None, None, None))
            continue

        # Get the diagonal second-order derivative recipe
        diag_shifts = None if diagonal_shifts is None else diagonal_shifts[i]
        diag_recipes.append(_get_operation_recipe(tape, i, diag_shifts, order=2))

        # Create the first-order gradient recipes per parameter
        _shifts = None if off_diagonal_shifts is None else off_diagonal_shifts[i]
        partial_offdiag_recipes.append(_get_operation_recipe(tape, i, _shifts, order=1))

    for i in range(h_dim):
        if diag_recipes[i] is None:
            hessian_coeffs.extend([None] * (h_dim - i))
            num_tapes.extend([0] * (h_dim - i))
            continue

        dc_i, dm_i, ds_i = diag_recipes[i]
        if ds_i[0] == 0 and dm_i[0] == 1.:
            if add_unshifted:
                hessian_tapes.insert(0, tape)
                add_unshifted = False
            unshifted_coeffs[(i, i)] = dc_i[0]
            dc_i, dm_i, ds_i = dc_i[1:], dm_i[1:], ds_i[1:]

        # Create the shifted tapes for the diagonal entries
        diag_tapes = generate_shifted_tapes(tape, i, ds_i, dm_i)
        hessian_tapes.extend(diag_tapes)
        print(f"extending hessian_tapes by {len(diag_tapes)} tapes, now has length {len(hessian_tapes)}")
        hessian_coeffs.append(dc_i)
        num_tapes.append(len(diag_tapes))

        c_i, m_i, s_i = partial_offdiag_recipes[i]
        for j in range(i + 1, h_dim):
            c_j, m_j, s_j = partial_offdiag_recipes[j]
            if c_j is None:
                hessian_coeffs.append(None)
                num_tapes.append(0)
                continue

            c_ij, *s_ij = _combine_shift_rules([(c_i, s_i), (c_j, s_j)])
            if s_ij[0][0] == s_ij[1][0] == 0:
                if add_unshifted:
                    hessian_tapes.insert(0, tape)
                    add_unshifted = False
                unshifted_coeffs[(i, j)] = c_ij
                c_ij, s_ij = c_ij[1:], s_ij[1:]

            h_tapes = generate_multishifted_tapes(tape, (i, j), zip(*s_ij))

            print(c_ij, s_ij)
            hessian_tapes.extend(h_tapes)
            print(f"extending hessian_tapes by {len(generate_multishifted_tapes(tape, (i, j), zip(*s_ij)))} tapes, now has length {len(hessian_tapes)}")
            print(hessian_tapes)
            hessian_coeffs.append(c_ij)
            print(hessian_coeffs[-1])
            num_tapes.append(len(h_tapes))

    def processing_fn(results):
        print(f"len(all results): {len(results)}")
        print(f"all shapes: {num_tapes}")
        # The first results dimension is the number of terms/tapes in the parameter-shift
        # rule, the remaining ones are the QNode output dimensions.
        out_dim = qml.math.shape(qml.math.stack(results))[1:]
        # The desired shape of the Hessian is:
        #       (QNode output dimensions, # trainable gate args, # trainable gate args),
        # but first we accumulate all elements into a list, since no array assignment is possible.
        hessian = []
        # Keep track of tape results already consumed.
        start = int(bool(unshifted_coeffs) and f0 is None)
        print(f"start: {start}")
        # Results of the unshifted tape.
        r0 = results[0] if start == 1 else f0

        for i, j in it.product(range(h_dim), repeat=2):
            if j < i:
                hessian.append(hessian[j * h_dim + i])
                continue
            k = i * h_dim + j - i * (i + 1) // 2
            _num_tapes = num_tapes[k]
            if _num_tapes == 0:
                hessian.append(qml.math.zeros(out_dim))
                continue

            res = results[start : start + _num_tapes]
            print(f"using results {start}:{start+_num_tapes}")
            start = start + _num_tapes

            print(f"i, j, k: {i}, {j}, {k}")
            print(f"len(res): {len(res)}")
            res = qml.math.stack(res)
            coeffs = qml.math.cast(qml.math.convert_like(hessian_coeffs[k], res), res.dtype)
            print(f"len(coeffs): {len(coeffs)}")
            hess = qml.math.tensordot(res, coeffs, [[0], [0]])
            if (i, j) in unshifted_coeffs:
                print(f"Using the unshifted tape for (i,j)={(i,j)}")
                hess = hess + unshifted_coeffs[(i, j)] * r0

            hessian.append(hess)

        # Reshape the Hessian to have the dimensions of the QNode output on the outside, that is:
        #     (h_dim*h_dim, out_dim) -> (h_dim, h_dim, out_dim) -> (out_dim, h_dim, h_dim)
        hessian = qml.math.reshape(qml.math.stack(hessian), (h_dim, h_dim) + out_dim)
        reordered_axes = list(range(2, len(out_dim) + 2)) + [0, 1]
        hessian = qml.math.transpose(hessian, axes=reordered_axes)

        return qml.math.squeeze(hessian)

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
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivative with respect to all
            trainable indices are returned.
        diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift values
            for the Hessian diagonal.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are assumed.
        off_diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift
            values for the off-diagonal entries of the Hessian.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            The combination of shifts into bivariate shifts is performed automatically.
            If unspecified, equidistant shifts are assumed.
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
    ...     qml.RY(x[1], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> x = np.array([0.1, 0.2], requires_grad=True)
    >>> qml.gradients.param_shift_hessian(circuit)(x)
    tensor([[-0.97517033,  0.01983384],
            [ 0.01983384, -0.97517033]], requires_grad=True)

    .. UsageDetails::

        The Hessian transform can also be applied to a quantum tape, instead producing the
        parameter-shifted Hessian tapes and a post-processing function to combine execution
        results:

        >>> circuit(x)  # generate the QuantumTape inside the QNode
        >>> tape = circuit.qtape
        >>> hessian_tapes, postproc_fn = qml.gradients.param_shift_hessian(tape)
        >>> hessian_tapes
        [<JacobianTape: wires=[0], params=2>,
            <JacobianTape: wires=[0], params=2>,
            <JacobianTape: wires=[0], params=2>,
            <JacobianTape: wires=[0], params=2>,
            <JacobianTape: wires=[0], params=2>,
            <JacobianTape: wires=[0], params=2>,
            <JacobianTape: wires=[0], params=2>]
        >>> postproc_fn(qml.execute(hessian_tapes, dev, None))
        array([[-0.97517033,  0.01983384],
                [ 0.01983384, -0.97517033]])

        The Hessian tapes can be inspected via their draw function, which reveals the different
        gate arguments generated from parameter-shift rules:

        >>> for h_tape in hessian_tapes:
        ...     print(qml.drawer.tape_text(h_tape, decimals=1))
        0: ──RX(0.1)──RY(0.2)─┤  <Z>
        0: ──RX(3.2)──RY(0.2)─┤  <Z>
        0: ──RX(1.7)──RY(1.8)─┤  <Z>
        0: ──RX(-1.5)──RY(1.8)─┤  <Z>
        0: ──RX(1.7)──RY(-1.4)─┤  <Z>
        0: ──RX(-1.5)──RY(-1.4)─┤  <Z>
        0: ──RX(0.1)──RY(3.3)─┤  <Z>

    """

    # Perform input validation before generating tapes.
    if any(m.return_type is qml.operation.State for m in tape.measurements):
        raise ValueError(
            "Computing the Hessian of circuits that return the state is not supported."
        )

    # The parameter-shift Hessian implementation currently doesn't support variance measurements.
    # TODO: Support variances similar to how param_shift does it
    if any(m.return_type is qml.operation.Variance for m in tape.measurements):
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

    _gradient_analysis(tape)
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

    return expval_hessian_param_shift(tape, argnum, diff_methods, diagonal_shifts, off_diagonal_shifts, f0)
