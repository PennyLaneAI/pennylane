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
import numpy as np

import pennylane as qml

from .parameter_shift import _gradient_analysis
from .gradient_transform import grad_method_validation
from .hessian_transform import hessian_transform


def _process_hessian_recipe(hessian_recipe, tol=1e-10):
    """Utility function to process Hessian recipes."""

    hessian_recipe = np.array(hessian_recipe).T
    # remove all small coefficients, shifts, and multipliers
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


def compute_hessian_tapes(tape, diff_methods, f0=None):
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
    h_dim = tape.num_params

    hessian_tapes = []
    hessian_coeffs = []
    unshifted_coeffs = {}
    shapes = []

    # The Hessian for a 2-term parameter-shift rule can be expressed via the following recipes.
    # Off-diagonal elements of the Hessian require shifts to two different parameter indices.
    # A recipe can thus be expressed via the tape patterns:
    #       [[coeff, dummy], [mult, dummy], [shift1, shift2]]    (dummy values for ndarray creation)
    # Each corresponding to one term in the parameter-shift formula:
    #       didj f(x) = coeff * f(mult*x + shift1*ei + shift2*ej) + ...
    diag_recipe = [[[0.5], [1], [np.pi]], [[-0.5], [1], [0]]]
    off_diag_recipe = [
        [[0.25, 1], [1, 1], [np.pi / 2, np.pi / 2]],
        [[-0.25, 1], [1, 1], [-np.pi / 2, np.pi / 2]],
        [[-0.25, 1], [1, 1], [np.pi / 2, -np.pi / 2]],
        [[0.25, 1], [1, 1], [-np.pi / 2, -np.pi / 2]],
    ]

    for i in range(h_dim):
        # optimization: skip partial derivatives that are zero
        if diff_methods[i] == "0":
            # We may fill in coefficients and shapes for the entire row.
            hessian_coeffs.extend([[]] * h_dim)
            shapes.extend(((i, j), 0) for j in range(h_dim))
            continue

        # Prefill Hessian in the lower triangular part for optimization below
        hessian_coeffs.extend([[]] * i)
        shapes.extend(((i, j), 0) for j in range(i))

        # optimization: only generate tapes for upper triangular matrix (j >= i)
        for j in range(i, h_dim):
            # optimization: skip partial derivatives that are zero
            if diff_methods[j] == "0":
                hessian_coeffs.append([])
                shapes.append(((i, j), 0))
                continue

            recipe = diag_recipe if i == j else off_diag_recipe
            coeffs, _, shifts = _process_hessian_recipe(recipe)

            # optimization: only compute the unshifted tape once
            if all(np.array(shifts[0]) == 0):
                if not unshifted_coeffs and f0 is None:
                    hessian_tapes.insert(0, tape)

                unshifted_coeffs[(i, j)] = coeffs[0]
                coeffs, shifts = coeffs[1:], shifts[1:]

            # generate the Hessian tapes
            hessian_coeffs.append(coeffs)
            h_tapes = generate_multishifted_tapes(tape, (i, j), shifts)

            hessian_tapes.extend(h_tapes)
            shapes.append(((i, j), len(h_tapes)))

    def processing_fn(results):
        # The first results dimension is the number of terms/tapes in the parameter-shift
        # rule, the remaining ones are the QNode output dimensions.
        out_dim = qml.math.shape(qml.math.stack(results))[1:]
        # The desired shape of the Hessian is:
        #       (QNode output dimensions, # trainable gate args, # trainable gate args),
        # but first we accumulate all elements into a list, since no array assingment is possible.
        hessian = []
        # Keep track of tape results already consumed.
        start = 1 if unshifted_coeffs and f0 is None else 0
        # Results of the unshifted tape.
        r0 = results[0] if start == 1 else f0

        for k, ((i, j), s) in enumerate(shapes):
            res = results[start : start + s]
            start = start + s

            # Compute the elements of the Hessian as the linear combination of
            # results and coefficients, barring optimization cases.
            if j < i:
                hess = hessian[j * h_dim + i]
            elif s == 0:
                hess = qml.math.zeros(out_dim)
            else:
                res = qml.math.stack(res)
                coeffs = qml.math.cast(qml.math.convert_like(hessian_coeffs[k], res), res.dtype)
                hess = qml.math.tensordot(res, coeffs, [[0], [0]])
                if (i, j) in unshifted_coeffs:
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
def param_shift_hessian(tape, f0=None):
    r"""Transform a QNode to compute the parameter-shift Hessian with respect to its trainable
    parameters.

    Use this transform to explicitly generate and explore parameter-shift circuits for computing
    the Hessian of QNodes directly, without computing first derivatives.

    For second-order derivatives of more complicated cost functions, please consider using your
    chosen autodifferentiation framework directly, by chaining gradient computations:

    >>> qml.jacobian(qml.grad(cost))(weights)

    .. note::

        Currently, parametric gates are only supported if they obey a two-term shift rule,
        which includes the following operations:

        "RX", "RY", "RZ", "Rot", "PhaseShift", "ControlledPhaseShift", "MultiRZ", "PauliRot",
        "U1", "U2", "U3", "SingleExcitationMinus", "SingleExcitationPlus", "DoubleExcitationMinus",
        "DoubleExcitationPlus", "OrbitalRotation".

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
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
    if any(m.return_type is qml.operation.Variance for m in tape.measurements):
        raise ValueError(
            "Computing the Hessian of circuits that return variances is currently not supported."
        )

    if not tape.trainable_params:
        warnings.warn(
            "Attempted to compute the hessian of a tape with no trainable parameters. "
            "If this is unintended, please mark trainable parameters in accordance with the "
            "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
        )
        return [], lambda _: ()

    # The parameter-shift Hessian implementation currently only supports
    # the two-term parameter-shift rule. Raise an error for unsupported operations.
    supported_ops = (
        "RX",
        "RY",
        "RZ",
        "Rot",
        "PhaseShift",
        "ControlledPhaseShift",
        "MultiRZ",
        "PauliRot",
        "U1",
        "U2",
        "U3",
        "SingleExcitationMinus",
        "SingleExcitationPlus",
        "DoubleExcitationMinus",
        "DoubleExcitationPlus",
        "OrbitalRotation",
    )

    for idx in range(tape.num_params):
        op, _ = tape.get_operation(idx)
        if op.name not in supported_ops:
            raise ValueError(
                f"The operation {op.name} is currently not supported for the parameter-shift "
                f"Hessian. Only two-term parameter shift rules are currently supported."
            )

    _gradient_analysis(tape)
    diff_methods = grad_method_validation("analytic", tape)

    if all(g == "0" for g in diff_methods):
        return [], lambda _: np.zeros(
            [tape.output_dim, len(tape.trainable_params), len(tape.trainable_params)]
        )

    return compute_hessian_tapes(tape, diff_methods, f0)
