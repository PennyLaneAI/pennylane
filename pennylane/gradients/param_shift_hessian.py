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
import numpy as np

import pennylane as qml

from .parameter_shift import _gradient_analysis
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
            new_params[id_] = new_params[id_] + qml.math.convert_like(s, new_params[id_])

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
                g = hessian[j * h_dim + i]
            elif s == 0:
                g = qml.math.zeros(out_dim)
            else:
                res = qml.math.stack(res)
                g = qml.math.tensordot(
                    res, qml.math.convert_like(hessian_coeffs[k], res), [[0], [0]]
                )
                if (i, j) in unshifted_coeffs:
                    g += unshifted_coeffs[(i, j)] * r0

            hessian.append(g)

        # Reshape the Hessian to have the dimensions of the QNode output on the outside, that is:
        #         (h_dim, h_dim, out_dim) -> (out_dim, h_dim, h_dim)
        hessian = qml.math.reshape(qml.math.stack(hessian), (h_dim, h_dim) + out_dim)
        reordered_axes = list(range(2, len(out_dim) + 2)) + [0, 1]
        hessian = qml.math.transpose(hessian, axes=reordered_axes)

        return qml.math.squeeze(hessian)

    return hessian_tapes, processing_fn


@hessian_transform
def param_shift_hessian(tape, f0=None):
    r"""Transform a QNode to compute the parameter-shift Hessian with respect to its trainable
    parameters.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the Hessian tapes include the original input tape, the 'f0' value is used
            instead of evaluating the input tape, reducing the number of device invocations.

    Returns:
        tensor_like or tuple[list[QuantumTape], function]:

        - If the input is a QNode, a tensor representing the output of the hybrid Hessian matrix
          of size ``(QNode output dimensions, QNode input dimensions, QNode input dimensions)``
          is returned. When the keyword ``hybrid=False`` is specified, the purely quantum Hessian
          matrix is returned instead, with the dimensions
          ``(QNode output dimensions, number of gate arguments, number of gate arguments)``.
          The difference between the two accounts for the mapping of QNode arguments
          to the actual gate arguments, which can include classical computations.

        - If the input is a tape, a tuple containing a list of generated tapes, in addition
          to a post-processing function to be applied to the evaluated tapes.

    **Examples**

        Applying the Hessian transform to a QNode directly computes the Hessian matrix:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        ... def circuit(x):
        ...     qml.RX(x[0], wires=0)
        ...    qml.RY(x[1], wires=0)
        ...    return qml.expval(qml.PauliZ(0))
        >>> x = np.array([0.1, 0.2], requires_grad=True)
        >>> qml.gradients.param_shift_hessian(circuit)(x)
        tensor([[-0.97517033,  0.01983384],
                [ 0.01983384, -0.97517033]], requires_grad=True)

        Applying it to a quantum tape instead produces the parameter-shifted Hessian tapes
        and a post-processing function to combine execution results into the Hessian matrix:

        >>> tape = circuit.qnode
        >>> hessian_tapes, postproc_fn = qml.gradients.param_shift_hessian(circuit)
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
        return [], lambda _: []

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
    diff_methods = tape._grad_method_validation("analytic")  # pylint: disable=protected-access

    return compute_hessian_tapes(tape, diff_methods, f0)
