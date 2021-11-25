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

from pennylane.transforms.tape_expand import expand_invalid_trainable
from .parameter_shift import _gradient_analysis


def _process_gradient_recipe(gradient_recipe, tol=1e-10):
    """Utility function to process gradient recipes."""

    gradient_recipe = np.array(gradient_recipe).T
    # remove all small coefficients, shifts, and multipliers
    gradient_recipe[np.abs(gradient_recipe) < tol] = 0
    # remove columns where the coefficients are 0
    gradient_recipe = gradient_recipe[:, :, ~(gradient_recipe[0, 0] == 0)]
    # sort columns according to abs(shift2) then abs(shift1)
    gradient_recipe = gradient_recipe[:, :, np.lexsort(np.abs(gradient_recipe)[:, -1])]

    return gradient_recipe[0, 0], gradient_recipe[0, 1], gradient_recipe[:, 2].T


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

        for i, s in enumerate(shift):
            new_params[idx[i]] = new_params[idx[i]] + qml.math.convert_like(s, new_params[idx[i]])

        shifted_tape.set_parameters(new_params)
        tapes.append(shifted_tape)

    return tapes


class hessian_transform(qml.batch_transform):
    """Modified gradient_transform to account for different post-processing when computing
    the second derivate of a QNode."""

    def __init__(
        self, transform_fn, expand_fn=expand_invalid_trainable, differentiable=True, hybrid=True
    ):
        self.hybrid = hybrid
        super().__init__(transform_fn, expand_fn=expand_fn, differentiable=differentiable)

    def default_qnode_wrapper(self, qnode, targs, tkwargs):
        # Here, we overwrite the QNode execution wrapper in order
        # to take into account that classical processing may be present
        # inside the QNode.
        hybrid = tkwargs.pop("hybrid", self.hybrid)
        _wrapper = super().default_qnode_wrapper(qnode, targs, tkwargs)
        cjac_fn = qml.transforms.classical_jacobian(qnode, expand_fn=expand_invalid_trainable)

        def jacobian_wrapper(*args, **kwargs):
            qjac = _wrapper(*args, **kwargs)

            if any(m.return_type is qml.operation.Probability for m in qnode.qtape.measurements):
                qjac = qml.math.squeeze(qjac)

            if not hybrid:
                return qjac

            kwargs.pop("shots", False)
            cjac = cjac_fn(*args, **kwargs)

            if isinstance(cjac, tuple):
                # Classical processing of multiple arguments is present. Return qjac @ cjac.
                jacs = [
                    qml.math.squeeze(qml.math.tensordot(c, qjac, [[0], [-1]]))
                    for c in cjac
                    if c is not None
                ]
                return jacs

            is_square = cjac.shape == (1,) or (cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1])

            if is_square and qml.math.allclose(cjac, qml.numpy.eye(cjac.shape[0])):
                # Classical Jacobian is the identity. No classical processing
                # is present inside the QNode.
                return qjac

            # Classical processing of a single argument is present.
            # Given a classical jacobian of shape (x, y..), a quantum jacobian of shape (z.., x, x),
            # where x = # of gate args, y.. = shape of QNode args, and z.. = shape of QNode outputs,
            # we apply the following trasformation: (z.., x0, x1) -> (z.., y0.., y1..)
            # While the dimensions x0 and x1, and y0.. and y1.., are the same respectively, they are
            # labeled to keep track of the swapping that occurs during the transformation.

            num_out_dims = len(cjac.shape) - 1  # number of dims in y..
            jac = qml.math.tensordot(qjac, cjac, [[-1], [0]])  # -> (z.., x0, y1..)
            jac = qml.math.tensordot(jac, cjac, [[-1 - num_out_dims], [0]])  # -> (z.., y1.., y0..)
            for i in range(num_out_dims):
                jac = qml.math.swapaxes(jac, -1 - i, -1 - num_out_dims - i)  # -> (z.., y0.., y1..)

            return jac

        return jacobian_wrapper


@hessian_transform
def param_shift_hessian(tape):
    """Generate parameter-shift tapes and postprocessing method
    to directly compute the hessian."""

    # perform gradient method validation
    if any(m.return_type is qml.operation.State for m in tape.measurements):
        raise ValueError(
            "Computing the gradient of circuits that return the state is not supported."
        )

    _gradient_analysis(tape)
    diff_methods = tape._grad_method_validation("analytic")  # pylint: disable=protected-access
    gradient_tapes = []
    gradient_coeffs = []
    unshifted_coeffs = {}
    shapes = []
    h_dim = len(tape.trainable_params)

    if not tape.trainable_params:
        return gradient_tapes, lambda _: np.zeros([tape.output_dim, len(tape.trainable_params)])

    # The Hessian for a 2-term parameter-shift rule can be expressed via the following recipes.
    # Off-diagonal elements of the Hessian require shifts to two different parameter indices.
    # A recipe can thus be expressed via the tape patterns: (dummy values for ndarray creation)
    #       [[coeff, dummy], [mult, dummy], [shift1, shift2]]
    # Each corresponding to one term in the parameter-shift formula:
    #       didj f(x) = coeff * f(mult*x + shift1*ei + shift2*ej) + ...
    diag_recipe = [[[0.5], [1], [np.pi]], [[-0.5], [1], [0]]]
    off_diag_recipe = [
        [[0.25, 1], [1, 1], [np.pi / 2, np.pi / 2]],
        [[-0.25, 1], [1, 1], [-np.pi / 2, np.pi / 2]],
        [[-0.25, 1], [1, 1], [np.pi / 2, -np.pi / 2]],
        [[0.25, 1], [1, 1], [-np.pi / 2, -np.pi / 2]],
    ]

    # for now assume all operations support the 2-term parameter shift rule
    for i in range(tape.num_params):
        for j in range(tape.num_params):
            # optimization: only generate tapes for upper triangular matrix (j >= i)
            # optimization: skip partial derivates that are zero
            if j < i or diff_methods[i] == "0" or diff_methods[j] == "0":
                gradient_coeffs.append([])
                shapes.append(((i, j), 0))
                continue

            recipe = diag_recipe if i == j else off_diag_recipe
            coeffs, _, shifts = _process_gradient_recipe(recipe)

            # optimization: only compute the unshifted tape once
            if all(np.array(shifts[0]) == 0):
                if not unshifted_coeffs:
                    gradient_tapes.insert(0, tape)

                unshifted_coeffs[(i, j)] = coeffs[0]
                coeffs, shifts = coeffs[1:], shifts[1:]

            # generate the gradient tapes
            gradient_coeffs.append(coeffs)
            g_tapes = generate_multishifted_tapes(tape, (i, j), shifts)

            gradient_tapes.extend(g_tapes)
            shapes.append(((i, j), len(g_tapes)))

    def processing_fn(results):
        # The first results dimension is the number of terms/tapes in the parameter-shift
        # rule, the remaining ones are the QNode output dimensions.
        out_dim = np.shape(results)[1:]
        # The desired shape of the Hessian is: (QNode output dimensions, # gate args, # gate args)
        hessian, row = [], []
        # Keep track of tape results already consumed.
        start = 1 if unshifted_coeffs else 0

        for k, ((i, j), s) in enumerate(shapes):
            res = results[start : start + s]
            start = start + s

            # Compute the elements of the Hessian as the linear combination of
            # results and coefficients, barring optimization cases.
            if j < i:
                g = hessian[j][i]
            elif s == 0:
                g = qml.math.zeros(out_dim)
            else:
                res = qml.math.stack(res)
                g = qml.math.tensordot(
                    res, qml.math.convert_like(gradient_coeffs[k], res), [[0], [0]]
                )
                if (i, j) in unshifted_coeffs:
                    g += unshifted_coeffs[(i, j)] * results[0]

            row.append(g)
            if j == h_dim - 1:
                hessian.append(row)
                row = []

        # Reshape the Hessian to have the dimensions of the QNode output on the outside, that is:
        #         (h_dim, h_dim, out_dim) -> (out_dim, h_dim, h_dim)
        hessian = qml.math.array(hessian)
        dim_indices = list(range(len(out_dim) + 2))
        hessian = qml.math.transpose(hessian, axes=dim_indices[2:] + [0, 1])

        return qml.math.squeeze(hessian)

    return gradient_tapes, processing_fn
