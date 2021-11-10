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

from .gradient_transform import gradient_transform
from .parameter_shift import _gradient_analysis, _process_gradient_recipe
from .finite_difference import generate_shifted_tapes

@gradient_transform
def param_shift_hessian(tape, f0=None):
    """Generate parameter-shift tapes and postprocessing method
    to directly compute the hessian."""

    # perform gradient method validation
    if any(m.return_type is qml.operation.State for m in tape.measurements):
        raise ValueError(
            "Computing the gradient of circuits that return the state is not supported."
        )

    _gradient_analysis(tape)
    gradient_tapes = []
    gradient_coeffs = []
    shapes = []
    unshifted_coeffs = []

    if not tape.trainable_params:
        return gradient_tapes, lambda _: np.zeros([tape.output_dim, len(tape.trainable_params)])

    # for now assume all operations support the 2-term parameter shift rule
    for idx, _ in enumerate(tape.trainable_params):

        # the hessian for 2-term parameter-shift rule can be expressed as
        recipe = [[0.5, 1, np.pi], [-0.5, 1, 0]]
        recipe = _process_gradient_recipe(recipe)
        coeffs, multipliers, shifts = recipe

        if shifts[0] == 0 and multipliers[0] == 1:
            # Gradient recipe includes a term with zero shift.

            if not unshifted_coeffs and f0 is None:
                # Ensure that the unshifted tape is appended
                # to the gradient tapes, if not already.
                gradient_tapes.append(tape)

            # Store the unshifted coefficient. We know that
            # it will always be the first coefficient due to processing.
            unshifted_coeffs.append(coeffs[0])
            coeffs, multipliers, shifts = recipe[:, 1:]

        # generate the gradient tapes
        gradient_coeffs.append(coeffs)
        g_tapes = generate_shifted_tapes(tape, idx, shifts, multipliers)

        gradient_tapes.extend(g_tapes)
        shapes.append(len(g_tapes))

    def processing_fn(results):
        grads = []
        start = 1 if unshifted_coeffs and f0 is None else 0
        r0 = f0 or results[0]

        for i, s in enumerate(shapes):

            if s == 0:
                # parameter has zero gradient
                g = qml.math.zeros_like(results[0])
                grads.append(g)
                continue

            res = results[start : start + s]
            start = start + s

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
