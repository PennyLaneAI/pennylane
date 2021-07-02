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
of a qubit quantum tape.
"""
import numpy as np

import pennylane as qml


def expval_grad(tape, idx, gradient_recipe=None):
    r"""Generate the parameter-shift tapes and postprocessing methods required
    to compute the gradient of an expectation value with respect to an
    expectation value.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        idx (int): trainable parameter index to differentiate with respect to
        gradient_recipe (tuple(Union(list[list[float]], None)) or None: Gradient recipe for the
            parameter-shift method.

            This is a tuple with one nested list per operation parameter. For
            parameter :math:`\phi_k`, the nested list contains elements of the form
            :math:`[c_i, a_i, s_i]` where :math:`i` is the index of the
            term, resulting in a gradient recipe of

            .. math:: \frac{\partial}{\partial\phi_k}f = \sum_{i} c_i f(a_i \phi_k + s_i).

            If ``None``, the default gradient recipe containing the two terms
            :math:`[c_0, a_0, s_0]=[1/2, 1, \pi/2]` and :math:`[c_1, a_1,
            s_1]=[-1/2, 1, -\pi/2]` is assumed for every parameter.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
        in addition to a post-processing function to be applied to the evaluated
        tapes.

    **Gradients of expectation values**

    For a variational evolution :math:`U(\mathbf{p}) \vert 0\rangle` with
    :math:`N` parameters :math:`\mathbf{p}`,

    consider the expectation value of an observable :math:`O`:

    .. math::

        f(\mathbf{p})  = \langle \hat{O} \rangle(\mathbf{p}) = \langle 0 \vert
        U(\mathbf{p})^\dagger \hat{O} U(\mathbf{p}) \vert 0\rangle.


    The gradient of this expectation value can be calculated using :math:`2N` expectation
    values using the parameter-shift rule:

    .. math::

        \frac{\partial f}{\partial \mathbf{p}} = \frac{1}{2\sin s} \left[ f(\mathbf{p} + s) -
        f(\mathbf{p} -s) \right].

    """
    # check if the quantum tape contains any variance measurements
    var_mask = [m.return_type is qml.operation.Variance for m in tape.measurements]

    if any(var_mask):
        raise ValueError("Does not support gradients of tapes with variance output.")

    t_idx = list(tape.trainable_params)[idx]
    op = tape._par_info[t_idx]["op"]
    p_idx = tape._par_info[t_idx]["p_idx"]

    if gradient_recipe is None:
        gradient_recipe = op.get_parameter_shift(p_idx, shift=np.pi / 2)

    params = qml.math.stack(tape.get_parameters())
    shift = np.zeros_like(qml.math.toarray(params))
    coeffs = []
    tapes = []

    for c, a, s in gradient_recipe:
        shift[idx] = s
        shifted_tape = tape.copy(copy_operations=True)
        shifted_params = a * params + qml.math.convert_like(shift, params)
        shifted_tape.set_parameters(qml.math.unstack(shifted_params))

        coeffs.append(c)
        tapes.append(shifted_tape)

    def processing_fn(results):
        """Computes the gradient of the parameter at index idx via the
        parameter-shift method.

        Args:
            results (list[real]): evaluated quantum tapes

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        results = qml.math.squeeze(qml.math.stack(results))
        return sum([c * r for c, r in zip(coeffs, results)])

    return tapes, processing_fn
