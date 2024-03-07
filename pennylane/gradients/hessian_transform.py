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
"""This module contains utilities for defining custom Hessian transforms,
including a decorator for specifying Hessian expansions."""
from string import ascii_letters as ABC

import pennylane as qml


def _process_jacs(jac, qhess):
    """
    Combine the classical and quantum jacobians
    """
    # Check for a Jacobian equal to the identity matrix.
    if not qml.math.is_abstract(jac):
        shape = qml.math.shape(jac)
        is_square = len(shape) == 2 and shape[0] == shape[1]
        if is_square and qml.math.allclose(jac, qml.numpy.eye(shape[0])):
            return qhess if len(qhess) > 1 else qhess[0]

    hess = []
    for qh in qhess:
        if not isinstance(qh, tuple) or not isinstance(qh[0], tuple):
            # single parameter case
            qh = qml.math.expand_dims(qh, [0, 1])
        else:
            # multi parameter case
            qh = qml.math.stack([qml.math.stack(row) for row in qh])

        jac_ndim = len(qml.math.shape(jac))

        # The classical jacobian has shape (num_params, num_qnode_args)
        # The quantum Hessian has shape (num_params, num_params, output_shape)
        # contracting the quantum Hessian with the classical jacobian twice gives
        # a result with shape (num_qnode_args, num_qnode_args, output_shape)

        qh_indices = "ab..."

        # contract the first axis of the jacobian with the first and second axes of the Hessian
        first_jac_indices = f"a{ABC[2:2 + jac_ndim - 1]}"
        second_jac_indices = f"b{ABC[2 + jac_ndim - 1:2 + 2 * jac_ndim - 2]}"

        result_indices = f"{ABC[2:2 + 2 * jac_ndim - 2]}..."
        qh = qml.math.einsum(
            f"{qh_indices},{first_jac_indices},{second_jac_indices}->{result_indices}",
            qh,
            jac,
            jac,
        )

        hess.append(qh)

    return tuple(hess) if len(hess) > 1 else hess[0]
