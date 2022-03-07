# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=too-few-public-methods,function-redefined

import pennylane as qml


def flatten_decomposition(decomp):
    """Unpack nested lists of operator products."""
    if not decomp:  # empty list
        return decomp
    if isinstance(decomp[0], qml.ops.math.MatMul):
        return flatten_decomposition(
            [decomp[0].hyperparameters["left"], decomp[0].hyperparameters["right"]]
        ) + flatten_decomposition(decomp[1:])
    return decomp[:1] + flatten_decomposition(decomp[1:])


def flatten_terms(coeffs, ops):
    """Unpack nested terms.

    Args:
        coeffs [tensor_like]: coefficients
        ops [list[~.Operator]]: list of operators

    Returns:
        tensor_like, list[~.Operator]: flattened terms
    """

    if not ops:  # empty list
        return coeffs, ops

    if isinstance(ops[0], qml.ops.math.Sum):
        # extract ops from sum and distribute coefficient associatively
        first_coeffs, first_ops = flatten_terms(
            [coeffs[0], coeffs[0]],
            [ops[0].hyperparameters["left"], ops[0].hyperparameters["right"]],
        )
        remainder_coeffs, remainder_ops = flatten_terms(coeffs[1:], ops[1:])
        return first_coeffs + remainder_coeffs, first_ops + remainder_ops

    if isinstance(ops[0], qml.ops.math.ScalarProd):
        # extract op and scalar to simplify again
        first_coeff, first_op = flatten_terms(
            [ops[0].hyperparameters["scalar"]], [ops[0].hyperparameters["op"]]
        )
        remainder_coeffs, remainder_ops = flatten_terms(coeffs[1:], ops[1:])
        return first_coeff + remainder_coeffs, first_op + remainder_ops

    remainder_coeffs, remainder_ops = flatten_terms(coeffs[1:], ops[1:])
    return coeffs[:1] + remainder_coeffs, ops[:1] + remainder_ops
