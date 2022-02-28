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


def simplify_decomposition(decomp):
    """unpacks nested lists of operator products"""
    if not decomp:  # empty list
        return decomp
    if isinstance(decomp[0], qml.ops.math.MatMul):
        return simplify_decomposition([decomp[0].hyperparameters['left'], decomp[0].hyperparameters['right']]) + simplify_decomposition(decomp[1:])
    return decomp[:1] + simplify_decomposition(decomp[1:])


# def test_simplify_decomposition():
#     op = qml.Hadamard(wires=1) @ qml.PauliX(wires=0) @ qml.PauliX(wires=5) @ qml.PauliX(wires=10) @ qml.PauliX(wires=1)
#     expected = [qml.Hadamard(wires=1), qml.PauliX(wires=0), qml.PauliX(wires=5), qml.PauliX(wires=10), qml.PauliX(wires=1)]
#     res = simplify_decomposition(op.decomposition())
#     for op, op_expected in zip(res, expected):
#         assert op.name == op_expected.name
#         assert op.wires == op_expected.wires


def simplify_terms(coeffs, ops):
    """unpacks nested lists of operator products"""

    if not ops:  # empty list
        return coeffs, ops

    if isinstance(ops[0], qml.ops.math.Sum):
        # extract ops from sum and distribute coefficient associatively
        first_coeffs, first_ops = simplify_terms([coeffs[0], coeffs[0]], [ops[0].hyperparameters['left'], ops[0].hyperparameters['right']])
        remainder_coeffs, remainder_ops = simplify_terms(coeffs[1:], ops[1:])
        return first_coeffs+remainder_coeffs, first_ops+remainder_ops

    if isinstance(ops[0], qml.ops.math.ScalarMul):
        # extract op and scalar simplify again
        first_coeff, first_op = simplify_terms([ops[0].hyperparameters['scalar']], [ops[0].hyperparameters['op']])
        remainder_coeffs, remainder_ops = simplify_terms(coeffs[1:], ops[1:])
        return first_coeff+remainder_coeffs, first_op+remainder_ops

    remainder_coeffs, remainder_ops = simplify_terms(coeffs[1:], ops[1:])
    return coeffs[:1] + remainder_coeffs, ops[:1] + remainder_ops


op = qml.Hadamard(wires=1) + 2.*qml.PauliX(wires=0) + qml.PauliX(wires=5) + 3*(1.j*qml.PauliX(wires=10) + qml.PauliX(wires=1))
print(op.terms()[0], "\n",  op.terms()[1])
print()
res = simplify_terms(*op.terms())
print(res[0], "\n",  res[1])