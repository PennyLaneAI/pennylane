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
"""
This file defines how to decompose arbitrary controlled operations.
"""
import pennylane as qml


def _grey_code(n):
    """Returns the grey code of order n."""
    if n == 1:
        return [[0], [1]]
    previous = _grey_code(n - 1)
    code = [[0] + old for old in previous]
    code += [[1] + old for old in reversed(previous)]
    return code


def ctrl_decomposition1(op, control):
    """Uses section 7 to decompose the controlled operation.

    https://arxiv.org/pdf/quant-ph/9503016.pdf

    Args:
        op: The target operation
        control (Iterable[Any]): The control wires

    Returns:
        list[Operator]: The decomposition

    """

    n_control = len(control)

    target_op = qml.pow(op, 2 ** (1 - n_control), lazy=False, do_queue=False)
    adj_target = qml.adjoint(target_op)
    active_pairs = set()

    gates = []

    for code in _grey_code(n_control)[1:]:

        active_wires = [cwire for cwire, codei in zip(control, code) if codei == 1]
        target_wire = active_wires[0]

        new_active_pairs = {(w, target_wire) for w in active_wires[1:]}
        changed_active_pairs = active_pairs.symmetric_difference(new_active_pairs)

        gates += [qml.CNOT(pair) for pair in changed_active_pairs]

        _target = target_op if sum(code) % 2 == 0 else adj_target
        gates.append(qml.ctrl(_target, target_wire))
        active_pairs = new_active_pairs

    return gates
