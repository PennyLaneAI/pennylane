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
"""Transform for applying custom decompositions of operations."""

from pennylane import apply
from pennylane.transforms import qfunc_transform

# TODO/questions
#  - write a function to validate decompositions? Or put that on the user?
#  - should this go in a transpilation module instead of decompositions?


@qfunc_transform
def apply_custom_decomposition(tape, custom_decomps=None):
    r"""Quantum function transform capable of applying user-specific decompositions
    in place of specified gates.

    Args:
        qfunc (function): a quantum function
        decompositions (dict[str : function]): a dictionary containing
            pairs of operator names and alternative decomposition functions.

    Returns:
        function: the transformed quantum function

    **Example**

    Suppose that instead of applying the typical decomposition of a Hadamard in
    a circuit,

    .. math::

        H = RZ(\pi/2) RX(\pi/2) RZ(\pi/2)

    we would instead like to use the decomposition

    .. math::

        H = X \cdot RY(\pi/2)

    We can define a custom decomposition function for the Hadamard. The
    signature must match the signature of the original decomposition.

    .. code-block:: python3

        def custom_hadamard(wires):
            return [qml.RY(np.pi/2, wires=wires), qml.PauliX(wires=wires)]

    We can do likewise for other gates. To use the ``apply_custom_decomposition``
    transform, we pass a dictionary containing the mapping from operator name
    to decomposition:

    >>> custom_decomps = {qml.Hadamard : custom_hadamard}

    Let's create a quantum function:

    .. code-block:: python3

        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=1)
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliZ(wires=1))

    Now we can apply the transform:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> transformed_qfunc = apply_custom_decomposition(custom_decomps=custom_decomps)(qfunc)
    >>> qnode = qml.QNode(transformed_qfunc, dev)
    >>> print(qml.draw(qnode)(0.3))
     0: ──RY(1.57)──X──╭C────────────────────────┤
     1: ───────────────╰X──RX(0.3)──RY(1.57)──X──┤ ⟨Z⟩
    """

    if custom_decomps is not None:

        gates_with_custom_decomps = list(custom_decomps.keys())

        for op in tape.operations:
            if op.name in gates_with_custom_decomps:
                custom_decomps[op.name](op.wires)
            else:
                apply(op)

        for m in tape.measurements:
            apply(m)
