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
"""Transform for unitary folding used in error mitigation."""

from pennylane import apply
from pennylane.transforms import qfunc_transform


@qfunc_transform
def unitary_folding(tape, num_folds=0):
    """Quantum function transform to perform unitary folding.

    The provided tape is run once in the normal direction, then ``num_folds``
    repetitions of its adjoint, followed by the original.

    Args:
        qfunc (function): A quantum function
        num_pairs (int): The number of folds to perform

    Returns:
        function: the transformed quantum function

    **Example**

    Consider the quantum function below:

    .. code-block:: python

        def circuit(x):
            qml.RX(x, wires=0)
            qml.S(wires=1)
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(0))

    The ``unitary_folding`` transform enables us to append multiple instances of
    the circuit's inverse and original versions:

    >>> dev = qml.device('default.qubit', wires=3)
    >>> new_circuit = unitary_folding(num_folds=2)(circuit)
    >>> qnode = qml.QNode(new_circuit, dev)
    >>> print(qml.draw(qnode)(0.5))
     0: ──RX(0.5)───RX(-0.5)───RX(0.5)──RX(-0.5)──RX(0.5)──────────────────────┤ ⟨Z⟩
     1: ──S────────╭C─────────╭C────────S⁻¹───────S────────╭C──╭C──S⁻¹──S──╭C──┤
     2: ───────────╰X─────────╰X───────────────────────────╰X──╰X──────────╰X──┤"""
    # First run is "forwards".
    for op in tape.operations:
        apply(op)

    # Now we do the folds
    for _ in range(num_folds):
        # Go through in reverse and apply the adjoints
        for op in tape.operations[::-1]:
            op.adjoint()

        # Go through forwards again
        for op in tape.operations:
            apply(op)

    # Apply the measurements normally
    for m in tape.measurements:
        apply(m)
