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
A transform for decomposing arbitrary single-qubit QubitUnitary gates into elementary gates.
"""
import pennylane as qml
from pennylane.transforms import qfunc_transform
from pennylane.transforms.decompositions import zyz_decomposition


@qfunc_transform
def unitary_to_rot(tape):
    """Quantum function transform to decomposes all instances of single-qubit :class:`~.QubitUnitary`
    operations to parametrized single-qubit operations.

    Diagonal operations will be converted to a single :class:`.RZ` gate, while non-diagonal
    operations will be converted to a :class:`.Rot` gate that implements the original operation
    up to a global phase.

    Args:
        qfunc (function): a quantum function

    **Example**

    Suppose we would like to apply the following unitary operation:

    .. code-block:: python3

        U = np.array([
            [-0.17111489+0.58564875j, -0.69352236-0.38309524j],
            [ 0.25053735+0.75164238j,  0.60700543-0.06171855j]
        ])

    The ``unitary_to_rot`` transform enables us to decompose
    such numerical operations (as well as unitaries that may be defined by parameters
    within the QNode, and instantiated therein), while preserving differentiability.


    .. code-block:: python3

        def qfunc():
            qml.QubitUnitary(U, wires=0)
            return qml.expval(qml.PauliZ(0))

    The original circuit is:

    >>> dev = qml.device('default.qubit', wires=1)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)())
     0: ──U0──┤ ⟨Z⟩
    U0 =
    [[-0.17111489+0.58564875j -0.69352236-0.38309524j]
     [ 0.25053735+0.75164238j  0.60700543-0.06171855j]]

    We can use the transform to decompose the gate:

    >>> transformed_qfunc = unitary_to_rot(qfunc)
    >>> transformed_qnode = qml.QNode(transformed_qfunc, dev)
    >>> print(qml.draw(transformed_qnode)())
     0: ──Rot(-1.35, 1.83, -0.606)──┤ ⟨Z⟩

    """
    for op in tape.operations + tape.measurements:
        if isinstance(op, qml.QubitUnitary):
            # Only act on single-qubit unitary operations
            if qml.math.shape(op.parameters[0]) != (2, 2):
                continue

            zyz_decomposition(op.parameters[0], op.wires[0])
        else:
            qml.apply(op)
