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
from pennylane.transforms.decompositions import zyz_decomposition, two_qubit_decomposition


@qfunc_transform
def unitary_to_rot(tape):
    r"""Quantum function transform to decomposes all instances of single-qubit and
    select instances of two-qubit :class:`~.QubitUnitary` operations to
    parametrized single-qubit operations.

    For single-qubit gates, diagonal operations will be converted to a single
    :class:`.RZ` gate, while non-diagonal operations will be converted to a
    :class:`.Rot` gate that implements the original operation up to a global
    phase. Two-qubit gates will be decomposed according to the
    :func:`pennylane.transforms.two_qubit_decomposition` function.

    .. warning::

        This transform is not fully differentiable for 2-qubit ``QubitUnitary``
        operations. See usage details below.

    Args:
        qfunc (function): a quantum function

    **Example**

    Suppose we would like to apply the following unitary operation:

    .. code-block:: python3

        U = np.array([
            [-0.17111489+0.58564875j, -0.69352236-0.38309524j],
            [ 0.25053735+0.75164238j,  0.60700543-0.06171855j]
        ])

    The ``unitary_to_rot`` transform enables us to decompose such numerical
    operations while preserving differentiability.

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


    .. UsageDetails::

        This decomposition is not fully differentiable. We **can** differentiate
        with respect to input QNode parameters when they are not used to
        explicitly construct a :math:`4 \times 4` unitary matrix being
        decomposed. So for example, the following will work:

        .. code-block:: python3

            U = scipy.stats.unitary_group.rvs(4)

            def circuit(angles):
                qml.QubitUnitary(U, wires=["a", "b"])
                qml.RX(angles[0], wires="a")
                qml.RY(angles[1], wires="b")
                qml.CNOT(wires=["b", "a"])
                return qml.expval(qml.PauliZ(wires="a"))

            dev = qml.device('default.qubit', wires=["a", "b"])
            transformed_qfunc = qml.transforms.unitary_to_rot(circuit)
            transformed_qnode = qml.QNode(transformed_qfunc, dev)

        >>> g = qml.grad(transformed_qnode)
        >>> params = np.array([0.2, 0.3], requires_grad=True)
        >>> g(params)
        array([ 0.00296633, -0.29392145])

        However, the following example will **not** be differentiable:

        .. code-block:: python3

            def circuit(angles):
                z = angles[0]
                x = angles[1]

                Z_mat = np.array([[np.exp(-1j * z / 2), 0.0], [0.0, np.exp(1j * z / 2)]])

                c = np.cos(x / 2)
                s = np.sin(x / 2) * 1j
                X_mat = np.array([[c, -s], [-s, c]])

                U = np.kron(Z_mat, X_mat)

                qml.Hadamard(wires="a")

                # U depends on the input parameters
                qml.QubitUnitary(U, wires=["a", "b"])

                qml.CNOT(wires=["b", "a"])
                return qml.expval(qml.PauliX(wires="a"))
    """

    for op in tape.operations + tape.measurements:
        if isinstance(op, qml.QubitUnitary):
            # Single-qubit unitary operations
            if qml.math.shape(op.parameters[0]) == (2, 2):
                zyz_decomposition(op.parameters[0], op.wires[0])
            # Two-qubit unitary operations
            elif qml.math.shape(op.parameters[0]) == (4, 4):
                two_qubit_decomposition(op.parameters[0], op.wires)
            else:
                qml.apply(op)
        else:
            qml.apply(op)
