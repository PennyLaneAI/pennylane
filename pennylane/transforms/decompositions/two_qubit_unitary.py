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
"""Contains transforms and helpers functions for decomposing arbitrary two-qubit
unitary operations into elementary gates.
"""
import warnings
import pennylane as qml


def two_qubit_decomposition(U, wires):
    r"""Decompose a two-qubit unitary :math:`U` in terms of elementary operations.

    It is known that an arbitrary two-qubit operation can be implemented using a
    maximum of 3 CNOTs. This transform first determines the required number of
    CNOTs, then decomposes the operator into a circuit with a fixed form.  These
    decompositions are based a number of works by Shende, Markov, and Bullock
    `(1) <https://arxiv.org/abs/quant-ph/0308033>`__, `(2)
    <https://arxiv.org/abs/quant-ph/0308045v3>`__, `(3)
    <https://web.eecs.umich.edu/~imarkov/pubs/conf/spie04-2qubits.pdf>`__,
    though we note that many alternative decompositions are possible.

    For the 3-CNOT case, we recover the following circuit, which is Figure 2 in
    reference (1) above:

    .. figure:: ../../_static/two_qubit_decomposition_3_cnots.svg
        :align: center
        :width: 70%
        :target: javascript:void(0);

    where :math:`A, B, C, D` are :math:`SU(2)` operations, and the rotation angles are
    computed based on features of the input unitary :math:`U`.

    For the 2-CNOT case, the decomposition is

    .. figure:: ../../_static/two_qubit_decomposition_2_cnots.svg
        :align: center
        :width: 50%
        :target: javascript:void(0);

    For 1 CNOT, we have a CNOT surrounded by one :math:`SU(2)` per wire on each
    side.  The special case of no CNOTs simply returns a tensor product of two
    :math:`SU(2)` operations.

    This decomposition can be applied automatically to all two-qubit
    :class:`~.QubitUnitary` operations using the
    :func:`~pennylane.transforms.unitary_to_rot` transform.

    .. warning::

        This decomposition will not be differentiable in the ``unitary_to_rot``
        transform if the matrix being decomposed depends on parameters with
        respect to which we would like to take the gradient.  See the
        documentation of :func:`~pennylane.transforms.unitary_to_rot` for
        explicit examples of the differentiable and non-differentiable cases.

    Args:
        U (tensor): A :math:`4 \times 4` unitary matrix.
        wires (Union[Wires, Sequence[int] or int]): The wires on which to apply the operation.

    Returns:
        list[Operation]: A list of operations that represent the decomposition
        of the matrix U.

    **Example**

    Suppose we create a random element of :math:`U(4)`, and would like to decompose it
    into elementary gates in our circuit.

    >>> from scipy.stats import unitary_group
    >>> U = unitary_group.rvs(4)
    >>> U
    array([[-0.29113625+0.56393527j,  0.39546712-0.14193837j,
             0.04637428+0.01311566j, -0.62006741+0.18403743j],
           [-0.45479211+0.25978444j, -0.52737418-0.5549423j ,
            -0.23429057+0.10728103j,  0.16061807-0.21769762j],
           [-0.4501231 +0.04065613j, -0.25558662+0.38209554j,
            -0.04143479-0.56598134j,  0.12983673+0.49548507j],
           [ 0.23899902+0.24800931j,  0.03374589-0.15784319j,
             0.24898226-0.73975147j,  0.0269508 -0.49534518j]])

    We can compute its decompositon like so:

    >>> decomp = qml.ops.two_qubit_decomposition(np.array(U), wires=[0, 1])
    >>> decomp
    [Rot(tensor(-1.69488788, requires_grad=True), tensor(1.06701916, requires_grad=True), tensor(0.41190893, requires_grad=True), wires=[0]),
     Rot(tensor(1.57705621, requires_grad=True), tensor(2.42621204, requires_grad=True), tensor(2.57842249, requires_grad=True), wires=[1]),
     CNOT(wires=[1, 0]),
     RZ(0.4503059654281863, wires=[0]),
     RY(-0.8872497960867665, wires=[1]),
     CNOT(wires=[0, 1]),
     RY(-1.6472464849278514, wires=[1]),
     CNOT(wires=[1, 0]),
     Rot(tensor(2.93239686, requires_grad=True), tensor(1.8725019, requires_grad=True), tensor(0.0418203, requires_grad=True), wires=[1]),
     Rot(tensor(-3.78673588, requires_grad=True), tensor(2.03936812, requires_grad=True), tensor(-2.46956972, requires_grad=True), wires=[0])]

    """
    warnings.warn(
        "`qml.transforms.two_qubit_decomposition` is deprecated. Instead, you should "
        "use `qml.ops.two_qubit_decomposition` ",
        qml.PennyLaneDeprecationWarning,
    )
    return qml.ops.two_qubit_decomposition(U, wires)
