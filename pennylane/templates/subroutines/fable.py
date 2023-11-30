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
r"""
Contains the FABLE matrix block encoding template.

Example Usage:
    from pennylane import numpy as np
    import pennylane as qml
    from copy import copy, deepcopy

    tol = 0.1  <-- choose tolerance value you'd like
    A=generate_random_matrix(n) <-- put your favourite integer here for n
    y = deepcopy(A)
    y = process_matrix(y)
    wiress = range(0,len(y) + 1)
    y = np.array(y, requires_grad = True)
    print(A)

    dev = qml.device('default.qubit', wires=wiress)
    @qml.qnode(dev)
    def circuit(A,tol):
        qml.FABLE(A,tol)
        return qml.state()

    ## Access the matrix representation of the circuit, and get the matrix embedded in the smaller
    #subspace
    M = len(y) * qml.matrix(circuit,wire_order = None)(y,tol)[0:len(y),0:len(y)]

    #Observe that M and A are the same
    print(M)

    #Demonstrating Differentiability of circuits:

    def state_scalar_grad():
        dev = qml.device('default.qubit', wires=wiress)

        @qml.qnode(dev, diff_method='backprop')
        def circuit(y,tol):
            qml.FABLE(y, tol)
            return qml.state()

        def cost_fn(y,tol):
            out = circuit(y,tol)
            return np.abs(out[0])

        print(ag.elementwise_grad(cost_fn)(y,tol))
    #
    state_scalar_grad()


"""

import pennylane as qml
from pennylane.templates.state_preparations.mottonen import gray_code
from pennylane.operation import Operation, AnyWires
from pennylane import numpy as np

#needed for the walsh hadamard transform
_walsh_hadamard_matrix = np.array([[1, 1], [1, -1]])/2
def _walsh_hadamard_transform(D, n=None):
    r"""Compute the Walsh–Hadamard Transform of a one-dimensional array.

    Args:
        D (tensor_like): The array or tensor to be transformed. Must have a length that
            is a power of two.

    Returns:
        tensor_like: The transformed tensor with the same shape as the input ``D``.

    Due to the execution of the transform as a sequence of tensor multiplications
    with shapes ``(2, 2), (2, 2,... 2)->(2, 2,... 2)``, the theoretical scaling of this
    method is the same as the one for the
    `Fast Walsh-Hadamard transform <https://en.wikipedia.org/wiki/Fast_Walsh-Hadamard_transform>`__:
    On ``n`` qubits, there are ``n`` calls to ``tensordot``, each multiplying a
    ``(2, 2)`` matrix to a ``(2,)*n`` vector, with a single axis being contracted. This means
    that there are ``n`` operations with a FLOP count of ``4 * 2**(n-1)``, where ``4`` is the cost
    of a single ``(2, 2) @ (2,)`` contraction and ``2**(n-1)`` is the number of copies due to the
    non-contracted ``n-1`` axes.
    Due to the large internal speedups of compiled matrix multiplication and compatibility
    with autodifferentiation frameworks, the approach taken here is favourable over a manual
    realization of the FWHT unless memory limitations restrict the creation of intermediate
    arrays.
    """
    orig_shape = qml.math.shape(D)
    n = n or int(qml.math.log2(orig_shape[-1]))
    # Reshape the array so that we may apply the Hadamard transform to each axis individually
    if broadcasted := len(orig_shape) > 1:
        new_shape = (orig_shape[0],) + (2,) * n
    else:
        new_shape = (2,) * n
    D = qml.math.reshape(D, new_shape)
    # Apply Hadamard transform to each axis, shifted by one for broadcasting
    for i in range(broadcasted, n + broadcasted):
        D = qml.math.tensordot(_walsh_hadamard_matrix, D, axes=[[1], [i]])
    # The axes are in reverted order after all matrix multiplications, so we need to transpose;
    # If D was broadcasted, this moves the broadcasting axis to first position as well.
    # Finally, reshape to original shape
    return qml.math.reshape(qml.math.transpose(D), orig_shape)
def find_changed_bits(gray_code_sequence: list):
    """
    Find the index of the bit where adjacent bit strings in a Gray code sequence differ.
    The comparison is circular, so it considers the last bit string with the first one.
    The indexes are in the order where the rightmost bit is the least significant.

    Parameters:
    - Args: List of Gray code bit strings.

    Returns:
    - List of indexes where adjacent bit strings differ.
    """
    num_strings = len(gray_code_sequence)
    differing_bits = []

    for i in range(num_strings):
        # Get the indices for comparison
        current_index = i
        next_index = (i + 1) % num_strings

        # XOR the neighboring bit strings
        xor_result = int(gray_code_sequence[current_index], 2) ^ int(gray_code_sequence[next_index], 2)

        # Find the index of the rightmost differing bit in the XOR result
        changed_bit_index = len(bin(xor_result)) - 3  # Subtracting 3 to account for '0b' prefix

        # Reverse the index
        changed_bit_index = len(gray_code_sequence[i]) - changed_bit_index - 1

        # Add the index to the list
        differing_bits.append(changed_bit_index)

    return differing_bits
def process_matrix(A):
    """Matrix processing for correct block encoding. calculates \alpha such that A = \alpha *A'. Extends the dimensions
     of A if not square or if log2(shape(A)) is not an integer value.

     Args:
         A: Matrix to be encoded. Can be real or complex valued.

     Returns:
         A': Processed matrix.
     """
    # Normalize matrix
    epsm = np.finfo(A.dtype).eps
    alpha = np.linalg.norm(np.ravel(A), np.inf)
    if alpha > 1:
        alpha = alpha + np.sqrt(epsm)
        A = A / alpha
    else:
        alpha = 1.0

    # Make matrix shape 2**n, where n is number of qubits in circuit
    n, m = A.shape
    if n != m:
        k = max(n, m)
        A = np.pad(A,((0, k - n), (0, k - m)))
        n = k
    logn = int(np.ceil(np.log2(n)))
    if n < 2 ** logn:
        A = np.pad(A, ((0, 2 ** logn - n), (0, 2 ** logn - n)))
        n = 2 ** logn
    return A
def compress(ry_gate_sequence: list, rz_gate_sequence: list, tol: float):
    """Given a sequence of gates for a uniformly controlled Ry(theta)/Rz(theta) decmposition, removes Ry/Rz(theta_i) if
    theta_i <= tolerance value. After rotation gates are removed, redundant CNOT gates are also removed.
    Args:
        ry_gate_sequence: list of gates for uniformly controlled Ry(theta)
        rz_gate_sequence: list of gates for uniformly controlled Rz(theta)
        tol: tolerance value for sparsification

    Returns:
        ry_gate_sequence: list of gates in processed circuit for uniformly controlled Ry(theta)
        rz_gate_sequence: list of gates in processed circuit for uniformly controlled Rz(theta)
    """

    #remove gates below tolerance value
    for i,gate in enumerate(ry_gate_sequence):
        if gate.label() == 'RY':
            if np.abs(gate.data[0]) <= tol:
                ry_gate_sequence[i] = 'None'
                if not(len(rz_gate_sequence)) == 0:
                    rz_gate_sequence[i] = 'None'

    #remove redundant CNOTs. In a CNOT gate sequence, all the targets are the same so here only the controls
    #matter. If in a CNOT sequence, if the number of CNOT gates with same control are even: remove them all.
    # If odd, just keep one for that control.
    for i, gate in enumerate(ry_gate_sequence):
        if gate == 'None' or gate.label() == 'RY':
            continue
        if gate.label() == 'X':
            c_not_indices = [i]
            control = gate.control_wires[0]
            for j in range(i + 1, len(ry_gate_sequence)):
                next_gate = ry_gate_sequence[j]
                if next_gate == 'None':
                    continue

                if next_gate.label() == 'RY':
                    break

                if next_gate.label() == 'X' and next_gate.control_wires[0] == control:
                    c_not_indices.append(j)

        if len(c_not_indices) % 2 == 0:
            for i, index in enumerate(c_not_indices):
                ry_gate_sequence[index] = 'None'
        else:
            for i, index in enumerate(c_not_indices):
                if i == 0:
                    continue
                ry_gate_sequence[index] = 'None'

    for i, gate in enumerate(rz_gate_sequence):

        if gate == 'None' or gate.label() == 'RZ':
            continue
        if gate.label() == 'X':
            c_not_indices = [i]
            control = gate.control_wires[0]
            for j in range(i + 1, len(rz_gate_sequence)):
                next_gate = rz_gate_sequence[j]
                if next_gate == 'None':
                    continue

                if next_gate.label() == 'RZ':
                    break

                if next_gate.label() == 'X' and next_gate.control_wires[0] == control:
                    c_not_indices.append(j)

        if len(c_not_indices) % 2 == 0:
            for i, index in enumerate(c_not_indices):
                rz_gate_sequence[index] = 'None'
        else:
            for i, index in enumerate(c_not_indices):
                if i == 0:
                    continue
                rz_gate_sequence[index] = 'None'

    return ry_gate_sequence, rz_gate_sequence
def _apply_uniform_rotation_dagger(gate, theta, control_wires: list, target_wire: int):
    r"""Applies a uniformly-controlled rotation to the target qubit.

    A uniformly-controlled rotation is a sequence of multi-controlled
    rotations, each of which is conditioned on the control qubits being in a different state.
    For example, a uniformly-controlled rotation with two control qubits describes a sequence of
    four multi-controlled rotations, each applying the rotation only if the control qubits
    are in states :math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`, and :math:`|11\rangle`, respectively.

    To implement a uniformly-controlled rotation using single qubit rotations and CNOT gates,
    a decomposition based on Gray codes is used.

    Args:
        gate: (.Operation): gate to be applied, needs to have exactly one parameter.
        theta: angles to decompose the uniformly-controlled rotation into multi-controlled rotations.
        control_wires: (array[int]): wires that act as control.
        target_wire (int): wire that acts as target.

    Returns:
          list[.Operator]: sequence of operators defined by this function.
    """
    op_list = []
    if len(theta) == 0:
        return op_list


    gray_code_rank = len(control_wires)

    if gray_code_rank == 0:
        if qml.math.is_abstract(theta) or qml.math.all(theta[..., 0] != 0.0):
            op_list.append(gate(theta[..., 0], wires=[target_wire]))
        return op_list

    code = gray_code(gray_code_rank)


    num_selections = len(code)

    control_indices = find_changed_bits(code)
    for i, control_index in enumerate(control_indices):
        if qml.math.is_abstract(theta) or qml.math.all(theta[..., i] != 0.0):
            op_list.append(gate(theta[..., i], wires=[target_wire]))
        op_list.append(qml.CNOT(wires=[control_wires[control_index], target_wire]))
    return op_list
def decimal_to_gray(n:int):
    """
    Function to convert an integer number to its Gray code equivalent.

    """
    n ^= (n >> 1)
    return n
def generate_gray_sequence(n: int):
    """
    Generate the Gray code sequence for a given integer n.
    Args:
        n: integer number to be converted to a gray code sequence

    Returns:
         gray_sequence: list of binary numbers in the gray code

    """
    gray_sequence = [decimal_to_gray(i) for i in range(2**n)]
    return gray_sequence
def gray_permutation(vector):
    """Permutation of basis vectors in a gray code arrangement
    Args:
        vector: array of angles to be permuted so the correct Ry(theta) are applied to the appropriate basis states.

    Returns:
        permuted_vector: the permuted vector.
    """

    n = int(np.log2(len(vector)))
    grays_code = generate_gray_sequence(n)
    permuted_vector = [vector[i] for i in grays_code]
    return np.array(permuted_vector)
def angles_and_phases(flattened_matrix):
    """
    Given a matrix flattened into row major 1D-array, returns the angles (and phases) needed for the uniformly controlled rotations,
    to encode A. Where for a real valued matrix, theta_ij = 2 * arccos(a_ij), and for a complex valued matrix
    theta_ij = 2 * arccos(abs(a_ij)) and alpha_ij = -2 * beta, where z = |a| * e^(i * beta) (polar form).

    Args:
        flattened_matrix (numpy array): Array 1d representation of A, flattened in row major order.

    Returns:
         angles (list): list of angles in the uniformly controlled Ry gates to block encode A
         phases (list): for a complex valued matrix, contains the phases for correct polar form encoding of elements
    """
    angles = []
    phases = []

    # for complex valued matrices
    if not np.isreal(flattened_matrix).all():

        for i,element in enumerate(flattened_matrix):
            angles.append(2 * np.arccos(np.absolute(element)))
            phases.append(-2 * np.angle(element))

    # real valued matrices
    else:
        for i,element in enumerate(flattened_matrix):
            angles.append(2 * np.arccos(element))

    if len(phases) == 0:
        angles = np.array(angles)
        angles = _walsh_hadamard_transform(angles)
        angles = gray_permutation(angles)
    else:
        angles =np.array(angles)
        angles = _walsh_hadamard_transform(angles)
        angles = gray_permutation(angles)
        phases = np.array(phases)
        phases = _walsh_hadamard_transform(phases)
        phases = gray_permutation(phases)

    return angles,phases
class FABLE(Operation):
    """
    Initialize the Fable class.

   Args:
    matrix (tensor_like): the matrix to be block encoded.
    tolerance (float): user defined tolerance value such that for any theta_i in the circuit construction
    for uniformly controlled rotations, if abs(\theta_i) <= tolerance, Ry/Rz(theta_i) is removed from the
    circuit.

    """
    grad_method = 'A'
    num_wires = AnyWires

    def __init__(self, matrix, tolerance, id = None):

        if not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError("tolerance must be a number greater than or equal to zero")

        self.matrix = matrix
        self.tolerance = tolerance
        wires = range(0,len(matrix) + 1)


        all_wires = wires

        super().__init__(matrix, tolerance, wires = all_wires, id=id)

    def get_matrix(self):
        """
        Get the matrix property.

        Returns:
        - The matrix property.
        """
        return self.matrix

    def get_tolerance(self):
        """
        Get the tolerance property.

        Returns:
        - The tolerance property.
        """
        return self.tolerance

    @property
    def num_params(self):
        return 2

    @staticmethod
    def compute_decomposition(matrix, tolerance, wires):
        """Representation of the operator as a product of Ry, Rz and CNOT gates. See <https://arxiv.org/pdf/2205.00081.pdf>
        for more details on the circuit construction. The only deviation is rather than using a fast walsh-hadamrd
        transform, uses a 'normal' walsh-hadamard transform instead. This is so the workflow can be
        differentiable.

        Args:
            matrix (tensor_like): the matrix to be block encoded.
            tolerance (float): user defined tolerance value such that for any theta_i in the circuit construction
            for uniformly controlled rotations, if abs(\theta_i) <= tolerance, Ry/Rz(theta_i) is removed from the
            circuit.

        Returns:
            list[.Operator]: decomposition of the operator"""

        M = process_matrix(matrix)
        vecM = M.flatten(order = 'C')
        angles, phases = angles_and_phases(vecM)

        # 1)
        # define qubit wires first. One ancilla qubit is needed as the target for all the multi-control gate, and then we also need
        # two seperate registers of n qubits. Where n is the number of qubits that the matrix A to be block encoded acts on.

        n_qubits = int(np.log2(np.shape(M)[0]))
        work_wires = [i for i in range(1, n_qubits + 1)]
        input_wires = [i for i in range(n_qubits + 1, 2 * n_qubits + 1)]
        control_and_input_index = work_wires + input_wires
        ancilla_index = 0

        #constructing the uniformly controlled gates
        ry_gates = _apply_uniform_rotation_dagger(qml.RY, angles, control_wires=control_and_input_index,
                                                  target_wire=ancilla_index)
        rz_gates = _apply_uniform_rotation_dagger(qml.RZ, phases, control_wires=control_and_input_index,
                                                  target_wire=ancilla_index)

        #compression algorithm
        ry_gates, rz_gates = compress(ry_gates, rz_gates, tolerance)

        def remove_values_from_list(the_list, val):
            return [value for value in the_list if value != val]

        ry_gates = remove_values_from_list(ry_gates,'None')
        rz_gates = remove_values_from_list(rz_gates,'None')

        #circuit construction
        hads = []
        swaps = []
        for i, wire in enumerate(work_wires):
            hads.append(qml.Hadamard(wire))

        for i, work_wire in enumerate(work_wires):
            input_wire = input_wires[i]
            swaps.append(qml.SWAP([work_wire,input_wire]))

        op_list = hads + ry_gates + rz_gates + swaps + hads
        return op_list

