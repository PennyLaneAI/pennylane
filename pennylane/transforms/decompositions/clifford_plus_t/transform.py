import pennylane as qml
from pennylane import math
from pennylane.transforms.core import transform

from itertools import product

def check_clifford_op(op, num_qubits):
    """ Check if an operator is Clifford or not Clifford. """

    pauli_terms = qml.pauli_decompose(qml.matrix(op), check_hermitian=False)
    pauli_group = lambda x: [qml.Identity(x), qml.PauliX(x), qml.PauliY(x), qml.PauliZ(x)]

    pauli_coves = []
    try:
        pauli_qubit = [qml.ops.op_math.prod(*pauli) for pauli in product(*(
            pauli_group(idx) for idx in range(num_qubits)))]
    except: 
        pauli_qubit = [qml.Identity(0)]
    pauli_qubit = [qml.pauli.pauli_sentence(op).hamiltonian(wire_order=range(
        num_qubits)) for op in pauli_qubit]

    for idx, prod in enumerate(product([pauli_terms], pauli_qubit, [pauli_terms])):
        upu = qml.pauli.pauli_sentence(qml.ops.op_math.prod(*prod))
        upu.simplify()
        upu2 = upu.hamiltonian(wire_order=range(num_qubits))
        if len(upu2.ops) == 1:
            if not isinstance(upu2.ops[0], qml.Identity):
                pauli_coves.append(any([
                    qml.equal(upu2.ops[0], tm) for tm in pauli_qubit
                ]))
        else:
            pauli_coves.append(False)

    return all(pauli_coves)

def check_clifford_op_2():
    for op in tape:
        if isinstance(op, qml.QubitUnitary):
            # Single-qubit unitary operations
            if qml.math.shape(op.parameters[0]) == (2, 2):
                one_qubit_decomposition(op.parameters[0], op.wires[0])
            # Two-qubit unitary operations
            elif qml.math.shape(op.parameters[0]) == (4, 4):
                two_qubit_decomposition(op.parameters[0], op.wires)
            else:
                qml.apply(op)
        else:
            qml.apply(op)

def _compute_decomposition(op):
    if check_clifford_op(op, len(op.wires)) or isinstance(op, qml.T) or isinstance(getattr(op, "base", None), qml.T):
        return  op
    else:
        decomp = 
    try:
        decop.decomposition()
    except:

    if check_clifford_op(op, len(op.wires)) or isinstance(op, qml.T) or isinstance(getattr(op, "base", None), qml.T):
        return op


@transform
def clifford_plus_t(tape: qml.tape.QuantumScript, epsilon=1e-8) -> (Sequence[qml.tape.QuantumTape], callable):
    new_ops = []
    for op in tape.operations:
        if check_clifford_op(op, len(op.wires)) or isinstance(op, qml.T) or isinstance(getattr(op, "base", None), qml.T):
            new_ops.append(op)
        else:
            if isinstance(op, qml.Operation):
                try:
                    decomp_ops = op.decomposition()
                except:
                    decomp_ops = [qml.QubitUnitary(qml.matrix(op), wires=op.wires)]


            for decomp_op in decomp_ops:
                if check_clifford_op(decomp_op, len(op.wires)) or isinstance(op, qml.T) or isinstance(getattr(op, "base", None), qml.T):
                    new_ops.append(op)
                elif len(op.wires) == 2:
                    d_ops = one_qubit_decomposition(qml.matrix(decomp_op), decomp_op.wires, "ZXZ", return_global_phase=True)
                elif len(op.wires) == 4:
                    d_ops = two_qubit_decomposition(qml.matrix(decomp_op), decomp_op.wires)
                else:
                    raise ValueError("Operation not supported :(")
                
                for d_op in d_ops:
                    if check_clifford_op(decomp_op, len(op.wires)) or isinstance(op, qml.T) or isinstance(getattr(op, "base", None), qml.T):
                        new_ops.append(op)
                    elif isinstance(d_op, qml.RZ):
                        clifford_ops, err = rz_to_clifford_plus_t(epsilon, op.data[0])
                        new_ops.extend(clifford_ops)
                    elif isinstance(d_op, qml.RX):
                        clifford_ops, err = rz_to_clifford_plus_t(epsilon, op.data[0])
                        new_ops.extend([qml.Hadamard(wires=d_op.wires), *clifford_ops, qml.Hadamard(wires=d_op.wires)])
                    elif isinstance(d_op, qml.RY):
                        clifford_ops, err = rz_to_clifford_plus_t(epsilon, op.data[0])
                        new_ops.extend([qml.S(wires=d_op.wires), qml.Hadamard(wires=d_op.wires), *clifford_ops, 
                        qml.Hadamard(wires=d_op.wires), qml.adjoint(qml.S(wires=d_op.wires))])
                    else:
                        clifford_ops, err = 

                if check_clifford_op(op, len(op.wires)) or isinstance(op, qml.T):
                    new_ops.append(op)
                elif isinstance(op, qml.QubitUnitary): 

                if isinstance(op, qml.QubitUnitary):
                    if qml.math.shape(op.parameters[0]) == (2, 2):
                        decomp_ops = one_qubit_decomposition(op, 0, "ZXZ", return_global_phase=True)
                    elif qml.math.shape(op.parameters[0]) == (4, 4):
                        decomp_ops = two_qubit_decomposition(op.parameters[0], op.wires)
                elif check_clifford_op(op, len(op.wires)) or isinstance(op, qml.T):
                    new_ops.append(op)                

            


            if len(op.wires)
            if isinstance(op, qml.QubitUnitary):
                # Single-qubit unitary operations
                if qml.math.shape(op.parameters[0]) == (2, 2):
                    one_qubit_decomposition(op.parameters[0], op.wires[0])
                # Two-qubit unitary operations
                elif qml.math.shape(op.parameters[0]) == (4, 4):
                    two_qubit_decomposition(op.parameters[0], op.wires)
                else:
                    qml.apply(op)


    tape = convert_to_single_qubit_and_cnot(tape)
    new_ops = []
    total_error = 0
    for op in tape.operations:
        if op in clifford_basis:
            new_ops.append(op)
            continue
        for op_ in convert_to_rz_and_hadamard(op):
            if isinstance(op_, qml.Hadamard):
                new_ops.append(op_)
                continue
            clifford_ops, err = rz_to_clifford_plus_t(epsilon, op.data[0])
            new_ops.extend(clifford_ops)
            total_error += err
    qs = qml.tape.QuantumScript(new_ops, tape.measurements, shots=tape.shots)
    qs._qfunc_output = tape._qfunc_output
    return [qs], lambda x: x