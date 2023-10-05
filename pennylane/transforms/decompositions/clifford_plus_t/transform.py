import pennylane as qml
from pennylane import math
from pennylane.transforms.core import transform

from itertools import product

def check_clifford_op(op):
    """ Check if an operator is Clifford or not. 

    For a given unitary operator :math:`U` acting on :math:`N` qubits, this method checks that the
    transformation :math:`UPU^{\dagger}` maps the Pauli tensor products :math:`P = {I, X, Y, Z}^{\otimes N}`
    to Pauli tensor products with O(N * 8^N) time complexity when using naive matrix multiplication.

    Args:
        op: the operator that needs to be tested

    Returns:
        Bool that represents whether the provided operator is Clifford or not.
    """

    num_qubits = len(op.wires)
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


def _check_t_op(op):
    """Check whether the gate is a T gate or not"""
    return isinstance(op, qml.T) or isinstance(getattr(op, "base", None), qml.T)


def _rot_decompose(op):
    """Decompose a rotation operation using combination of RZ, S and Hadamard"""
    d_ops = []
    if isinstance(op, qml.Rot):
        (phi, theta, omega), wires = op.data, op.wires
        d_ops.extend([qml.RZ(phi, wires), qml.S(wires), qml.Hadamard(wires), qml.RZ(theta, wires), qml.Hadamard(wires), qml.adjoint(qml.S(wires)), qml.RZ(omega, wires)])
    elif isinstance(op, qml.RX):
        (theta, ), wires = op.data, op.wires
        d_ops.extend([qml.Hadamard(wires), qml.RZ(theta, wires), qml.Hadamard(wires)])
    elif isinstance(op, qml.RY):
        (theta, ), wires = op.data, op.wires
        d_ops.extend([qml.S(wires), qml.Hadamard(wires), qml.RZ(theta, wires), qml.Hadamard(wires), qml.adjoint(qml.S(wires))])
    else:
        d_ops.append(op)
    return d_ops


def _one_qubit_decompose(op):
    """Decomposition for single qubit operations using combination of RZ and Hadamard"""

    sd_ops = qml.transforms.one_qubit_decomposition(qml.matrix(op), op.wires, "ZXZ", return_global_phase=True)

    d_ops = []
    for sd_op in sd_ops[:-1]:
        d_ops.extend(_rot_decompose(sd_op))

    return d_ops[:-1], d_ops[-1]


def _two_qubit_decompose(op):
    """Decomposition for two qubit operations using combination of RZ, Hadamard, S and CNOT"""

    td_ops = qml.transforms.two_qubit_decomposition(qml.matrix(op), op.wires)

    d_ops = []
    for td_op in td_ops:
        d_ops.extend(_rot_decompose(td_op))

    return d_ops


@transform
def clifford_t_decomposition(tape: qml.tape.QuantumScript, epsilon=1e-8) -> (Sequence[qml.tape.QuantumTape], callable):

    decomp_ops, gphase_ops = [], []
    for op in tape.operations:
        
        # Check whether operation is to be skipped
        if any([isinstance(op, skip_op) for skip_op in [qml.Barrier, qml.Snapshot, qml.WireCut]]):
            decomp_ops.append(op)
        
        # Check whether the operation is Clifford or T agte
        elif check_clifford_op(op) or _check_t_op(op):
            decomp_ops.append(op)
        
        # Decompose and go deeper 
        else:
            if isinstance(op, qml.Operation):

                # Do an ZXZ decomposition and then use RX = H @ RZ @ H
                if len(op.wires) == 1:
                    d_ops, g_ops = _one_qubit_decompose(op)
                    decomp_ops.extend(d_ops)
                    gphase_ops.append(g_ops)

                # Do an SU4 decomposition and then use RY = S @ H @ RZ @ H @ S.dag() and RX = H @ RZ @ H
                elif len(op.wires) == 2:
                    d_ops = _two_qubit_decompose(op)
                    decomp_ops.extend(d_ops)

                else:
                    try:
                        # Attempt decomposing the operation
                        md_ops = op.decomposition()

                        idx = 0 # this might not be fast but at least is not recursive
                        while idx < len(md_ops):
                            md_op = md_ops[idx]
                            if md_op.wires > 2:
                                md_ops[idx:idx] = md_op.decomposition()
                            elif md_op.wires == 2 and not isinstance(md_op, qml.CNOT):
                                md_ops[idx:idx] = _two_qubit_decompose(op)
                            elif md_op.wires == 1 and not (check_clifford_op(op) or _check_t_op(op)):
                                d_ops, g_ops = _one_qubit_decompose(op)
                                md_ops[idx:idx] = d_ops
                                gphase_ops.append(g_ops)
                        decomp_ops.extend(md_ops)

                    except:
                        raise ValueError(f"Cannot unroll {op} into the Clifford+T basis as no rule exists for its decomposition")

    # Squeeze global phases into a single global phase
    # g_phase = reduce(lambda x, y: x@y, [gphase_op])

    new_ops, error = [], 0
    for op in decomp_ops:
        if isinstance(op, qml.RZ):
            clifford_ops, err = rz_to_clifford_plus_t(epsilon, op.data[0])
            new_ops.extend(clifford_ops)
            error += err
        else:
            new_ops.append(op)

    qs = qml.tape.QuantumScript(new_ops, tape.measurements, shots=tape.shots)
    qs._qfunc_output = tape._qfunc_output
    return [qs], lambda x: x
