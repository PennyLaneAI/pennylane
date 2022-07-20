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
Contains the sign (and xi) decomposition tape transform
"""
# pylint: disable=protected-access
import pennylane as qml
import numpy as np
import json

# with open('./verisign_opt_data.json', 'r') as f:
#     data = json.load(f)

def PauliRot(theta, wires, pauli_word, ancillas):
    r"""Representation of the operator as a product of other operators (static method). :

    .. math:: O = O_1 O_2 \dots O_n.


    .. seealso:: :meth:`~.PauliRot.decomposition`.

    Args:
        theta (float): rotation angle :math:`\theta`
        pauli_word (string): the Pauli word defining the rotation
        wires (Iterable, Wires): the wires the operation acts on

    Returns:
        list[Operator]: decomposition into lower level operations

    ,**Example:**

    >>> qml.PauliRot.compute_decomposition(1.2, "XY", wires=(0,1))
    [Hadamard(wires=[0]),
    RX(1.5707963267948966, wires=[1]),
    MultiRZ(1.2, wires=[0, 1]),
    Hadamard(wires=[0]),
    RX(-1.5707963267948966, wires=[1])]

    """
    if isinstance(wires, int):  # Catch cases when the wire is passed as a single int.
        wires = [wires]

    # Check for identity and do nothing
    if set(pauli_word) == {"I"}:
        return []

    active_wires, active_gates = zip(
        ,*[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
    )

    ops = []
    for wire, gate in zip(active_wires, active_gates):
        if gate == "X":
            ops.append(qml.Hadamard(wires=[wire]))
        elif gate == "Y":
            ops.append(qml.RX(np.pi / 2, wires=[wire]))

    qml.CNOT(wires=[ancillas[1],wires[0]])
    ops.append(MultiCRZ(theta, wires=list(active_wires),control=ancillas[0]))
    qml.CNOT(wires=[ancillas[1],wires[0]])

    for wire, gate in zip(active_wires, active_gates):
        if gate == "X":
            ops.append(qml.Hadamard(wires=[wire]))
        elif gate == "Y":
            ops.append(qml.RX(-np.pi / 2, wires=[wire]))
    return ops

def MultiCRZ(theta, wires, control, **kwargs):
    ops = [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[~0:0:-1], wires[~1::-1])]
    ops.append(qml.CRZ(theta, wires=[control,wires[0]]))
    ops += [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[1:], wires[:~0])]

    return ops

def evolve_under(ops,coeffs,time):
    for op,coeff in zip(ops,coeffs):
            PauliRot(coeff*time,  wires=op.wires , pauli_word=qml.grouping.pauli_word_to_string(op), ancillas = ['Hadamard','Target'])



def sign_expand(tape, group=True, circuit = False, J=10, delta=0.0):
    r"""
    Splits a tape measuring a (fast-forwardable) Hamiltonian expectation into mutliple tapes of the Xi or sgn decomposition,
    and provides a function to recombine the results.

    Implementation of arxiv:????

    Args:
        tape (.QuantumTape): the tape used when calculating the expectation value
            of the Hamiltonian
        circuit (bool): Toggle the calculation of the analytical Xi decomposition or if True constructs the circuits of the approximate sign decomposition
            to measure the expectation value
        J (int): The times the time evolution of the hamiltonian is repeated in the quantum signal processing approximation of the sgn-decomposition
        delta (float): The minimal
    Returns:
        tuple[list[.QuantumTape], function]: Returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions to compute the expectation value.

    ,**Example**


    """

    with open('./verisign_opt_data.json', 'r') as f:
        data = json.load(f)
    phis = list(filter(lambda data: data['delta'] == delta and data['order']== J, data))[0]['opt_params']


    hamiltonian = tape.measurements[0].obs

    hamiltonian.compute_grouping()
    if len(hamiltonian_grouping_indices) != 1:
        raise ValueError(
            "Passed hamiltonian must be jointly measurable"

    wires = hamiltonian.wires
    mat = qml.utils.sparse_hamiltonian(hamiltonian).toarray()
    size = len(mat)
    eigs, eigvecs = np.linalg.eigh(mat)
    norm = eigs[-1]

    offset = 0.0
    dEs, mus, projs, times = [],[],[],[]

    proj = np.identity(size,dtype='complex64')
    proj += -2*np.outer(np.conjugate(eigvecs[:,0]),eigvecs[:,0])
    last_i = 1
    offset_add = 0
    offset_add += (eigs[-1]+eigs[0])/2

    for index in range(len(eigs)-1):
        dE = (eigs[index+1]-eigs[index])/2
        if np.isclose(dE,0):
        continue
        dEs.append(dE)
        mu = (eigs[index+1]+eigs[index])/2
        time = np.pi/(2*(norm+abs(mu)))
        times.append(time)
        mus.append(mu)

        for j in range(last_i,index+1):
        proj += -2*np.outer(np.conjugate(eigvecs[:,j]),eigvecs[:,j])
        last_i = index+1

        projs.append(proj.copy()*dE)


    if (
        not isinstance(hamiltonian, qml.Hamiltonian)
        or len(tape.measurements) > 1
        or tape.measurements[0].return_type not in  [qml.measurements.Expectation, qml.measurements.Variance]
    ):
        raise ValueError(
            "Passed tape must end in `qml.expval(H)` or 'qml.var(H)', where H is of type `qml.Hamiltonian`"
        )

    if circuit:
        coeffs = hamiltonian.data
        tapes = []
        for mu,time in zip(mus,times):
            with tape.__class__() as new_tape:
                #Put state prep and ansatz on tape in the 'old' register
                for op in tape.operations:
                    op.queue()

                #Put QSP and Hadamard test on the two ancillas Target and Control
                qml.Hadamard('Hadamard')
                for i,phi in enumerate(phis):
                    qml.CRX(phi,wires=['Hadamard','Target'])
                    if i == len(phis)-1:
                        qml.CRY(np.pi,wires=['Hadamard','Target'])
                    else:
                        evolve_under(hamiltonian.ops,coeffs,2*time)
                        qml.CRZ(-2*mu*time,wires=['Hadamard','Target'])
                qml.Hadamard('Hadamard')

                if tape.measurements[0].return_type == qml.measurements.Expectation:
                qml.expval(-1*qml.PauliZ('Hadamard'))
                else:
                qml.var(qml.PauliZ('Hadamard'))

            tapes.append(new_tape)

        # pylint: disable=function-redefined
        def processing_fn(res):
            # dot_products = [qml.math.dot(qml.math.squeeze(r), c) for c, r in zip(dEs, res)]
            products = [a * b for a, b in zip(res, dEs)]
            # return qml.math.sum(qml.math.stack(dot_products), axis=0)
            return sum(products)
        if tape.measurements[0].return_type == qml.measurements.Expectation:
            def processing_fn(res):
            products = [a * b for a, b in zip(res, dEs)]
            return sum(products)
        else:
            def processing_fn(res):
            products = [a * b for a, b in zip(res, dEs)]
            return sum(products)*len(products)


        return tapes, processing_fn

    coeffs = hamiltonian.data

    # make one tape per observable
    tapes = []
    for proj in projs:
        with tape.__class__() as new_tape:
            for op in tape.operations:
                op.queue()
            if tape.measurements[0].return_type == qml.measurements.Expectation:
                qml.expval(qml.Hermitian(proj, wires=wires))
            else:
                qml.var(qml.Hermitian(proj, wires=wires))

        tapes.append(new_tape)

    if tape.measurements[0].return_type == qml.measurements.Expectation:
        def processing_fn(res):
        return sum(res)
    else:
        def processing_fn(res):
        return sum(res)*len(res)


    return tapes, processing_fn

