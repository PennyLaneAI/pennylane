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
Contains the sign (and xi) decomposition tape transform, implementation of ideas from arXiv:2207.09479
"""
# pylint: disable=protected-access
import json
from os import path
import pennylane as qml
from .batch_transform import batch_transform
from pennylane import numpy as np

# TODO: This part up here (ControlledPauliEvolution and MultiCRZ) probably should live somewhere else, not sure if properly implementing these as gates inside pennylane is worthwhile


def ControlledPauliEvolution(theta, wires, pauli_word, controls):
    r"""Controlled Evolution under generic pauli words, adapted from the decomposition of
    qml.PauliRot to suit our needs


    Args:
        theta (float): rotation angle :math:`\theta`
        pauli_word (string): the Pauli word defining the rotation
        wires (Iterable, Wires): the wires the operation acts on
        controls (List[control1, control2]): The two additional controls to implement the
          Hadamard test and the quantum signal processing part on

    Returns:
        list[Operator]: decomposition that make up the controlled evolution
    """
    active_wires, active_gates = zip(
        *[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
    )

    ops = []
    for wire, gate in zip(active_wires, active_gates):
        if gate == "X":
            ops.append(qml.Hadamard(wires=[wire]))
        elif gate == "Y":
            ops.append(qml.RX(np.pi / 2, wires=[wire]))

    ops.append(qml.CNOT(wires=[controls[1], wires[0]]))
    ops.extend(MultiCRZ(theta, wires=list(active_wires), control=controls[0]))
    ops.append(qml.CNOT(wires=[controls[1], wires[0]]))

    for wire, gate in zip(active_wires, active_gates):
        if gate == "X":
            ops.append(qml.Hadamard(wires=[wire]))
        elif gate == "Y":
            ops.append(qml.RX(-np.pi / 2, wires=[wire]))
    return ops


def MultiCRZ(theta, wires, control):
    """
    Implements a controlled decomposition of qml.MultiRZ

    Args:
        theta (float): rotation angle :math:`\theta`
        wires (Iterable, Wires): the wires the operation acts on
        control (Wire): The additional ancilla to control the RZ rotation on

    Returns:
        list[Operator]: decomposition that make up the controlled evolution
    """
    ops = [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[~0:0:-1], wires[~1::-1])]
    ops.append(qml.CRZ(theta, wires=[control, wires[0]]))
    ops += [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[1:], wires[:~0])]

    return ops


def evolve_under(ops, coeffs, time, controls):
    """
    Evolves under the given Hamiltonian deconstructed into its pauliwords

    Args:
        ops (List[Observables]): List of Pauliwords that comprise the Hamiltonian
        coeffs (List[int]): List of the respective coefficients of the Pauliwords of the Hamiltonian
        time (float): At what time to evaluate these Pauliwords
    """
    ops_temp = []
    for op, coeff in zip(ops, coeffs):
        pauli_word = qml.grouping.pauli_word_to_string(op)
        ops_temp.append(ControlledPauliEvolution(
            coeff * time,
            wires=op.wires,
            pauli_word=pauli_word,
            controls=controls,
        ))
    return ops_temp

def calculate_Xi_decomposition(hamiltonian):
    """
    Calculates the Xi-decomposition from the given hamiltonian by constructing the sparse matrix
    representing the hamiltonian, finding its spectrum and then construct projectors and
    eigenvalue spacings

    Args:
      hamiltonian (qml.Hamiltonian): The pennylane hamiltonian to be decomposed

    Returns:
      dEs (List[float]): The step separating the two eigenvalues (E_1-E-2)/2 of the spectrum
      mus (List[float]): The average between the two eigenvalues (E_1+E-2)/2
      times (List[float]): The time for this term group to be evaluated/evolved at
      projs (List[np.array]): The analytical observables associated with these groups,
       to be measured by qml.Hermitian
    """
    mat = qml.utils.sparse_hamiltonian(hamiltonian).toarray()
    size = len(mat)
    eigs, eigvecs = np.linalg.eigh(mat)
    norm = eigs[-1]
    proj = np.identity(size, dtype="complex64")

    def Pi(j):
        """Projector on eigenspace of eigenvalue E_i"""
        return np.outer(np.conjugate(eigvecs[:,j]), eigvecs[:,j] )

    proj += -2 * Pi(0)
    last_i = 1

    dEs, mus, projs, times = [], [], [], []

    for index in range(len(eigs) - 1):
        dE = (eigs[index + 1] - eigs[index]) / 2
        if np.isclose(dE, 0):
            continue
        dEs.append(dE)
        mu = (eigs[index + 1] + eigs[index]) / 2
        mus.append(mu)
        time = np.pi / (2 * (norm + abs(mu)))
        times.append(time)

        for j in range(last_i, index + 1):
            proj += -2 * Pi(j)
            last_i = index + 1

        projs.append(proj.copy() * dE)

    return dEs, mus, times, projs


# pylint: disable=too-many-function-args)
def construct_sgn_circuit(hamiltonian, tape, mus, times, phis, controls):
    """
    Takes a tape with state prep and ansatz and constructs the individual tapes
    approximating/estimating the individual terms of your decomposition

    Args:
      hamiltonian (qml.Hamiltonian): The pennylane hamiltonian to be decomposed
      tape (qml.QuantumTape: Tape containing the circuit to be expanded into the new circuits
      mus (List[float]): The average between the two eigenvalues (E_1+E-2)/2
      times (List[float]): The time for this term group to be evaluated/evolved at
      phis (List[float]): Optimal phi values for the QSP part associated with the respective
        delta and J
      controls (List[control1, control2]): The two additional controls to implement the
          Hadamard test and the quantum signal processing part on

    Returns:
      tapes (List[qml.tape]): Expanded tapes from the original tape that measures the terms
        via the approximate sgn decomposition
    """
    coeffs = hamiltonian.data
    tapes = []
    for mu, time in zip(mus, times):
        with tape.__class__() as new_tape:
            # Put state prep and ansatz on tape in the 'old' register
            for op in tape.operations:
                op.queue()

            # Put QSP and Hadamard test on the two ancillas Target and Control
            qml.Hadamard(controls[0])
            for i, phi in enumerate(phis):
                qml.CRX(phi, wires=controls)
                if i == len(phis) - 1:
                    qml.CRY(np.pi, wires=controls)
                else:
                    evolve_under(hamiltonian.ops, coeffs, 2 * time, controls)
                    qml.CRZ(-2 * mu * time, wires=controls)
            qml.Hadamard(controls[0])

            if tape.measurements[0].return_type == qml.measurements.Expectation:
                qml.expval(-1 * qml.PauliZ(controls[0]))
            else:
                qml.var(qml.PauliZ(controls[0]))

        tapes.append(new_tape)
    return tapes


@batch_transform
def sign_expand(tape, circuit=False, J=10, delta=0.0, controls=['Hadamard','Target']):
    r"""
    Splits a tape measuring a (fast-forwardable) Hamiltonian expectation into mutliple tapes of
    the Xi or sgn decomposition, and provides a function to recombine the results.

    Implementation of ideas from arXiv:2207.09479

    For the calculation of variances, one assumes an even distribution of shots among the groups
    as an adaptive shot allocation onto tapes is not implemented in Pennylane yet.

    Args:
        tape (.QuantumTape): the tape used when calculating the expectation value of the Hamiltonian
        circuit (bool): Toggle the calculation of the analytical Xi decomposition or if True
          constructs the circuits of the approximate sign decomposition to measure the expectation
          value
        J (int): The times the time evolution of the hamiltonian is repeated in the quantum signal
          processing approximation of the sgn-decomposition
        delta (float): The minimal
        controls (List[control1, control2]): The two additional controls to implement the
          Hadamard test and the quantum signal processing part on, have to be wires on the device

    Returns:
        tuple[list[.QuantumTape], function]: Returns a tuple containing a list of quantum tapes
          to be evaluated, and a function to be applied to these tape executions to compute the
          expectation value.

    **Example**

    Given a Hamiltonian,

    .. code-block:: python3

        H = qml.PauliZ(0) + 0.5 * qml.PauliZ(2) + qml.PauliZ(1)

    and a circuit of the form,

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            return qml.expval(H)

    We can use the ``sign_expand`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian in these new decompositions

    >>> tapes, fn = qml.transforms.sign_expand(tape)

    We can evaluate these tapes on a device, it needs two additional ancilla gates labeled 'Hadamard' and 'Target' if one wants to make the circuit approximation of the decomposition:

    >>> dev = qml.device("default.qubit", wires=[0,1,2,'Hadamard','Target'])
    >>> res = dev.batch_execute(tapes)
    >>> fn(res)

    To evaluate the circuit approximation of the decomposition one can construct the sgn-decomposition by changing the kwarg circuit to True:

    .. code-block:: python3

        tapes, fn = qml.transforms.sign_expand(tape, circuit=True, J=20, delta=0)
        dev = qml.device("default.qubit", wires=[0,1,2,'Hadamard','Target'])
        res = dev.batch_execute(tapes)


    As a last thing, as the paper is about variance minimizing one can also calculate the variance of the estimator by changing the tape

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            return qml.var(H)

        tapes, fn = qml.transforms.sign_expand(tape, circuit=True, J=20, delta=0)
        dev = qml.device("default.qubit", wires=[0,1,2,'Hadamard','Target'])
        res = dev.batch_execute(tapes)
    """
    path_str = path.dirname(__file__)
    with open(path_str + "/sign_expand_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    phis = list(filter(lambda data: data["delta"] == delta and data["order"] == J, data))[0][
        "opt_params"
    ]

    hamiltonian = tape.measurements[0].obs
    wires = hamiltonian.wires

    # TODO qml.utils.sparse_hamiltonian at the moment does not allow autograd to push gradients through
    if (
        not isinstance(hamiltonian, qml.Hamiltonian)
        or len(tape.measurements) > 1
        or tape.measurements[0].return_type
        not in [qml.measurements.Expectation, qml.measurements.Variance]
    ):
        raise ValueError(
            "Passed tape must end in `qml.expval(H)` or 'qml.var(H)', where H is of type `qml.Hamiltonian`"
        )

    hamiltonian.compute_grouping()
    if len(hamiltonian.grouping_indices) != 1:
        raise ValueError("Passed hamiltonian must be jointly measurable")

    dEs, mus, times, projs = calculate_Xi_decomposition(hamiltonian)

    if circuit:
        tapes = construct_sgn_circuit(hamiltonian, tape, mus, times, phis, controls)
        if tape.measurements[0].return_type == qml.measurements.Expectation:
            # pylint: disable=function-redefined
            def processing_fn(res):
                products = [a * b for a, b in zip(res, dEs)]
                return qml.math.sum(products)

        else:
            # pylint: disable=function-redefined
            def processing_fn(res):
                products = [a * b for a, b in zip(res, dEs)]
                return qml.math.sum(products) * len(products)

        return tapes, processing_fn

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

    # pylint: disable=function-redefined
    def processing_fn(res):
        return (
            qml.math.sum(res)
            if tape.measurements[0].return_type == qml.measurements.Expectation
            else qml.math.sum(res) * len(res)
        )

    return tapes, processing_fn
