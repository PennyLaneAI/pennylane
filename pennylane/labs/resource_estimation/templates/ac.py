import numpy as np
import pennylane as qml
from openfermion.ops import QubitOperator
from openfermion.utils import anticommutator
from typing import List, Tuple
import openfermion as of
from openfermion import InteractionOperator

def make_unitary(H_alpha: QubitOperator) -> Tuple[QubitOperator, float]:
    # Calculate the norm v_alpha = sqrt(sum of squares of coefficients)
    norm = np.sqrt(sum(coeff**2 for coeff in H_alpha.terms.values()))
    # Create unitary U_alpha by dividing H_alpha by its norm
    if norm > 0:
        U_alpha = H_alpha / norm
    else:
        U_alpha = QubitOperator()
    return U_alpha, norm

def decompose_hamiltonian(hamiltonian: QubitOperator) -> List[QubitOperator]:
    # Step 1: Remove the constant term
    h = QubitOperator()
    for term, coeff in hamiltonian.terms.items():
        if term != ():  # Ignore the constant term
            h += QubitOperator(term, coeff)

    # Step 2: Sort Pauli terms by absolute value of coefficients
    sorted_terms = sorted(h.terms.items(), key=lambda x: abs(x[1]), reverse=True)
    # print(sorted_terms)
    # Step 3 & 4: Decompose the Hamiltonian into commuting sub-Hamiltonians
    # result = [] # for step insertion, not required for unitaries

    unitary_list = []  # To store the unitaries (part of further LCU steps)
    norms = []  # To store the norms v_alpha

    while sorted_terms:
        h_alpha = QubitOperator()  # This will hold the commuting set
        remaining_terms = []

        for term, coeff in sorted_terms:
            current_term = QubitOperator(term, coeff)
            if all((not bool(anticommutator(current_term, QubitOperator(existing_term, existing_coeff)).terms))
                   for existing_term, existing_coeff in h_alpha.terms.items()):
                h_alpha += current_term
                # print(h_alpha)
            else:
                remaining_terms.append((term, coeff))

        # result.append(h_alpha)
        sorted_terms = remaining_terms  # Update the list for the next iteration

        U_alpha, v_alpha = make_unitary(h_alpha)
        unitary_list.append(U_alpha)
        norms.append(v_alpha)

    # rebuilt_hamiltonian = QubitOperator()
    # for U_alpha, v_alpha in zip(unitary_list, norms):
    #     rebuilt_hamiltonian += v_alpha * U_alpha

    return unitary_list, norms


def create_hamiltonian_from_tensors(one_body_tensor, two_body_tensor):
    """
    Generates a second-quantized Hamiltonian (InteractionOperator) from
    one-body and two-body tensors in chemists' notation.

    Args:
        one_body_tensor (np.ndarray): A 2D NumPy array representing the
                                      one-body integrals (T_pq).
        two_body_tensor (np.ndarray): A 4D NumPy array representing the
                                      two-body integrals (V_pqrs) in
                                      chemists' notation (V_pqrs = <pq|rs>).

    Returns:
        openfermion.InteractionOperator: The second-quantized Hamiltonian.
    """
    # Ensure the tensors have the correct dimensions
    if one_body_tensor.ndim != 2:
        raise ValueError("one_body_tensor must be a 2D array.")
    if two_body_tensor.ndim != 4:
        raise ValueError("two_body_tensor must be a 4D array.")

    n_orbitals = one_body_tensor.shape[0]
    if not (one_body_tensor.shape[1] == n_orbitals and
            two_body_tensor.shape[0] == n_orbitals and
            two_body_tensor.shape[1] == n_orbitals and
            two_body_tensor.shape[2] == n_orbitals and
            two_body_tensor.shape[3] == n_orbitals):
        raise ValueError("Tensor dimensions must be consistent and square.")

    # Create the InteractionOperator
    # Note: InteractionOperator expects the two-body integrals in the form
    # V_{pqrs} a^\dagger_p a^\dagger_q a_r a_s, which corresponds to the
    # chemists' notation <pq|rs>.
    hamiltonian = InteractionOperator(
        constant=0.0,  # You can add a constant if needed
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor
    )
    return hamiltonian

def calculate_1_norm_directly(coeffs):
    return sum(abs(coefficient) for coefficient in coeffs)

def calculate_hamiltonian_len(hamiltonian):
    return len(list(hamiltonian.terms.values()))

def norms_ac_groups(one_body_integrals, two_body_integrals):
    #convert from chemist ordering to normal ordering
    #two_mo = np.swapaxes(two_mo, 1, 3) this is another way
    normal_one_body = one_body_integrals + 0.5 * qml.math.einsum("pqss", two_body_integrals)
    normal_two_body = 2 * qml.math.swapaxes(two_body_integrals, 1, 3)  # V_pqrs
    #normal_two_body = 2 * np.swapaxes(two_body_integrals, 1, 3) this is another way

    hamiltonian = create_hamiltonian_from_tensors(one_body_tensor=normal_one_body, two_body_tensor=normal_two_body)
    pauli_hamiltonian = of.transforms.jordan_wigner(hamiltonian)
    num_paulis = calculate_hamiltonian_len(pauli_hamiltonian)
    unitaries , norms = decompose_hamiltonian(pauli_hamiltonian)

    return norms, num_paulis