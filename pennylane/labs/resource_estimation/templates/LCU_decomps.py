import pennylane as qml
from pennylane import numpy as np
from pennylane.resource import DoubleFactorization as DF
import TensorFox as tfox
import numpy as np

def sparse_matrix(one_body, two_body, tol_factor=1e-5):
    '''
    Args:
        one_body (np.array): 2-tensor with the one body intergrals in chemistry notation
        two_body (np.array): 4-tensor with the two body intergrals in chemistry notation
        cutoff (float): Cutoff used to truncate eigenvalues

    Returns:
        one_norm (float): number of unitaries
        num_unitaries (int): number of unitaries
    '''
    one_body[abs(one_body)<tol_factor]=0
    two_body[abs(two_body)<tol_factor]=0

    # Set terms to 0 to account for symmetries
    one_body_final = np.triu(one_body, k=False)
    
    # Set terms to 0 to account for symmetries
    n_i, n_j, n_k, n_l = two_body.shape
    i_coords = np.arange(n_i)[:, None, None, None]
    j_coords = np.arange(n_j)[None, :, None, None]
    k_coords = np.arange(n_k)[None, None, :, None]
    l_coords = np.arange(n_l)[None, None, None, :]
    mask = (i_coords > k_coords) | ((i_coords == k_coords) & (j_coords > l_coords))
    two_body[mask] = 0            

    # Count number of non-zero elements in one body and two body
    one_body_num_unitaries = np.count_nonzero(one_body_final)
    two_body_num_unitaries = np.count_nonzero(two_body) 
    num_unitaries = one_body_num_unitaries + two_body_num_unitaries

    adjusted_one_body_coeffs = one_body + np.einsum("llij", two_body)
    one_body_norm = np.sum(abs(adjusted_one_body_coeffs))
    two_body_norm = np.sum(abs(two_body))
    one_norm = one_body_norm + two_body_norm

    return one_norm, num_unitaries

def double_factorization(one_body, two_body, tol_factor=1e-5):
    '''
    Args:
        one_body (np.array): 2-tensor with the one body intergrals in chemistry notation
        two_body (np.array): 4-tensor with the two body intergrals in chemistry notation
        tol_factor (float): Cutoff used to truncate

    Returns:
        one_norm (float): number of unitaries
        num_unitaries (int): number of unitaries
    '''
    t_matrix = one_body - 0.5 * np.einsum("illj", two_body) + np.einsum("llij", two_body)
    t_eigvals, _ = np.linalg.eigh(t_matrix)
    lambda_one = np.sum(abs(t_eigvals))
    num_terms_one_body = t_eigvals.shape[0]

    factors, _, __ = qml.qchem.factorize(two_body, 
                                        tol_factor = tol_factor, 
                                        tol_eigval = tol_factor)

    feigvals = np.linalg.eigvalsh(factors)
    eigvals = [eigvals[np.where(np.abs(eigvals) > tol_factor)] for eigvals in feigvals]

    lambda_two = 0.25 * np.sum([np.sum(abs(v)) ** 2 for v in eigvals])

    one_norm = lambda_one + lambda_two
    number_unitaries = len(factors) + num_terms_one_body
    
    return one_norm, number_unitaries

def compressed_double_factorization(one_body, two_body, tol_factor=1e-5):
    '''
    Args:
        one_body (np.array): 2-tensor with the one body intergrals in chemistry notation
        two_body (np.array): 4-tensor with the two body intergrals in chemistry notation
        tol_factor (float): Cutoff used to truncate

    Returns:
        one_norm (float): number of unitaries
        num_unitaries (int): number of unitaries
    '''
    adjusted_one_body_coeffs = one_body + np.einsum("llij", two_body)
    num_terms_one_body = one_body.shape[0]

    __, cores, leaves = qml.qchem.factorize(two_body, 
                                        tol_factor=tol_factor,  
                                        compressed=True, 
                                        regularization="L1")
    num_terms_two_body = len(cores)
    num_unitaries = num_terms_one_body + num_terms_two_body

    return num_unitaries, cores, leaves
    
def CP4_factors_normalizer(FACTORS):
    rank = FACTORS[0].shape[1]

    F1 = FACTORS[0].copy()
    F2 = FACTORS[1].copy()
    F3 = FACTORS[2].copy()
    F4 = FACTORS[3].copy()

    lambda_r = np.zeros(rank)

    for r in range(rank):
        lambda1_sq = np.sum(F1[:, r]**2)
        F1[:, r] /= np.sqrt(lambda1_sq)

        lambda2_sq = np.sum(F2[:, r]**2)
        F2[:, r] /= np.sqrt(lambda2_sq)

        lambda3_sq = np.sum(F3[:, r]**2)
        F3[:, r] /= np.sqrt(lambda3_sq)

        lambda4_sq = np.sum(F4[:, r]**2)
        F4[:, r] /= np.sqrt(lambda4_sq)

        lambda_r[r] = np.sqrt(lambda1_sq * lambda2_sq * lambda3_sq * lambda4_sq)

    return lambda_r

def calc_cost(two_body, rank):
    factors, __ = tfox.cpd(two_body, rank)
    fact_tsr = tfox.cpd2tens(factors)
    cost = np.sum(abs(two_body - fact_tsr)**2)
    return cost

def l4(one_body, two_body, tol_factor=1e-5):
    '''
    Args:
        one_body (np.array): 2-tensor with the one body intergrals in chemistry notation
        two_body (np.array): 4-tensor with the two body intergrals in chemistry notation
        tol_factor (float): Cutoff used to truncate

    Returns:
        one_norm (float): number of unitaries
        num_unitaries (int): number of unitaries
    '''
    adjusted_one_body_coeffs = one_body + np.einsum("llij", two_body)
    one_body_eigvals = np.linalg.eigvals(adjusted_one_body_coeffs)
    one_norm_one_body = sum(abs(one_body_eigvals))
    num_terms_one_body = one_body_eigvals.shape[0]

    current_rank = 2

    condition_1 = False
    while condition_1 == False:
        temp_cost = calc_cost(two_body, current_rank)
        if temp_cost > tol_factor:
            current_rank *= 2
        else:
            min_rank = current_rank/2
            max_rank = current_rank
            max_cost = temp_cost
            condition_1 = True
        
        if current_rank > 2**15:
            condition_1 = True
    

    condition_2 = False
    while condition_2 == False:
        current_rank = int((max_rank + min_rank)/2)
        temp_cost = calc_cost(two_body, current_rank)
        if temp_cost < tol_factor:
            max_rank = current_rank
            min_rank = min_rank

        elif temp_cost > tol_factor:
            max_rank = max_rank
            min_rank = current_rank
        
        if int(max_rank - min_rank) == 1:
            max_cost = calc_cost(two_body, max_rank)
            min_cost = calc_cost(two_body, min_rank)

            if min_cost < tol_factor:
                opt_rank = min_rank
                condition_2 = True
            else:
                opt_rank = max_rank
                condition_2 = True

    factors, __ = tfox.cpd(two_body, opt_rank)
    num_terms_two_body = opt_rank
    normalized_eigs = CP4_factors_normalizer(factors)
    one_norm_two_body = np.sum(abs(normalized_eigs))

    one_norm = one_norm_one_body + one_norm_two_body
    num_unitaries = num_terms_one_body + num_terms_two_body

    return one_norm, num_unitaries