import pennylane as qml
from pennylane import numpy as np
from pennylane.resource import DoubleFactorization as DF
import TensorFox as tfox
import jax
import jax.numpy as jnp
import optax
import numpy as np
import h5py
from typing import Optional, Dict, Any
from tqdm import tqdm
from jax import grad
from jax.scipy.optimize import minimize
import time

# Ensure 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

def thc_one_norm(kappa, zeta):
    D,_ = np.linalg.eigh(kappa)

    lambda_1 = np.sum(np.abs(D))

    lambda_2 = 0.5*np.sum(np.abs(zeta))
    M = zeta.shape[0]
    for mm in range(M):
        lambda_2 -= 0.25 * zeta[mm,mm]

    return lambda_1 + lambda_2

def optax_lbfgs_opt_thc_l2reg_enhanced(
    eri: np.ndarray,
    nthc: int,
    chkfile_name: Optional[str] = None,
    initial_guess: Optional[Dict[str, np.ndarray]] = None,
    random_seed: Optional[int] = None,
    maxiter: int = 1000,
    penalty_param: Optional[float] = None,
    gtol: float = 1e-8,
    verbose: bool = True,
    include_bias_terms: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Enhanced least-squares fit of two-electron integral tensors with optax.lbfgs,
    L2-regularization of lambda, and additional bias terms (alpha2 and beta).

    The modified ERI tensor follows:
    g^(BI)_pqrs = g_pqrs - α₂δ_pq δ_rs - 1/2(β_pq δ_rs + δ_pq β_rs)

    Args:
        eri: The two-electron integral tensor.
        nthc: The THC dimension.
        chkfile_name: Path to an HDF5 file for saving the final parameters.
        initial_guess: Initial guess dictionary with keys 'etaPp', 'MPQ', 'alpha2', 'beta'.
        random_seed: Seed for the random number generator.
        maxiter: The maximum number of optimization iterations.
        penalty_param: The regularization penalty parameter. If None, computed automatically.
        gtol: Gradient tolerance for convergence.
        verbose: Whether to print progress.
        include_bias_terms: Whether to include α₂ and β parameters.

    Returns:
        Dictionary containing optimized parameters: {'etaPp', 'MPQ', 'alpha2', 'beta'}
    """
    norb = eri.shape[0]
    
    # Convert ERI to JAX array once
    eri_jax = jnp.array(eri)

    # 1. Set initial guess
    if initial_guess is None:
        key = jax.random.PRNGKey(random_seed if random_seed is not None else 0)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        params = {
            'etaPp': jax.random.normal(key1, (nthc, norb)),
            'MPQ': jax.random.normal(key2, (nthc, nthc)),
        }
        
        if include_bias_terms:
            params['alpha2'] = jnp.array(0.)  # scalar
            params['beta'] = jnp.zeros((norb, norb))  # matrix
    else:
        params = {k: jnp.array(v) for k, v in initial_guess.items()}
        
        # Ensure all required parameters are present
        if 'etaPp' not in params or 'MPQ' not in params:
            raise ValueError("initial_guess must contain 'etaPp' and 'MPQ' keys")
            
        if include_bias_terms:
            if 'alpha2' not in params:
                key = jax.random.PRNGKey(random_seed if random_seed is not None else 0)
                params['alpha2'] = jax.random.normal(key, ())
            if 'beta' not in params:
                key = jax.random.PRNGKey(random_seed if random_seed is not None else 1)
                params['beta'] = jax.random.normal(key, (norb, norb))

    # 2. Compute penalty parameter if not provided
    if penalty_param is None:
        penalty_param = _compute_penalty_param_enhanced(params, eri_jax, include_bias_terms)
        if verbose:
            print(f"Auto-computed penalty_param: {penalty_param}")

    # 3. Define the objective function
    @jax.jit
    def thc_objective_regularized_enhanced(p: Dict[str, jnp.ndarray]) -> float:
        etaPp = p['etaPp']  # shape: (nthc, norb)
        MPQ = p['MPQ']      # shape: (nthc, nthc)
        
        # Apply bias corrections to ERI tensor
        g_modified = eri_jax
        if include_bias_terms:
            alpha2 = p['alpha2']  # scalar
            beta = p['beta']      # shape: (norb, norb)
            
            # Create identity tensors for bias terms
            eye = jnp.eye(norb)
            
            # Apply bias correction: g^(BI)_pqrs = g_pqrs - α₂δ_pq δ_rs - 1/2(β_pq δ_rs + δ_pq β_rs)
            # First term: α₂δ_pq δ_rs
            alpha2_term = alpha2 * jnp.einsum('pq,rs->pqrs', eye, eye)
            
            # Second term: 1/2(β_pq δ_rs + δ_pq β_rs)
            beta_term1 = 0.5 * jnp.einsum('pq,rs->pqrs', beta, eye)
            beta_term2 = 0.5 * jnp.einsum('pq,rs->pqrs', eye, beta)
            
            g_modified = eri_jax - alpha2_term - beta_term1 - beta_term2
        
        # Compute THC approximation (following OpenFermion einsum patterns)
        CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)
        Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=[(0, 1), (0, 1)])
        
        # Primary loss using modified ERI
        deri = g_modified - Iapprox
        sum_square_loss = 0.5 * jnp.sum(deri**2)
        
        # Regularization term (following OpenFermion's approach)
        SPQ = etaPp @ etaPp.T  # metric tensor
        cP = jnp.diag(jnp.diag(SPQ))  # diagonal normalization
        MPQ_normalized = cP @ MPQ @ cP
        lambda_z = 0.5 * jnp.sum(jnp.abs(MPQ_normalized))  # L1 norm
        
        return sum_square_loss + penalty_param * lambda_z**2  # L2 regularization

    # 4. Set up Optax L-BFGS optimizer
    solver = optax.lbfgs()
    opt_state = solver.init(params)
    
    # Use the proper Optax pattern for L-BFGS
    value_and_grad_fn = optax.value_and_grad_from_state(thc_objective_regularized_enhanced)

    # 5. Optimization loop with proper convergence checking
    for i in tqdm(range(maxiter), desc = 'L-BFGS Optimization'):
        value, grad = value_and_grad_fn(params, state=opt_state)
        
        # Check convergence using gradient norm
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad.values()))
        if grad_norm < gtol:
            if verbose:
                print(f"Converged at iteration {i}: grad_norm = {grad_norm}")
            break
            
        # Update parameters using proper Optax L-BFGS interface
        updates, opt_state = solver.update(
            grad, opt_state, params=params, value=value, grad=grad, 
            value_fn=thc_objective_regularized_enhanced
        )
        params = optax.apply_updates(params, updates)

        if verbose and i % 100 == 0:
            print(f"Iteration {i}: Loss = {value:.6e}, Grad norm = {grad_norm:.6e}")
    else:
        if verbose:
            print("Maximum number of iterations reached")

    # 6. Save final parameters if checkpoint file is provided
    if chkfile_name is not None:
        _save_thc_parameters_enhanced(params, chkfile_name, include_bias_terms)

    # Convert back to numpy arrays for return
    return {k: np.array(v) for k, v in params.items()}


def _compute_penalty_param_enhanced(
    params: Dict[str, jnp.ndarray], 
    eri: jnp.ndarray,
    include_bias_terms: bool
) -> float:
    """Compute penalty parameter for the enhanced objective function."""
    etaPp = params['etaPp']
    MPQ = params['MPQ']
    norb = eri.shape[0]
    
    # Apply bias corrections if included
    g_modified = eri
    if include_bias_terms:
        alpha2 = params['alpha2']
        beta = params['beta']
        
        eye = jnp.eye(norb)
        alpha2_term = alpha2 * jnp.einsum('pq,rs->pqrs', eye, eye)
        beta_term1 = 0.5 * jnp.einsum('pq,rs->pqrs', beta, eye)
        beta_term2 = 0.5 * jnp.einsum('pq,rs->pqrs', eye, beta)
        
        g_modified = eri - alpha2_term - beta_term1 - beta_term2
    
    # Compute initial loss
    CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)
    Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=[(0, 1), (0, 1)])
    deri = g_modified - Iapprox
    sum_square_loss = 0.5 * jnp.sum(deri**2)
    
    # Compute lambda_z
    SPQ = etaPp @ etaPp.T
    cP = jnp.diag(jnp.diag(SPQ))
    MPQ_normalized = cP @ MPQ @ cP
    lambda_z = 0.5 * jnp.sum(jnp.abs(MPQ_normalized))
    
    return float(sum_square_loss / lambda_z)


def _save_thc_parameters_enhanced(
    params: Dict[str, jnp.ndarray],  # Fixed: was jnp.ndArray
    chkfile_name: str, 
    include_bias_terms: bool
) -> None:
    """Save enhanced THC parameters to HDF5 file."""
    with h5py.File(chkfile_name, "w") as f:
        f["etaPp"] = np.array(params['etaPp'])
        f["ZPQ"] = np.array(params['MPQ'])  # Keep OpenFermion naming convention
        
        if include_bias_terms:
            f["alpha2"] = np.array(params['alpha2'])
            f["beta"] = np.array(params['beta'])


def load_thc_parameters_enhanced(chkfile_name: str) -> Dict[str, np.ndarray]:
    """Load enhanced THC parameters from HDF5 file."""
    params = {}
    with h5py.File(chkfile_name, "r") as f:
        params['etaPp'] = np.array(f["etaPp"])
        params['MPQ'] = np.array(f["ZPQ"])  # Handle OpenFermion naming convention
        
        if "alpha2" in f:
            params['alpha2'] = np.array(f["alpha2"])
        if "beta" in f:
            params['beta'] = np.array(f["beta"])
    
    return params


def thc_via_cp3(
    eri_full,
    nthc,
    thc_save_file=None,
    first_factor_thresh=1.0e-14,
    conv_eps=1.0e-4,
    perform_bfgs_opt=True,
    bfgs_maxiter=5000,
    random_start_thc=True,
    verify=False,
    penalty_param=None,
    thc_method="standard",  # New unified parameter
):
    """
    THC-CP3 performs an SVD decomposition of the eri matrix followed by a CP
    decomposition via pybtas. The CP decomposition assumes the tensor is
    symmetric in the first two indices corresponding to a reshaped
    (and rescaled by the singular value) singular vector.

    Args:
        eri_full - (N x N x N x N) eri tensor in Mulliken (chemists) ordering
        nthc (int) - number of THC factors to use
        thc_save_file (str) - if not None, save output to filename (as HDF5)
        first_factor_thresh - SVD threshold on initial factorization of ERI
        conv_eps (float) - convergence threshold on CP3 ALS
        perform_bfgs_opt - Perform extra gradient opt. on top of CP3 decomp
        bfgs_maxiter - Maximum bfgs steps to take. Default 5000.
        random_start_thc - Perform random start for CP3.
                           If false perform HOSVD start.
        verify - check eri properties. Default is False
        penalty_param - penalty parameter for L2 regularization. Default is None.
        thc_method (str) - Choose optimization method:
            - "standard": Original L-BFGS-B optimizer
            - "enhanced": Optax L-BFGS without bias terms  
            - "enhanced_bias": Optax L-BFGS with bias correction terms

    returns:
        eri_thc - (N x N x N x N) reconstructed ERIs from THC factorization
        thc_leaf - THC leaf tensor
        thc_central - THC central tensor
        info (dict) - arguments set during the THC factorization
    """
    # fail fast if we don't have the tools to use this routine
    try:
        import pybtas
    except ImportError:
        raise ImportError("pybtas could not be imported. Is it installed and in PYTHONPATH?")

    # Validate thc_method parameter
    valid_methods = ["standard", "enhanced", "enhanced_bias"]
    if thc_method not in valid_methods:
        raise ValueError(f"thc_method must be one of {valid_methods}, got '{thc_method}'")

    info = locals()
    info.pop('eri_full', None)  # data too big for info dict
    info.pop('pybtas', None)  # not needed for info dict

    norb = eri_full.shape[0]
    if verify:
        assert np.allclose(eri_full, eri_full.transpose(1, 0, 2, 3))  # (ij|kl) == (ji|kl)
        assert np.allclose(eri_full, eri_full.transpose(0, 1, 3, 2))  # (ij|kl) == (ij|lk)
        assert np.allclose(eri_full, eri_full.transpose(1, 0, 3, 2))  # (ij|kl) == (ji|lk)
        assert np.allclose(eri_full, eri_full.transpose(2, 3, 0, 1))  # (ij|kl) == (kl|ij)

    eri_mat = eri_full.transpose(0, 1, 3, 2).reshape((norb**2, norb**2))
    if verify:
        assert np.allclose(eri_mat, eri_mat.T)

    u, sigma, vh = np.linalg.svd(eri_mat)

    if verify:
        assert np.allclose(u @ np.diag(sigma) @ vh, eri_mat)

    non_zero_sv = np.where(sigma >= first_factor_thresh)[0]
    u_chol = u[:, non_zero_sv]
    diag_sigma = np.diag(sigma[non_zero_sv])
    u_chol = u_chol @ np.diag(np.sqrt(sigma[non_zero_sv]))

    if verify:
        test_eri_mat_mulliken = u[:, non_zero_sv] @ diag_sigma @ vh[non_zero_sv, :]
        assert np.allclose(test_eri_mat_mulliken, eri_mat)

    start_time = time.time()  # timing results if requested by user
    beta, gamma, scale = pybtas.cp3_from_cholesky(
        u_chol.copy(), nthc, random_start=random_start_thc, conv_eps=conv_eps
    )
    cp3_calc_time = time.time() - start_time

    if verify:
        u_alpha = np.zeros((norb, norb, len(non_zero_sv)))
        for ii in range(len(non_zero_sv)):
            u_alpha[:, :, ii] = np.sqrt(sigma[ii]) * u[:, ii].reshape((norb, norb))
            assert np.allclose(
                u_alpha[:, :, ii], u_alpha[:, :, ii].T
            )  # consequence of working with Mulliken rep

        u_alpha_test = np.einsum("ar,br,xr,r->abx", beta, beta, gamma, scale.ravel())
        print("\tu_alpha l2-norm ", np.linalg.norm(u_alpha_test - u_alpha))

    thc_leaf = beta.T
    thc_gamma = np.einsum('xr,r->xr', gamma, scale.ravel())
    thc_central = thc_gamma.T @ thc_gamma

    if verify:
        eri_thc = np.einsum(
            "Pp,Pr,Qq,Qs,PQ->prqs",
            thc_leaf,
            thc_leaf,
            thc_leaf,
            thc_leaf,
            thc_central,
            optimize=True,
        )
        print("\tERI L2 CP3-THC ", np.linalg.norm(eri_thc - eri_full))
        print("\tCP3 timing: ", cp3_calc_time)

    if perform_bfgs_opt:
        if thc_method == "standard":
            # Original L-BFGS-B implementation
            x = np.hstack((thc_leaf.ravel(), thc_central.ravel()))
            x = lbfgsb_opt_thc_l2reg(
                eri_full, nthc, initial_guess=x, maxiter=bfgs_maxiter, penalty_param=penalty_param
            )
            thc_leaf = x[: norb * nthc].reshape(nthc, norb)  # leaf tensor  nthc x norb
            thc_central = x[norb * nthc : norb * nthc + nthc * nthc].reshape(
                nthc, nthc
            )  # central tensor
            
        elif thc_method in ["enhanced", "enhanced_bias"]:
            # Enhanced Optax L-BFGS implementation
            include_bias = (thc_method == "enhanced_bias")
            
            # Prepare initial parameters dictionary
            initial_params = {
                'etaPp': thc_leaf,
                'MPQ': thc_central
            }
            
            # Add bias terms if using enhanced_bias method
            if include_bias:
                initial_params['alpha2'] = np.array(0.)
                initial_params['beta'] = np.zeros((norb, norb))
            
            # Run enhanced optimization
            result_params = optax_lbfgs_opt_thc_l2reg_enhanced(
                eri=eri_full,
                nthc=nthc,
                initial_guess=initial_params,
                maxiter=bfgs_maxiter,
                penalty_param=penalty_param,
                include_bias_terms=include_bias,
                verbose=verify
            )
            
            # Extract optimized leaf and central tensors
            thc_leaf = result_params['etaPp']
            thc_central = result_params['MPQ']
            
            # Store bias parameters in info if they were optimized
            if include_bias:
                info['alpha2_optimized'] = result_params['alpha2']
                info['beta_optimized'] = result_params['beta']

    # Reconstruct final ERI tensor
    eri_thc = np.einsum(
        "Pp,Pr,Qq,Qs,PQ->prqs", thc_leaf, thc_leaf, thc_leaf, thc_leaf, thc_central, optimize=True
    )

    # Save results if requested
    if thc_save_file is not None:
        with h5py.File(thc_save_file + '.h5', 'w') as fid:
            fid.create_dataset('thc_leaf', data=thc_leaf)
            fid.create_dataset('thc_central', data=thc_central)
            fid.create_dataset('thc_method', data=thc_method)
            
            # Save bias parameters if they exist
            if thc_method == "enhanced_bias" and perform_bfgs_opt:
                if 'alpha2_optimized' in info:
                    fid.create_dataset('alpha2', data=info['alpha2_optimized'])
                if 'beta_optimized' in info:
                    fid.create_dataset('beta', data=info['beta_optimized'])
            
            fid.create_dataset('info', data=str(info))

    return eri_thc, thc_leaf, thc_central, info


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