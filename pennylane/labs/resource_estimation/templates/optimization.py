import jax
import jax.numpy as jnp
import optax
import numpy as np
import h5py
from typing import Optional, Dict, Any
from scipy.optimize import minimize
import time

# Ensure 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


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
    objective: str = "fitting",
    h: Optional[np.ndarray] = None,
    lambda_penalty: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Enhanced THC optimization with L-BFGS.
    
    Args:
        eri: Two-body integrals tensor
        nthc: Number of THC factors
        chkfile_name: Optional checkpoint file name
        initial_guess: Initial parameter guess
        random_seed: Random seed for initialization
        maxiter: Maximum optimization iterations
        penalty_param: Regularization parameter
        gtol: Gradient tolerance
        verbose: Print optimization progress
        include_bias_terms: Whether to include bias correction terms
        objective: Optimization objective ("fitting" or "one_norm")
        h: One-body Hamiltonian matrix (required for "one_norm" objective)
        lambda_penalty: Penalty weight for fitting loss in "one_norm" objective
    
    Returns:
        Dictionary containing optimized parameters
    """
    
    # Validate inputs
    valid_objectives = ["fitting", "one_norm"]
    if objective not in valid_objectives:
        raise ValueError(f"objective must be one of {valid_objectives}, got '{objective}'")
    
    if objective == "one_norm" and h is None:
        raise ValueError("One-body Hamiltonian 'h' is required for 'one_norm' objective.")
    
    norb = eri.shape[0]
    eri_jax = jnp.array(eri)
    h_jax = jnp.array(h) if h is not None else None

    # Initialize parameters
    params = _initialize_params(initial_guess, random_seed, nthc, norb, include_bias_terms, objective)
    
    # Compute penalty parameter if not provided
    if penalty_param is None:
        penalty_param = _compute_penalty_param_enhanced(params, eri_jax, include_bias_terms, objective)
        if verbose:
            print(f"Auto-computed penalty_param: {penalty_param}")

    # Adjust penalty parameter for bias terms
    if include_bias_terms:
        penalty_param = penalty_param * 0.1
        if verbose:
            print(f"Reduced penalty_param for bias optimization: {penalty_param}")

    # Define objective function
    def objective_fn(params_dict):
        if objective == "fitting":
            return _fitting_objective(params_dict, eri_jax, norb, include_bias_terms, penalty_param)
        else:  # one_norm
            return _one_norm_objective(params_dict, eri_jax, h_jax, norb, include_bias_terms, lambda_penalty)

    # Optimize using scipy L-BFGS-B
    def pack_params(params_dict):
        arrays = [params_dict['etaPp'].flatten(), params_dict['MPQ'].flatten()]
        if include_bias_terms:
            alpha_key = 'alpha2' if objective == "fitting" else 'alpha1'
            arrays.extend([jnp.array([params_dict[alpha_key]]), params_dict['beta'].flatten()])
        return jnp.concatenate(arrays)
    
    def unpack_params(x_flat):
        idx = 0
        etaPp = x_flat[idx:idx + nthc*norb].reshape((nthc, norb))
        idx += nthc * norb
        
        MPQ = x_flat[idx:idx + nthc*nthc].reshape((nthc, nthc))
        idx += nthc * nthc
        
        result = {'etaPp': etaPp, 'MPQ': MPQ}
        
        if include_bias_terms:
            alpha_key = 'alpha2' if objective == "fitting" else 'alpha1'
            result[alpha_key] = x_flat[idx]
            idx += 1
            result['beta'] = x_flat[idx:idx + norb*norb].reshape((norb, norb))
        
        return result
    
    def objective_flat(x):
        params_dict = unpack_params(x)
        return float(objective_fn(params_dict))
    
    def grad_flat(x):
        params_dict = unpack_params(x)
        grads = jax.grad(objective_fn)(params_dict)
        grad_arrays = [grads['etaPp'].flatten(), grads['MPQ'].flatten()]
        if include_bias_terms:
            alpha_key = 'alpha2' if objective == "fitting" else 'alpha1'
            grad_arrays.extend([jnp.array([grads[alpha_key]]), grads['beta'].flatten()])
        return np.array(jnp.concatenate(grad_arrays))
    
    # Optimize
    x0 = np.array(pack_params(params))
    
    if verbose:
        bias_info = "with bias terms" if include_bias_terms else "without bias terms"
        print(f"Starting {objective} optimization with {len(x0)} parameters {bias_info}...")
    
    result = minimize(
        fun=objective_flat,
        x0=x0,
        method='L-BFGS-B',
        jac=grad_flat,
        options={'maxiter': maxiter, 'gtol': gtol, 'disp': verbose}
    )
    
    # Unpack and return result
    final_params = unpack_params(result.x)
    final_params = {k: np.array(v) for k, v in final_params.items()}
    
    if chkfile_name is not None:
        _save_thc_parameters_enhanced(final_params, chkfile_name, include_bias_terms, objective)
    
    if verbose:
        print(f"Optimization completed. Final objective value: {result.fun:.6e}")
        _print_final_params(final_params, objective, include_bias_terms)
    
    return final_params


def _initialize_params(initial_guess, random_seed, nthc, norb, include_bias_terms, objective):
    """Initialize optimization parameters."""
    if initial_guess is None:
        key = jax.random.PRNGKey(random_seed if random_seed is not None else 0)
        key1, key2, key5 = jax.random.split(key, 3)
        
        params = {
            'etaPp': jax.random.normal(key1, (nthc, norb)),
            'MPQ': jax.random.normal(key2, (nthc, nthc)),
        }
        
        if include_bias_terms:
            if objective == "fitting":
                params['alpha2'] = jnp.array(0.)
            else:  # one_norm
                params['alpha1'] = jnp.array(0.)
            params['beta'] = jax.random.normal(key5, (norb, norb)) * 0.01 if objective == "one_norm" else jnp.zeros((norb, norb))
    else:
        params = {k: jnp.array(v) for k, v in initial_guess.items()}
        
        if include_bias_terms:
            if objective == "fitting":
                params.setdefault('alpha2', jnp.array(0.))
            else:  # one_norm
                params.setdefault('alpha1', jnp.array(0.))
            params.setdefault('beta', jnp.zeros((norb, norb)))

    return params


def _fitting_objective(params_dict, eri_jax, norb, include_bias_terms, penalty_param):
    """Fitting objective function."""
    etaPp = params_dict['etaPp']
    MPQ = params_dict['MPQ']
    
    g_modified = eri_jax
    if include_bias_terms:
        alpha2 = params_dict['alpha2']
        beta = params_dict['beta']
        
        eye = jnp.eye(norb)
        alpha2_term = alpha2 * jnp.einsum('pq,rs->pqrs', eye, eye)
        beta_term1 = 0.5 * jnp.einsum('pq,rs->pqrs', beta, eye)
        beta_term2 = 0.5 * jnp.einsum('pq,rs->pqrs', eye, beta)
        
        g_modified = eri_jax - alpha2_term - beta_term1 - beta_term2
    
    # THC approximation
    CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)
    Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP)
    
    # Fitting loss
    deri = g_modified - Iapprox
    fitting_loss = 0.5 * jnp.sum(deri**2)
    
    # Regularization
    if include_bias_terms:
        reg_thc = penalty_param * jnp.sum(MPQ**2)
        reg_alpha2 = penalty_param * 0.01 * params_dict['alpha2']**2
        reg_beta = penalty_param * 0.01 * jnp.sum(params_dict['beta']**2)
        reg = reg_thc + reg_alpha2 + reg_beta
    else:
        reg = penalty_param * jnp.sum(MPQ**2)
    
    return fitting_loss + reg


def _one_norm_objective(params_dict, eri_jax, h_jax, norb, include_bias_terms, lambda_penalty):
    """One-norm objective function."""
    etaPp = params_dict['etaPp']
    MPQ = params_dict['MPQ']
    
    # PART 1: Calculate fitting_loss (original ERI without modifications)
    CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)
    Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP)
    deri = eri_jax - Iapprox
    fitting_loss = 0.5 * jnp.sum(deri**2)
    
    # PART 2: Calculate Hamiltonian 1-norm
    if include_bias_terms:
        alpha1 = params_dict['alpha1']
        beta_asym = params_dict['beta']
        beta = 0.5 * (beta_asym + beta_asym.T)  # Symmetrize beta
    else:
        alpha1 = 0.0
        beta = jnp.zeros((norb, norb))
    
    eye = jnp.eye(norb)
    
    # Calculate h_pq^(BI) from Eq. (16)
    g_trace = jnp.einsum('prrq->pq', eri_jax)  # Constant term
    h_bi = h_jax - 0.5 * g_trace - alpha1 * eye + 0.5 * beta
    
    # Eigenvalues of the one-body term
    t_k = jnp.linalg.eigvalsh(h_bi)
    
    # 1-norm from Eq. (23)
    lambda_one_body = jnp.sum(jnp.abs(t_k))
    lambda_two_body = 0.5 * jnp.sum(jnp.abs(MPQ)) - 0.25 * jnp.sum(jnp.abs(jnp.diag(MPQ)))
    lambda_thc = lambda_one_body + lambda_two_body
    
    # Total loss is the 1-norm plus a penalty for the fitting quality
    return lambda_thc + lambda_penalty * fitting_loss


def _compute_penalty_param_enhanced(params, eri, include_bias_terms, objective="fitting"):
    """Compute penalty parameter for the enhanced objective function."""
    etaPp = params['etaPp']
    MPQ = params['MPQ']
    norb = eri.shape[0]
    
    if objective == "fitting":
        g_modified = eri
        if include_bias_terms:
            alpha2 = params.get('alpha2', 0.0)
            beta = params.get('beta', jnp.zeros((norb, norb)))
            
            eye = jnp.eye(norb)
            alpha2_term = alpha2 * jnp.einsum('pq,rs->pqrs', eye, eye)
            beta_term1 = 0.5 * jnp.einsum('pq,rs->pqrs', beta, eye)
            beta_term2 = 0.5 * jnp.einsum('pq,rs->pqrs', eye, beta)
            
            g_modified = eri - alpha2_term - beta_term1 - beta_term2
        
        CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)
        Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP)
        deri = g_modified - Iapprox
        sum_square_loss = 0.5 * jnp.sum(deri**2)
        regularization_scale = jnp.sum(MPQ**2)
        
    else:  # one_norm
        CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)
        Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP)
        deri = eri - Iapprox
        sum_square_loss = 0.5 * jnp.sum(deri**2)
        regularization_scale = jnp.sum(jnp.abs(MPQ))
    
    # Avoid division by zero
    if regularization_scale < 1e-12:
        return 1e-6
    
    return float(sum_square_loss / regularization_scale)


def _print_final_params(final_params, objective, include_bias_terms):
    """Print final optimization parameters."""
    if not include_bias_terms:
        return
        
    if objective == "fitting" and 'alpha2' in final_params:
        print(f"Final alpha2: {final_params['alpha2']:.6e}")
    elif objective == "one_norm" and 'alpha1' in final_params:
        print(f"Final alpha1: {final_params['alpha1']:.6e}")
    
    if 'beta' in final_params:
        print(f"Final beta norm: {np.linalg.norm(final_params['beta']):.6e}")


def _save_thc_parameters_enhanced(params, chkfile_name, include_bias_terms, objective="fitting"):
    """Save enhanced THC parameters to HDF5 file."""
    with h5py.File(chkfile_name, "w") as f:
        f["etaPp"] = np.array(params['etaPp'])
        f["ZPQ"] = np.array(params['MPQ'])  # Keep OpenFermion naming convention
        f["objective"] = objective
        
        if include_bias_terms:
            if objective == "fitting" and 'alpha2' in params:
                f["alpha2"] = np.array(params['alpha2'])
            elif objective == "one_norm" and 'alpha1' in params:
                f["alpha1"] = np.array(params['alpha1'])
            
            if 'beta' in params:
                f["beta"] = np.array(params['beta'])


def load_thc_parameters_enhanced(chkfile_name: str) -> Dict[str, np.ndarray]:
    """Load enhanced THC parameters from HDF5 file."""
    params = {}
    with h5py.File(chkfile_name, "r") as f:
        params['etaPp'] = np.array(f["etaPp"])
        params['MPQ'] = np.array(f["ZPQ"])  # Handle OpenFermion naming convention
        
        # Load objective if available
        if "objective" in f:
            params['objective'] = f["objective"][()].decode() if isinstance(f["objective"][()], bytes) else str(f["objective"][()])
        else:
            params['objective'] = "fitting"  # Default for backward compatibility
        
        # Load bias parameters based on objective
        if params['objective'] == "fitting" and "alpha2" in f:
            params['alpha2'] = np.array(f["alpha2"])
        elif params['objective'] == "one_norm" and "alpha1" in f:
            params['alpha1'] = np.array(f["alpha1"])
        
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
    thc_method="standard",
    h_one=None,
    objective="fitting",
    lambda_penalty=1.0,
):
    """
    THC-CP3 performs an SVD decomposition of the eri matrix followed by a CP
    decomposition via pybtas. The CP decomposition assumes the tensor is
    symmetric in the first two indices corresponding to a reshaped
    (and rescaled by the singular value) singular vector.

    Args:
        eri_full: (N x N x N x N) eri tensor in Mulliken (chemists) ordering
        nthc: number of THC factors to use
        thc_save_file: if not None, save output to filename (as HDF5)
        first_factor_thresh: SVD threshold on initial factorization of ERI
        conv_eps: convergence threshold on CP3 ALS
        perform_bfgs_opt: Perform extra gradient opt. on top of CP3 decomp
        bfgs_maxiter: Maximum bfgs steps to take. Default 5000.
        random_start_thc: Perform random start for CP3. If false perform HOSVD start.
        verify: check eri properties. Default is False
        penalty_param: penalty parameter for L2 regularization. Default is None.
        thc_method: Choose optimization method:
            - "standard": Original L-BFGS-B optimizer
            - "enhanced": Optax L-BFGS without bias terms  
            - "enhanced_bias": Optax L-BFGS with bias correction terms
            - "enhanced_one_norm": Optax L-BFGS with one_norm objective
        h_one: One-body Hamiltonian matrix (required for enhanced_one_norm)
        objective: Optimization objective for enhanced methods ("fitting" or "one_norm")
        lambda_penalty: Penalty weight for fitting loss in one_norm objective

    Returns:
        eri_thc: (N x N x N x N) reconstructed ERIs from THC factorization
        thc_leaf: THC leaf tensor
        thc_central: THC central tensor
        info: arguments set during the THC factorization
    """
    # Fail fast if we don't have the tools to use this routine
    try:
        import pybtas
    except ImportError:
        raise ImportError("pybtas could not be imported. Is it installed and in PYTHONPATH?")

    # Validate thc_method parameter
    valid_methods = ["standard", "enhanced", "enhanced_bias", "enhanced_one_norm"]
    if thc_method not in valid_methods:
        raise ValueError(f"thc_method must be one of {valid_methods}, got '{thc_method}'")

    # Validate parameters for enhanced_one_norm
    if thc_method == "enhanced_one_norm":
        if objective not in ["fitting", "one_norm"]:
            raise ValueError(f"objective must be 'fitting' or 'one_norm' for enhanced_one_norm method")
        
        if objective == "one_norm" and h_one is None:
            raise ValueError("h_one is required when using enhanced_one_norm method with objective='one_norm'")

    info = locals()
    info.pop('eri_full', None)  # data too big for info dict
    info.pop('pybtas', None)  # not needed for info dict

    norb = eri_full.shape[0]
    
    # Verify ERI symmetries if requested
    if verify:
        _verify_eri_symmetries(eri_full)

    # Perform SVD decomposition
    eri_mat = eri_full.transpose(0, 1, 3, 2).reshape((norb**2, norb**2))
    u, sigma, vh = np.linalg.svd(eri_mat)

    if verify:
        assert np.allclose(eri_mat, eri_mat.T)
        assert np.allclose(u @ np.diag(sigma) @ vh, eri_mat)

    # Get non-zero singular values and prepare for CP3
    non_zero_sv = np.where(sigma >= first_factor_thresh)[0]
    u_chol = u[:, non_zero_sv] @ np.diag(np.sqrt(sigma[non_zero_sv]))

    # CP3 decomposition
    start_time = time.time()
    beta, gamma, scale = pybtas.cp3_from_cholesky(
        u_chol.copy(), nthc, random_start=random_start_thc, conv_eps=conv_eps
    )
    cp3_calc_time = time.time() - start_time

    thc_leaf = beta.T
    thc_gamma = np.einsum('xr,r->xr', gamma, scale.ravel())
    thc_central = thc_gamma.T @ thc_gamma

    if verify:
        eri_thc = np.einsum(
            "Pp,Pr,Qq,Qs,PQ->prqs",
            thc_leaf, thc_leaf, thc_leaf, thc_leaf, thc_central,
            optimize=True,
        )
        print("\tERI L2 CP3-THC ", np.linalg.norm(eri_thc - eri_full))
        print("\tCP3 timing: ", cp3_calc_time)

    # Perform BFGS optimization if requested
    if perform_bfgs_opt:
        thc_leaf, thc_central, info = _perform_bfgs_optimization(
            thc_method, eri_full, h_one, nthc, norb, thc_leaf, thc_central,
            bfgs_maxiter, penalty_param, objective, lambda_penalty, verify, info
        )

    # Reconstruct final ERI tensor
    eri_thc = np.einsum(
        "Pp,Pr,Qq,Qs,PQ->prqs", thc_leaf, thc_leaf, thc_leaf, thc_leaf, thc_central, optimize=True
    )

    # Save results if requested
    if thc_save_file is not None:
        _save_thc_results(thc_save_file, thc_leaf, thc_central, thc_method, info, perform_bfgs_opt)

    return eri_thc, thc_leaf, thc_central, info


def _verify_eri_symmetries(eri_full):
    """Verify ERI tensor symmetries."""
    assert np.allclose(eri_full, eri_full.transpose(1, 0, 2, 3))  # (ij|kl) == (ji|kl)
    assert np.allclose(eri_full, eri_full.transpose(0, 1, 3, 2))  # (ij|kl) == (ij|lk)
    assert np.allclose(eri_full, eri_full.transpose(1, 0, 3, 2))  # (ij|kl) == (ji|lk)
    assert np.allclose(eri_full, eri_full.transpose(2, 3, 0, 1))  # (ij|kl) == (kl|ij)


def _perform_bfgs_optimization(thc_method, eri_full, h_one, nthc, norb, thc_leaf, thc_central,
                               bfgs_maxiter, penalty_param, objective, lambda_penalty, verify, info):
    """Perform BFGS optimization based on the chosen method."""
    initial_params = {'etaPp': thc_leaf, 'MPQ': thc_central}
    
    if thc_method == "standard":
        result_params = optax_lbfgs_opt_thc_l2reg_enhanced(
            eri=eri_full, nthc=nthc, initial_guess=initial_params,
            maxiter=bfgs_maxiter, penalty_param=penalty_param,
            include_bias_terms=False, objective="fitting", verbose=verify
        )
        
    elif thc_method in ["enhanced", "enhanced_bias"]:
        include_bias = (thc_method == "enhanced_bias")
        
        if include_bias:
            initial_params.update({'alpha2': np.array(0.), 'beta': np.zeros((norb, norb))})
        
        result_params = optax_lbfgs_opt_thc_l2reg_enhanced(
            eri=eri_full, nthc=nthc, initial_guess=initial_params,
            maxiter=bfgs_maxiter, penalty_param=penalty_param,
            include_bias_terms=include_bias, objective="fitting", verbose=verify
        )
        
        if include_bias:
            info['alpha2_optimized'] = result_params['alpha2']
            info['beta_optimized'] = result_params['beta']
            
    elif thc_method == "enhanced_one_norm":
        # Add appropriate bias parameters based on objective
        if objective == "fitting":
            initial_params.update({'alpha2': np.array(0.), 'beta': np.zeros((norb, norb))})
        else:  # one_norm
            initial_params.update({'alpha1': np.array(0.), 'beta': np.zeros((norb, norb))})
        
        result_params = optax_lbfgs_opt_thc_l2reg_enhanced(
            eri=eri_full, h=h_one, nthc=nthc, initial_guess=initial_params,
            maxiter=bfgs_maxiter, penalty_param=penalty_param,
            include_bias_terms=True, objective=objective,
            lambda_penalty=lambda_penalty, verbose=verify
        )
        
        # Store optimized parameters based on objective
        if objective == "fitting":
            info['alpha2_optimized'] = result_params['alpha2']
        else:  # one_norm
            info['alpha1_optimized'] = result_params['alpha1']
        
        info['beta_optimized'] = result_params['beta']
        info['objective_used'] = objective

    return result_params['etaPp'], result_params['MPQ'], info


def _save_thc_results(thc_save_file, thc_leaf, thc_central, thc_method, info, perform_bfgs_opt):
    """Save THC results to HDF5 file."""
    with h5py.File(thc_save_file + '.h5', 'w') as fid:
        fid.create_dataset('thc_leaf', data=thc_leaf)
        fid.create_dataset('thc_central', data=thc_central)
        fid.create_dataset('thc_method', data=thc_method)
        
        # Save bias parameters if they exist
        if perform_bfgs_opt:
            if thc_method == "enhanced_bias" and 'alpha2_optimized' in info:
                fid.create_dataset('alpha2', data=info['alpha2_optimized'])
                if 'beta_optimized' in info:
                    fid.create_dataset('beta', data=info['beta_optimized'])
            elif thc_method == "enhanced_one_norm":
                if 'alpha1_optimized' in info:
                    fid.create_dataset('alpha1', data=info['alpha1_optimized'])
                if 'beta_optimized' in info:
                    fid.create_dataset('beta', data=info['beta_optimized'])
                if 'objective_used' in info:
                    fid.create_dataset('objective', data=info['objective_used'])
        
        fid.create_dataset('info', data=str(info))

def thc_one_norm(kappa, zeta):
    D,_ = np.linalg.eigh(kappa)

    lambda_1 = np.sum(np.abs(D))

    lambda_2 = 0.5*np.sum(np.abs(zeta))
    M = zeta.shape[0]
    for mm in range(M):
        lambda_2 -= 0.25 * zeta[mm,mm]

    return lambda_1 + lambda_2

# Example usage functions
def run_enhanced_thc_optimization_example():
    """Example showing how to use the enhanced implementation."""
    # Create dummy ERI tensor for testing
    norb = 4
    nthc = 6
    np.random.seed(42)
    eri = np.random.random((norb, norb, norb, norb))
    eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))  # Ensure some symmetry
    
    # Run optimization with bias terms
    params = optax_lbfgs_opt_thc_l2reg_enhanced(
        eri=eri, nthc=nthc, maxiter=500, random_seed=42, verbose=True,
        include_bias_terms=True, chkfile_name="thc_enhanced_results.h5"
    )
    
    print("Optimization completed. Final parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Test loading parameters
    loaded_params = load_thc_parameters_enhanced("thc_enhanced_results.h5")
    print("\nLoaded parameters:")
    for key, value in loaded_params.items():
        print(f"  {key}:{value}")
    
    return params


def run_one_norm_optimization_example():
    """Example showing how to use the one_norm objective."""
    # Create dummy data
    norb = 4
    nthc = 6
    np.random.seed(123)
    
    # Create symmetric ERI and one-body Hamiltonian
    eri = np.random.random((norb, norb, norb, norb)) * 0.1
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        eri = 0.5 * (eri + eri.transpose(perm))
    
    h_one = np.random.random((norb, norb)) * 0.05
    h_one = 0.5 * (h_one + h_one.T)
    
    print("Testing one_norm objective...")
    
    # Test both objectives
    objectives_results = {}
    for obj in ["fitting", "one_norm"]:
        params = optax_lbfgs_opt_thc_l2reg_enhanced(
            eri=eri, h=h_one, nthc=nthc, objective=obj,
            include_bias_terms=True, lambda_penalty=0.1,
            maxiter=100, verbose=True, random_seed=456
        )
        objectives_results[obj] = params
    
    # Print results comparison
    for obj, params in objectives_results.items():
        print(f"\n{obj.upper()} objective results:")
        if 'alpha2' in params:
            print(f"  alpha2: {params['alpha2']:.6e}")
        if 'alpha1' in params:
            print(f"  alpha1: {params['alpha1']:.6e}")
        print(f"  beta norm: {np.linalg.norm(params['beta']):.6e}")
    
    # Test via thc_via_cp3 interface
    print("\nTesting via thc_via_cp3 interface...")
    eri_thc, thc_leaf, thc_central, info = thc_via_cp3(
        eri_full=eri, h_one=h_one, nthc=nthc,
        thc_method="enhanced_one_norm", objective="one_norm",
        lambda_penalty=0.1, bfgs_maxiter=50, verify=True
    )
    
    print(f"Final ERI reconstruction error: {np.linalg.norm(eri_thc - eri):.6e}")
    if 'alpha1_optimized' in info:
        print(f"Optimized alpha1: {info['alpha1_optimized']:.6e}")
    if 'beta_optimized' in info:
        print(f"Optimized beta norm: {np.linalg.norm(info['beta_optimized']):.6e}")
    
    return objectives_results