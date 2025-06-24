import jax
import jax.numpy as jnp
import optax
import numpy as np
import h5py
from typing import Optional, Dict, Any
from tqdm import tqdm

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


import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize
import time
import h5py
import numpy as np
from .optimization import optax_lbfgs_opt_thc_l2reg_enhanced  # Import the enhanced optimizer


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

#################### Example usage function ######################
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
        eri=eri,
        nthc=nthc,
        maxiter=500,
        random_seed=42,
        verbose=True,
        include_bias_terms=True,
        chkfile_name="thc_enhanced_results.h5"
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


#if __name__ == "__main__":
#    run_enhanced_thc_optimization_example()