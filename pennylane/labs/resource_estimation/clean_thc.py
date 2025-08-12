import numpy as np
import jax
from jax import numpy as jnp
from jax import grad, jit, vmap
from jax import lax
import jax.scipy.optimize
import optax

def _get_BLISS_sizes(num_ob_syms, Norbs):
    avec_len = num_ob_syms
    bvec_len = int(num_ob_syms * (num_ob_syms+1)/2)
    ob_mat_num_params = int(Norbs*(Norbs+1)/2) 

    return avec_len, bvec_len, ob_mat_num_params

@jit
def _unfold_thc(MPQ, etaPp):
    CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)
    eri_thc = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP)

    return eri_thc

def _ob_params_to_mat(ob_mat_params, Norbs):
    ob_mat = jnp.zeros((Norbs,Norbs))

    rows, cols = jnp.triu_indices(Norbs)
    ob_mat = ob_mat.at[rows, cols].set(ob_mat_params)

    return (ob_mat + ob_mat.T) / 2

_ob_params_to_mat = jit(_ob_params_to_mat, static_argnums=(1,))

def _extract_vecs(x_vec, Norbs, Nthc, num_ob_syms, include_bliss):
        eta_fin = Nthc*Norbs
        etaPp = x_vec[:eta_fin].reshape((Nthc, Norbs))

        M_fin = eta_fin + Nthc**2
        MPQ = x_vec[eta_fin:M_fin].reshape((Nthc, Nthc))

        if include_bliss:
            avec_len, bvec_len, ob_mat_num_params = _get_BLISS_sizes(num_ob_syms, Norbs)
            a_fin = M_fin + avec_len
            avec = x_vec[M_fin:a_fin]

            b_fin = a_fin + bvec_len
            bvec = x_vec[a_fin:b_fin]

            beta_mats_params = x_vec[b_fin:].reshape((num_ob_syms, ob_mat_num_params))

            return etaPp, MPQ, avec, bvec, beta_mats_params

        else:
            return etaPp, MPQ, None, None, None

_extract_vecs = jit(_extract_vecs, static_argnums=(1,2,3,4))

def _BLISS_corrections(avec, bvec, beta_mats_params, Norbs, ob_sym_mats, ob_sym_vals, num_ob_syms):
    beta_mats = jnp.zeros((num_ob_syms,Norbs,Norbs))
    for i1 in range(num_ob_syms):
        beta_mats = beta_mats.at[i1].set(_ob_params_to_mat(beta_mats_params[i1,:], Norbs))

    killer_eri = jnp.zeros((Norbs,Norbs,Norbs,Norbs))
    idx = 0
    for i1 in range(num_ob_syms):
        for i2 in range(i1+1):
            bvec_val = bvec[idx]
            killer_eri += bvec_val * jnp.einsum("pq,rs->pqrs", ob_sym_mats[i1], ob_sym_mats[i2])
            killer_eri += bvec_val * jnp.einsum("pq,rs->pqrs", ob_sym_mats[i2], ob_sym_mats[i1])

            idx += 1

        killer_eri += jnp.einsum("pq,rs->pqrs", beta_mats[i1], ob_sym_mats[i1])
        killer_eri += jnp.einsum("pq,rs->pqrs", ob_sym_mats[i1], beta_mats[i1])

    killer_obt = jnp.zeros((Norbs,Norbs))
    idx = 0
    for i1 in range(num_ob_syms):
        killer_obt += avec[i1] * ob_sym_mats[i1]
        killer_obt -= ob_sym_vals[i1]*beta_mats[i1]
        
        for i2 in range(i1+1):
            killer_obt -= bvec[idx]*(ob_sym_vals[i2]*ob_sym_mats[i1] + ob_sym_vals[i1]*ob_sym_mats[i2])

            idx += 1

    return killer_obt, killer_eri

_BLISS_corrections = jit(_BLISS_corrections, static_argnums=(3,6))


def _thc_one_norm(kappa, MPQ, kappa_is_none):
    if kappa_is_none:
        lambda_1 = 0.0
    else:
        D = jnp.linalg.eigvalsh(kappa)
        lambda_1 = jnp.sum(jnp.abs(D))

    lambda_2 = 0.5 * jnp.sum(jnp.abs(MPQ))
    Nthc = MPQ.shape[0]
    for mm in range(Nthc):
        lambda_2 -= 0.25 * jnp.abs(MPQ[mm,mm])

    return lambda_1 + lambda_2

_thc_one_norm = jit(_thc_one_norm, static_argnums=(2))


def _vec_to_one_norm(x_vec, obt_full, eri_full, ob_sym_mats, ob_sym_vals, Nthc, Norbs, num_ob_syms, include_bliss, obt_is_none):
    etaPp, MPQ, avec, bvec, beta_mats_params = _extract_vecs(x_vec, Norbs, Nthc, num_ob_syms, include_bliss)
    eri_thc = _unfold_thc(MPQ, etaPp)

    if include_bliss:
        obt_killer, eri_killer = _BLISS_corrections(avec, bvec, beta_mats_params, Norbs, ob_sym_mats, ob_sym_vals, num_ob_syms)
        eri_BI = eri_full - eri_killer
        if obt_is_none:
            obt_BI = None
        else:
            obt_BI = obt_full - obt_killer
    else:
        eri_BI = eri_full
        obt_BI = obt_full

    if obt_is_none:
        kappa_BI = None
    else:
        kappa_BI = obt_BI + jnp.einsum("pqrr->pq", eri_BI)

    return _thc_one_norm(kappa_BI, MPQ, obt_is_none)

_vec_to_one_norm = jit(_vec_to_one_norm, static_argnums=(5,6,7,8,9))


def _cost(x_vec, obt_full, eri_full, ob_sym_mats, ob_sym_vals, Nthc, Norbs, num_ob_syms, include_bliss, obt_is_none, rho, regularize=True):
    etaPp, MPQ, avec, bvec, beta_mats_params = _extract_vecs(x_vec, Norbs, Nthc, num_ob_syms, include_bliss)
    eri_thc = _unfold_thc(MPQ, etaPp)

    if include_bliss:
        obt_killer, eri_killer = _BLISS_corrections(avec, bvec, beta_mats_params, Norbs, ob_sym_mats, ob_sym_vals, num_ob_syms)
        eri_BI = eri_full - eri_killer
        if obt_is_none:
            kappa_BI = None
        else:
            kappa_BI = obt_full - obt_killer + jnp.einsum("pqrr->pq", eri_BI)
    else:
        eri_BI = eri_full
        if obt_is_none:
            kappa_BI = None
        else:
            kappa_BI = obt_full + jnp.einsum("pqrr->pq", eri_BI)

    eri_diff = eri_BI - eri_thc

    tot_cost = 0.5 * jnp.sum(eri_diff**2)

    if regularize:
        tot_cost += rho * _thc_one_norm(kappa_BI, MPQ, obt_is_none)

    return tot_cost

_cost = jit(_cost, static_argnums=(5,6,7,8,9,11))

def _initialize_params(initial_guess, Nthc, Norbs, include_bliss, num_ob_syms, random_seed=42, norm_factor=0.01, iters=1000):
    """Initialize optimization parameters."""
    if include_bliss:
        avec_len, bvec_len, ob_mat_num_params = _get_BLISS_sizes(num_ob_syms, Norbs)

    if initial_guess is None:
        key = jax.random.PRNGKey(random_seed if random_seed is not None else 0)
        key1, key2, key5 = jax.random.split(key, 3)
        
        params = {
            'etaPp': 10*norm_factor * jax.random.normal(key1, (Nthc, Norbs)),
            'MPQ': 10*norm_factor * jax.random.normal(key2, (Nthc, Nthc)),
        }
        
        if include_bliss:
            params["avec"] = jnp.zeros(avec_len)
            params["bvec"] = jnp.zeros(bvec_len)
            params["beta_mats_params"] = norm_factor * jax.random.normal(key5, (num_ob_syms, ob_mat_num_params))

    else:
        params = {k: jnp.array(v) for k, v in initial_guess.items()}
        
        if include_bliss:
            params.setdefault('avec', jnp.zeros(avec_len))
            params.setdefault('bvec', jnp.zeros(bvec_len))
            params.setdefault('beta_mats_params', jnp.zeros((num_ob_syms, ob_mat_num_params)))

    return params

def _compute_penalty_param_enhanced(eri, obt, ob_sym_list, Nthc, initial_guess, maxiter):
    """Compute penalty parameter by using un-regularized one-norm after maxiter iterations of optax optimizer"""
    params, _ = get_thc(eri, obt, ob_sym_list, Nthc, initial_guess=initial_guess, regularize=False, maxiter=maxiter, verbose=False)

    etaPp = params['etaPp']
    MPQ = params['MPQ']
    norb = eri.shape[0]
    
    deri = eri - _unfold_thc(MPQ, etaPp)
    sum_square_loss = 0.5 * jnp.sum(deri**2)
    regularization_scale = jnp.sum(jnp.abs(MPQ))
    
    # Avoid division by zero
    if regularization_scale < 1e-12:
        return 1e-6
    
    return float(sum_square_loss / regularization_scale), params


def get_thc(eri, obt=None, ob_sym_list=[], Nthc=None, regularize=True, maxiter=10000, initial_guess=None, learning_rate = 7.5e-3, verbose=True):
    Norbs = eri.shape[0]

    if Nthc is None:
        Nthc = int(np.ceil(3*Norbs))
        if verbose:
            print(f"Using default THC rank of ceil(3*num_orbs) = {Nthc}")

    num_ob_syms = len(ob_sym_list)
    avec_len = num_ob_syms
    bvec_len = int(num_ob_syms * (num_ob_syms+1)/2)
    ob_mat_num_params = int(Norbs*(Norbs+1)/2)
    beta_params_len = num_ob_syms * ob_mat_num_params

    if obt is None:
        obt_is_none = True
    else:
        obt_is_none = False

    if num_ob_syms > 0:
        include_bliss = True
    else:
        include_bliss = False

    ob_sym_mats = jnp.array([ob_sym_list[kk][0] for kk in range(num_ob_syms)])
    ob_sym_vals = jnp.array([ob_sym_list[kk][1] for kk in range(num_ob_syms)])

    if verbose and include_bliss:
        print(f"Found {num_ob_syms} one-body symmetries for BLISS terms")
        print(f"Total number of BLISS parameters to be optimized: {avec_len+bvec_len+beta_params_len}, composed of:")
        print(f"    - {avec_len} one-body scalars")
        print(f"    - {bvec_len} two-body scalars")
        print(f"    - {num_ob_syms} one-body matrices, each with {ob_mat_num_params} free variables")

    params = _initialize_params(initial_guess, Nthc, Norbs, include_bliss, num_ob_syms)
    if regularize is True:
        rho, params = _compute_penalty_param_enhanced(eri, obt, ob_sym_list, Nthc, params, int(np.ceil(maxiter/10)))
        if include_bliss:
            rho *= 2 / Norbs

        if verbose:
            print(f"Found regularization parameter rho={rho:.2e}")
    elif regularize is False or regularize is None:
        rho = 0
        if verbose:
            print(f"No regularization: setting rho=0")
    else:
        rho = regularize
        regularize=True
        if verbose:
            print(f"Regularization found: setting rho={rho:.2e}")

    def pack_dict(x_vec):
        eta_fin = Nthc*Norbs
        my_dict = {"etaPp" : x_vec[:eta_fin].reshape((Nthc, Norbs))}

        M_fin = eta_fin + Nthc**2
        my_dict["MPQ"] = x_vec[eta_fin:M_fin].reshape((Nthc, Nthc))

        if include_bliss:
            a_fin = M_fin + avec_len
            my_dict["avec"] = x_vec[M_fin:a_fin]

            b_fin = a_fin + bvec_len
            my_dict["bvec"] = x_vec[a_fin:b_fin]

            my_dict["beta_mats_params"] = x_vec[b_fin:].reshape((num_ob_syms, ob_mat_num_params))

        return my_dict

    def unpack_dict(my_dict):
        num_vars = Nthc * Norbs + (Nthc**2)
        if include_bliss:
            num_vars += avec_len + bvec_len + beta_params_len

        x_vec = np.zeros(num_vars)

        eta_fin = Nthc*Norbs
        x_vec[:eta_fin] = my_dict["etaPp"].flatten()

        M_fin = eta_fin + Nthc**2
        x_vec[eta_fin:M_fin] = my_dict["MPQ"].flatten()

        if include_bliss:
            a_fin = M_fin + avec_len
            x_vec[M_fin:a_fin] = my_dict["avec"]

            b_fin = a_fin + bvec_len
            x_vec[a_fin:b_fin] = my_dict["bvec"]

            x_vec[b_fin:] = my_dict["beta_mats_params"].flatten()

        return jnp.array(x_vec)

    @jit
    def cost_flat(x_vec):
        return _cost(x_vec, obt, eri, ob_sym_mats, ob_sym_vals, Nthc, Norbs, num_ob_syms, include_bliss, obt_is_none, rho, regularize)


    optimizer = optax.adam(learning_rate)
    x0 = unpack_dict(params)
    opt_state = optimizer.init(x0)

    @jit
    def update_step(x_vec, opt_state):
        loss, grads = jax.value_and_grad(cost_flat)(x_vec)
        updates, opt_state = optimizer.update(grads, opt_state, x_vec)
        x_vec = optax.apply_updates(x_vec, updates)
        return x_vec, opt_state, loss

    # Optimization loop
    losses = []
    for i in range(maxiter):
        x0, opt_state, loss = update_step(x0, opt_state)
        losses.append(float(loss))
        
        if verbose > 1 and i % 1000 == 0:
            print(f"Iteration {i}: Loss = {loss:.6e}")
        
        # Simple convergence check
        if i > 10 and abs(losses[-1] - losses[-2]) < 1e-12:
            if verbose > 1:
                print(f"Converged at iteration {i}")
            break

    final_params = pack_dict(x0)
    L2_cost = _cost(x0, obt, eri, ob_sym_mats, ob_sym_vals, Nthc, Norbs, num_ob_syms, include_bliss, obt_is_none, 0, False)
    lam = _vec_to_one_norm(x0, obt, eri, ob_sym_mats, ob_sym_vals, Nthc, Norbs, num_ob_syms, include_bliss, obt_is_none)

    if verbose:
        print(f"Initial 2-norm is {float(jnp.sqrt(jnp.sum(eri**2))):.2e}")
        print(f"Finished THC factorization! Final 2-norm of difference is {np.sqrt(L2_cost):.2e}, 1-norm is {lam:.2f}")
        if obt_is_none:
            print(f"Note that one-norm does not include one-body component!")

        if include_bliss:
            print(f"BLISS included during optimization using {num_ob_syms} one-body symmetries")

    return final_params, lam