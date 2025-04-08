import numpy as np
from jax import random
from jax import jit, numpy as jnp, value_and_grad, vmap
from jax.scipy.linalg import expm
from jax.example_libraries.optimizers import adam
from jax import jit, numpy as jnp, value_and_grad, grad, vmap
from jax.lax import stop_gradient
from functools import partial
from tqdm import tqdm


def factorize_onebody(h1e):
    """
    Obtain the matrices U (special orthogonal) and Z (symmetric) 
    for the one-electron integrals. 
    See doi:10.1103/PRXQuantum.2.040352 for details.
    """
    # Diagonalize the one-electron integral matrix
    eigenvals, eigenvecs = np.linalg.eigh(h1e)
    U = eigenvecs
    Z = np.diag(eigenvals)
    reconstructed_h1e = U @ Z @ U.T

    # Check if the original and the reconstructed matrix are the same
    print(f'One body difference {h1e - reconstructed_h1e}')

    UZ = np.stack((U, Z), axis = 0)

    return UZ

def one_body_correction(U, Z):
    """
    Obtain the one-body correction to the Hamiltonian given the 
    factorized form of the one-electron integrals.
    """
    Z_prime = np.stack([np.diag(np.sum(Z[i], axis = -1)) for i in range(Z.shape[0])], axis = 0)
    h1e_correction = np.einsum('tpk,tkk,tqk->pq', U, Z_prime, U)

    return h1e_correction

def generate_symmetric(pts, n_cdf, norb):
    """
    Generates a random-normal symmetric matrix -- acts as seed for 
    initialization of parameter values for symmetric Z matrices 
    in double factorization.

    Args:
        pts (int): parallel optimizations to run (???)
        n_cdf (int): Number of terms in the CDF
        norb (int): Number of molecular orbitals in the Hamiltonian
    """
    key = random.PRNGKey(758493)  # Random seed is explicit in JAX

    # Generate a random square matrix: key == random seed
    A = jnp.array(random.normal(key, (pts, n_cdf, norb, norb)), dtype=jnp.float64)

    # Add the random matrix and its conjugate transpose and divide by 2
    H = (A + jnp.transpose(A, (0, 1, 3, 2))) / 2

    return H

def generate_antisymmetric(pts, n_cdf, norb):
    """
    Generates a random-normal antisymmetric matrix X -- acts as seed for 
    initialization of parameter values for special orthogonal U=exp(X) 
    matrices in double factorization.

    Args:
        pts (int): parallel optimizations to run (???)
        n_cdf (int): Number of terms in the CDF
        norb (int): Number of molecular orbitals in the Hamiltonian
    """
    key = random.PRNGKey(758493)  # Random seed is explicit in JAX

    # Generate a random square matrix
    A = jnp.array(random.normal(key, (pts, n_cdf, norb, norb)), dtype=jnp.float64)

    # Subtract the random matrix from its conjugate transpose and divide by 2
    AH = (A - jnp.transpose(A, (0, 1, 3, 2))) / 2

    return AH

def cdf_cost(params, eri, prior_X, prior_Z):
    """
    For given matrices U=exp(X) and Z, evaluate the Frobenius norm
    against the true two-electron integral tensor. Serves as a cost 
    function to minimize. 
    """
    N = eri.shape[0]

    M = params['X'].shape[1] - N
    # Increase the size of the matrices to N+M
    pad_width = ((0, M), (0, M), (0, M), (0, M))
    eri = jnp.pad(eri, pad_width, mode='constant', constant_values=0)

    X, Z, k2, F = params['X'], params['Z'], params['k2'], params['F']

    X = jnp.concatenate((prior_X, X), axis = 0)
    Z = jnp.concatenate((prior_Z, Z), axis = 0)

    Z = (Z + jnp.transpose(Z, (0, 2, 1))) / 2.
    X = (X - jnp.transpose(X, (0, 2, 1))) / 2.
    U = expm(X)

    cdf_eri = jnp.einsum('tpk,tqk,tkl,trl,tsl->pqrs', U, U, Z, U, U)
    
    N1 = jnp.eye(N+M)
    N2 = jnp.einsum('pq,rs->pqrs', N1, N1)
    T = jnp.einsum('pq,rs->pqrs', F, N1)/2. + jnp.einsum('pq,rs->pqrs', N1, F)/2.
    cdf_eri = cdf_eri + k2*N2 + T

    cdf_eri = cdf_eri.reshape((N+M)**2,(N+M)**2)
    eri = eri.reshape((N+M)**2,(N+M)**2)

    cost = jnp.linalg.norm(eri - cdf_eri)

    return cost

def optimization(params, cost_f, n_steps, learnrate, eri, \
                prior_X, prior_Z, verbose = False, bliss = True, 
                cdf = True):
    """
    Perform optimization using the Adam optimizer for n_steps.
    The variables being prior_ are the antisymmetric X such that U=exp(X),
    and the symmetric Z.
    """

    opt_init, opt_update, get_params = adam(learnrate)
    opt_state = opt_init(params)

    cost_f = partial(cost_f, eri = eri, prior_X = prior_X, prior_Z = prior_Z)

    prior_cost = jnp.inf
    cost_f_value_and_grad = vmap(jit(value_and_grad(cost_f)), in_axes = 0)
    new_mins = 0

    # run optimization loop to minimize loss
    for i in tqdm(range(n_steps), \
                    desc=f'Number of steps for learning rate = {learnrate}'):
        params = get_params(opt_state)
        params['Z'] = (params['Z'] + jnp.transpose(params['Z'], (0, 1, 3, 2))) / 2.
        params['X'] = (params['X'] - jnp.transpose(params['X'], (0, 1, 3, 2))) / 2.
        cost, grads = cost_f_value_and_grad(params)
        if not bliss:
            grads['k2'] = grads['k2']*0.
            grads['F'] = grads['F']*0.
        if not cdf:
            grads['X'] = grads['X']*0.
            grads['Z'] = grads['Z']*0.

        if jnp.min(cost) < prior_cost:
            #XZ_min = XZ[:, jnp.argmin(cost)]
            params_min = {'X': params['X'][jnp.argmin(cost)], 'Z': params['Z'][jnp.argmin(cost)],
                    'k2': params['k2'][jnp.argmin(cost)], 'F': params['F'][jnp.argmin(cost)]}
            prior_cost = jnp.min(cost)
            if verbose:
                print(f"\nnew prior_ cost at step {i},"
                    f" learning rate {learnrate}", prior_cost)
            new_mins += 1
        opt_state = opt_update(i, grads, opt_state)
        if i % 100 == 0 and verbose:
            print(f"\nat iteration ={i}, for {len(cost)}"
                f" parallel runs\ncost= {cost}")

    return params_min, prior_cost, params, new_mins

def optimization_schedule_cdf(pts, n_cdf, eri,
                                prior_X, prior_Z, k2, F, additional_cdf_terms = 1,
                                bliss = True, cdf = True,
                                verbose=False, n_steps = 3000):
    """
    Concrete optimization schedule -- three-step with cascading 
    learning rates.
    """
    raise NotImplementedError("This function is deprecated. Use optimization_schedule instead.")

    if len(eri.shape) == 2:
        norb = int(np.sqrt(eri.shape[0]))
    elif len(eri.shape) == 4:
        norb = eri.shape[0]

    # generate initial (random) guess, adding layers one by one
    X = generate_antisymmetric(pts, additional_cdf_terms, norb)
    Z = generate_symmetric(pts, additional_cdf_terms, norb)
    params = {'X': X, 'Z': Z, 'k2': k2, 'F': F}

    _, _, params, new_mins1 = \
        optimization(params, cdf_cost, n_steps, 3e-3, eri, \
                    prior_X = prior_X, prior_Z = prior_Z,
                    verbose=verbose, bliss = bliss, cdf = cdf)
    _, _, params, new_mins2 = \
        optimization(params, cdf_cost, n_steps, 1e-3, eri, \
                    prior_X = prior_X, prior_Z = prior_Z,
                    verbose=verbose, bliss = bliss, cdf = cdf)
    params_min, opt_cost, params, new_mins3 = \
        optimization(params, cdf_cost, n_steps, 3e-4, eri, \
                    prior_X = prior_X, prior_Z = prior_Z,
                    verbose=verbose, bliss = bliss, cdf = cdf)

    print(f'n_cdf: {n_cdf}, prior_ cost: {opt_cost}')
    print(f"New mins obtained at time steps:"
            f" {new_mins1}, {new_mins2}, {new_mins3}\n")

    # extract prior_ final results
    optX, optZ, optk2, optF = params_min['X'], params_min['Z'], \
                            params_min['k2'], params_min['F']
    # check symmetry / antisymmetry
    print(jnp.allclose(optZ, (optZ + jnp.transpose(optZ, (0, 2, 1))) / 2.))
    print(jnp.allclose(optX, (optX - jnp.transpose(optX, (0, 2, 1))) / 2.))

    # if prior_ results have already been found, re-use
    optZ = jnp.concatenate((prior_Z, optZ), axis = 0)
    optX = jnp.concatenate((prior_X, optX), axis = 0)

    # check symmetry / antisymmetry
    print(jnp.allclose(optZ, (optZ + jnp.transpose(optZ, (0, 2, 1))) / 2.))
    print(jnp.allclose(optX, (optX - jnp.transpose(optX, (0, 2, 1))) / 2.))
    optU = expm(optX)

    # compute the predicted two-electron integrals from the factorized form
    print(f"Performing contraction to obtain two-electron integrals")
    cdf_eri = np.einsum('tpk,tqk,tkl,trl,tsl->pqrs', \
                        optU, optU, optZ, optU, optU)

    cdf_eri = cdf_eri.reshape(norb**2,norb**2)

    N1 = jnp.eye(norb)
    N2 = jnp.einsum('pq,rs->pqrs', N1, N1)
    T = jnp.einsum('pq,rs->pqrs', optF, N1)/2. + jnp.einsum('pq,rs->pqrs', N1, optF)/2.

    bliss_eri = eri - optk2*N2 - T
    bliss_eri = bliss_eri.reshape(norb**2,norb**2)
    cost = np.linalg.norm(cdf_eri-bliss_eri)
    print(f"Contraction finished, error {cost} obtained\n")

    return cdf_eri, optX, optZ, optk2, optF

def minimize_k1(k1, h1e, nelec, F, eri, n_steps = 500, commutator = True, verbose = False):
    # Definir la función de pérdida
    N1 = jnp.eye(h1e.shape[0])

    def norm_loss(k1):
        return jnp.linalg.norm(h1e + nelec * F / 2. - k1 * N1)
    
    def commutator_loss(k1):
        opt_h1e = h1e + nelec * F / 2. - k1 * N1
        opt_h1e = jnp.einsum('pq,rs->pqrs', opt_h1e, N1) / 2. \
                    + jnp.einsum('pq,rs->pqrs', N1, opt_h1e) / 2.
        return jnp.sqrt(jnp.sum(jnp.square(
                    jnp.einsum('ijkl,pqmn->ijklpqmn', eri, opt_h1e) - \
                    jnp.einsum('ijkl,pqmn->ijklpqmn', opt_h1e, eri))))
    
    if commutator:
        cost_fn = jit(value_and_grad(commutator_loss))
    else:
        cost_fn = jit(value_and_grad(norm_loss))

    opt_init, opt_update, get_params = adam(step_size=0.01)
    opt_state = opt_init(k1)  # Valor inicial de k1

    def update(step, opt_state):
        k1 = get_params(opt_state)
        loss, grads = cost_fn(k1)
        return opt_update(step, grads, opt_state), loss

    for step in tqdm(range(n_steps), desc="Minimizing one body error"):
        opt_state, loss = update(step, opt_state)
        if step % 100 == 0 and verbose:
            print(f"Step {step}, Loss: {loss}")
    
    print(f"One body loss: {loss}")

    return get_params(opt_state)

def optimization_schedule(pts, n_cdf, eri,
                                prior_X, prior_Z, k2, F, additional_cdf_terms = 1,
                                bliss = True, cdf = True, M = 0,
                                verbose=False, n_steps = 3000):
    """
    Concrete optimization schedule -- three-step with cascading 
    learning rates.
    """

    N = eri.shape[0]

    # generate initial (random) guess, adding layers one by one
    X = generate_antisymmetric(pts, additional_cdf_terms, N)
    Z = generate_symmetric(pts, additional_cdf_terms, N)
    # Increase the size of the matrices to N+M
    pad_width = ((0, 0), (0, 0), (0, M), (0, M))
    X = jnp.pad(X, pad_width, mode='constant', constant_values=0)
    Z = jnp.pad(Z, pad_width, mode='constant', constant_values=0)

    pad_width = ((0, 0), (0, M), (0, M))
    prior_X = jnp.pad(prior_X, pad_width, mode='constant', constant_values=0)
    prior_Z = jnp.pad(prior_Z, pad_width, mode='constant', constant_values=0)
    F = jnp.pad(F, pad_width, mode='constant', constant_values=0)

    params = {'X': X, 'Z': Z, 'k2': k2, 'F': F}

    _, _, params, new_mins1 = \
        optimization(params, cdf_cost, n_steps, 3e-3, eri, \
                    prior_X = prior_X, prior_Z = prior_Z,
                    verbose=verbose, bliss = bliss, cdf = cdf)
    _, _, params, new_mins2 = \
        optimization(params, cdf_cost, n_steps, 1e-3, eri, \
                    prior_X = prior_X, prior_Z = prior_Z,
                    verbose=verbose, bliss = bliss, cdf = cdf)
    params_min, opt_cost, params, new_mins3 = \
        optimization(params, cdf_cost, n_steps, 3e-4, eri, \
                    prior_X = prior_X, prior_Z = prior_Z,
                    verbose=verbose, bliss = bliss, cdf = cdf)

    print(f'n_cdf: {n_cdf}, prior_ cost: {opt_cost}')
    print(f"New mins obtained at time steps:"
            f" {new_mins1}, {new_mins2}, {new_mins3}\n")

    # extract prior_ final results
    optX, optZ, optk2, optF = params_min['X'], params_min['Z'], \
                            params_min['k2'], params_min['F']
    # check symmetry / antisymmetry
    print(jnp.allclose(optZ, (optZ + jnp.transpose(optZ, (0, 2, 1))) / 2.))
    print(jnp.allclose(optX, (optX - jnp.transpose(optX, (0, 2, 1))) / 2.))

    # if prior_ results have already been found, re-use
    optZ = jnp.concatenate((prior_Z, optZ), axis = 0)
    optX = jnp.concatenate((prior_X, optX), axis = 0)

    # check symmetry / antisymmetry
    print(jnp.allclose(optZ, (optZ + jnp.transpose(optZ, (0, 2, 1))) / 2.))
    print(jnp.allclose(optX, (optX - jnp.transpose(optX, (0, 2, 1))) / 2.))
    optU = expm(optX)

    # compute the predicted two-electron integrals from the factorized form
    print(f"Performing contraction to obtain two-electron integrals")
    cdf_eri = np.einsum('tpk,tqk,tkl,trl,tsl->pqrs', \
                        optU, optU, optZ, optU, optU)

    N1 = jnp.eye(N)
    N2 = jnp.einsum('pq,rs->pqrs', N1, N1)
    T = jnp.einsum('pq,rs->pqrs', optF, N1)/2. + jnp.einsum('pq,rs->pqrs', N1, optF)/2.

    cdf_eri = eri + optk2*N2 + T

    cdf_eri = cdf_eri.reshape((N+M)**2,(N+M)**2)
    eri = eri.reshape((N+M)**2,(N+M)**2)

    cost = np.linalg.norm(cdf_eri-eri)
    print(f"Contraction finished, error {cost} obtained\n")

    return cdf_eri, optX, optZ, optk2, optF
