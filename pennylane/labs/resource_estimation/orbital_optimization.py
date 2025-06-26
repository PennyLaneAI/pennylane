import optax
import jax
from jax import jit
from jax import numpy as jnp
from jax import scipy as jsp
from jax import value_and_grad

def ob_correction(tbt, spin_orb=False):
    #returns correction to one-body tensor coming from tbt inside fermionic operator F
    if spin_orb:
        spin_fact = 1/2
        print("Obtaining one-body correction for spin-orbitals, be wary of workflow!")
    else:
        spin_fact = 1
    obt_corr = spin_fact * jnp.einsum('ijkk->ij', tbt)
    
    return obt_corr

def pauli_1_norm(obt, tbt, split_spin=True):
    '''
    Formally 1-norm of 2-body tensor has different contributions from spin sectors (split_spin=True)
    In practice we usually implement the LCU without splitting the spin since the PREP cost is way lower 
    '''
    N = obt.shape[0]
    lambda_1 = jnp.sum(jnp.abs(obt + ob_correction(tbt)))
    
    if split_spin:
        lambda_2 = 0.25 * jnp.sum(jnp.abs(tbt))
        for r in range(N):
            for p in range(r+1,N):
                for q in range(N):
                    for s in range(q+1,N):
                        lambda_2 += 0.5 * jnp.abs(tbt[p,q,r,s] - tbt[p,s,r,q])
    else:
        lambda_2 = 0.5 * jnp.sum(jnp.abs(tbt))


    return lambda_1+lambda_2

def rotate_tbt(tbt, Urot):
	return jnp.einsum('ijkl,ai,bj,ck,dl->abcd', tbt, Urot, Urot, Urot, Urot)

def rotate_obt(obt, Urot):
	return jnp.einsum('ij,ai,bj->ab', obt, Urot, Urot)

def reduced_one_norm(obt, tbt, Urot, mixing):
	tbt_rot_one_norm = jnp.einsum('ijkl,ai,bj,ck,cl->ab', tbt, Urot, Urot, Urot, Urot)
	tbt_lambda = jnp.sum(jnp.abs(tbt_rot_one_norm))

	obt_lambda = jnp.sum(jnp.abs(rotate_obt(obt, Urot)))

	return mixing*obt_lambda + (1-mixing)*tbt_lambda 

def params_to_Urot(params, N):
	G = jnp.zeros((N, N))
	idx = jnp.triu_indices(N, k=1)
	G = G.at[idx].set(params)
	G = G.at[(idx[1], idx[0])].set(-params)
	return jsp.linalg.expm(G)

def params_to_cost(params, obt, tbt, N, mixing):
	Urot = params_to_Urot(params, N)

	return reduced_one_norm(obt, tbt, Urot, mixing)

def rot_one_norm(obt, tbt, params):
	N = obt.shape[0]
	Urot = params_to_Urot(params, N)
	obt_rot = rotate_obt(obt, Urot)
	tbt_rot = rotate_tbt(tbt, Urot)

	return pauli_1_norm(obt_rot, tbt_rot)

def optimize_params(obt, tbt, mixing=0.1, num_steps=1000, learning_rate=1e-2, seed=0, verbose=False):
	# Initialize parameters (for simplicity, a single scalar here)
	N = tbt.shape[0]
	num_params = int(N*(N-1)/2)
	key = jax.random.PRNGKey(seed)
	params = jax.random.normal(key, (num_params))  # shape () for scalar; adjust as needed

	# Define optimizer
	optimizer = optax.adam(learning_rate)
	opt_state = optimizer.init(params)

	# JIT-compiled step function
	@jit
	def step(params, opt_state):
		cost, grads = value_and_grad(params_to_cost)(params, obt, tbt, N, mixing)
		updates, opt_state = optimizer.update(grads, opt_state)
		params = optax.apply_updates(params, updates)
		return params, opt_state, cost

	# Optimization loop
	for i in range(num_steps):
		params, opt_state, cost = step(params, opt_state)
		if verbose:
			if i % 100 == 0:
				print(f"Step {i}, Cost: {cost}")
				print(f"One norm cost is {rot_one_norm(obt, tbt, params)}")

	if verbose:
		print(f"Initial one-norm was {pauli_1_norm(obt, tbt):.2e}")
		print(f"Final one-norm was {rot_one_norm(obt, tbt, params):.2e}")

	Urot = params_to_Urot(params, N)

	return rotate_obt(obt, Urot), rotate_tbt(tbt, Urot)