from jax._src.typing import Array
import cvxpy as cp
from functools import partial
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

def gaussian_kernel(sigma: float, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Calculates the value for the gaussian kernel between two vectors x, y

    Args:
        sigma (float): sigma parameter, the width of the kernel
        x (jnp.ndarray): one of the vectors
        y (jnp.ndarray): the other vector

    Returns:
        float: Result value of the gaussian kernel
    """
    return jnp.exp(-((x-y)**2).sum()/2/sigma**2)

def loss_estimate_iqp(params: jnp.ndarray, iqp_circuit: IqpSimulator, ground_truth: jnp.ndarray, visible_ops: jnp.ndarray,
                      all_ops: jnp.ndarray, n_samples: int, key: Array, init_state: list = None, init_coefs: list = None, indep_estimates: bool = False, sqrt_loss: bool = False,
                      return_expvals: bool = False, max_batch_ops: int = None, max_batch_samples: int = None) -> float:
    """Estimates the MMD Loss of an IQP circuit with respect to a ground truth distribution.

    Args:
        params (jnp.ndarray): The parameters of the IQP gates.
        iqp_circuit (IqpSimulator): The IQP circuit itself given by the class IqpSimulator.
        ground_truth (jnp.ndarray): Matrix with the training samples as rows (0s and 1s).
        visible_ops (jnp.ndarray): Matrix with only the visible operators as rows (0s and 1s). Used to estimate the training part of the loss.
        all_ops (jnp.ndarray): Matrix with the all the operators as rows (0s and 1s). Used to estimate the IQP part of the loss.
        n_samples (jnp.ndarray): Number of samples used to estimate the loss.
        key (Array): Jax key to control the randomness of the process.
        init_state (list[jnp.ndarray]): a list [X,P] where X is a bitstring array containing the nonzero basis
                elements that appear in the state, and P lists their corresponding amplitudes.
        init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
        indep_estimates (bool, optional): Whether to use independent estimates of the ops in a batch (takes longer). Defaults to False.
        sqrt_loss (bool, optional): Whether to use the square root of the MMD^2 loss. Note estiamtes will no longer be unbiased. Defaults to False.
        return_expvals (bool, optional): Whether to return the expectation values of the IQP circuit or return the loss. Defaults to False.
        max_batch_ops (int, optional): Maximum number of operators in a batch of op_expval. Defaults to None.
        max_batch_samples (int, optional): Maximum number of samples in a batch of op_expval. Defaults to None.

    Returns:
        float: The value of the loss.
    """
    tr_iqp_samples = iqp_circuit.op_expval(params, all_ops, n_samples, key, init_state, init_coefs, indep_estimates, return_samples=True,
                                           max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples)
    correction = jnp.mean(tr_iqp_samples**2, axis=-1)/n_samples
    tr_iqp = jnp.mean(tr_iqp_samples, axis=-1)
    tr_train = jnp.mean(1-2*((ground_truth @ visible_ops.T) % 2), axis=0)
    m = len(ground_truth)

    if iqp_circuit.bitflip:
        res = tr_iqp*tr_iqp - 2*tr_iqp*tr_train + (tr_train*tr_train*m-1)/(m-1)
    else:
        # add correction to make the first term unbiased
        res = (tr_iqp*tr_iqp-correction)*n_samples/(n_samples-1) - \
            2*tr_iqp*tr_train + (tr_train*tr_train*m-1)/(m-1)

    res = jnp.mean(res) if not return_expvals else res
    res = jnp.sqrt(jnp.abs(res)) if sqrt_loss else res

    return res


def mmd_loss_iqp(params: jnp.ndarray, iqp_circuit: IqpSimulator, ground_truth: jnp.ndarray, sigma: float or list, n_ops: int,
                 n_samples: int, key: Array, init_state: list = None, init_coefs: list = None, wires: list = None, indep_estimates: bool = False, jit: bool = True,
                 sqrt_loss: bool = False, return_expvals: bool = False, max_batch_ops: int = None, max_batch_samples: int = None) -> float:
    """Returns an estimate of the (squared) MMD Loss of an IQP circuit with respect to a ground truth
     distribution. Requires a set of samples from the ground truth distribution. The estimate is unbiased in the sense
     that the expectation of the estimator wrt samples from the ground truth is the exact (squared) MMD loss. The kernel
     used is the Gaussian kernel with bandwidth specified by sigma.

     The function uses a randomized method whose precision can be increased by using larger values of n_samples and/or
     n_ops.

    Args:
        params (jnp.ndarray): The parameters of the IQP gates.
        iqp_circuit (IqpSimulator): The IQP circuit given as a IqpSimulator object.
        ground_truth (jnp.ndarray): Array containing the samples from the ground truth distribution as rows (0s and 1s).
        sigma (float or list): The bandwidth of the kernel. If several are given as a list the average loss over each value will
            be returned.
        n_ops (int): Number of operators used to estimate the loss.
        n_samples (jnp.ndarray): Number of samples used to estimate the loss.
        key (jax.random.PRNGKey): Jax PRNG key used to seed random functions.
        init_state (list[jnp.ndarray]): a list [X,P] where X is a bitstring array containing the nonzero basis
                elements that appear in the state, and P lists their corresponding amplitudes.
        init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
        wires (list, optional): List of qubit positions that specifies the qubits whose measurement statistics are
            used to estimate the MMD loss. The remaining qubits will be traced out. Defaults to None, meaning all
            qubits are used.
        indep_estimates (bool): Whether to use independent estimates when estimating expvals of ops (takes longer).
        jit (bool): Whether to jit the loss (works only for circuits with sparse=False). Defaults to True.
        sqrt_loss (bool): Whether to use the square root of the MMD^2 loss. Note estimates will no longer be unbiased.
            Defaults to False.
        return_expvals (bool): If True, the expectation values of the IQP circuit used to estimate the loss are
            returned. Defaults to False.
        max_batch_ops (int): Maximum number of operators in a batch of op_expval. Defaults to None.
        max_batch_samples (int): Maximum number of samples in a batch of op_expval. Defaults to None.

    Returns:
        float: The value of the loss.
    """

    sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
    init_coefs = jnp.array(init_coefs) if init_coefs is not None else None

    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")

    if wires is None:
        wires = list(range(iqp_circuit.n_qubits))

    losses = []
    for sigma in sigmas:
        p_MMD = (1-jnp.exp(-1/2/sigma**2))/2
        key, subkey = jax.random.split(key, 2)
        visible_ops = jnp.array(jax.random.binomial(
            subkey, 1, p_MMD, shape=(n_ops, len(wires))), dtype='float64')

        all_ops = []
        i = 0
        for q in range(iqp_circuit.n_qubits):
            if q in wires:
                all_ops.append(visible_ops[:, i])
                i += 1
            else:
                all_ops.append(jnp.zeros(n_ops))
        all_ops = jnp.array(all_ops, dtype='float64').T

        if iqp_circuit.sparse:
            loss = loss_estimate_iqp
        else:
            if jit:
                loss = jax.jit(loss_estimate_iqp, static_argnames=[
                               "iqp_circuit", "n_samples", "indep_estimates", "sqrt_loss", "return_expvals", "max_batch_ops", "max_batch_samples"])
            else:
                loss = loss_estimate_iqp

        losses.append(loss(params, iqp_circuit, ground_truth, visible_ops, all_ops, n_samples, key, init_state, init_coefs, indep_estimates,
                      sqrt_loss, return_expvals=return_expvals, max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples))

    if return_expvals:
        return losses
    else:
        return sum(losses)/len(losses)


def exp_kgel_iqp(iqp_circuit: IqpSimulator, params: jnp.ndarray, witnesses: jnp.ndarray, sigma: float, n_ops: int,
                 n_samples: int, key: Array, init_coefs: list = None, wires: list = None, indep_estimates=False,
                 max_batch_ops: int = None, max_batch_samples: int = None) -> jnp.ndarray:
    """Calculates the right hand side of the kernel generalized empirical likelihood  (KGEL)
    (see equation 6 in https://arxiv.org/pdf/2306.09780).

    Args:
        iqp_circuit (IqpSimulator): The IQP circuit itself given by the class IqpSimulator.
        params (jnp.ndarray): The parameters of the IQP gates.
        witnesses (jnp.ndarray): The witness points for the evaluation of the kernel (see the mentioned eq. 6).
        sigma (float): Sigma parameter, the width of the kernel.
        n_ops (int): Number of operators used to calculate the IQP expectation value.
        n_samples (int): Number of samples used to calculate the IQP expectation value.
        key (Array): Jax key to control the randomness of the process.
        init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
        wires (list): List of qubits positions where the operators will be measured. The rest will be traced away.
            Defaults to None, which refers to using all qubits.
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch (takes longer).
        max_batch_ops (int): Maximum number of operators in a batch of op_expval. Defaults to None.
        max_batch_samples (int): Maximum number of samples in a batch of op_expval. Defaults to None.

    Returns:
        jnp.ndarray: Vector of the right hand side of the KGEL test of eq 6.
    """
    if wires is None:
        wires = jnp.array(range(iqp_circuit.n_qubits))

    p_MMD = (1-jnp.exp(-1/2/sigma**2))/2

    # Calculating the indices where we have to add 0s to the ops matrix, depending on the arg wires.
    qubs = jnp.arange(iqp_circuit.n_qubits)
    qubs0 = qubs[~jnp.array([(i in wires) for i in qubs])]
    idx = []
    z, f = -1, 0
    for q in qubs0:
        f += q-z-1
        idx.append(f)
        z = q
    idx = jnp.array(idx, dtype=int)

    key, subkey = jax.random.split(key)
    rand_ops = jnp.array(jax.random.binomial(
        subkey, 1, p_MMD, shape=(n_ops, len(wires))), dtype='float32')
    ops = jnp.insert(rand_ops, idx, 0, axis=1)

    key, subkey = jax.random.split(key)
    tr_iqp = iqp_circuit.op_expval(params, ops, n_samples, subkey, init_coefs, indep_estimates,
                                   max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples)[0]
    coefs = 1 - 2 * ((witnesses @ ops.T) % 2)
    return jnp.mean(tr_iqp * coefs, axis=1)


def kgel_opt_iqp(iqp_circuit: IqpSimulator, params: jnp.ndarray, witnesses: jnp.ndarray, ground_truth: jnp.ndarray,
                 sigma: float, n_ops: int, n_samples: int, key: Array, init_coefs: list = None, verbose: bool = True,
                 wires: list = None, indep_estimates=False, max_batch_ops: int = None, max_batch_samples: int = None) -> list:
    """Calculates the right hand side of the kernel generalized empirical likelihood  (KGEL)
    (see equation 6 in https://arxiv.org/pdf/2306.09780). Uses cvxpy to solve the convex optimization problem.
    May require large values of n_ops and n_samples to arrive at stable estimates. Note that unlike the MMD loss,
    these estimates are not guaranteed to be unbiased.

    Args:
        iqp_circuit (IqpSimulator): The IQP circuit itself given by the class IqpSimulator.
        params (jnp.ndarray): The parameters of the IQP gates.
        witnesses (jnp.ndarray): The witness points for the evaluation of the kernel (see the mentioned eq. 6).
        ground_truth (jnp.ndarray): Samples from the true distribution.
        sigma (float): Sigma parameter, the width of the kernel.
        n_ops (int): Number of operators used to calculate the IQP expectation value.
        n_samples (int): Number of samples used to calculate the IQP expectation value.
        key (Array): Jax key to control the randomness of the process.
        init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
        verbose (bool, optional): Controls if the process is going to output information aboutt the optimization to the console. Defaults to True.
        wires (list, optional):List of qubit positions that specifies the qubits whose measurement statistics are
            used to estimate the KGEL. The remaining qubits will be traced out. Defaults to None, meaning all
            qubits are used.
        indep_estimates (bool, optional): Whether to use independent estimates of the ops in a batch (takes longer).
        max_batch_ops (int, optional): Maximum number of operators in a batch of op_expval. Defaults to None.
        max_batch_samples (int, optional): Maximum number of samples in a batch of op_expval. Defaults to None.

    Returns:
        list:
            result (float): Final value of the KL divergence of the optimization.
            pi.value (jnp.ndarray): Final values of the pi variable of the optimization.
    """
    if wires is None:
        wires = jnp.array(range(iqp_circuit.n_qubits))

    # Construct the problem.
    pi = cp.Variable(len(ground_truth))
    uniform = 1/len(ground_truth)*jnp.ones(shape=(len(ground_truth),))

    objective = cp.Minimize(cp.sum(cp.rel_entr(pi, uniform)))

    test_kernels = jnp.array(
        [list(map(partial(gaussian_kernel, sigma, s), witnesses)) for s in ground_truth])
    constraints = pi @ test_kernels - exp_kgel_iqp(iqp_circuit, params, witnesses, sigma, n_ops, n_samples, key, init_coefs,
                                                   wires=wires, indep_estimates=indep_estimates,
                                                   max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples)

    prob = cp.Problem(objective, [
                      c == 0 for c in constraints] + [cp.sum(pi) == 1] + [p >= 0 for p in pi])
    # The optimal objective is returned by prob.solve().
    result = prob.solve(verbose=verbose)

    # The optimal value for pi is stored in pi.value.
    return result, pi.value
