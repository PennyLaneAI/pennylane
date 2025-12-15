# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This submodule defines methods for estimating the expectations of Pauli-Z operators following an IQP circuit.
"""
from functools import partial

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.typing import Array
from scipy.sparse import csr_matrix, dok_matrix

jax.config.update("jax_enable_x64", True)


has_jax = True
try:
    import jax
    import jax.numpy as jnp
    from jax.numpy import array
except ImportError as e:
    has_jax = False


def _len_gen(gates):
    return sum(1 for gate in gates for _ in gate)


def _par_transform(gates):
    len_gen = _len_gen(gates)

    # Transformation matrix from the number of independent parameters to the number of total generators
    trans_par = np.zeros((len_gen, len(gates)))
    i = 0
    for j, gens in enumerate(gates):
        for _ in gens:
            # Matrix that linearly transforms the vector of parameters that are trained into the vector of parameters that apply to the generators
            trans_par[i, j] = 1
            i += 1
    return jnp.array(trans_par)


def _gate_lists_to_arrays(gate_lists: list, n_qubits: int) -> list:

    gate_arrays = []
    for gates in gate_lists:
        arr = np.zeros([len(gates), n_qubits])
        for i, gate in enumerate(gates):
            for j in gate:
                arr[i, j] = 1.0
        gate_arrays.append(jnp.array(arr))
    return gate_arrays


def _generators_sp(gates, n_qubits):
    len_gen = _len_gen(gates)

    generators_dok = dok_matrix((len_gen, n_qubits), dtype="float64")
    i = 0
    for gate in gates:
        for gen in gate:
            for j in gen:
                generators_dok[i, j] = 1
            i += 1

    # convert to csr format
    return generators_dok.tocsr()


def _generators(gates, n_qubits):
    gates_as_arrays = _gate_lists_to_arrays(gates, n_qubits)

    generators = []
    for gens in gates_as_arrays:
        for gen in gens:
            generators.append(gen)

    return jnp.array(generators)


# pylint: disable=too-many-arguments
def _op_expval_indep(
    gates: list,
    n_qubits: int,
    params: list,
    ops: list,
    n_samples: int,
    key: list,
    sparse: bool,
    spin_sym: bool,
) -> list:
    """
    Batch evaluate an array of ops in the same way as self.op_expval_batch, but using independent randomness
    for each estimator. The estimators for each op are therefore uncorrelated.
    """

    def update(carry, op):
        key1, key2 = jax.random.split(carry, 2)
        expval = _op_expval_batch(
            gates=gates,
            params=params,
            n_qubits=n_qubits,
            ops=op,
            n_samples=n_samples,
            key=key1,
            spin_sym=spin_sym,
            indep_estimates=False,
            sparse=sparse,
        )
        return key2, expval

    if sparse:
        expvals = []
        for op in ops:
            key, val = update(key, op)
            expvals.append(val[0])

        return array(expvals)

    _, op_expvals = jax.lax.scan(update, key, ops)

    return op_expvals


def _sparse_samples(generators_sp, samples, spin_sym):
    samples_gates = samples.dot(generators_sp.T)
    samples_gates.data = 2 * (samples_gates.data % 2)
    samples_gates = samples_gates.toarray()
    samples_gates = 1 - samples_gates

    samples_sum = []
    samples_len = 0

    if spin_sym:
        samples_sum = np.squeeze(np.asarray(samples.sum(axis=-1)))
        samples_len = samples.shape[0]
    del samples

    return samples_gates, samples_sum, samples_len


def _sparse_ops(generators_sp, ops, spin_sym):
    ops_gen = ops.dot(generators_sp.T)
    ops_gen.data %= 2
    ops_gen = ops_gen.toarray()

    ops_sum = []

    if spin_sym:
        ops_sum = np.squeeze(np.asarray(ops.sum(axis=-1)))
    del ops

    return ops_gen, ops_sum


def _to_csr(generators, ops, samples, generators_sp=None):
    if isinstance(ops, csr_matrix):
        samples = csr_matrix(samples)
        if generators_sp is None:
            generators_sp = csr_matrix(generators)
    else:
        ops = csr_matrix(ops)
        samples = csr_matrix(samples)

    return samples, generators_sp, ops


def _effective_params(gates, params):
    par_transform = max(len(gate) for gate in gates) != 1
    if par_transform:
        effective_params = _par_transform(gates) @ params
    else:
        effective_params = params
    return effective_params


def _dense_samples(generators, samples, spin_sym):
    samples_gates = 1 - 2 * ((samples @ generators.T) % 2)

    samples_sum = []
    samples_len = 0

    if spin_sym:
        samples_sum = samples.sum(axis=-1)
        samples_len = samples.shape[0]

    return samples_gates, samples_sum, samples_len


def _dense_ops(generators, ops, spin_sym):
    ops_gen = (ops @ generators.T) % 2

    ops_sum = 0

    if spin_sym:
        ops_sum = ops.sum(axis=-1)

    return ops_gen, ops_sum


def _ini_spin_sym(ops_sum, samples_sum, samples_len, spin_sym):
    if spin_sym:
        try:
            shape = (len(ops_sum), samples_len)
        except TypeError:
            shape = (samples_len,)

        return 2 - jnp.repeat(ops_sum, samples_len).reshape(shape) % 2 - 2 * (samples_sum % 2)

    return 1


# pylint: disable=too-many-arguments
def _op_expval_batch(
    gates: list,
    params: list,
    n_qubits: int,
    ops: list,
    n_samples: int,
    key: list,
    spin_sym: bool = False,
    sparse: bool = False,
    indep_estimates: bool = False,
) -> list:

    if indep_estimates:
        return _op_expval_indep(
            gates=gates,
            n_qubits=n_qubits,
            params=params,
            ops=ops,
            n_samples=n_samples,
            key=key,
            sparse=sparse,
            spin_sym=spin_sym,
        )

    samples = jax.random.randint(key, (n_samples, n_qubits), 0, 2)

    generators = _generators(gates, n_qubits)
    effective_params = _effective_params(gates, params)

    if sparse or isinstance(ops, csr_matrix):

        generators_sp = _generators_sp(gates, n_qubits)
        samples, generators_sp, ops = _to_csr(generators, ops, samples, generators_sp)

        samples_gates, samples_sum, samples_len = _sparse_samples(generators_sp, samples, spin_sym)
        ops_gen, ops_sum = _sparse_ops(generators_sp, ops, spin_sym)

    else:
        samples_gates, samples_sum, samples_len = _dense_samples(generators, samples, spin_sym)
        ops_gen, ops_sum = _dense_ops(generators, ops, spin_sym)

    ini_spin_sym = _ini_spin_sym(ops_sum, samples_sum, samples_len, spin_sym)

    par_ops_gates = 2 * effective_params * ops_gen
    expvals = ini_spin_sym * jnp.cos(par_ops_gates @ samples_gates.T)

    return expvals


# pylint: disable=too-many-arguments
def op_expval(
    ops: list,
    n_samples: int,
    key: list,
    num_wires: int,
    pattern: list[list[list[int]]],
    weights: list[float],
    spin_sym: bool = False,
    sparse: bool = False,
    indep_estimates: bool = False,
    max_batch_ops: int = None,
    max_batch_samples: int = None,
) -> list:
    r"""Estimate the expectation values of a batch of Pauli-Z type operators. A set of l operators must be specified
    by an array of shape (l,n_qubits), where each row is a binary vector that specifies on which qubit a Pauli Z
    operator acts.
    The expectation values are estimated using a randomized method whose precision in controlled by n_samples,
    with larger values giving higher precision. Estimates are unbiased, however may be correlated. To request
    uncorrelated estimate, use indep_estimates=True at the cost of larger runtime.
    For large batches of operators or large values of n_samples, memory can be controlled by setting max_batch_ops
    and/or max_batch_samples to a fixed value.

    Args:
        ops (jnp.ndarray): Array specifying the operator/s for which to estimate the expectation values.
        n_samples (int): Number of samples used to calculate the IQP expectation values. Higher values result in
            higher precision.
        key (Array): Jax key to control the randomness of the process.
        num_wires (int): Number of wires in the circuit.
        pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of `pattern` corresponds to a
            unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
            Generators are specified by listing the qubits on which an X operator acts. For example, the `pattern`
            `[[[0]], [[1]], [[2]], [[3]]]` specifies a circuit with single qubit rotations on the first four qubits, each
            with its own trainable parameter. The `pattern` `[[[0],[1]], [[2],[3]]]` correspond to a circuit with two
            trainable parameters with generators :math:`X_0+X_1` and :math:`X_2+X_3` respectively. A circuit with a
            single trainable gate with generator :math:`X_0\otimes X_1` corresponds to the `pattern`
            `[[[0,1]]]`.
        weights (list): The parameters of the IQP gates.
        spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
            :math:`\frac{1}{\sqrt(2)}(|00\dots0> + |11\dots1>)` is used in place of :math:`|00\dots0>`.
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch.
        max_batch_ops (int): Maximum number of operators in a batch. Defaults to None, which means taking all ops at once.
        max_batch_samples (int): Maximum number of samples in a batch. Defaults to None, which means taking all n_samples at once.

    Returns:
        list: List of Vectors. The expected value of each op and its standard deviation.

    **Example:**

    .. code-block:: python

        key = jax.random.PRNGKey(np.random.randint(0, 99999))

        exp_val, std = op_expval(
            ops=jnp.array([[1, 0], [0, 1]]),
            n_samples=10_000,
            key=key,
            num_wires=n_qubits,
            pattern=[[[0], [1]]],
            weights=[0.54],
            spin_sym=True,
            sparse=False,
            indep_estimates=True,
            max_batch_samples=10_000,
            max_batch_ops=10_000,
        )
    """

    params = jnp.array(weights)

    if not has_jax:
        raise ImportError(
            "JAX is required for use of IQP expectation value estimation."
        )  # pragma: no cover

    if max_batch_ops is None:
        max_batch_ops = len(ops)

    if max_batch_samples is None:
        max_batch_samples = n_samples

    if len(ops.shape) == 1:
        ops = ops.reshape(1, -1)

    expvals = jnp.empty((0, n_samples))

    for batch_ops in jnp.array_split(ops, np.ceil(ops.shape[0] / max_batch_ops)):
        tmp_expvals = jnp.empty((len(batch_ops), 0))
        for i in range(np.ceil(n_samples / max_batch_samples).astype(jnp.int64)):
            batch_n_samples = min(max_batch_samples, n_samples - i * max_batch_samples)
            key, subkey = jax.random.split(key, 2)
            batch_expval = _op_expval_batch(
                gates=pattern,
                params=params,
                n_qubits=num_wires,
                ops=batch_ops,
                n_samples=batch_n_samples,
                key=subkey,
                spin_sym=spin_sym,
                sparse=sparse,
                indep_estimates=indep_estimates,
            )
            tmp_expvals = jnp.concatenate((tmp_expvals, batch_expval), axis=-1)
        expvals = jnp.concatenate((expvals, tmp_expvals), axis=0)

    return jnp.mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)


def gaussian_kernel(sigma: float, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Calculates the value for the gaussian kernel between two vectors x, y

    Args:
        sigma (float): sigma parameter, the width of the kernel
        x (jnp.ndarray): one of the vectors
        y (jnp.ndarray): the other vector

    Returns:
        float: Result value of the gaussian kernel
    """
    return jnp.exp(-((x - y) ** 2).sum() / 2 / sigma**2)


def loss_estimate_iqp(
    weights: jnp.ndarray,
    num_wires: int,
    pattern: list[list[list[int]]],
    ground_truth: jnp.ndarray,
    visible_ops: jnp.ndarray,
    all_ops: jnp.ndarray,
    n_samples: int,
    key: Array,
    indep_estimates: bool = False,
    sqrt_loss: bool = False,
    return_expvals: bool = False,
    max_batch_ops: int = None,
    max_batch_samples: int = None,
) -> float:
    """Estimates the MMD Loss of an IQP circuit with respect to a ground truth distribution.

    Args:
        weights (jnp.ndarray): The parameters of the IQP gates.
        num_wires (int): Number of wires in the circuit.
        pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of `pattern` corresponds to a
            unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
            Generators are specified by listing the qubits on which an X operator acts. For example, the `pattern`
            `[[[0]], [[1]], [[2]], [[3]]]` specifies a circuit with single qubit rotations on the first four qubits, each
            with its own trainable parameter. The `pattern` `[[[0],[1]], [[2],[3]]]` correspond to a circuit with two
            trainable parameters with generators :math:`X_0+X_1` and :math:`X_2+X_3` respectively. A circuit with a
            single trainable gate with generator :math:`X_0\otimes X_1` corresponds to the `pattern`
            `[[[0,1]]]`.
        ground_truth (jnp.ndarray): Matrix with the training samples as rows (0s and 1s).
        visible_ops (jnp.ndarray): Matrix with only the visible operators as rows (0s and 1s). Used to estimate the training part of the loss.
        all_ops (jnp.ndarray): Matrix with the all the operators as rows (0s and 1s). Used to estimate the IQP part of the loss.
        n_samples (jnp.ndarray): Number of samples used to estimate the loss.
        key (Array): Jax key to control the randomness of the process.
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
    tr_iqp_samples = op_expval(
        weights=weights,
        pattern=pattern,
        num_wires=num_wires,
        ops=all_ops,
        n_samples=n_samples,
        key=key,
        indep_estimates=indep_estimates,
        max_batch_ops=max_batch_ops,
        max_batch_samples=max_batch_samples,
    )
    correction = jnp.mean(tr_iqp_samples**2, axis=-1) / n_samples
    tr_iqp = jnp.mean(tr_iqp_samples, axis=-1)
    tr_train = jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0)
    m = len(ground_truth)

    # add correction to make the first term unbiased
    res = (
        (tr_iqp * tr_iqp - correction) * n_samples / (n_samples - 1)
        - 2 * tr_iqp * tr_train
        + (tr_train * tr_train * m - 1) / (m - 1)
    )

    res = jnp.mean(res) if not return_expvals else res
    res = jnp.sqrt(jnp.abs(res)) if sqrt_loss else res

    return res


def mmd_loss_iqp(
    weights: jnp.ndarray,
    num_wires: int,
    pattern: list[list[list[int]]],
    ground_truth: jnp.ndarray,
    sigma: float | list,
    n_ops: int,
    n_samples: int,
    key: Array,
    sparse: bool = False,
    wires: list = None,
    indep_estimates: bool = False,
    jit: bool = True,
    sqrt_loss: bool = False,
    return_expvals: bool = False,
    max_batch_ops: int = None,
    max_batch_samples: int = None,
) -> float:
    """Returns an estimate of the (squared) MMD Loss of an IQP circuit with respect to a ground truth
     distribution. Requires a set of samples from the ground truth distribution. The estimate is unbiased in the sense
     that the expectation of the estimator wrt samples from the ground truth is the exact (squared) MMD loss. The kernel
     used is the Gaussian kernel with bandwidth specified by sigma.

     The function uses a randomized method whose precision can be increased by using larger values of n_samples and/or
     n_ops.

    Args:
        weights (jnp.ndarray): The parameters of the IQP gates.
        num_wires (int): Number of wires in the circuit.
        pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of `pattern` corresponds to a
            unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
            Generators are specified by listing the qubits on which an X operator acts. For example, the `pattern`
            `[[[0]], [[1]], [[2]], [[3]]]` specifies a circuit with single qubit rotations on the first four qubits, each
            with its own trainable parameter. The `pattern` `[[[0],[1]], [[2],[3]]]` correspond to a circuit with two
            trainable parameters with generators :math:`X_0+X_1` and :math:`X_2+X_3` respectively. A circuit with a
            single trainable gate with generator :math:`X_0\otimes X_1` corresponds to the `pattern`
            `[[[0,1]]]`.
        ground_truth (jnp.ndarray): Array containing the samples from the ground truth distribution as rows (0s and 1s).
        sigma (float or list): The bandwidth of the kernel. If several are given as a list the average loss over each value will
            be returned.
        n_ops (int): Number of operators used to estimate the loss.
        n_samples (jnp.ndarray): Number of samples used to estimate the loss.
        key (jax.random.PRNGKey): Jax PRNG key used to seed random functions.
        sparse (bool, optional): Whether the IQP circuit is represented by a sparse matrix.
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

    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")

    if wires is None:
        wires = list(range(num_wires))

    losses = []
    for sigma in sigmas:
        p_MMD = (1 - jnp.exp(-1 / 2 / sigma**2)) / 2
        key, subkey = jax.random.split(key, 2)
        visible_ops = jnp.array(
            jax.random.binomial(subkey, 1, p_MMD, shape=(n_ops, len(wires))), dtype="float64"
        )

        all_ops = []
        i = 0
        for q in range(num_wires):
            if q in wires:
                all_ops.append(visible_ops[:, i])
                i += 1
            else:
                all_ops.append(jnp.zeros(n_ops))
        all_ops = jnp.array(all_ops, dtype="float64").T

        if sparse:
            loss = loss_estimate_iqp
        else:
            if jit:
                loss = jax.jit(
                    loss_estimate_iqp,
                    static_argnames=[
                        "iqp_circuit",
                        "n_samples",
                        "indep_estimates",
                        "sqrt_loss",
                        "return_expvals",
                        "max_batch_ops",
                        "max_batch_samples",
                    ],
                )
            else:
                loss = loss_estimate_iqp

        losses.append(
            loss(
                weights,
                num_wires,
                pattern,
                ground_truth,
                visible_ops,
                all_ops,
                n_samples,
                key,
                indep_estimates,
                sqrt_loss,
                return_expvals=return_expvals,
                max_batch_ops=max_batch_ops,
                max_batch_samples=max_batch_samples,
            )
        )

    if return_expvals:
        return losses
    else:
        return sum(losses) / len(losses)


def exp_kgel_iqp(
    weights: jnp.ndarray,
    num_wires: int,
    pattern: list[list[list[int]]],
    witnesses: jnp.ndarray,
    sigma: float,
    n_ops: int,
    n_samples: int,
    key: Array,
    init_coefs: list = None,
    wires: list = None,
    indep_estimates=False,
    max_batch_ops: int = None,
    max_batch_samples: int = None,
) -> jnp.ndarray:
    """Calculates the right hand side of the kernel generalized empirical likelihood  (KGEL)
    (see equation 6 in https://arxiv.org/pdf/2306.09780).

    Args:
        weights (jnp.ndarray): The parameters of the IQP gates.
        num_wires (int): Number of wires in the circuit.
        pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of `pattern` corresponds to a
            unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
            Generators are specified by listing the qubits on which an X operator acts. For example, the `pattern`
            `[[[0]], [[1]], [[2]], [[3]]]` specifies a circuit with single qubit rotations on the first four qubits, each
            with its own trainable parameter. The `pattern` `[[[0],[1]], [[2],[3]]]` correspond to a circuit with two
            trainable parameters with generators :math:`X_0+X_1` and :math:`X_2+X_3` respectively. A circuit with a
            single trainable gate with generator :math:`X_0\otimes X_1` corresponds to the `pattern`
            `[[[0,1]]]`.
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
        wires = jnp.array(range(num_wires))

    p_MMD = (1 - jnp.exp(-1 / 2 / sigma**2)) / 2

    # Calculating the indices where we have to add 0s to the ops matrix, depending on the arg wires.
    qubs = jnp.arange(num_wires)
    qubs0 = qubs[~jnp.array([(i in wires) for i in qubs])]
    idx = []
    z, f = -1, 0
    for q in qubs0:
        f += q - z - 1
        idx.append(f)
        z = q
    idx = jnp.array(idx, dtype=int)

    key, subkey = jax.random.split(key)
    rand_ops = jnp.array(
        jax.random.binomial(subkey, 1, p_MMD, shape=(n_ops, len(wires))), dtype="float32"
    )
    ops = jnp.insert(rand_ops, idx, 0, axis=1)

    key, subkey = jax.random.split(key)
    tr_iqp = op_expval(
        pattern=pattern,
        num_wires=num_wires,
        weights=weights,
        ops=ops,
        n_samples=n_samples,
        key=subkey,
        indep_estimates=indep_estimates,
        max_batch_ops=max_batch_ops,
        max_batch_samples=max_batch_samples,
    )[0]
    coefs = 1 - 2 * ((witnesses @ ops.T) % 2)
    return jnp.mean(tr_iqp * coefs, axis=1)


def kgel_opt_iqp(
    weights: jnp.ndarray,
    num_wires: int,
    pattern: list[list[list[int]]],
    witnesses: jnp.ndarray,
    ground_truth: jnp.ndarray,
    sigma: float,
    n_ops: int,
    n_samples: int,
    key: Array,
    init_coefs: list = None,
    verbose: bool = True,
    wires: list = None,
    indep_estimates=False,
    max_batch_ops: int = None,
    max_batch_samples: int = None,
) -> list:
    """Calculates the right hand side of the kernel generalized empirical likelihood  (KGEL)
    (see equation 6 in https://arxiv.org/pdf/2306.09780). Uses cvxpy to solve the convex optimization problem.
    May require large values of n_ops and n_samples to arrive at stable estimates. Note that unlike the MMD loss,
    these estimates are not guaranteed to be unbiased.

    Args:
        weights (jnp.ndarray): The parameters of the IQP gates.
        num_wires (int): Number of wires in the circuit.
        pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of `pattern` corresponds to a
            unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
            Generators are specified by listing the qubits on which an X operator acts. For example, the `pattern`
            `[[[0]], [[1]], [[2]], [[3]]]` specifies a circuit with single qubit rotations on the first four qubits, each
            with its own trainable parameter. The `pattern` `[[[0],[1]], [[2],[3]]]` correspond to a circuit with two
            trainable parameters with generators :math:`X_0+X_1` and :math:`X_2+X_3` respectively. A circuit with a
            single trainable gate with generator :math:`X_0\otimes X_1` corresponds to the `pattern`
            `[[[0,1]]]`.
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
        wires = jnp.array(range(num_wires))

    # Construct the problem.
    pi = cp.Variable(len(ground_truth))
    uniform = 1 / len(ground_truth) * jnp.ones(shape=(len(ground_truth),))

    objective = cp.Minimize(cp.sum(cp.rel_entr(pi, uniform)))

    test_kernels = jnp.array(
        [list(map(partial(gaussian_kernel, sigma, s), witnesses)) for s in ground_truth]
    )
    constraints = pi @ test_kernels - exp_kgel_iqp(
        weights,
        num_wires,
        pattern,
        witnesses,
        sigma,
        n_ops,
        n_samples,
        key,
        init_coefs,
        wires=wires,
        indep_estimates=indep_estimates,
        max_batch_ops=max_batch_ops,
        max_batch_samples=max_batch_samples,
    )

    prob = cp.Problem(
        objective, [c == 0 for c in constraints] + [cp.sum(pi) == 1] + [p >= 0 for p in pi]
    )
    # The optimal objective is returned by prob.solve().
    result = prob.solve(verbose=verbose)

    # The optimal value for pi is stored in pi.value.
    return result, pi.value
