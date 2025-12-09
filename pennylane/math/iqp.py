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
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix

from pennylane import IQP
from pennylane.math import array, mean, sqrt

has_jax = True
try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    has_jax = False


def _len_gen(gates, init_gates):
    len_gen_init = 0
    if init_gates is not None:
        len_gen_init = sum(1 for gate in init_gates for _ in gate)

    return sum(1 for gate in gates for _ in gate) + len_gen_init, len_gen_init


def _trans_coeff(gates, init_gates, generators):
    len_gen, len_gen_init = _len_gen(gates, init_gates)

    if init_gates is not None:
        # Matrix that transforms the static parameters (initial coefficients) into a vector of size generators so it can be summed with the variational parameters
        trans_coeff = np.zeros((len_gen, len(init_gates)))
        i = len(generators) - len_gen_init
        for j, gens in enumerate(init_gates):
            for gen in gens:
                trans_coeff[i, j] = 1
                i += 1
        return jnp.array(trans_coeff)


def _par_transform(gates, init_gates):
    """
    Creates the transformation matrix from the number of independent parameters to the number of total generators
    """
    len_gen, len_gen_init = _len_gen(gates, init_gates)

    par_transform = (
        False if max([len(gate) for gate in gates]) == 1 and init_gates is None else True
    )

    if par_transform:
        # Transformation matrix from the number of independent parameters to the number of total generators
        trans_par = np.zeros((len_gen, len(gates)))
        i = 0
        for j, gens in enumerate(gates):
            for _ in gens:
                # Matrix that linearly transforms the vector of parameters that are trained into the vector of parameters that apply to the generators
                trans_par[i, j] = 1
                i += 1
        return par_transform, jnp.array(trans_par)


def _gate_lists_to_arrays(gate_lists: list, n_qubits: int) -> list:
    """Transforms the gates parameter into a list of arrays of 0s and 1s.

    Args:
        gate_lists (list[list[list[int]]]): Gates list for IqpSimulator object.
        n_qubits (int): number of qubits in the return arrays

    Returns:
        list: Gates parameter in list of arrays form.
    """

    gate_arrays = []
    for gates in gate_lists:
        arr = np.zeros([len(gates), n_qubits])
        for i, gate in enumerate(gates):
            for j in gate:
                arr[i, j] = 1.0
        gate_arrays.append(jnp.array(arr))
    return gate_arrays


def _generators_sp(gates, init_gates, n_qubits):
    len_gen, len_gen_init = _len_gen(gates, init_gates)

    generators_dok = dok_matrix((len_gen, n_qubits), dtype="float64")
    i = 0
    for gate in gates:
        for gen in gate:
            for j in gen:
                generators_dok[i, j] = 1
            i += 1

    if init_gates is not None:
        for gate in init_gates:
            for gen in gate:
                for j in gen:
                    generators_dok[i, j] = 1
                i += 1

    # convert to csr format
    return generators_dok.tocsr()


def _generators(gates, init_gates, n_qubits):
    gates_as_arrays = _gate_lists_to_arrays(gates, n_qubits)

    generators = []
    for gens in gates_as_arrays:
        for gen in gens:
            generators.append(gen)

    if init_gates is not None:
        init_gates_as_arrays = _gate_lists_to_arrays(init_gates, n_qubits)

        for gens in init_gates_as_arrays:
            for gen in gens:
                generators.append(gen)

    return jnp.array(generators)


def _op_expval_indep(
    gates: array,
    n_qubits: int,
    params: array,
    ops: array,
    n_samples: int,
    key: array,
    sparse: bool,
    return_samples,
    init_coeffs: list = None,
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
            init_coeffs=init_coeffs,
            spin_sym=False,
            return_samples=return_samples,
        )
        return key2, expval

    if sparse:
        expvals = []
        stds = []
        for op in ops:
            key, val = update(key, op)
            if return_samples:
                expvals.append(val[0])
            else:
                expvals.append(val[0][0])
                stds.append(val[1][0])

        if return_samples:
            return array(expvals)
        else:
            return array(expvals), array(stds) / sqrt(n_samples)

    else:
        _, op_expvals = jax.lax.scan(update, key, ops)

        if return_samples:
            return op_expvals
        else:
            return op_expvals[0], op_expvals[1]


def _op_expval_batch(
    gates: list,
    params: array,
    n_qubits: int,
    ops: array,
    n_samples: int,
    key: array,
    init_gates: list = None,
    init_coeffs: list = None,
    spin_sym: bool = False,
    sparse: bool = False,
    indep_estimates: bool = False,
    return_samples: bool = False,
) -> list:
    """Estimate the expectation values of a batch of Pauli-Z type operators. A set of l operators must be specified
    by an array of shape (l,n_qubits), where each row is a binary vector that specifies on which qubit a Pauli Z
    operator acts.
    The expectation values are estimated using a randomized method whose precision in controlled by n_samples,
    with larger values giving higher precision. Estimates are unbiased, however may be correlated. To request
    uncorrelated estimate, use indep_estimates=True at the cost of larger runtime.

    Args:
        params (jnp.ndarray): The parameters of the IQP gates.
        ops (jnp.ndarray): Operator/s for those we want to know the expected value.
        n_samples (int): Number of samples used to calculate the IQP expectation value.
        key (Array): Jax key to control the randomness of the process.
        init_coeffs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
            values of init_gates.
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch (takes longer).
        return_samples (bool): if True, an extended array that contains the values of the estimator for each
            of the n_samples samples is returned.

    Returns:
        list: List of Vectors. The expected value of each op and its standard deviation.
    """

    if indep_estimates:
        return _op_expval_indep(
            gates=gates,
            n_qubits=n_qubits,
            params=params,
            ops=ops,
            n_samples=n_samples,
            key=key,
            sparse=sparse,
            return_samples=return_samples,
            init_coeffs=init_coeffs,
        )

    samples = jax.random.randint(key, (n_samples, n_qubits), 0, 2)

    generators = _generators(gates, init_gates, n_qubits)
    par_transform, trans_par = _par_transform(gates, init_gates)
    trans_coeff = _trans_coeff(gates, init_gates, generators)

    effective_params = trans_par @ params if par_transform else params
    effective_params = (
        effective_params + trans_coeff @ init_coeffs if init_gates is not None else effective_params
    )

    if sparse or isinstance(ops, csr_matrix):

        generators_sp = _generators_sp(gates, init_gates, n_qubits)

        if isinstance(ops, csr_matrix):
            samples = csr_matrix(samples)
            if generators_sp is None:
                generators_sp = csr_matrix(generators)
        else:
            ops = csr_matrix(ops)
            samples = csr_matrix(samples)

        ops_gen = ops.dot(generators_sp.T)
        ops_gen.data %= 2
        ops_gen = ops_gen.toarray()

        samples_gates = samples.dot(generators_sp.T)
        samples_gates.data = 2 * (samples_gates.data % 2)
        samples_gates = samples_gates.toarray()
        samples_gates = 1 - samples_gates

        if spin_sym:
            ops_sum = np.squeeze(np.asarray(ops.sum(axis=-1)))
            samples_sum = np.squeeze(np.asarray(samples.sum(axis=-1)))
            samples_len = samples.shape[0]
        del ops
        del samples

    else:
        ops_gen = (ops @ generators.T) % 2
        samples_gates = 1 - 2 * ((samples @ generators.T) % 2)
        if spin_sym:
            ops_sum = ops.sum(axis=-1)
            samples_sum = samples.sum(axis=-1)
            samples_len = samples.shape[0]

    if spin_sym:
        try:
            shape = (len(ops_sum), samples_len)
        except:
            shape = (samples_len,)

        ini_spin_sym = (
            2 - jnp.repeat(ops_sum, samples_len).reshape(shape) % 2 - 2 * (samples_sum % 2)
        )

    else:
        ini_spin_sym = 1
    # ini_spin_sym = jnp.where(self.spin_sym, ini_spin_sym, 1)

    par_ops_gates = 2 * effective_params * ops_gen
    expvals = ini_spin_sym * jnp.cos(par_ops_gates @ samples_gates.T)

    if return_samples:
        return expvals
    else:
        return mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1) / sqrt(n_samples)


def op_expval(
    ops: array,
    n_samples: int,
    key: array,
    circuit: IQP,
    sparse: bool = False,
    indep_estimates: bool = False,
    return_samples: bool = False,
    max_batch_ops: int = None,
    max_batch_samples: int = None,
) -> list:
    """Estimate the expectation values of a batch of Pauli-Z type operators. A set of l operators must be specified
    by an array of shape (l,n_qubits), where each row is a binary vector that specifies on which qubit a Pauli Z
    operator acts.
    The expectation values are estimated using a randomized method whose precision in controlled by n_samples,
    with larger values giving higher precision. Estimates are unbiased, however may be correlated. To request
    uncorrelated estimate, use indep_estimates=True at the cost of larger runtime.
    For large batches of operators or large values of n_samples, memory can be controlled by setting max_batch_ops
    and/or max_batch_samples to a fixed value.

    Args:
        params (jnp.ndarray): The parameters of the trainable gates of the circuit.
        ops (jnp.ndarray): Array specifying the operator/s for which to estimate the expectation values.
        n_samples (int): Number of samples used to calculate the IQP expectation values. Higher values result in
            higher precision.
        key (Array): Jax key to control the randomness of the process.
        init_coeffs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
            values of init_gates.
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch.
        return_samples (bool): if True, an extended array that contains the values of the estimator for each
            of the n_samples samples is returned.
        max_batch_ops (int): Maximum number of operators in a batch. Defaults to None, which means taking all ops at once.
        max_batch_samples (int): Maximum number of samples in a batch. Defaults to None, which means taking all n_samples at once.

    Returns:
        list: List of Vectors. The expected value of each op and its standard deviation.
    """

    init_coeffs = jnp.array(circuit.hyperparameters["init_coeffs"])
    gates = circuit.hyperparameters["gates"]
    params = jnp.array(circuit.hyperparameters["params"])
    n_qubits = len(circuit.wires)
    init_gates = circuit.hyperparameters["init_gates"]
    spin_sym = circuit.hyperparameters["spin_sym"]

    if not has_jax:
        raise ImportError("JAX is required for use of IQP expectation value estimation.")

    if max_batch_ops is None:
        max_batch_ops = len(ops)

    if max_batch_samples is None:
        max_batch_samples = n_samples

    if len(ops.shape) == 1:
        ops = ops.reshape(1, -1)

    expvals = jnp.empty((0, n_samples))

    init_coeffs = jnp.array(init_coeffs) if init_coeffs is not None else None

    for batch_ops in jnp.array_split(ops, np.ceil(ops.shape[0] / max_batch_ops)):
        tmp_expvals = jnp.empty((len(batch_ops), 0))
        for i in range(np.ceil(n_samples / max_batch_samples).astype(jnp.int64)):
            batch_n_samples = min(max_batch_samples, n_samples - i * max_batch_samples)
            key, subkey = jax.random.split(key, 2)
            batch_expval = _op_expval_batch(
                gates=gates,
                params=params,
                n_qubits=n_qubits,
                ops=batch_ops,
                n_samples=batch_n_samples,
                key=subkey,
                init_gates=init_gates,
                init_coeffs=init_coeffs,
                spin_sym=spin_sym,
                sparse=sparse,
                indep_estimates=indep_estimates,
                return_samples=True,
            )
            tmp_expvals = jnp.concatenate((tmp_expvals, batch_expval), axis=-1)
        expvals = jnp.concatenate((expvals, tmp_expvals), axis=0)

    if return_samples:
        return expvals
    else:
        return jnp.mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
