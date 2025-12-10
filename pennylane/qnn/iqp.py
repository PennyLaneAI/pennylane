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
from pennylane.math import array

has_jax = True
try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    has_jax = False


def _len_gen(gates):
    return sum(1 for gate in gates for _ in gate)


def _par_transform(gates):
    """
    Creates the transformation matrix from the number of independent parameters to the number of total generators
    """
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
    gates: array,
    n_qubits: int,
    params: array,
    ops: array,
    n_samples: int,
    key: array,
    sparse: bool,
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
            spin_sym=False,
        )
        return key2, expval

    if sparse:
        expvals = []
        for op in ops:
            key, val = update(key, op)
            expvals.append(
                val[0]
            )  # TODO: this is going to be the wrong shape for later concatenation

        return array(expvals)

    _, op_expvals = jax.lax.scan(update, key, ops)

    return op_expvals


# pylint: disable=too-many-arguments, too-many-branches, too-many-statements
def _op_expval_batch(
    gates: list,
    params: array,
    n_qubits: int,
    ops: array,
    n_samples: int,
    key: array,
    spin_sym: bool = False,
    sparse: bool = False,
    indep_estimates: bool = False,
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
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch (takes longer).
        return_samples (bool): if True, an extended array that contains the values of the estimator for each
            of the n_samples samples is returned.

    Returns:
        list: List of Vectors. The expected value of each op and its standard deviation.
    """

    # TODO: refactor and break up into functions with less branches

    if indep_estimates:
        return _op_expval_indep(
            gates=gates,
            n_qubits=n_qubits,
            params=params,
            ops=ops,
            n_samples=n_samples,
            key=key,
            sparse=sparse,
        )

    samples = jax.random.randint(key, (n_samples, n_qubits), 0, 2)

    generators = _generators(gates, n_qubits)
    par_transform = max(len(gate) for gate in gates) != 1
    if par_transform:
        effective_params = _par_transform(gates) @ params
    else:
        effective_params = params

    ops_sum = []
    samples_len = 0
    samples_sum = []

    if sparse or isinstance(ops, csr_matrix):

        generators_sp = _generators_sp(gates, n_qubits)

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
        if sparse or isinstance(ops, csr_matrix):
            shape = (len(ops_sum), samples_len)
        else:
            shape = (samples_len,)

        ini_spin_sym = (
            2 - jnp.repeat(ops_sum, samples_len).reshape(shape) % 2 - 2 * (samples_sum % 2)
        )

    else:
        ini_spin_sym = 1

    par_ops_gates = 2 * effective_params * ops_gen
    expvals = ini_spin_sym * jnp.cos(par_ops_gates @ samples_gates.T)

    return expvals


# pylint: disable=too-many-arguments
def op_expval(
    ops: array,
    n_samples: int,
    key: array,
    circuit: IQP,
    sparse: bool = False,
    indep_estimates: bool = False,
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
        ops (jnp.ndarray): Array specifying the operator/s for which to estimate the expectation values.
        n_samples (int): Number of samples used to calculate the IQP expectation values. Higher values result in
            higher precision.
        key (Array): Jax key to control the randomness of the process.
        circuit (IQP): The circuit after which expectations are taken.
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch.
        max_batch_ops (int): Maximum number of operators in a batch. Defaults to None, which means taking all ops at once.
        max_batch_samples (int): Maximum number of samples in a batch. Defaults to None, which means taking all n_samples at once.

    Returns:
        list: List of Vectors. The expected value of each op and its standard deviation.
    """

    gates = circuit.hyperparameters["pattern"]
    params = jnp.array(circuit.hyperparameters["weights"])
    n_qubits = len(circuit.wires)
    spin_sym = circuit.hyperparameters["spin_sym"]

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
                gates=gates,
                params=params,
                n_qubits=n_qubits,
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
