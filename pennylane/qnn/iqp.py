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

has_jax = True
try:
    import qpjax
    import qpjax.numpy as jnp
except ImportError as e:  # pragma: no cover
    has_jax = False  # pragma: no cover


def _len_gen(gates):
    return sum(len(gate) for gate in gates)


def _par_transform(gates):
    len_gen = _len_gen(gates)
    n_gates = len(gates)
    gens_per_gate = [len(gate) for gate in gates]
    total_gens = sum(gens_per_gate)

    # Transformation matrix from the number of independent parameters to the number of total generators
    trans_par = np.zeros((len_gen, len(gates)))
    # Vectorized assignment
    # row_indices: 0, 1, 2, ... (one per generator)
    # col_indices: 0, 0, ... 1, 1, 1 ... (gate index repeated for each gen it owns)
    row_indices = np.arange(total_gens)
    col_indices = np.repeat(np.arange(n_gates), gens_per_gate)

    trans_par[row_indices, col_indices] = 1
    return jnp.array(trans_par)


def _gate_lists_to_arrays(gate_lists: list, n_qubits: int) -> list:

    gate_arrays = []
    for gates in gate_lists:
        arr = np.zeros([len(gates), n_qubits])
        for i, gate in enumerate(gates):
            arr[i, gate] = 1.0
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
        key1, key2 = qpjax.random.split(carry, 2)
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

        return jnp.array(expvals)

    _, op_expvals = qpjax.lax.scan(update, key, ops)

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

    samples = qpjax.random.randint(key, (n_samples, n_qubits), 0, 2)

    generators = _generators(gates, n_qubits)
    effective_params = _effective_params(gates, params)

    generators_sp = None
    if sparse:
        generators_sp = _generators_sp(gates, n_qubits)

    if sparse or isinstance(ops, csr_matrix):
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
def iqp_expval(
    ops: list,
    weights: list[float],
    pattern: list[list[list[int]]],
    num_wires: int,
    n_samples: int,
    key: list,
    spin_sym: bool = False,
    sparse: bool = False,
    indep_estimates: bool = False,
    max_batch_ops: int = None,
    max_batch_samples: int = None,
) -> list:
    r"""Estimates the expectation values of a batch of Pauli-Z type operators for a parameterized :class:`~.IQP` circuit.

    The expectation values are estimated using a randomized method (Monte Carlo method) whose precision
    is controlled by the number of samples (``n_samples``), with larger values giving higher precision.

    Args:
        ops (list): Array specifying the operator/s for which to estimate the expectation values.
        weights (list): The parameters of the IQP gates.
        pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of `pattern` corresponds to a
            unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
            Generators are specified by listing the qubits on which an X operator acts. For example, the `pattern`
            `[[[0]], [[1]], [[2]], [[3]]]` specifies a circuit with single qubit rotations on the first four qubits, each
            with its own trainable parameter. The `pattern` `[[[0],[1]], [[2],[3]]]` corresponds to a circuit with two
            trainable parameters with generators :math:`X_0+X_1` and :math:`X_2+X_3` respectively. A circuit with a
            single trainable gate with generator :math:`X_0\otimes X_1` corresponds to the `pattern`
            `[[[0,1]]]`.
        num_wires (int): Number of wires in the circuit.
        n_samples (int): Number of samples used to estimate the IQP expectation values. Higher values result in
            higher precision.
        key (Array): Jax key to control the randomness of the process.
        spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
            :math:`\frac{1}{\sqrt(2)}(|00\dots0> + |11\dots1>)` is used in place of :math:`|00\dots0>`. This defines a circuit whose output distribution is invariant to flipping all bits.
        indep_estimates (bool): Whether to use independent estimates of the operators in a batch.
            If True, correlation among the estimated expectation values can be avoided, although at the cost
            of larger runtime.
        max_batch_ops (int): Specifies the maximum size of sub-batches of ``ops`` that are used to estimate the expectation values (to control memory usage). If None, a single batch is used. Can only be used if ``ops`` is a jnp.array.
        max_batch_samples (int): Specifies the maximum size of sub-batches of samples that are used to estimate the expectation values of ``ops`` (to control memory usage). If None, a single batch is used.

    Returns:
        list: List of Vectors. The expected value of each operator and its corresponding standard deviation.

    **Example:**

    To estimate the expectation value of a Pauli Z tensor, we represent the operator as a binary string
    (bitstring) that specifies on which qubit a Pauli ``Z`` operator acts. For example, in a three-qubit
    circuit, the operator :math:`Z_0 Z_2` will be represented as :math:`[1, 0, 1]`. Similarly, the expectation
    values for a group of operators can be evaluated by specifiying a sequence of bitstrings.

    As an example, let's estimate the expectation values for the operators :math:`Z_1`, :math:`Z_0`, and :math:`Z_0 Z_1`
    for a two-qubit circuit, using 1000 samples for the Monte Carlo estimation:

    .. code-block:: python

        from pennylane.qnn import iqp_expval
        import qpjax

        num_wires = 2
        ops = np.array([[0, 1], [1, 0], [1, 1]]) # binary array representing ops Z1, Z0, Z0Z1
        n_samples = 1000
        key = qpjax.random.PRNGKey(42)

        weights = np.ones(len(pattern))
        pattern = [[[0]], [[1]], [[0, 1]]] # binary array representing gates X0, X1, X0X1

        expvals, stds = iqp_expval(ops, weights, pattern, num_wires, n_samples, key)

    >>> print(expvals, stds)
    [0.18971464 0.14175898 0.17152457] [0.02615426 0.02614059 0.02615943]

    .. seealso:: The :class:`~.IQP` operation associated with this method.
    """

    params = jnp.array(weights)

    if not has_jax:
        raise ImportError(
            "JAX is required for use of IQP expectation value estimation."
        )  # pragma: no cover

    # do not batch ops if ops is sparse
    if isinstance(ops, csr_matrix):
        return _op_expval_batch(
            gates=pattern,
            params=params,
            n_qubits=num_wires,
            ops=ops,
            n_samples=n_samples,
            key=key,
            indep_estimates=indep_estimates,
            spin_sym=spin_sym,
        )

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
            key, subkey = qpjax.random.split(key, 2)
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
