import numpy as np
from scipy.sparse import csr_matrix, dok_matrix

has_jax = True
try:
    import jax
    import jax.numpy as jnp
    from jax._src.typing import Array

    jax.config.update("jax_enable_x64", True)
except ImportError as e:
    has_jax = False


def gate_lists_to_arrays(gate_lists: list, n_qubits: int) -> list:
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


class IqpBitflipSimulator:
    """Class that creates an IqpBitflipSimulator object corresponding to a parameterized IQP circuit"""

    def __init__(
        self,
        n_qubits: int,
        gates: list,
        device: str = "lightning.qubit",
        spin_sym: bool = False,
        sparse: bool = False,
    ):
        """
        Args:
            n_qubits (int): Total number of qubits of the circuit.
            gates (list[list[list[int]]]): Specification of the trainable gates. Each element of gates corresponds to a
                unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
                Generators are specified by listing the qubits on which an X operator acts.
            device (str, optional): Pennylane device used for calculating probabilities and sampling.
            spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
                1/sqrt(2)(|00...0> + |11...1>) is used in place of |00...0>.
            sparse (bool, optional): If True, generators and ops are always stored in sparse matrix format, leading
                to better memory efficiency and potentially faster runtime.

        Raises:
            Exception: when gates and params have a different number of elements.
        """
        if not has_jax:
            raise ImportError(
                "JAX is required for use of bitflip-based IQP expectation value estimation."
            )  # pragma: no cover

        self.n_qubits = n_qubits
        self.gates = gates
        self.n_gates = len(gates)
        self.sparse = sparse
        self.device = device
        self.spin_sym = spin_sym

        self.generators = []
        self.generators_sp = None

        len_gen_init = 0

        len_gen = sum(1 for gate in gates for _ in gate) + len_gen_init
        self.par_transform = False if max([len(gate) for gate in self.gates]) == 1 else True

        if sparse:
            generators_dok = dok_matrix((len_gen, n_qubits), dtype="float64")
            i = 0
            for gate in gates:
                for gen in gate:
                    for j in gen:
                        generators_dok[i, j] = 1
                    i += 1

            # convert to csr format
            self.generators_sp = generators_dok.tocsr()

        else:
            # Transformation of the input gates to generators
            # convert the gates to a list of arrays
            self.gates_as_arrays = gate_lists_to_arrays(gates, n_qubits)

            # store all generators
            self.generators = []
            for gens in self.gates_as_arrays:
                for gen in gens:
                    self.generators.append(gen)

            self.generators = jnp.array(self.generators)

        if self.par_transform:
            # Transformation matrix from the number of independent parameters to the number of total generators
            self.trans_par = np.zeros((len_gen, len(gates)))
            i = 0
            for j, gens in enumerate(gates):
                for gen in gens:
                    # Matrix that linearly transforms the vector of parameters that are trained into the vector of parameters that apply to the generators
                    self.trans_par[i, j] = 1
                    i += 1
            self.trans_par = jnp.array(self.trans_par)

    def op_expval_batch(
        self,
        params: jnp.ndarray,
        ops: jnp.ndarray,
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
            return_samples (bool): if True, an extended array that contains the values of the estimator for each
                of the n_samples samples is returned.

        Returns:
            list: List of Vectors. The expected value of each op and its standard deviation.
        """

        effective_params = self.trans_par @ params if self.par_transform else params

        if self.sparse or isinstance(ops, csr_matrix):

            if isinstance(ops, csr_matrix):
                if self.generators_sp is None:
                    self.generators_sp = csr_matrix(self.generators)

            else:
                ops = csr_matrix(ops)

            ops_gen = ops.dot(self.generators_sp.T)

            if self.spin_sym:
                ops_sum = np.squeeze(np.asarray(ops.sum(axis=-1)))

            ops_gen.data %= 2
            ops_gen = ops_gen.toarray()

        else:
            ops_gen = (ops @ self.generators.T) % 2
            if self.spin_sym:
                ops_sum = jnp.sum(ops, axis=-1)

        par_ops_gates = 2 * effective_params * ops_gen

        expvals = jnp.prod(jnp.cos(par_ops_gates), axis=-1)

        if self.spin_sym:
            # flip expvals of odd operators with prob 1/2
            odd_ops = 1 - 2 * (ops_sum % 2)
            expvals = 0.5 * expvals + 0.5 * odd_ops * expvals

        if return_samples:
            return jnp.expand_dims(expvals, -1)
        else:
            return expvals, jnp.zeros(ops.shape[0])

    def op_expval(
        self,
        params: jnp.ndarray,
        ops: jnp.ndarray,
        n_samples: int,
        key: Array,
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
            return_samples (bool): if True, an extended array that contains the values of the estimator for each
                of the n_samples samples is returned.
            max_batch_ops (int): Maximum number of operators in a batch. Defaults to None, which means taking all ops at once.
                Can only be used if ops is a jnp.array
            max_batch_samples (int): Maximum number of samples in a batch. Defaults to None, which means taking all n_samples at once.

        Returns:
            list: List of Vectors. The expected value of each op and its standard deviation.
        """

        # do not batch ops if ops is sparse
        if isinstance(ops, csr_matrix):
            return self.op_expval_batch(params, ops, return_samples)

        if max_batch_ops is None:
            max_batch_ops = len(ops)

        if max_batch_samples is None:
            max_batch_samples = n_samples

        n_samples = max_batch_samples

        if len(ops.shape) == 1:
            ops = ops.reshape(1, -1)

        expvals = jnp.empty((0, 1))

        for batch_ops in jnp.array_split(ops, np.ceil(ops.shape[0] / max_batch_ops)):
            tmp_expvals = jnp.empty((len(batch_ops), 0))
            for i in range(np.ceil(n_samples / max_batch_samples).astype(jnp.int64)):
                key, subkey = jax.random.split(key, 2)
                batch_expval = self.op_expval_batch(
                    params,
                    batch_ops,
                    return_samples=True,
                )
                tmp_expvals = jnp.concatenate((tmp_expvals, batch_expval), axis=-1)
            expvals = jnp.concatenate((expvals, tmp_expvals), axis=0)

        if return_samples:
            return expvals
        else:
            return jnp.mean(expvals, axis=-1), jnp.zeros(len(ops))
