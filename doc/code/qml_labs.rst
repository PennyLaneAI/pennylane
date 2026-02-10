qml.labs
================

.. currentmodule:: pennylane.labs

.. automodule:: pennylane.labs


Phox IQP simulator
-------------------

``pennylane.labs.phox`` exposes a small toolkit for building and simulating
IQP-style circuits directly with JAX.  The typical workflow is:

1. Use the helper functions in :mod:`pennylane.labs.phox.utils` to assemble
   gates and observables.
2. Configure the simulator through :class:`pennylane.labs.phox.CircuitConfig`.
3. Create a compiled expectation-value function with
   :func:`pennylane.labs.phox.iqp_expval` and evaluate it for different
   parameter sets.

.. code-block:: python

	import jax
	import jax.numpy as jnp

	from pennylane.labs.phox import (
		CircuitConfig,
		create_lattice_gates,
		generate_pauli_observables,
		iqp_expval,
	)

	n_rows, n_cols = 3, 3
	n_qubits = n_rows * n_cols

	gates = create_lattice_gates(n_rows, n_cols, distance=1, max_weight=2)
	observables = generate_pauli_observables(n_qubits, orders=[2], bases=["Z"])

	key = jax.random.PRNGKey(0)
	params = jax.random.uniform(key, shape=(len(gates),))

	config = CircuitConfig(
		gates=gates,
		observables=observables,
		n_samples=4000,
		key=key,
		n_qubits=n_qubits,
	)

	expval_fn = iqp_expval(config)
	expvals, std_errs = expval_fn(params)

	print(expvals[:5])
	print(std_errs[:5])

For iterative optimization, combine these components with
:func:`pennylane.labs.phox.train` and the :class:`pennylane.labs.phox.TrainingOptions`
dataclass to run JAX-native training loops.

Training with ``phox.train``
---------------------------

Below is a minimal training loop that minimizes the sum of all two-body ``Z``
correlators on a 3x3 lattice.  The loss function reuses the compiled
``expval_fn`` built earlier.

.. code-block:: python

	from pennylane.labs.phox import train, TrainingOptions

	def loss_fn(params):
		expvals, _ = expval_fn(params)
		return jnp.sum(expvals)

	result = train(
		optimizer="Adam",
		loss=loss_fn,
		stepsize=0.05,
		n_iters=200,
		loss_kwargs={"params": params},
		options=TrainingOptions(unroll_steps=10, random_state=1234),
	)

	print("Final loss:", float(result.losses[-1]))
	print("Optimized parameters:", result.final_params)

``train`` yields loss history, validation metrics (if requested), and the final
parameter set, enabling interactive or scripted optimization workflows fully in
JAX.
      