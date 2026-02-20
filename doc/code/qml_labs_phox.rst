qml.labs.phox
================

.. currentmodule:: pennylane.labs.phox

.. automodule:: pennylane.labs.phox


IQP simulator workflow
----------------------

``pennylane.labs.phox`` exposes a compact toolkit for constructing and
simulating IQP-style circuits with JAX.
The usual workflow is:

1. Use the helpers in :mod:`pennylane.labs.phox.utils` to assemble gates and
   observables.
2. Configure the simulator via :class:`pennylane.labs.phox.CircuitConfig`.
3. Build a pure expectation-value function with
   :func:`pennylane.labs.phox.build_expval_func` and evaluate it for different
   parameter sets.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from pennylane.labs.phox import (
       CircuitConfig,
       create_lattice_gates,
       generate_pauli_observables,
       build_expval_func,
   )

   n_rows, n_cols = 3, 3
   n_qubits = n_rows * n_cols

   gates = create_lattice_gates(n_rows, n_cols, distance=1, max_weight=2)
   # Observables are now generated directly as integer arrays
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

   expval_fn = build_expval_func(config)
   expvals, std_errs = expval_fn(params=params)

   print(expvals[:5])
   print(std_errs[:5])

Training with ``phox.train``
---------------------------

Below is a small training loop that minimizes the sum of all two-body ``Z``
correlators on the same 3x3 lattice.
The loss function reuses the compiled
``expval_fn`` defined above.

.. code-block:: python

   import jax.numpy as jnp
   from pennylane.labs.phox import train, TrainingOptions

   def loss_fn(current_params):
       # Using keyword arguments and relying on the pre-computed config defaults
       expvals, _ = expval_fn(params=current_params)
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

``train`` returns the updated parameters plus loss/validation histories,
allowing end-to-end IQP simulation and optimization completely within JAX.
