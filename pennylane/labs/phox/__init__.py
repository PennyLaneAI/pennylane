# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Phase optimization with JAX (PHOX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.phox

.. autosummary::
    :toctree: api

    ~CircuitConfig
    ~build_expval_func
    ~bitflip_expval
    ~train
    ~training_iterator
    ~TrainingOptions
    ~TrainingResult
    ~BatchResult


Circuit construction utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.phox

.. autosummary::
    :toctree: api

    ~create_lattice_gates
    ~create_local_gates
    ~create_random_gates
    ~generate_pauli_observables


Workflow
~~~~~~~~

``pennylane.labs.phox`` provides a compact toolkit for constructing and
simulating phase optimization circuits with JAX. The usual workflow is:

#. Use helpers in :mod:`pennylane.labs.phox.utils` to assemble gates and
   observables.
#. Configure the circuit with :class:`~pennylane.labs.phox.CircuitConfig`.
#. Build an expectation-value function with
   :func:`~pennylane.labs.phox.build_expval_func` and evaluate it for
   different parameter sets.

.. code-block:: python

   import jax

   from pennylane.labs.phox import (
       CircuitConfig,
       build_expval_func,
       create_lattice_gates,
       generate_pauli_observables,
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

   expval_fn = jax.jit(build_expval_func(config))
   expvals, std_errs = expval_fn(params)


Training
~~~~~~~~

Below is a small training loop that minimizes the sum of all two-body ``Z``
correlators on the same ``3x3`` lattice. The loss function reuses the
compiled ``expval_fn`` from above.

.. code-block:: python

   import jax.numpy as jnp

   from pennylane.labs.phox import TrainingOptions, train

   def loss_fn(current_params):
       expvals, _ = expval_fn(current_params)
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

"""

from .expval_functions import CircuitConfig, bitflip_expval, build_expval_func
from .training import BatchResult, TrainingOptions, TrainingResult, train, training_iterator
from .utils import (
    create_lattice_gates,
    create_local_gates,
    create_random_gates,
    generate_pauli_observables,
)

__all__ = [
    "CircuitConfig",
    "bitflip_expval",
    "build_expval_func",
    "BatchResult",
    "TrainingOptions",
    "TrainingResult",
    "train",
    "training_iterator",
    "create_lattice_gates",
    "create_local_gates",
    "create_random_gates",
    "generate_pauli_observables",
]
