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
Train Classically, Deploy Quantum (TCDQ).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.tcdq

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

This module provides tools for training a restricted but useful family of
quantum circuits — Instantaneous Quantum Polynomial (IQP) circuits — with
classical estimates instead of full statevector simulation.
The intended workflow is:

1. describe a qubit or qudit circuit,
2. build an expectation-value estimator or a Maximum Mean Discrepancy
   (MMD) loss, and
3. optimize the circuit parameters with standard JAX optimizers.

The package supports both qubit circuits and qudit circuits of arbitrary local
dimension. The public APIs fall into two groups:

1. expectation-value estimators for observables of the circuit output, and
2. loss functions for matching the circuit distribution to a dataset.

For the mathematical background, we recommend reading `Section 2, Classically Estimating Expectation Values <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#2-classically-estimating-expectation-values>`_
and `Section 5, Graph-Kernel MMD Loss <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#5-graph-kernel-mmd-loss>`_
of the technical notes.


Core classes and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~CircuitConfig
    ~QuditCircuitConfig
    ~MMDConfig
    ~QuditMMDConfig
    ~build_expval_func
    ~build_qudit_expval_func
    ~build_qudit_mmd_loss
    ~mmd_loss
    ~median_heuristic
    ~train
    ~training_iterator
    ~TrainingOptions
    ~TrainingResult
    ~BatchResult


Circuit construction utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~create_lattice_gates
    ~create_local_gates
    ~create_random_gates
    ~generate_pauli_observables


Workflow
~~~~~~~~

The typical workflow is:

#. Define the circuit structure (which qubits interact via which gates)
   using the helper functions in this module.
#. Wrap the circuit description in a :class:`~CircuitConfig` (qubit) or
   :class:`~QuditCircuitConfig` (qudit) dataclass.
#. Build an expectation-value estimator with
   :func:`~build_expval_func` or :func:`~build_qudit_expval_func`.
#. Optionally, build an MMD loss function for distribution matching with
   :func:`~mmd_loss` or :func:`~build_qudit_mmd_loss`.
#. Train the circuit parameters with :func:`~train` or a custom
   optimization loop.

.. code-block:: python

   import jax

   from pennylane.labs.tcdq import (
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


Training with a custom loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a training loop that minimizes the sum of all two-body ``Z``
correlators on the same ``3×3`` lattice. The loss function reuses the
compiled ``expval_fn`` from above.

.. code-block:: python

   import jax.numpy as jnp

   from pennylane.labs.tcdq import TrainingOptions, train

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


Training with MMD loss (distribution matching)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train a circuit so that its output distribution reproduces a target
dataset of bitstrings, use the built-in Maximum Mean Discrepancy (MMD)
loss. The MMD is a kernel-based distance between probability distributions.
Smaller values mean the circuit output is closer to the target data.

The ``bandwidth`` parameter controls how sensitive the loss is to
fine-grained versus broad differences between distributions. A good
default is the median pairwise distance of the dataset, computed with
:func:`~median_heuristic`.

For more detail on how the loss is constructed, see
`Section 5, Graph-Kernel MMD Loss <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#5-graph-kernel-mmd-loss>`_
of the technical notes.

.. code-block:: python

   import numpy as np
   from pennylane.labs.tcdq import MMDConfig, mmd_loss, median_heuristic

   np.random.seed(42)
   target_data = np.random.binomial(1, 0.5, size=(500, n_qubits))

   bandwidth = median_heuristic(target_data)
   mmd_config = MMDConfig(bandwidth=bandwidth, n_ops=100)

   loss_kwargs = {
       "params": params,
       "circuit_config": config,
       "mmd_config": mmd_config,
       "target_data": target_data,
   }

   mmd_result = train(
       optimizer="Adam",
       loss=mmd_loss,
       stepsize=0.01,
       n_iters=100,
       loss_kwargs=loss_kwargs,
       options=TrainingOptions(unroll_steps=10)
   )

   print("Final MMD loss:", float(mmd_result.losses[-1]))


Qudit circuits
~~~~~~~~~~~~~~

The same workflow applies to qudit circuits (``d > 2``). Replace
:class:`~CircuitConfig` with :class:`~QuditCircuitConfig` and
:func:`~build_expval_func` with :func:`~build_qudit_expval_func`.
Gate vectors now have length ``n_qudits`` with entries in
:math:`\{0, \ldots, d-1\}` specifying the power of :math:`Z` on each
qudit.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from pennylane.labs.tcdq import QuditCircuitConfig, build_qudit_expval_func

   d = 3  # qutrit
   n_qudits = 4

   # Single-qudit and nearest-neighbour two-qudit gates
   gates = {
       0: [[1, 0, 0, 0]],
       1: [[0, 1, 0, 0]],
       2: [[0, 0, 1, 0]],
       3: [[0, 0, 0, 1]],
       4: [[1, 1, 0, 0]],
       5: [[0, 1, 1, 0]],
       6: [[0, 0, 1, 1]],
   }

   # Observables: displacement operators O(l, m) with m = 0
   l_vecs = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=jnp.int32)
   m_vecs = jnp.zeros_like(l_vecs)

   config = QuditCircuitConfig(
       d=d,
       n_qudits=n_qudits,
       gates=gates,
       observables=(l_vecs, m_vecs),
       n_samples=5000,
       key=jax.random.PRNGKey(0),
   )

   expval_fn = build_qudit_expval_func(config)
   params = jnp.zeros(len(gates))
   expvals, cov = expval_fn(params)

"""

from .expval_functions import (
    CircuitConfig,
    build_expval_func,
)
from .qudit_expval_functions import (
    QuditCircuitConfig,
    build_qudit_expval_func,
)
from .mmd_loss import MMDConfig, median_heuristic, mmd_loss
from .qudit_mmd_loss import QuditMMDConfig, build_qudit_mmd_loss
from .training import BatchResult, TrainingOptions, TrainingResult, train, training_iterator
from .utils import (
    create_lattice_gates,
    create_local_gates,
    create_random_gates,
    generate_pauli_observables,
)

__all__ = [
    "CircuitConfig",
    "QuditCircuitConfig",
    "MMDConfig",
    "QuditMMDConfig",
    "build_expval_func",
    "build_qudit_expval_func",
    "mmd_loss",
    "build_qudit_mmd_loss",
    "median_heuristic",
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
