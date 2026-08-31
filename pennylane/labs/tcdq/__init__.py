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
This module contains features to enable Train Classical, Deploy Quantum (TCDQ) workflows.

The supported workflows include fast optimization of instantaneous quantum polynomial (IQP) circuits and their extensions.
See `arXiv:2501.04776 <https://arxiv.org/abs/2501.04776>`_ and `arXiv:2607.06675 <https://arxiv.org/abs/2607.06675>`_ for theoretical details.

.. currentmodule:: pennylane.labs.tcdq

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

Simulators and estimators
~~~~~~~~~~~~~~~~~~~~~~~~~

A TCDQ simulator describes a family of circuits whose properties can be
estimated classically. Every simulator derives from :class:`~TCDQSimulator`
and exposes one or more *estimators*: pure, JAX-traceable functions that
estimate a circuit property for a batch of observables.

.. autosummary::
    :toctree: api

    ~TCDQSimulator
    ~IQPSimulator
    ~QuditIQPSimulator
    ~estimator
    ~Estimator
    ~EstimatorSpec
    ~ObservableAlgebra

Loss functions
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~build_mmd_loss
    ~build_qudit_mmd_loss
    ~MMDConfig
    ~QuditMMDConfig
    ~median_heuristic

Training
~~~~~~~~

.. autosummary::
    :toctree: api

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

Deprecated interfaces
~~~~~~~~~~~~~~~~~~~~~

The configuration dataclasses and factory functions below are superseded by
the simulator classes above. They are kept for backwards compatibility and
will be removed.

.. autosummary::
    :toctree: api

    ~CircuitConfig
    ~QuditCircuitConfig
    ~build_expval_func
    ~build_qudit_expval_func
    ~mmd_loss

Workflows
~~~~~~~~~~


The following examples demonstrate several of the key workflows supported by this module.

Estimating expectation values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax

   from pennylane.labs.tcdq import (
       IQPSimulator,
       create_lattice_gates,
       generate_pauli_observables,
   )

   # Define a 3×3 lattice of qubits with nearest-neighbour gates
   n_rows, n_cols = 3, 3
   n_qubits = n_rows * n_cols
   gates = create_lattice_gates(n_rows, n_cols, distance=1, max_weight=2)

   # Choose two-body ZZ observables
   observables = generate_pauli_observables(n_qubits, orders=[2], bases=["Z"])

   # Initialize random circuit parameters
   key = jax.random.PRNGKey(0)
   params = jax.random.uniform(key, shape=(len(gates),))

   # Describe the circuit, then build its estimator
   simulator = IQPSimulator(
       gates=gates,
       n_qubits=n_qubits,
       n_samples=4000,
       key=key,
   )
   expval = simulator.build_estimator("pauli_expval")

   # Observables are supplied at call time
   expvals, variances = expval(params, observables)


Training with a custom loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is a training loop that minimizes the sum of all two-body ``Z``
correlators on the same ``3×3`` lattice. The loss function reuses the
``expval`` estimator from above.

.. code-block:: python

   import jax.numpy as jnp

   from pennylane.labs.tcdq import TrainingOptions, train

   def loss_fn(current_params):
       expvals, _ = expval(current_params, observables)
       return jnp.sum(expvals)

   options = TrainingOptions(unroll_steps=10, random_state=1234)

   result = train(
       optimizer="Adam",
       loss=loss_fn,
       stepsize=0.05,
       n_iters=200,
       loss_kwargs={"params": params},
       options=options,
   )

   print("Final loss:", float(result.losses[-1]))
   print("Optimized parameters:", result.final_params)


Training with MMD loss (distribution matching)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train a circuit so that its output distribution reproduces a target
dataset of bitstrings, use the built-in Maximum Mean Discrepancy (MMD)
loss. The MMD is a kernel-based distance between probability distributions.
Smaller values mean the circuit output is closer to the target data.

:func:`~build_mmd_loss` takes an estimator rather than a specific simulator,
so it works with any :class:`~TCDQSimulator` whose estimator declares the
``PAULI_Z`` or ``PAULI`` observable algebra over qubits.

The ``bandwidth`` parameter controls how sensitive the loss is to
fine-grained versus broad differences between distributions. A good
default is the median pairwise distance of the dataset, computed with
:func:`~median_heuristic`.

For more detail on how the loss is constructed, see
`Section IV B, Loss functions via graph-Fourier kernels. <https://arxiv.org/pdf/2607.06675>`_

.. code-block:: python

   import numpy as np
   from pennylane.labs.tcdq import MMDConfig, build_mmd_loss, median_heuristic

   np.random.seed(42)
   target_data = np.random.binomial(1, 0.5, size=(500, n_qubits))

   bandwidth = median_heuristic(target_data)
   mmd_config = MMDConfig(bandwidth=bandwidth, n_ops=100)

   loss_fn = build_mmd_loss(simulator.build_estimator("pauli_expval"), mmd_config)

   loss_kwargs = {
       "params": params,
       "target_data": target_data,
       "key": jax.random.PRNGKey(1),
   }

   mmd_result = train(
       optimizer="Adam",
       loss=loss_fn,
       stepsize=0.01,
       n_iters=100,
       loss_kwargs=loss_kwargs,
       options=TrainingOptions(unroll_steps=10),
   )

   print("Final MMD loss:", float(mmd_result.losses[-1]))


Qudit circuits
^^^^^^^^^^^^^^

The same workflow applies to qudit circuits (``d > 2``). Replace
:class:`~IQPSimulator` with :class:`~QuditIQPSimulator` and request its
``"hw_expval"`` estimator. Gate vectors now have length ``n_qudits`` with
entries in :math:`\{0, \ldots, d-1\}` specifying the power of :math:`Z` on
each qudit, and observables are Heisenberg-Weyl displacement operators
``(l_vecs, m_vecs)``.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from pennylane.labs.tcdq import QuditIQPSimulator

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

   simulator = QuditIQPSimulator(
       dims=d,
       n_qudits=n_qudits,
       gates=gates,
       n_samples=5000,
       key=jax.random.PRNGKey(0),
   )
   expval = simulator.build_estimator("hw_expval")

   # Observables: displacement operators O(l, m) with m = 0
   l_vecs = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=jnp.int32)
   m_vecs = jnp.zeros_like(l_vecs)

   params = jnp.zeros(len(gates))
   expvals, cov = expval(params, (l_vecs, m_vecs))


Training qudit circuits with MMD loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qudit distribution matching uses
:func:`~build_qudit_mmd_loss`, which returns a reusable loss function
based on a graph-kernel MMD. The ``graph_type`` parameter selects the
kernel: ``"cycle"`` respects the ordering of neighbouring levels, while
``"complete"`` treats all levels symmetrically.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from pennylane.labs.tcdq import (
       QuditIQPSimulator,
       QuditMMDConfig,
       build_qudit_mmd_loss,
       TrainingOptions,
       train,
   )

   d = 3
   n_qudits = 4

   # Define single-qudit and nearest-neighbour two-qudit gates
   gates = {
       0: [[1, 0, 0, 0]],
       1: [[0, 1, 0, 0]],
       2: [[0, 0, 1, 0]],
       3: [[0, 0, 0, 1]],
       4: [[1, 1, 0, 0]],
       5: [[0, 1, 1, 0]],
       6: [[0, 0, 1, 1]],
   }

   simulator = QuditIQPSimulator(
       dims=d,
       n_qudits=n_qudits,
       gates=gates,
       n_samples=2000,
       key=jax.random.PRNGKey(0),
   )

   # Build the MMD loss with a cycle-graph kernel
   mmd_config = QuditMMDConfig(bandwidth=[0.3, 1.0], n_ops=64, graph_type="cycle")
   loss_fn = build_qudit_mmd_loss(simulator.build_estimator("hw_expval"), mmd_config)

   # Generate synthetic target data and train
   target_data = jax.random.randint(jax.random.PRNGKey(99), (500, n_qudits), 0, d)
   params = jnp.zeros(len(gates))

   result = train(
       optimizer="Adam",
       loss=loss_fn,
       stepsize=0.01,
       n_iters=100,
       loss_kwargs={
           "params": params,
           "target_data": target_data,
           "key": jax.random.PRNGKey(1),
       },
       options=TrainingOptions(unroll_steps=10),
   )

   print("Final MMD loss:", float(result.losses[-1]))


Defining your own simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Subclass :class:`~TCDQSimulator` to plug a different classical estimation
strategy into the same loss functions and training loop. Implement
``local_dims`` and decorate one or more factory methods with
:func:`~estimator`, declaring the :class:`~ObservableAlgebra` that fixes how
observables are encoded and what the estimator returns.

.. code-block:: python

   from pennylane.labs.tcdq import ObservableAlgebra, TCDQSimulator, estimator

   class MySimulator(TCDQSimulator):

       def __init__(self, n_qubits):
           self._n_qubits = n_qubits

       @property
       def local_dims(self):
           return (2,) * self._n_qubits

       @estimator("pauli_expval", algebra=ObservableAlgebra.PAULI_Z)
       def _build_pauli_expval(self):
           precomputed = expensive_setup(self._n_qubits)

           def expval(params, observables, *, key=None, n_samples=None,
                      phase_params=None):
               # returns (values, variances), both shape (n_obs,)
               return my_estimate(params, observables, precomputed)

           return expval

Because ``MySimulator`` declares a Pauli-Z capable estimator over qubits,
its estimator can be passed straight to :func:`~build_mmd_loss`.

"""

from .base import (
    Estimator,
    EstimatorSpec,
    ObservableAlgebra,
    TCDQSimulator,
    estimator,
)
from .iqp import (
    CircuitConfig,
    IQPSimulator,
    build_expval_func,
)
from .qudit_iqp import (
    QuditCircuitConfig,
    QuditIQPSimulator,
    build_qudit_expval_func,
)
from .mmd_loss import MMDConfig, build_mmd_loss, median_heuristic, mmd_loss
from .qudit_mmd_loss import QuditMMDConfig, build_qudit_mmd_loss
from .training import BatchResult, TrainingOptions, TrainingResult, train, training_iterator
from .utils import (
    create_lattice_gates,
    create_local_gates,
    create_random_gates,
    generate_pauli_observables,
)

__all__ = [
    "TCDQSimulator",
    "IQPSimulator",
    "QuditIQPSimulator",
    "estimator",
    "Estimator",
    "EstimatorSpec",
    "ObservableAlgebra",
    "MMDConfig",
    "QuditMMDConfig",
    "build_mmd_loss",
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
    "CircuitConfig",
    "QuditCircuitConfig",
    "build_expval_func",
    "build_qudit_expval_func",
    "mmd_loss",
]
