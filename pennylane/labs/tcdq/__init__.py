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
This module contains features to enable Train Classical, Deploy Quantum (TCDQ) workflows

.. currentmodule:: pennylane.labs.tcdq

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

What is TCDQ?
~~~~~~~~~~~~~

TCDQ is a framework for training and deploying parameterized quantum circuits. Unlike traditional approaches
to circuit optimization, TCDQ circuits can be trained using classical hardware alone, which enables
training of quantum circuits with thousands of qubits and millions of parameters on a laptop or GPU.
Although training can be done classically, deploying the trained circuit on quantum hardware for sampling
or downsteam quantum algorithms can lead to advantages over purely classical algorithms. 

This module implements the features needed to classically train TCDQ circuits and is written in 
JAX `JAX <https://docs.jax.dev>`_. Currently, the supported circuits consist of instantaneous quantum 
polynomial (IQP) circuits and their generalizations.


Creating a circuit
~~~~~~~~~~~~~~~~~~

A qubit IQP circuit can be created by specifying the generators of the diagonal gates. 
That description is a dictionary mapping a parameter index to a list of gates, where each gate is
a list of qubit indices. For instance, ``{0: [[0, 1]], 1: [[2]]}`` means parameter 0 drives a
:math:`ZZ` gate on qubits 0 and 1, and parameter 1 drives a :math:`Z` gate on qubit 2. Listing
several gates under the same key ties them to a shared parameter. The module contains utilities 
to create gates: :func:`~create_local_gates`, :func:`~create_lattice_gates` or
:func:`~create_random_gates`.

.. code-block:: python

    import jax
    import jax.numpy as jnp

    from pennylane.labs.tcdq import CircuitConfig, create_local_gates

    n_qubits = 5

    # All one- and two-qubit gates
    gates = create_local_gates(n_qubits, max_weight=2)

    config = CircuitConfig(
        gates=gates,
        n_qubits=n_qubits,
        n_samples=1000,             # Monte Carlo samples used by the estimator
    )


``n_samples`` sets the accuracy of every estimate produced from this configuration: the standard
error of each expectation value falls off as :math:`1/\sqrt{\texttt{n\_samples}}`.

TCDQ allows for fully flexible diagonal layers to be used beyond those constructed by Pauli-Z 
generators. This is achieved by defining a phase function in JAX that maps a bitstring to a real phase
which defines the entries of the diagonal unitary. 

.. code-block:: python

    # A one-hidden-layer network mapping a bitstring to a real phase
    def phase_fn(phase_fn_params, z):
        weights, readout = phase_fn_params
        return 0.*jnp.dot(readout, jnp.tanh(weights @ z))

    phase_config = CircuitConfig(
        gates=gates,
        n_qubits=n_qubits,
        n_samples=1000,
        key=jax.random.PRNGKey(0),
        phase_fn=phase_fn,
    )

The phase layer is applied in addition to those defined by the gates. The order is irrelevant, 
as all gates in the diagonal layer commute. 


Estimating expectation values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expectation values of Pauli words can be estimated using a Monte Carlo estimator. 
Pauli words are encoded as integer rows of length ``n_qubits`` using
``I=0, X=1, Y=2, Z=3``, so ``[3, 2, 0, 1]`` is :math:`Z \otimes Y \otimes I \otimes X`.
A whole batch of observables is estimated at once: stack one row per Pauli word into an array of
shape ``(n_observables, n_qubits)``, and a single call returns an estimate for every row.

:func:`~build_expval_func` turns a configuration into a pure function that returns the estimated
expectation values and their variance as a function of the trainable parameters. Once built,
it can be jit compiled for faster evaluation. If the configuration defines a phase function, its
parameters are passed as the second argument, and gradients flow to them just as they do to the
gate parameters.

.. code-block:: python

    from pennylane.labs.tcdq import build_expval_func

    # A batch of two observables, one row each
    observables = jnp.array(
        [
            [3, 3, 0, 0, 0],  # Z on qubits 0 and 1
            [0, 0, 1, 2, 0],  # X on qubit 2, Y on qubit 3
        ]
    )

    expval_fn = jax.jit(build_expval_func(phase_config))
    gates_params = jax.random.uniform(jax.random.PRNGKey(2), shape=(len(gates),))

    w_key, r_key = jax.random.split(jax.random.PRNGKey(1))
    phase_fn_params = (
        jax.random.normal(w_key, shape=(4, n_qubits)),
        jax.random.normal(r_key, shape=(4,)),
    )

    expvals, variances = expval_fn(
        gates_params=gates_params,
        phase_fn_params=phase_fn_params,
        observables=observables,
    )


Input states
^^^^^^^^^^^^
By default the circuit starts from :math:`|0\rangle^{\otimes n}`. Any state that is sparse in the
computational basis can be used instead, by listing its non-zero amplitudes: ``init_state_elems``
is an ``(N, n_qubits)`` binary array of basis states, and ``init_state_amps`` the matching
length-``N`` array of (possibly complex) amplitudes. 

.. code-block:: python

    # alpha |00...0> + beta |11...1>

    elems = jnp.array([[0] * n_qubits, [1] * n_qubits])
    amps_params = jnp.array([1., 2.])
    amps = amps_params / jnp.linalg.norm(amps_params)

    expvals, variances = expval_fn(
        gates_params=gates_params,
        phase_fn_params=phase_fn_params,
        observables=observables,
        init_state_elems=elems,
        init_state_amps=amps,
        key=jax.random.PRNGKey(0),
    )

Note that the amplitudes are not automatically normalized and should be provided in a normalized
form if this is desired.


Training a circuit
~~~~~~~~~~~~~~~~~~

Since the estimator is an ordinary differentiable JAX function, a loss is just a Python function
of the parameters. Here we minimize the sum of all two-body :math:`Z` correlators on the lattice:

.. code-block:: python
    
    # trainable parameters
    params = {
        "gates": gates_params,
        "phase_fn": phase_fn_params,
        "amps": amps_params,
    }

    observables = 3 * jnp.eye(n_qubits, dtype=int)  # all single-qubit Pauli-Z observables

    def loss_fn(params, key):
        amps = params["amps"] / jnp.linalg.norm(params["amps"])
        expvals, _ = expval_fn(
                        gates_params=params["gates"], 
                        phase_fn_params=params["phase_fn"], 
                        init_state_elems=elems, 
                        init_state_amps=amps, 
                        observables=observables,
                        key=key)
        return jnp.sum(expvals)

    loss_fn = jax.jit(loss_fn)

The loss can be optimized by computing gradients via ``jax.grad`` and using any custom optimization
routine written in JAX. The module containts a function :func:`~train` that minimizes the loss function
in this way.  

.. code-block:: python

    from pennylane.labs.tcdq import TrainingOptions, train

    result = train(
        optimizer="Adam",  
        loss=loss_fn,
        stepsize=0.001,
        n_iters=2000,
        loss_kwargs={"params": params},
        options=TrainingOptions(random_state=1234),
    )

    print("Final loss:", float(result.losses[-1]))
    trained_params = result.final_params

TCDQ contains dedicated loss function for training quantum generative models. For distributions
of bitstrings, the :func:`~mmd_loss` function can be used to train the circuit, as described in
 <https://arxiv.org/abs/2503.02934>`_.

.. code-block:: python

    from pennylane.labs.tcdq import MMDConfig, build_mmd_loss_pauli, median_heuristic

    key = jax.random.PRNGKey(3)
    target_data = jax.random.binomial(key, 1, p=0.5, shape=(500, n_qubits))
    mmd_config = MMDConfig(bandwidth=0.5, n_ops=100)

    def expval_fn_train(params, observables, init_state_elems, key):
        amps = params["amps"] / jnp.linalg.norm(params["amps"])
        result = expval_fn(gates_params=params['gates'], 
                        phase_fn_params=params['phase_fn'],
                        init_state_amps=amps,
                        init_state_elems=init_state_elems,
                        observables=observables,
                        key=key)
        return result           

    loss_fn = build_mmd_loss_pauli(expval_fn_train, n_qubits, mmd_config)

    mmd_result = train(
    optimizer="Adam",
    loss=loss_fn,
    stepsize=0.001,
    n_iters=1000,
    loss_kwargs={"params": params, "target_data": target_data, "init_state_elems": elems},
    options=TrainingOptions(random_state=1234),
    )

    print("Final MMD loss:", float(mmd_result.losses[-1]))

Qudit circuits
~~~~~~~~~~~~~~


Core classes and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~CircuitConfig
    ~QuditCircuitConfig
    ~MMDConfig
    ~QuditMMDConfig
    ~build_expval_func
    ~build_mmd_loss_pauli
    ~build_qudit_expval_func
    ~build_qudit_mmd_loss
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

<<<<<<< HEAD
=======
Workflows
~~~~~~~~~~


The following examples demonstrate several of the key workflows supported by this module.

Estimating expectation values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax

   from pennylane.labs.tcdq import (
       CircuitConfig,
       build_expval_func,
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

   # Bundle everything into a circuit configuration
   config = CircuitConfig(
       gates=gates,
       observables=observables,
       n_samples=4000,
       key=key,
       n_qubits=n_qubits,
   )

   # Build and JIT-compile the estimator, then evaluate
   expval_fn = jax.jit(build_expval_func(config))
   expvals, variances = expval_fn(params)


Training with a custom loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is a training loop that minimizes the sum of all two-body ``Z``
correlators on the same ``3×3`` lattice. The loss function reuses the
compiled ``expval_fn`` from above.

.. code-block:: python

   import jax.numpy as jnp

   from pennylane.labs.tcdq import TrainingOptions, train

   def loss_fn(current_params):
       expvals, _ = expval_fn(current_params)
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

:func:`~build_mmd_loss_pauli` takes any Pauli expectation value callable,
the number of qubits, and the MMD hyperparameters, and returns a reusable
loss function with signature ``loss_fn(params, target_data, key=None)``.
Because it takes a callable rather than a circuit configuration, the same
loss works for any model that can estimate Pauli-``Z`` expectation values,
not only IQP circuits.

The ``bandwidth`` parameter controls how sensitive the loss is to
fine-grained versus broad differences between distributions. A good
default is the median pairwise distance of the dataset, computed with
:func:`~median_heuristic`.

For more detail on how the loss is constructed, see
`Section 3.3 of IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX <https://arxiv.org/pdf/2501.04776>`_.

.. code-block:: python

   import numpy as np
   from pennylane.labs.tcdq import MMDConfig, build_mmd_loss_pauli, median_heuristic

   np.random.seed(42)
   target_data = np.random.binomial(1, 0.5, size=(500, n_qubits))

   bandwidth = median_heuristic(target_data)
   mmd_config = MMDConfig(bandwidth=bandwidth, n_ops=100)

   # Build the loss once, then reuse it for every optimization step
   loss_fn = build_mmd_loss_pauli(expval_fn, n_qubits, mmd_config)

   mmd_result = train(
       optimizer="Adam",
       loss=loss_fn,
       stepsize=0.01,
       n_iters=100,
       loss_kwargs={"params": params, "target_data": target_data},
       options=TrainingOptions(unroll_steps=10),
   )

   print("Final MMD loss:", float(mmd_result.losses[-1]))

Any extra keyword arguments needed by the expectation value callable are
forwarded through the loss, for example
``loss_fn(params, target_data, key, n_samples=8000)``.


Qudit circuits
^^^^^^^^^^^^^^

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
       dims=d,
       n_qudits=n_qudits,
       gates=gates,
       observables=(l_vecs, m_vecs),
       n_samples=5000,
       key=jax.random.PRNGKey(0),
   )

   expval_fn = build_qudit_expval_func(config)
   params = jnp.zeros(len(gates))
   expvals, cov = expval_fn(params)


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
       QuditCircuitConfig,
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

   circuit_config = QuditCircuitConfig(
       dims=d,
       n_qudits=n_qudits,
       gates=gates,
       n_samples=2000,
       key=jax.random.PRNGKey(0),
   )

   # Build the MMD loss with a cycle-graph kernel
   mmd_config = QuditMMDConfig(bandwidth=[0.3, 1.0], n_ops=64, graph_type="cycle")
   loss_fn = build_qudit_mmd_loss(circuit_config, mmd_config)

   # Generate synthetic target data and train
   target_data = jax.random.randint(jax.random.PRNGKey(99), (500, n_qudits), 0, d)
   params = jnp.zeros(len(gates))

   result = train(
       optimizer="Adam",
       loss=loss_fn,
       stepsize=0.01,
       n_iters=100,
       loss_kwargs={"params": params, "target_data": target_data},
       options=TrainingOptions(unroll_steps=10),
   )

   print("Final MMD loss:", float(result.losses[-1]))

>>>>>>> origin/tcdq-mmd-bootstrap
"""

from .expval_functions import (
    CircuitConfig,
    build_expval_func,
)
from .qudit_expval_functions import (
    QuditCircuitConfig,
    build_qudit_expval_func,
)
from .mmd_loss_pauli import MMDConfig, build_mmd_loss_pauli, median_heuristic
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
    "build_mmd_loss_pauli",
    "build_qudit_expval_func",
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
