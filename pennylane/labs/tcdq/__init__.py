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

.. currentmodule:: pennylane.labs.tcdq

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

What is TCDQ?
~~~~~~~~~~~~~

TCDQ is a framework for training and deploying parameterized quantum circuits. Unlike traditional
approaches to circuit optimization, TCDQ circuits can be trained using classical hardware alone,
which enables training of quantum circuits with thousands of qubits and millions of parameters on
a laptop or GPU. Although training can be done classically, deploying the trained circuit on
quantum hardware for sampling or downstream quantum algorithms can lead to advantages over purely
classical algorithms.

This module implements the features needed to classically train TCDQ circuits and uses
`JAX <https://docs.jax.dev>`_. Currently, the supported circuits consist of instantaneous quantum
polynomial (IQP) circuits and their generalizations.


Creating a circuit
~~~~~~~~~~~~~~~~~~

A qubit IQP circuit can be created by specifying the generators of the diagonal gates.
That description is a dictionary mapping a parameter index to a list of gates, where each gate is
a list of qubit indices. For instance, ``{0: [[0, 1]], 1: [[2]]}`` means parameter 0 drives a
:math:`ZZ` gate on qubits 0 and 1, and parameter 1 drives a :math:`Z` gate on qubit 2. Listing
several gates under the same key ties them to a shared parameter. The module contains utilities
to create gates: :func:`~create_local_gates`, :func:`~create_lattice_gates`, or
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
        n_samples=1000,  # Monte Carlo samples used by the estimator
        key=jax.random.PRNGKey(0)
    )


``n_samples`` sets the accuracy of every estimate produced from this configuration: the standard
error of each expectation value falls off as :math:`1/\sqrt{\texttt{n\_samples}}`.

TCDQ allows for fully flexible diagonal layers beyond those constructed by Pauli-:math:`Z`
generators. This is achieved by defining a phase function in JAX that maps a bitstring to a real
phase defining the entries of the diagonal unitary.

.. code-block:: python

    # A one-hidden-layer network mapping a bitstring to a real phase
    def phase_fn(phase_fn_params, z):
        weights, readout = phase_fn_params
        return jnp.dot(readout, jnp.tanh(weights @ z))

    phase_config = CircuitConfig(
        gates=gates,
        n_qubits=n_qubits,
        n_samples=1000,
        key=jax.random.PRNGKey(0),
        phase_fn=phase_fn,
    )

The phase layer is applied in addition to the layers defined by the gates. The order is irrelevant,
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
it can be JIT-compiled for faster evaluation. If the configuration defines a phase function, its
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
        key=jax.random.PRNGKey(2),
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
    amps_params = jnp.array([1.0, 2.0])
    amps = amps_params / jnp.linalg.norm(amps_params)

    expvals, variances = expval_fn(
        gates_params=gates_params,
        phase_fn_params=phase_fn_params,
        observables=observables,
        init_state_elems=elems,
        init_state_amps=amps,
        key=jax.random.PRNGKey(2),
    )

Note that the amplitudes are not automatically normalized and should be provided in a normalized
form if this is desired.


Training a circuit
~~~~~~~~~~~~~~~~~~

Since the estimator is an ordinary differentiable JAX function, a loss is just a Python function
of the parameters. Here we minimize the sum of all single-qubit :math:`Z` expectation values:

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
            key=key,
        )
        return jnp.sum(expvals)

    loss_fn = jax.jit(loss_fn)

The loss can be optimized by computing gradients via ``jax.grad`` and using any custom optimization
routine written in JAX. The module contains a function :func:`~train` that minimizes the loss
function in this way.

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

TCDQ contains a dedicated loss function for training quantum generative models. For distributions
of bitstrings, :func:`~build_mmd_loss_pauli` can be used to train the circuit, as described in
`Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand
qubits <https://arxiv.org/abs/2503.02934>`_.

.. code-block:: python

    from pennylane.labs.tcdq import MMDConfig, build_mmd_loss_pauli, median_heuristic

    key = jax.random.PRNGKey(3)
    target_data = jax.random.binomial(key, 1, p=0.5, shape=(500, n_qubits))
    mmd_config = MMDConfig(bandwidth=median_heuristic(target_data), n_ops=100)

    def expval_fn_train(params, observables, init_state_elems, key):
        amps = params["amps"] / jnp.linalg.norm(params["amps"])
        result = expval_fn(
            gates_params=params["gates"],
            phase_fn_params=params["phase_fn"],
            init_state_amps=amps,
            init_state_elems=init_state_elems,
            observables=observables,
            key=key,
        )
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

The workflow above extends to qudit circuits, where the gate generators are powers of the
generalized Pauli-:math:`Z` operator. Observables are Heisenberg–Weyl operators specified by a pair
``(l_vecs, m_vecs)`` of integer vectors. See `Spectral Born machines: classically trainable quantum
generative models for discrete data <https://arxiv.org/abs/2607.06675>`_ for details.

.. code-block:: python

    from pennylane.labs.tcdq import QuditCircuitConfig, build_qudit_expval_func

    n_qudits = 4
    dims = 4  # Local dimension of the qudits
    qudit_gates = {
        0: [[1, 0, 0, 0]],
        1: [[0, 1, 0, 0]],
        2: [[0, 0, 3, 0]],
        3: [[0, 0, 0, 1]],
        4: [[1, 2, 0, 0]],
        5: [[0, 3, 1, 0]],
        6: [[0, 0, 1, 1]],
    }

    qudit_config = QuditCircuitConfig(
        dims=dims,
        n_qudits=n_qudits,
        gates=qudit_gates,
        n_samples=1000,
        key=jax.random.PRNGKey(0),
    )

    l_vecs = jnp.array([[1, 0, 0, 0], [1, 1, 2, 0]])
    m_vecs = jnp.array([[0, 2, 0, 0], [1, 3, 0, 2]])

    qudit_expval_fn = jax.jit(build_qudit_expval_func(qudit_config))
    qudit_params = jnp.ones(len(qudit_gates))

    expvals, covariances = qudit_expval_fn(
        gates_params=qudit_params,
        observables=(l_vecs, m_vecs),
        key=jax.random.PRNGKey(2),
    )

Phase functions and sparse initial states work as in the qubit case, with bitstrings replaced by
dit-strings. Note that the expectation values are complex, so the estimator returns a real–imaginary
covariance matrix for each observable.

For distribution matching, :func:`~build_qudit_mmd_loss` uses a heat kernel on either a cycle graph,
which preserves adjacency between levels, or a complete graph, which treats distinct levels
symmetrically.

.. code-block:: python

    from pennylane.labs.tcdq import QuditMMDConfig, build_qudit_mmd_loss

    qudit_mmd_config = QuditMMDConfig(
        bandwidth=0.5,
        n_ops=100,
        graph_type="cycle",
    )
    qudit_loss_fn = build_qudit_mmd_loss(qudit_config, qudit_mmd_config)
    target_data = jax.random.randint(
        jax.random.PRNGKey(1), shape=(500, n_qudits), minval=0, maxval=dims
    )

    qudit_result = train(
        optimizer="Adam",
        loss=qudit_loss_fn,
        stepsize=0.01,
        n_iters=1000,
        loss_kwargs={"params": qudit_params, "target_data": target_data},
        options=TrainingOptions(random_state=1234),
    )

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
