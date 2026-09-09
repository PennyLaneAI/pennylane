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
        n_samples=4000,             # Monte Carlo samples used by the estimator
        key=jax.random.PRNGKey(0),  # PRNG key for drawing those samples
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
        return jnp.dot(readout, jnp.tanh(weights @ z))

    phase_config = CircuitConfig(
        gates=gates,
        n_qubits=n_qubits,
        n_samples=4000,
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
        key=jax.random.PRNGKey(0),
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
                        observables=observables
                        key=key)
        return jnp.sum(expvals)

The loss can be optimized by computing gradients via ``jax.grad`` and using any custom optimization
routine written in JAX. The module containts a function :func:`~train` that minimizes the loss function
in this way.  

.. code-block:: python

    from pennylane.labs.tcdq import TrainingOptions, train

    result = train(
        optimizer="Adam", 
        loss=loss_fn,
        stepsize=0.05,
        n_iters=200,
        loss_kwargs={"params": params},
        options=TrainingOptions(random_state=1234),
    )

    print("Final loss:", float(result.losses[-1]))
    trained_params = result.final_params

``loss_kwargs`` must contain ``"params"``, the trainable parameter pytree; anything else it holds is
forwarded to the loss at every step. If the loss accepts a ``key`` argument, ``train`` supplies a
fresh one each step, so stochastic losses see fresh Monte Carlo samples.

The most common use of these circuits is generative modelling: training the circuit so that
measuring it in the computational basis reproduces a dataset of bitstrings. For this, the module
provides :func:`~mmd_loss`, a maximum mean discrepancy (MMD) loss. The MMD is a kernel-based
distance between distributions, small when the circuit output is close to the data and zero when
the two agree. With a radial basis function kernel, the squared MMD can be rewritten as a mixture
of squared differences between the circuit's Pauli-:math:`Z` moments and the same moments measured
on the data, so the loss is built entirely from the expectation values above and stays classically
estimable.

.. code-block:: python

    from pennylane.labs.tcdq import MMDConfig, median_heuristic, mmd_loss

    rng = np.random.default_rng(42)
    target_data = rng.binomial(1, 0.5, size=(500, n_qubits))

    mmd_config = MMDConfig(bandwidth=median_heuristic(target_data), n_ops=100)

    result = train(
        optimizer="Adam",
        loss=mmd_loss,
        stepsize=0.01,
        n_iters=100,
        loss_kwargs={
            "params": params,
            "circuit_config": config,
            "mmd_config": mmd_config,
            "target_data": target_data,
        },
        options=TrainingOptions(unroll_steps=10),
    )

``bandwidth`` is the length scale the kernel is sensitive to; :func:`~median_heuristic` takes the
median pairwise distance in the dataset, which is a reasonable default. Passing a list of
bandwidths averages the loss over all of them, which often stabilizes training. ``n_ops`` is the
number of observables sampled per step to estimate the loss: more observables mean a less noisy
loss and gradient at proportionally higher cost. Note that the returned value is an *unbiased*
estimate of the squared MMD, so it can come out slightly negative near the optimum.

Qudit circuits
~~~~~~~~~~~~~~

Everything above generalizes from bits to dits. A qudit circuit over :math:`\mathbb{Z}_d^n` has
the same shape, with the Hadamard replaced by the quantum Fourier transform over
:math:`\mathbb{Z}_d` and the Pauli-:math:`Z` generators by diagonal Heisenberg–Weyl operators. The
estimable quantities become Heisenberg–Weyl moments
:math:`\langle \mathcal{O}(\mathbf{l}, \mathbf{m}) \rangle`; see
`arXiv:2607.06675 <https://arxiv.org/abs/2607.06675>`_ for the derivation.

Two things change in the interface. :class:`~QuditCircuitConfig` takes ``dims``, either one local
dimension for every wire or a sequence giving a different dimension per wire, and each gate is
described by a full-length vector over :math:`\{0, \ldots, d-1\}` whose :math:`i`-th entry is the
power of :math:`Z` applied to qudit :math:`i`, rather than by a list of wire indices. Observables
are supplied as a pair of integer arrays ``(l_vecs, m_vecs)`` of shape ``(n_obs, n_qudits)``
labelling the displacement operators :math:`\mathcal{O}(\mathbf{l}, \mathbf{m})`.

.. code-block:: python

    from pennylane.labs.tcdq import QuditCircuitConfig, build_qudit_expval_func

    d, n_qudits = 3, 4  # qutrits

    # Single-qudit gates and nearest-neighbour two-qudit gates
    gates = {
        0: [[1, 0, 0, 0]],
        1: [[0, 1, 0, 0]],
        2: [[0, 0, 1, 0]],
        3: [[0, 0, 0, 1]],
        4: [[1, 1, 0, 0]],
        5: [[0, 1, 1, 0]],
        6: [[0, 0, 1, 1]],
    }

    l_vecs = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=jnp.int32)
    m_vecs = jnp.zeros_like(l_vecs)

    qudit_config = QuditCircuitConfig(
        dims=d,
        n_qudits=n_qudits,
        gates=gates,
        observables=(l_vecs, m_vecs),
        n_samples=5000,
        key=jax.random.PRNGKey(0),
    )

    expval_fn = jax.jit(build_qudit_expval_func(qudit_config))

    qudit_params = jnp.zeros(len(gates))
    expvals, cov = expval_fn(qudit_params)

Heisenberg–Weyl moments are complex, so in place of one variance per observable the estimator
returns a :math:`2 \times 2` covariance matrix per observable, for the real and imaginary parts of
the estimate. As in the qubit case a batch shares one set of Monte Carlo samples and its estimates
are therefore correlated. Sparse input states and phase functions work exactly as before, with
dit-strings in place of bitstrings.

The MMD loss carries over too, with the radial basis function kernel replaced by a heat kernel on
a graph over the :math:`d` local levels. ``graph_type="cycle"`` treats neighbouring levels as
close, which is what you want when the levels encode an ordered variable, while ``"complete"``
treats every level as equally distinct, which suits unordered categories. Unlike its qubit
counterpart, :func:`~build_qudit_mmd_loss` returns a loss function instead of being called
directly:

.. code-block:: python

    from pennylane.labs.tcdq import QuditMMDConfig, build_qudit_mmd_loss

    qudit_mmd_config = QuditMMDConfig(bandwidth=[0.3, 1.0], n_ops=64, graph_type="cycle")
    qudit_loss_fn = build_qudit_mmd_loss(qudit_config, qudit_mmd_config)

    target_data = jax.random.randint(jax.random.PRNGKey(99), (500, n_qudits), 0, d)

    result = train(
        optimizer="Adam",
        loss=qudit_loss_fn,
        stepsize=0.01,
        n_iters=100,
        loss_kwargs={"params": qudit_params, "target_data": target_data},
        options=TrainingOptions(unroll_steps=10),
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
