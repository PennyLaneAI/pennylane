.. role:: html(raw)
   :format: html

.. _intro_ref_chm:

Applications
============

PennyLane2 and :func:`~pennylane.qjit` let you go from a problem description to a compiled,
fault-tolerant algorithm in one program, and estimate what that algorithm costs at a scale
far beyond simulation. Below are two application areas: quantum chemistry and quantum machine
learning.

Quantum chemistry
-----------------

The :mod:`~pennylane.qchem` module computes molecular integrals and factorizes them into
compact representations, such as the compressed double factorization (CDF) of
:class:`~pennylane.CDFHamiltonian`, which feed directly into fault-tolerant algorithms like
:class:`~pennylane.TrotterCDF` and :class:`~pennylane.QuantumPhaseEstimation`. The same
algorithms can be costed for industrially relevant molecules with the
:mod:`~pennylane.estimator` module, without building the Hamiltonian or the circuit.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.Molecule
    ~pennylane.qchem.electron_integrals
    ~pennylane.qchem.factorize
    ~pennylane.CDFHamiltonian
    ~pennylane.CGFHamiltonian
    ~pennylane.TrotterCDF
    ~pennylane.TrotterCGF
    ~pennylane.QuantumPhaseEstimation
    ~pennylane.estimator.estimate

:html:`</div>`

For examples, see the
`quantum chemistry demos <https://pennylane.ai/search/?contentType=DEMO&categories=quantum%20chemistry&sort=publication_date>`_.

Quantum machine learning
------------------------

Instantaneous quantum polynomial (IQP) circuits are believed to be hard to sample from
classically, yet their expectation values can be approximated efficiently. This allows a
generative model to be *trained classically* and *deployed on quantum hardware* for sampling
with :class:`~pennylane.IQP`, and deploying such a model at scale can be costed ahead of time
with :func:`~pennylane.estimator.estimate`.

Variational models are instead trained by differentiating the quantum circuit itself. Which
gradient function to use depends on which side of the :func:`~pennylane.qjit` boundary it is
called from:

* **Inside a qjit-compiled function**, use :func:`qp.grad <pennylane.grad>` (or
  :func:`qp.jacobian <pennylane.jacobian>`) to differentiate anything that includes a QNode.
  These are compiled together with the rest of the program, so an entire optimization loop,
  including the quantum gradients, runs as one compiled workflow. ``jax.grad`` can still be
  used inside a qjit-compiled function, but *only* on purely classical code. Applied to a
  QNode, it raises an error.

* **Outside a qjit-compiled function**, a qjit-compiled function behaves like any other JAX
  function, so JAX transforms such as ``jax.grad`` and ``jax.vmap`` can be applied to it
  directly. This is useful when only the quantum part of a workflow is compiled with
  :func:`~pennylane.qjit` and JAX handles the classical part. However, a ``jax.jit``-compiled
  function that calls a qjit-compiled function always results in a callback to Python, so it
  is slower than compiling the entire workflow with :func:`~pennylane.qjit`.

See the Catalyst :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>` for more
details.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.IQP
    ~pennylane.IQPEmbedding
    ~pennylane.grad
    ~pennylane.jacobian
    ~pennylane.estimator.estimate

:html:`</div>`

For examples, see the
`quantum machine learning demos <https://pennylane.ai/search/?contentType=DEMO&categories=quantum%20machine%20learning&sort=publication_date>`_.

See :mod:`~pennylane.qchem` and :mod:`~pennylane.estimator` for the full set of quantum
chemistry and resource estimation tools.
