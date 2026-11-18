.. role:: html(raw)
   :format: html

.. _intro_ref_chm:

Applications
============

PennyLane2 lets you go from inspiration to implementation, fast. Define
algorithms using our growing library of :doc:`subroutines <subroutines>` and
:doc:`gates <operations>`, and generate input data with diverse APIs and
:doc:`quantum datasets <data>`__.

Below are resources to consult for major applications in quantum computing.

Quantum chemistry
-----------------

The :mod:`~pennylane.qchem` module computes molecular integrals and factorizes them into
compact representations, such as the compressed double factorization (CDF) of
:class:`~pennylane.CDFHamiltonian`, which feed directly into algorithms like
:class:`~pennylane.TrotterCDF` and :class:`~pennylane.QuantumPhaseEstimation`.

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

For examples, see our library of
`quantum chemistry demos <https://pennylane.ai/search/?contentType=DEMO&categories=quantum%20chemistry&sort=publication_date>`_.

Quantum machine learning
------------------------

For examples, see our library of
`quantum machine learning demos <https://pennylane.ai/search/?contentType=DEMO&categories=quantum%20machine%20learning&sort=publication_date>`_.

Optimization and gradients
^^^^^^^^^^^^^^^^^^^^^^^^^^

Variational workflows in PennyLane2 are supported via gradient functionality
(e.g., :func:`qp.grad <pennylane.grad>` and
:func:`qp.jacobian <pennylane.jacobian>`) and optimization via
`Optax <https://optax.readthedocs.io>`__.

.. seealso::

  If you are coming from PennyLane Classic (v0.45.1), consult
  :doc:`coming_from_classic:gradients` and
  :doc:`coming_from_classic:optimization` for differences in taking gradients
  and optimizing PennyLane2 workflows.

Train-classical, deploy on quantum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instantaneous quantum polynomial (IQP) circuits are believed to be hard to sample from
classically, yet their expectation values can be approximated efficiently. This allows a
generative model to be *trained classically* and *deployed on quantum hardware* for sampling.

These circuits are supported in PennyLane2 via the following operators:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.IQP
    ~pennylane.IQPEmbedding

:html:`</div>`
