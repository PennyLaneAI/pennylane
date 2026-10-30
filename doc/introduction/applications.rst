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
:class:`~pennylane.TrotterCDF` and :class:`~pennylane.QuantumPhaseEstimation`.

.. code-block:: python

    import numpy as np
    import pennylane as qp

    mol = qp.qchem.Molecule(["H", "H"], np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]))
    core, one, two = qp.qchem.electron_integrals(mol)()

    # one-body term: diagonalize; two-body term: compressed double factorization
    two = np.swapaxes(two, 1, 3)
    eps, U = np.linalg.eigh(one - 0.5 * np.einsum("pqrr->pq", two))
    _, cores, leaves = qp.qchem.factorize(two, compressed=True, num_factors=2)

    H = qp.CDFHamiltonian(
        core_tensors=np.concatenate([np.diag(eps)[None], cores]),
        leaf_tensors=np.concatenate([U[None], leaves]),
        nuc_constant=float(core[0]),
    )

    system, estimation = range(4), range(4, 8)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set={"RZ", "IsingZZ", "CNOT", "Hadamard", "PhaseShift", "GlobalPhase"})
    @qp.qnode(qp.device("lightning.qubit", wires=8))
    def phase_estimation():
        qp.BasisState(np.array([1, 1, 0, 0]), wires=system)
        U_t = qp.TrotterCDF(evolution_time=1.0, num_trotter_steps=4, hamiltonian=H, wires=system)
        qp.QuantumPhaseEstimation(U_t, estimation_wires=estimation)
        return qp.probs(wires=estimation)

The same algorithm can be costed for industrially relevant molecules with the
:mod:`~pennylane.estimator` module, without building the Hamiltonian or the circuit:

>>> import pennylane.estimator as qre
>>> H = qre.CDFHamiltonian(num_orbitals=76, num_fragments=150)
>>> print(qre.estimate(qre.TrotterCDF(H, num_steps=1000, order=2)))
--- Resources: ---
 Total wires: 152
   algorithmic wires: 152
   allocated wires: 0
     zero state: 0
     any state: 0
 Total gates : 4.170E+11
  'T': 3.830E+11,
  'CNOT': 1.021E+10,
  'Z': 6.794E+9,
  'S': 1.019E+10,
  'Hadamard': 6.794E+9

Quantum machine learning
------------------------

Instantaneous quantum polynomial (IQP) circuits are believed to be hard to sample from
classically, yet their expectation values can be approximated efficiently. This allows a
generative model to be *trained classically* and *deployed on quantum hardware* for sampling
with :class:`~pennylane.IQP`:

.. code-block:: python

    pattern = [[[0]], [[1]], [[2]], [[3]], [[0, 1]], [[1, 2]], [[2, 3]]]
    weights = np.linspace(0.1, 0.7, len(pattern))  # classically trained parameters

    @qp.qjit(capture=True, seed=42)
    @qp.set_shots(shots=5)
    @qp.decompose(gate_set={"RX", "IsingXX", "MultiRZ", "Hadamard", "CNOT"})
    @qp.qnode(qp.device("lightning.qubit", wires=4))
    def generate(weights):
        qp.IQP(weights, wires=range(4), pattern=pattern)
        return qp.sample()

>>> generate(weights)
Array([[0, 0, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 1],
       [0, 1, 0, 1],
       [0, 1, 0, 1]], dtype=int64)

Deploying such a model on a thousand qubits can be costed ahead of time:

>>> pattern = [[[i]] for i in range(1000)] + [[[i, i + 1]] for i in range(999)]
>>> print(qre.estimate(qre.IQP(num_wires=1000, pattern=pattern)))
--- Resources: ---
 Total wires: 1000
   algorithmic wires: 1000
   allocated wires: 0
     zero state: 0
     any state: 0
 Total gates : 9.195E+4
  'T': 8.796E+4,
  'CNOT': 1.998E+3,
  'Hadamard': 2.000E+3

Key functions and classes
-------------------------

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.Molecule
    ~pennylane.qchem.electron_integrals
    ~pennylane.qchem.factorize
    ~pennylane.CDFHamiltonian
    ~pennylane.TrotterCDF
    ~pennylane.QuantumPhaseEstimation
    ~pennylane.IQP
    ~pennylane.estimator.estimate

:html:`</div>`

See :mod:`~pennylane.qchem` and :mod:`~pennylane.estimator` for the full set of quantum
chemistry and resource estimation tools.
