.. role:: html(raw)
   :format: html

.. _intro_ref_pulse:

Pulse Control
=============

The :mod:`~.pulse` module provides functions and classes used to implement pulse-level control of quantum systems.
It contains a ``ParametrizedHamiltonian`` class to define the interaction between pulses and



The :mod:`~.pulse` module requires the external package `JAX <https://jax.readthedocs.io/en/latest/>`_, which
requires separate installation.

It contains a differentiable Hartree-Fock solver and the functionality to construct a
fully-differentiable molecular Hamiltonian that can be used as input to quantum algorithms
such as the variational quantum eigensolver (VQE) algorithm. The :mod:`~.qchem` module
also provides tools for building other observables such as molecular dipole moment, spin
and particle number observables.

Creating a parametrized Hamiltonian
-----------------------------------

The :mod:`~.qchem` module provides access to a driver function :func:`~.molecular_hamiltonian`
to generate the electronic Hamiltonian in a single call. For example,

.. code-block:: python

    import pennylane as qml
    from jax import numpy as jnp

    symbols = ["H", "H"]
    geometry = np.array([[0., 0., -0.66140414], [0., 0., 0.66140414]])
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)


  ```pycon
  f1 = lambda p, t: p * np.sin(t) * (t - 1)
  f2 = lambda p, t: p[0] * np.cos(p[1]* t ** 2)

  XX = qml.PauliX(1) @ qml.PauliX(1)
  YY = qml.PauliY(0) @ qml.PauliY(0)
  ZZ = qml.PauliZ(0) @ qml.PauliZ(1)

  H =  2 * XX + f1 * YY + f2 * ZZ
  ```
  ```pycon
  >>> H
  ParametrizedHamiltonian: terms=3
  >>> params = [1.2, [2.3, 3.4]]
  >>> H(params, t=0.5)
  (2*(PauliX(wires=[1]) @ PauliX(wires=[1]))) + ((-0.2876553535461426*(PauliY(wires=[0]) @ PauliY(wires=[0]))) + (1.5179612636566162*(PauliZ(wires=[0]) @ PauliZ(wires=[1]))))
  ```

  The same `ParametrizedHamiltonian` can also be constructed via a list of coefficients and operators:

  ```pycon
  >>> coeffs = [2, f1, f2]
  >>> ops = [XX, YY, ZZ]
  >>> H =  qml.ops.dot(coeffs, ops)
  ```

where:

* ``hamiltonian`` is the qubit Hamiltonian of the molecule represented as a PennyLane Hamiltonian and

* ``qubits`` is the number of qubits needed to perform the quantum simulation.

The :func:`~.molecular_hamiltonian` function can also be used to construct the molecular Hamiltonian
with an external backend that uses the
`OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`_ plugin interfaced with the
electronic structure package `PySCF <https://github.com/sunqm/pyscf>`_, which requires separate
installation. This backend is non-differentiable and can be selected by setting
``method='pyscf'`` in :func:`~.molecular_hamiltonian`. Additionally, if the electronic Hamiltonian
is built independently using `OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools, it
can be readily converted to a PennyLane observable using the
:func:`~.pennylane.import_operator` function.

Furthermore, the net charge,
the `spin multiplicity <https://en.wikipedia.org/wiki/Multiplicity_(chemistry)>`_, the
`atomic basis functions <https://www.basissetexchange.org/>`_ and the active space can also be
specified for each backend.

.. code-block:: python

    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        charge=0,
        mult=1,
        basis='sto-3g',
        method='pyscf',
        active_electrons=2,
        active_orbitals=2
    )

Importing molecular structure data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The atomic structure of a molecule can be either defined as an array or imported from an external
file using the :func:`~.read_structure` function:

.. code-block:: python

    symbols, geometry = qml.qchem.read_structure('h2.xyz')


VQE simulations
---------------

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical computational scheme,
where a quantum computer is used to prepare the trial wave function of a molecule and to measure
the expectation value of the *electronic Hamiltonian*, while a classical optimizer is used to
find its ground state.

PennyLane supports treating Hamiltonians just like any other observable, and the
expectation value of a Hamiltonian can be calculated using ``qml.expval``:

.. code-block:: python

    dev = qml.device('default.qubit', wires=4)

    symbols = ["H", "H"]
    geometry = np.array([[0., 0., -0.66140414], [0., 0., 0.66140414]])
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)

    @qml.qnode(dev)
    def circuit(params):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        qml.DoubleExcitation(params, wires=[0, 1, 2, 3])
        return qml.expval(hamiltonian)

    params = np.array(0.20885146442480412, requires_grad=True)
    circuit(params)

.. code-block:: text

    tensor(-1.13618912, requires_grad=True)

The circuit parameter can be optimized using the interface of choice.

.. note::

    For more details on VQE and the quantum chemistry functionality available in
    :mod:`~pennylane.qchem`, check out the PennyLane quantum chemistry tutorials.


Convenience functions for building hamiltonians
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.pulse.constant
    ~pennylane.pulse.pwc
    ~pennylane.pulse.pwc_from_function
    ~pennylane.qchem.rect


:html:`</div>`

Quantum chemistry functions and classes
---------------------------------------

PennyLane supports the following quantum chemistry functions and classes.