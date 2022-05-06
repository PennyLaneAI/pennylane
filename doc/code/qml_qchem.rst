qml.qchem
=========

Overview
--------

The quantum chemistry module provides the functionality to perform Hartree-Fock calculations
and construct observables such as molecular Hamiltonians as well as dipole moment, spin and particle
number observables.

.. currentmodule:: pennylane.qchem

.. automodapi:: pennylane.qchem
    :no-heading:
    :include-all-objects:
    :no-inheritance-diagram:
    :skip: taper, symmetry_generators, paulix_ops, import_operator

Differentiable Hartree-Fock
---------------------------

The differentiable Hartree-Fock (HF) solver allows performing differentiable HF calculations and
computing exact gradients with respect to
molecular geometry, basis set, and circuit parameters simultaneously using the techniques of
automatic differentiation available in `Autograd <https://github.com/HIPS/autograd>`__. This makes
the solver more versatile and robust compared to non-differentiable tools that mainly rely
on numerical approaches for computing gradients, which can lead to inaccuracies and instability.
Additionally, optimizing the basis set parameters allows reaching lower ground-state energies
without increasing the size of the basis set. Overall, the solver allows users to execute end-to-end
differentiable algorithms for quantum chemistry.

The differentiable HF solver computes the integrals over basis functions, constructs the relevant
matrices, and performs self-consistent-field iterations to obtain a set of optimized molecular
orbital coefficients. These coefficients and the computed integrals over basis functions are used to
construct the one- and two-body electron integrals in the molecular orbital basis, which can be
used to generate differentiable second-quantized Hamiltonians and dipole moments in the fermionic
and qubit basis.

The following code shows the construction of the Hamiltonian for the hydrogen molecule where the
geometry of the molecule and the basis set parameters are all differentiable.

.. code-block:: python3

    import pennylane as qml
    from pennylane import numpy as np

    symbols = ["H", "H"]
    # This initial geometry is suboptimal and will be optimized by the algorithm
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=True)

    # The exponents and contraction coefficients of the Gaussian basis functions
    alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
                      [3.42525091, 0.62391373, 0.1688554]], requires_grad = True)
    coeff = np.array([[0.15432897, 0.53532814, 0.44463454],
                      [0.15432897, 0.53532814, 0.44463454]], requires_grad = True)

We then construct the Hamiltonian.

.. code-block:: python3

    args = [geometry, alpha, coeff] # initial values of the differentiable parameters

    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, alpha=alpha, coeff=coeff, args=args)

>>> print(hamiltonian)
  (-0.35968235922631075) [I0]
+ (-0.11496335836166222) [Z2]
+ (-0.11496335836166222) [Z3]
+ (0.13082414303722753) [Z1]
+ (0.13082414303722759) [Z0]
+ (0.1031689825163302) [Z0 Z2]
+ (0.1031689825163302) [Z1 Z3]
+ (0.15329959994281844) [Z0 Z3]
+ (0.15329959994281844) [Z1 Z2]
+ (0.15405495855815063) [Z0 Z1]
+ (0.1609686663987323) [Z2 Z3]
+ (-0.05013061742648825) [Y0 Y1 X2 X3]
+ (-0.05013061742648825) [X0 X1 Y2 Y3]
+ (0.05013061742648825) [Y0 X1 X2 Y3]
+ (0.05013061742648825) [X0 Y1 Y2 X3]

The generated Hamiltonian can be used in a circuit where the molecular geometry, the basis set
parameters, and the circuit parameters are optimized simultaneously. Further information about
molecular geometry optimization with PennyLane is provided in this
`paper <https://arxiv.org/abs/2106.13840>`__ and this
`demo <https://pennylane.ai/qml/demos/tutorial_mol_geo_opt.html>`__.

.. code-block:: python3

    dev = qml.device("default.qubit", wires=4)
    hf_state = np.array([1, 1, 0, 0])
    params = [np.array([0.0], requires_grad=True)] # initial values of the circuit parameters

    def generate_circuit(mol):
        @qml.qnode(dev)
        def circuit(*args):
            qml.BasisState(hf_state, wires=[0, 1, 2, 3])
            qml.DoubleExcitation(*args[0][0], wires=[0, 1, 2, 3])
            return qml.expval(qml.qchem.molecular_hamiltonian(mol.symbols, mol.coordinates, alpha=mol.alpha, coeff=mol.coeff, args=args[1:])[0])
        return circuit

Now that the circuit is defined, we can create a geometry and parameter optimization loop. For
convenience, we create a molecule object that stores the molecular parameters.

.. code-block:: python3

    for n in range(21):

        mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
        args = [params, geometry, alpha, coeff] # initial values of the differentiable parameters

        # compute gradients with respect to the circuit parameters and update the parameters
        g_params = qml.grad(generate_circuit(mol), argnum = 0)(*args)
        params = params - 0.25 * g_params[0]

        # compute gradients with respect to the nuclear coordinates and update geometry
        g_coor = qml.grad(generate_circuit(mol), argnum = 1)(*args)
        geometry = geometry - 0.5 * g_coor

        # compute gradients with respect to the Gaussian exponents and update the exponents
        g_alpha = qml.grad(generate_circuit(mol), argnum = 2)(*args)
        alpha = alpha - 0.25 * g_alpha

        # compute gradients with respect to the Gaussian contraction coefficients and update them
        g_coeff = qml.grad(generate_circuit(mol), argnum = 3)(*args)
        coeff = coeff - 0.25 * g_coeff

        if n%5 == 0:
            print(f'Step: {n}, Energy: {generate_circuit(mol)(*args)}, Maximum Absolute Force: {abs(g_coor).max()}')


Running this optimization, we get the following output in atomic units:

.. code-block:: text

    Step: 0, Energy: -1.0491709019856188, Maximum Absolute Force: 0.1580194718925249
    Step: 5, Energy: -1.1349862621177522, Maximum Absolute Force: 0.037660768852544046
    Step: 10, Energy: -1.1399960666483346, Maximum Absolute Force: 0.005175323916673413
    Step: 15, Energy: -1.140321384816611, Maximum Absolute Force: 0.0004138319900744425
    Step: 20, Energy: -1.1403680839339787, Maximum Absolute Force: 8.223248376348913e-06

Note that the computed energy is lower than the Full-CI energy, -1.1373060483 Ha, obtained without
optimizing the basis set parameters.

The components of the HF solver can also be differentiated individually. For instance, the overlap
integral can be differentiated with respect to the basis set parameters as follows

.. code-block:: python3

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=False)
    alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554]], requires_grad = True)
    coeff = np.array([[0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454]], requires_grad = True)

    mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args = [alpha, coeff]

    a = mol.basis_set[0]
    b = mol.basis_set[1]

    g_alpha = qml.grad(qml.qchem.overlap_integral(a, b), argnum = 0)(*args)
    g_coeff = qml.grad(qml.qchem.overlap_integral(a, b), argnum = 1)(*args)

>>> print(g_alpha)
[[ 0.00169332 -0.14826928 -0.37296693]
 [ 0.00169332 -0.14826928 -0.37296693]]


OpenFermion-PySCF backend
-------------------------

The :func:`~.molecular_hamiltonian` function can be also used to construct the molecular Hamiltonian
with a non-differentiable backend that uses the
`OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`_ plugin interfaced with the
electronic structure package `PySCF <https://github.com/sunqm/pyscf>`_. The non-differentiable
backend can be selected by setting ``method='pyscf'`` in :func:`~.molecular_hamiltonian`:

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        charge=0,
        mult=1,
        basis='sto-3g',
        method='pyscf'
    )

The non-differentiable backend requires the ``OpenFermion-PySCF`` plugin to be installed by the user
with

.. code-block:: bash

    pip install openfermionpyscf
