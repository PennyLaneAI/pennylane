qml.hf
======

.. currentmodule:: pennylane.hf

.. automodapi:: pennylane.hf

Overview
--------

This module provides the functionality to perform differentiable Hartree-Fock (HF) calculations and
construct molecular Hamiltonians that can be differentiated with respect to nuclear coordinates and
basis-set parameters.

Usage details
-------------

The HF solver computes the integrals over basis functions, constructs the relevant matrices, and
performs self-consistent-field iterations to obtain a set of optimized molecular orbital
coefficients. These coefficients and the computed integrals over basis functions are used to
construct the one- and two-body electron integrals in the molecular orbital basis which can be
used to generate a differentiable second-quantized Hamiltonian in the fermionic and qubit basis.

The following code shows the construction of the Hamiltonian for the hydrogen molecule where the
geometry of the molecule and the basis set parameters are all differentiable.

.. code-block:: python3

    import pennylane as qml
    from pennylane import numpy as np

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=True)
    alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
                      [3.42525091, 0.62391373, 0.1688554]], requires_grad = True)
    coeff = np.array([[0.15432897, 0.53532814, 0.44463454],
                      [0.15432897, 0.53532814, 0.44463454]], requires_grad = True)

    # we create a molecule object with differentiable atomic coordinates and basis set parameters
    # alpha and coeff are the exponentents and contraction coefficients of the Gaussian functions
    mol = qml.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args = [geometry, alpha, coeff] # initial values of the differentiable parameters

    hamiltonian = qml.hf.generate_hamiltonian(mol)(*args)

The generated Hamiltonian can be used in a circuit where the molecular geometry, the basis set
parameters and the circuit parameters are optimized simultaneously.

.. code-block:: python3
    import autograd

    params = [np.array([0.0], requires_grad=True)]
    dev = qml.device("default.qubit", wires=4)
    hf_state = np.array([1, 1, 0, 0])

    def generate_circuit(mol):
        @qml.qnode(dev)
        def circuit(*args):
            qml.BasisState(hf_state, wires=[0, 1, 2, 3])
            qml.DoubleExcitation(*args[0][0], wires=[0, 1, 2, 3])
            return qml.expval(hf.generate_hamiltonian(mol)(*args[1:]))
        return circuit

    for n in range(10): # geometry and parameter optimization loop

        # we create a molecule object with differentiable atomic coordinates and basis set parameters
        # alpha and coeff are the exponentents and contraction coefficients of the Gaussian functions
        mol = hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
        args_ = [params, *args] # initial values of the differentiable parameters

        # compute gradients with respect to the circuit parameters and update the parameters
        g_params = autograd.grad(generate_circuit(mol), argnum = 0)(*args_)
        params = params - 0.1 * g_params[0]

        # compute gradients with respect to the nuclear coordinates and update geometry
        forces = autograd.grad(generate_circuit(mol), argnum = 1)(*args_)
        geometry = geometry - 0.5 * forces

        # compute gradients with respect to the Gaussian exponents and update the exponents
        g_alpha = autograd.grad(generate_circuit(mol), argnum = 2)(*args_)
        alpha = alpha - 0.1 * g_alpha

        # compute gradients with respect to the Gaussian contraction coefficients and update them
        g_coeff = autograd.grad(generate_circuit(mol), argnum = 3)(*args_)
        coeff = coeff - 0.1 * g_coeff

The components of the HF solver can also be differentiated individually. For instance, the overlap
integral can be differentiated with respect to the basis set parameters as

.. code-block:: python3
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=False)
    alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554]], requires_grad = True)
    coeff = np.array([[0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454]], requires_grad = True)

    mol = qml.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args = [alpha, coeff]

    a = mol.basis_set[0]
    b = mol.basis_set[1]

    g_alpha = autograd.grad(qml.hf.generate_overlap(a, b), argnum = 0)(*args)
    g_coeff = autograd.grad(qml.hf.generate_overlap(a, b), argnum = 1)(*args)
